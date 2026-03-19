"""
Large-scale data generator — simulates 1M+ audio feature samples
with accent / noise variations for trigger classification.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from tqdm import trange

from src.features.audio_utils import AudioUtils
from src.features.extractor import FeatureExtractor


# Trigger-class prototypes — each class has a characteristic spectral
# "signature" expressed as mean / std offsets applied to a base waveform.
_CLASS_SIGNATURES: Dict[int, Dict[str, float]] = {
    0: {"freq_center": 300, "bandwidth": 80, "amplitude": 0.7},   # wake-word A
    1: {"freq_center": 800, "bandwidth": 120, "amplitude": 0.9},  # wake-word B
    2: {"freq_center": 1500, "bandwidth": 200, "amplitude": 0.6}, # command
    3: {"freq_center": 3000, "bandwidth": 400, "amplitude": 0.5}, # noise / negative
}


class DataGenerator:
    """Generate synthetic audio-like samples at scale.

    Parameters
    ----------
    config : dict
        Full project config dict.
    """

    def __init__(self, config: Dict[str, Any]):
        self.data_cfg = config.get("data", {})
        self.audio_cfg = config.get("audio", {})
        self.feat_cfg = config.get("features", {})

        self.num_samples = int(self.data_cfg.get("num_samples", 1_000_000))
        self.batch_size = int(self.data_cfg.get("batch_size", 4096))
        self.num_classes = int(self.data_cfg.get("num_classes", 4))
        self.accents: List[str] = self.data_cfg.get(
            "accents", ["neutral", "british", "indian", "australian"]
        )
        self.noise_levels: List[float] = self.data_cfg.get(
            "noise_levels", [40, 30, 20, 10]
        )
        self.sr: int = int(self.audio_cfg.get("sample_rate", 16_000))
        self.duration: float = float(self.audio_cfg.get("duration_sec", 2.0))
        self.signal_len = int(self.sr * self.duration)

        self.train_split = float(self.data_cfg.get("train_split", 0.7))
        self.val_split = float(self.data_cfg.get("val_split", 0.15))
        self.output_dir = self.data_cfg.get("output_dir", "data")

        self.extractor = FeatureExtractor(self.feat_cfg, self.sr)
        self.feature_dim = self.extractor.feature_dim

        logger.info(
            "DataGenerator ready — {} samples, {} classes, {} accents",
            self.num_samples,
            self.num_classes,
            len(self.accents),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate feature matrix X, labels y, accent ids, and noise ids.

        Returns
        -------
        X : (N, feature_dim)
        y : (N,)
        accent_ids : (N,)
        noise_ids : (N,)
        """
        N = num_samples or self.num_samples
        X = np.empty((N, self.feature_dim), dtype=np.float32)
        y = np.empty(N, dtype=np.int64)
        accent_ids = np.empty(N, dtype=np.int32)
        noise_ids = np.empty(N, dtype=np.int32)

        logger.info("Generating {} samples in batches of {} …", N, self.batch_size)

        idx = 0
        for start in trange(0, N, self.batch_size, desc="Generating data"):
            end = min(start + self.batch_size, N)
            bs = end - start

            labels = np.random.randint(0, self.num_classes, size=bs)
            accent_indices = np.random.randint(0, len(self.accents), size=bs)
            noise_indices = np.random.randint(0, len(self.noise_levels), size=bs)

            for i in range(bs):
                sig = self._synthesize_signal(int(labels[i]))
                accent = self.accents[accent_indices[i]]
                snr = self.noise_levels[noise_indices[i]]

                sig = AudioUtils.apply_accent_perturbation(sig, accent, self.sr)
                sig = AudioUtils.add_noise(sig, snr)
                sig = AudioUtils.normalize(sig)

                X[idx] = self.extractor.extract_all(sig)
                y[idx] = labels[i]
                accent_ids[idx] = accent_indices[i]
                noise_ids[idx] = noise_indices[i]
                idx += 1

        return X, y, accent_ids, noise_ids

    def generate_fast(
        self,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fast path: directly synthesize feature vectors analytically
        (bypasses waveform → feature extraction for 100x speed on 1M+ samples).
        """
        N = num_samples or self.num_samples
        logger.info("Fast-generating {} feature vectors in batches of {} …", N, self.batch_size)

        y = np.random.randint(0, self.num_classes, size=N).astype(np.int64)
        accent_ids = np.random.randint(0, len(self.accents), size=N).astype(np.int32)
        noise_ids = np.random.randint(0, len(self.noise_levels), size=N).astype(np.int32)

        X = np.empty((N, self.feature_dim), dtype=np.float32)

        for start in trange(0, N, self.batch_size, desc="Fast-generating data"):
            end = min(start + self.batch_size, N)
            bs = end - start
            
            y_batch = y[start:end]
            acc_batch = accent_ids[start:end]
            noise_batch = noise_ids[start:end]
            
            X_batch = np.empty((bs, self.feature_dim), dtype=np.float32)

            for cls_id in range(self.num_classes):
                mask = y_batch == cls_id
                n_cls = int(mask.sum())
                if n_cls == 0:
                    continue

                sig = _CLASS_SIGNATURES[cls_id % len(_CLASS_SIGNATURES)]
                base = np.random.randn(n_cls, self.feature_dim).astype(np.float32)

                # Class-specific bias on first few features
                base[:, :self.extractor.n_mfcc] += sig["freq_center"] / 1000.0
                base[:, self.extractor.n_mfcc: self.extractor.n_mfcc * 2] *= sig["bandwidth"] / 200.0
                base *= sig["amplitude"]

                # Accent perturbation (shift a subset of features)
                for acc_idx, accent in enumerate(self.accents):
                    acc_mask = acc_batch[mask] == acc_idx
                    if acc_mask.sum() == 0:
                        continue
                    shift = (acc_idx - 1) * 0.15
                    base[acc_mask, :4] += shift

                # Noise perturbation
                for nidx, snr in enumerate(self.noise_levels):
                    n_mask = noise_batch[mask] == nidx
                    if n_mask.sum() == 0:
                        continue
                    noise_scale = 1.0 / (snr / 10.0)
                    base[n_mask] += np.random.randn(int(n_mask.sum()), self.feature_dim).astype(np.float32) * noise_scale

                X_batch[mask] = base
                
            X[start:end] = X_batch

        return X, y, accent_ids, noise_ids

    def save(
        self,
        X: np.ndarray,
        y: np.ndarray,
        accent_ids: np.ndarray,
        noise_ids: np.ndarray,
    ) -> None:
        """Persist generated data to disk as ``.npz``."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Split
        N = len(y)
        idx = np.random.permutation(N)
        n_train = int(N * self.train_split)
        n_val = int(N * self.val_split)

        splits = {
            "train": idx[:n_train],
            "val": idx[n_train: n_train + n_val],
            "test": idx[n_train + n_val:],
        }

        for name, sidx in splits.items():
            path = os.path.join(self.output_dir, f"{name}.npz")
            np.savez_compressed(
                path,
                X=X[sidx],
                y=y[sidx],
                accent_ids=accent_ids[sidx],
                noise_ids=noise_ids[sidx],
            )
            logger.info("Saved {} split → {} ({} samples)", name, path, len(sidx))

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _synthesize_signal(self, class_id: int) -> np.ndarray:
        """Create a synthetic waveform for a given trigger class."""
        sig_params = _CLASS_SIGNATURES.get(
            class_id % len(_CLASS_SIGNATURES),
            _CLASS_SIGNATURES[0],
        )
        t = np.linspace(0, self.duration, self.signal_len, endpoint=False)
        freq = sig_params["freq_center"] + np.random.randn() * sig_params["bandwidth"]
        amplitude = sig_params["amplitude"] + np.random.randn() * 0.05

        # Base tone + harmonics
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        signal += 0.3 * amplitude * np.sin(2 * np.pi * 2 * freq * t)
        signal += 0.1 * amplitude * np.sin(2 * np.pi * 3 * freq * t)

        # Envelope
        envelope = np.exp(-t * (1.0 + np.random.rand()))
        signal *= envelope

        return signal.astype(np.float32)
