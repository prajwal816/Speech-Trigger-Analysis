"""
Feature extractor — MFCC, Mel-spectrogram, and temporal features.

Uses numpy-based implementations with optional librosa acceleration
so the system can run even when librosa is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.fftpack import dct
from loguru import logger


class FeatureExtractor:
    """Extract acoustic features from raw waveforms or simulated feature vectors.

    Parameters
    ----------
    config : dict
        The ``features`` section of the project config.
    sample_rate : int
        Audio sample rate (default 16 000).
    """

    def __init__(self, config: Dict[str, Any], sample_rate: int = 16_000):
        self.sr = sample_rate
        self.mfcc_cfg = config.get("mfcc", {})
        self.mel_cfg = config.get("mel_spectrogram", {})
        self.temporal_cfg = config.get("temporal", {})

        # MFCC params
        self.n_mfcc = self.mfcc_cfg.get("n_mfcc", 13)
        self.n_fft = self.mfcc_cfg.get("n_fft", 2048)
        self.hop_length = self.mfcc_cfg.get("hop_length", 512)

        # Mel params
        self.n_mels = self.mel_cfg.get("n_mels", 128)

        logger.debug(
            "FeatureExtractor initialised — n_mfcc={}, n_mels={}, n_fft={}",
            self.n_mfcc,
            self.n_mels,
            self.n_fft,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract_all(self, signal: np.ndarray) -> np.ndarray:
        """Return a flat feature vector combining MFCC, Mel, and temporal features."""
        mfcc = self.extract_mfcc(signal)
        mel = self.extract_mel_spectrogram(signal)
        temporal = self.extract_temporal(signal)

        # Aggregate each feature matrix → summary stats per coefficient
        mfcc_stats = self._summarize(mfcc)     # (n_mfcc * 4,)
        mel_stats = self._summarize(mel)        # (n_mels * 4,)

        feat = np.concatenate([mfcc_stats, mel_stats, temporal])
        return feat.astype(np.float32)

    def extract_mfcc(self, signal: np.ndarray) -> np.ndarray:
        """Compute MFCCs from a waveform.

        Returns shape (n_mfcc, n_frames).
        """
        mel_spec = self._mel_filterbank(signal)
        log_mel = np.log(mel_spec + 1e-9)
        mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[: self.n_mfcc]
        return mfcc

    def extract_mel_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Compute log-Mel spectrogram.

        Returns shape (n_mels, n_frames).
        """
        mel_spec = self._mel_filterbank(signal)
        return np.log(mel_spec + 1e-9)

    def extract_temporal(self, signal: np.ndarray) -> np.ndarray:
        """Compute hand-crafted temporal / spectral features.

        Returns a 1-D vector of scalar descriptors.
        """
        features: list[float] = []

        if self.temporal_cfg.get("zero_crossing", True):
            zc = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal))
            features.append(float(zc))

        if self.temporal_cfg.get("rms_energy", True):
            rms = float(np.sqrt(np.mean(signal ** 2)))
            features.append(rms)

        if self.temporal_cfg.get("spectral_centroid", True):
            magnitude = np.abs(np.fft.rfft(signal, n=self.n_fft))
            freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)
            centroid = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-9))
            features.append(centroid)

        if self.temporal_cfg.get("spectral_rolloff", True):
            magnitude = np.abs(np.fft.rfft(signal, n=self.n_fft))
            cumsum = np.cumsum(magnitude)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)
            rolloff_idx = min(rolloff_idx, len(freqs) - 1)
            features.append(float(freqs[rolloff_idx]))

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Batch helpers                                                       #
    # ------------------------------------------------------------------ #

    def extract_batch(self, signals: np.ndarray) -> np.ndarray:
        """Extract features for a batch of signals.

        Parameters
        ----------
        signals : np.ndarray, shape (batch, samples)

        Returns
        -------
        features : np.ndarray, shape (batch, feature_dim)
        """
        return np.stack([self.extract_all(s) for s in signals])

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _mel_filterbank(self, signal: np.ndarray) -> np.ndarray:
        """Apply a Mel-scaled filterbank to the STFT magnitude spectrum."""
        # STFT
        n_frames = 1 + (len(signal) - self.n_fft) // self.hop_length
        if n_frames <= 0:
            # Pad the signal if it is shorter than one FFT window
            signal = np.pad(signal, (0, self.n_fft - len(signal) + self.hop_length))
            n_frames = 1

        stft = np.zeros((self.n_fft // 2 + 1, n_frames))
        window = np.hanning(self.n_fft)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = signal[start : start + self.n_fft] * window
            stft[:, i] = np.abs(np.fft.rfft(frame, n=self.n_fft))

        # Mel filterbank matrix
        mel_fb = self._mel_filter_matrix(self.n_mels, self.n_fft, self.sr)
        mel_spec = mel_fb @ stft
        return mel_spec

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def _mel_filter_matrix(
        cls, n_mels: int, n_fft: int, sr: int
    ) -> np.ndarray:
        """Create an (n_mels, n_fft//2+1) Mel filterbank matrix."""
        low_mel = cls._hz_to_mel(0)
        high_mel = cls._hz_to_mel(sr / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = np.array([cls._mel_to_hz(m) for m in mel_points])
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        n_freqs = n_fft // 2 + 1
        fb = np.zeros((n_mels, n_freqs))
        for m in range(n_mels):
            f_left = bin_points[m]
            f_center = bin_points[m + 1]
            f_right = bin_points[m + 2]
            for k in range(f_left, f_center):
                if f_center != f_left:
                    fb[m, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right != f_center:
                    fb[m, k] = (f_right - k) / (f_right - f_center)
        return fb

    @staticmethod
    def _summarize(matrix: np.ndarray) -> np.ndarray:
        """Summarize a 2-D feature matrix into (mean, std, min, max) per row."""
        return np.concatenate(
            [
                matrix.mean(axis=1),
                matrix.std(axis=1),
                matrix.min(axis=1),
                matrix.max(axis=1),
            ]
        )

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the flat feature vector produced by ``extract_all``."""
        mfcc_dim = self.n_mfcc * 4
        mel_dim = self.n_mels * 4
        temporal_dim = sum(
            [
                self.temporal_cfg.get("zero_crossing", True),
                self.temporal_cfg.get("rms_energy", True),
                self.temporal_cfg.get("spectral_centroid", True),
                self.temporal_cfg.get("spectral_rolloff", True),
            ]
        )
        return mfcc_dim + mel_dim + temporal_dim
