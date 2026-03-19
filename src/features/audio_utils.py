"""
Audio utility functions — loading, normalization, augmentation helpers.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class AudioUtils:
    """Static helper methods for raw-audio manipulation."""

    @staticmethod
    def normalize(signal: np.ndarray) -> np.ndarray:
        """Peak-normalize a waveform to [-1, 1]."""
        peak = np.max(np.abs(signal))
        if peak == 0:
            return signal
        return signal / peak

    @staticmethod
    def add_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add white Gaussian noise at a given SNR (dB)."""
        rms_signal = np.sqrt(np.mean(signal ** 2))
        rms_noise = rms_signal / (10 ** (snr_db / 20))
        noise = np.random.normal(0, rms_noise, signal.shape)
        return signal + noise

    @staticmethod
    def apply_accent_perturbation(
        signal: np.ndarray,
        accent: str,
        sr: int = 16_000,
    ) -> np.ndarray:
        """Simulate accent variation via formant-like spectral shift.

        Each accent maps to a deterministic pitch-shift factor applied via
        resampling so we stay in the time domain (no external dependency
        beyond numpy / scipy).
        """
        accent_factors: dict[str, float] = {
            "neutral": 1.00,
            "british": 1.03,
            "indian": 0.97,
            "australian": 1.05,
        }
        factor = accent_factors.get(accent, 1.0)
        if factor == 1.0:
            return signal

        # Simple time-stretch by linear interpolation then trim/pad
        indices = np.arange(0, len(signal), factor)
        indices = indices[indices < len(signal)].astype(int)
        stretched = signal[indices]

        # Match original length
        if len(stretched) < len(signal):
            stretched = np.pad(stretched, (0, len(signal) - len(stretched)))
        else:
            stretched = stretched[: len(signal)]
        return stretched

    @staticmethod
    def frame_signal(
        signal: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Slice a 1-D signal into overlapping frames.

        Returns
        -------
        frames : np.ndarray, shape (n_frames, frame_length)
        """
        n_frames = 1 + (len(signal) - frame_length) // hop_length
        indices = (
            np.arange(frame_length)[None, :]
            + hop_length * np.arange(n_frames)[:, None]
        )
        return signal[indices]

    @staticmethod
    def compute_energy(signal: np.ndarray) -> float:
        """Root-mean-square energy of a signal."""
        return float(np.sqrt(np.mean(signal ** 2)))

    @staticmethod
    def trim_silence(
        signal: np.ndarray,
        threshold_db: float = -40.0,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Remove leading/trailing silence below *threshold_db*."""
        threshold_linear = 10 ** (threshold_db / 20)
        frames = AudioUtils.frame_signal(signal, frame_length, hop_length)
        energies = np.sqrt(np.mean(frames ** 2, axis=1))
        active = np.where(energies > threshold_linear)[0]
        if len(active) == 0:
            return signal
        start = active[0] * hop_length
        end = min(active[-1] * hop_length + frame_length, len(signal))
        return signal[start:end]
