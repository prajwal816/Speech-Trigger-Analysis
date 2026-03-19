"""
Efficient data-loading utilities — NumPy-backed with optional
PyTorch DataLoader wrapping for DNN training.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
from loguru import logger


class NumpyBatchLoader:
    """Iterate over ``.npz`` data in shuffled mini-batches (no PyTorch dependency)."""

    def __init__(
        self,
        npz_path: str,
        batch_size: int = 256,
        shuffle: bool = True,
    ):
        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]
        self.accent_ids = data.get("accent_ids", np.zeros(len(self.y), dtype=np.int32))
        self.noise_ids = data.get("noise_ids", np.zeros(len(self.y), dtype=np.int32))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.y)
        logger.debug("Loaded {} from {} — {} samples", npz_path, npz_path, self.n)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(idx)
        for start in range(0, self.n, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


class TorchDatasetWrapper:
    """Wrap NumPy arrays into a ``torch.utils.data.Dataset``."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class DataLoaderFactory:
    """Create data loaders from saved ``.npz`` splits.

    Parameters
    ----------
    config : dict
        Full project config.
    """

    def __init__(self, config: Dict[str, Any]):
        self.data_dir = config.get("data", {}).get("output_dir", "data")
        self.batch_size = config.get("training", {}).get("batch_size", 256)

    def get_numpy_loader(
        self,
        split: str = "train",
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> NumpyBatchLoader:
        path = os.path.join(self.data_dir, f"{split}.npz")
        return NumpyBatchLoader(path, batch_size or self.batch_size, shuffle)

    def get_torch_loaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
    ):
        """Return PyTorch DataLoaders for train / val / test."""
        import torch
        from torch.utils.data import DataLoader

        bs = batch_size or self.batch_size
        loaders = {}
        for split in ("train", "val", "test"):
            path = os.path.join(self.data_dir, f"{split}.npz")
            if not os.path.exists(path):
                logger.warning("Split file not found: {}", path)
                continue
            data = np.load(path)
            ds = TorchDatasetWrapper(data["X"], data["y"])
            loaders[split] = DataLoader(
                ds,
                batch_size=bs,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            logger.info("PyTorch {} loader — {} batches", split, len(loaders[split]))
        return loaders

    def load_arrays(
        self, split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load raw numpy arrays for a given split."""
        path = os.path.join(self.data_dir, f"{split}.npz")
        data = np.load(path)
        return (
            data["X"],
            data["y"],
            data.get("accent_ids", np.zeros(len(data["y"]), dtype=np.int32)),
            data.get("noise_ids", np.zeros(len(data["y"]), dtype=np.int32)),
        )
