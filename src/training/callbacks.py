"""
Training callbacks — early stopping and model checkpointing.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from loguru import logger


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        ``"min"`` for loss, ``"max"`` for accuracy / AUC.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping triggered — no improvement for {} epochs",
                    self.patience,
                )
                return True
        return False


class ModelCheckpoint:
    """Save model weights whenever the monitored metric improves.

    Parameters
    ----------
    save_dir : str
        Directory to save checkpoints.
    mode : str
        ``"min"`` for loss, ``"max"`` for accuracy / AUC.
    """

    def __init__(self, save_dir: str = "experiments/checkpoints", mode: str = "min"):
        self.save_dir = save_dir
        self.mode = mode
        self.best: Optional[float] = None
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, value: float, save_fn) -> bool:
        """Call with the new metric value and a callable ``save_fn(path)``."""
        if self.best is None:
            self.best = value
            save_fn(os.path.join(self.save_dir, "best_model"))
            return True

        improved = (
            value < self.best if self.mode == "min" else value > self.best
        )
        if improved:
            self.best = value
            path = os.path.join(self.save_dir, "best_model")
            save_fn(path)
            logger.info("Checkpoint saved (metric={:.5f}) → {}", value, path)
            return True
        return False
