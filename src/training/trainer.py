"""
Trainer — orchestrates model training for both sklearn and DNN paths.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from src.models.classifier import DNNClassifier, HybridClassifier, SklearnClassifier
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.evaluation.metrics import MetricsCalculator


class Trainer:
    """Unified trainer for sklearn, DNN, and hybrid classifiers.

    Parameters
    ----------
    config : dict
        Full project configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.train_cfg = config.get("training", {})
        self.epochs = self.train_cfg.get("epochs", 30)
        self.lr = self.train_cfg.get("learning_rate", 1e-3)
        self.wd = self.train_cfg.get("weight_decay", 1e-4)
        self.batch_size = self.train_cfg.get("batch_size", 256)
        self.ckpt_dir = self.train_cfg.get("checkpoint_dir", "experiments/checkpoints")

        es_cfg = self.train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 5),
            min_delta=es_cfg.get("min_delta", 0.001),
            mode="min",
        )
        self.checkpoint = ModelCheckpoint(save_dir=self.ckpt_dir, mode="min")
        self.metrics = MetricsCalculator()
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ------------------------------------------------------------------ #
    # Sklearn path                                                        #
    # ------------------------------------------------------------------ #

    def train_sklearn(
        self,
        model: SklearnClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> SklearnClassifier:
        """Fit the sklearn model and report validation metrics."""
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        logger.info("Sklearn training completed in {:.1f}s", elapsed)

        if X_val is not None and y_val is not None:
            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)
            report = self.metrics.compute_all(y_val, preds, proba)
            logger.info("Validation — acc={:.4f}  roc_auc={:.4f}", report["accuracy"], report["roc_auc"])
        return model

    # ------------------------------------------------------------------ #
    # DNN path                                                            #
    # ------------------------------------------------------------------ #

    def train_dnn(
        self,
        model: DNNClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> DNNClassifier:
        """Train the DNN with mini-batch SGD, early stopping, and checkpointing."""
        import torch
        import torch.nn as nn

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        scheduler = self._build_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()

        N = len(y_train)
        for epoch in range(1, self.epochs + 1):
            model.train_mode()
            idx = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                xb = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(model.device)
                yb = torch.tensor(y_train[batch_idx], dtype=torch.long).to(model.device)

                optimizer.zero_grad()
                logits = model.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["train_loss"].append(avg_loss)

            # Validation
            val_loss = avg_loss
            val_acc = 0.0
            if X_val is not None and y_val is not None:
                model.eval_mode()
                with torch.no_grad():
                    logits_v = model.forward(X_val)
                    yv = torch.tensor(y_val, dtype=torch.long).to(model.device)
                    val_loss = criterion(logits_v, yv).item()
                    val_acc = float((logits_v.argmax(1).cpu().numpy() == y_val).mean())
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            logger.info(
                "Epoch {}/{} — train_loss={:.4f}  val_loss={:.4f}  val_acc={:.4f}",
                epoch, self.epochs, avg_loss, val_loss, val_acc,
            )

            # Callbacks
            self.checkpoint(val_loss, model.save)
            if self.early_stopping(val_loss):
                break

        return model

    # ------------------------------------------------------------------ #
    # Hybrid path                                                         #
    # ------------------------------------------------------------------ #

    def train_hybrid(
        self,
        model: HybridClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> HybridClassifier:
        """Train both components of the hybrid model."""
        logger.info("Training sklearn component …")
        self.train_sklearn(model.sklearn_model, X_train, y_train, X_val, y_val)

        logger.info("Training DNN component …")
        self.early_stopping = EarlyStopping(
            patience=self.train_cfg.get("early_stopping", {}).get("patience", 5),
            min_delta=self.train_cfg.get("early_stopping", {}).get("min_delta", 0.001),
        )
        self.train_dnn(model.dnn_model, X_train, y_train, X_val, y_val)

        if X_val is not None and y_val is not None:
            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)
            report = self.metrics.compute_all(y_val, preds, proba)
            logger.info(
                "Hybrid validation — acc={:.4f}  roc_auc={:.4f}",
                report["accuracy"],
                report["roc_auc"],
            )
        return model

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _build_scheduler(self, optimizer):
        import torch.optim.lr_scheduler as sched

        sched_cfg = self.train_cfg.get("scheduler", {})
        stype = sched_cfg.get("type", "cosine")
        if stype == "cosine":
            return sched.CosineAnnealingLR(optimizer, T_max=self.epochs)
        elif stype == "step":
            return sched.StepLR(
                optimizer,
                step_size=sched_cfg.get("step_size", 10),
                gamma=sched_cfg.get("gamma", 0.5),
            )
        elif stype == "plateau":
            return sched.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return None
