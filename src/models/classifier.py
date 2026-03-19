"""
Classifier models — Scikit-learn, DNN (PyTorch), and a Hybrid ensemble.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


# ────────────────────────────────────────────────────────────────────────
# Scikit-learn classifier
# ────────────────────────────────────────────────────────────────────────

class SklearnClassifier:
    """Gradient-boosting (or other) classifier via scikit-learn."""

    def __init__(self, config: Dict[str, Any]):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

        model_cfg = config.get("model", {}).get("sklearn", {})
        estimator = model_cfg.get("estimator", "gradient_boosting")

        if estimator == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=model_cfg.get("n_estimators", 200),
                max_depth=model_cfg.get("max_depth", 6),
                learning_rate=model_cfg.get("learning_rate", 0.1),
                random_state=config.get("project", {}).get("seed", 42),
            )
        elif estimator == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=model_cfg.get("n_estimators", 200),
                max_depth=model_cfg.get("max_depth", 6),
                random_state=config.get("project", {}).get("seed", 42),
            )
        else:
            raise ValueError(f"Unknown sklearn estimator: {estimator}")

        self.is_fitted = False
        logger.info("SklearnClassifier created — {}", estimator)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnClassifier":
        logger.info("Fitting sklearn model on {} samples …", len(y))
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Sklearn model saved → {}", path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        logger.info("Sklearn model loaded ← {}", path)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# ────────────────────────────────────────────────────────────────────────
# PyTorch DNN classifier
# ────────────────────────────────────────────────────────────────────────

class DNNClassifier:
    """Fully-connected deep classifier (PyTorch)."""

    def __init__(self, config: Dict[str, Any], input_dim: int, num_classes: int):
        import torch
        import torch.nn as nn

        model_cfg = config.get("model", {}).get("dnn", {})
        hidden_dims: List[int] = model_cfg.get("hidden_dims", [256, 128, 64])
        dropout: float = model_cfg.get("dropout", 0.3)
        use_bn: bool = model_cfg.get("batch_norm", True)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.device = self._resolve_device(config)
        self.net.to(self.device)
        logger.info(
            "DNNClassifier — {} → {} layers → {} classes  (device={})",
            input_dim,
            hidden_dims,
            num_classes,
            self.device,
        )

    @staticmethod
    def _resolve_device(config: Dict[str, Any]) -> str:
        import torch
        dev = config.get("project", {}).get("device", "auto")
        if dev == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return dev

    def parameters(self):
        return self.net.parameters()

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def forward(self, X):
        import torch
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        self.net.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return F.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.net.state_dict(), path)
        logger.info("DNN model saved → {}", path)

    def load(self, path: str) -> None:
        import torch
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("DNN model loaded ← {}", path)


# ────────────────────────────────────────────────────────────────────────
# Hybrid ensemble
# ────────────────────────────────────────────────────────────────────────

class HybridClassifier:
    """Ensemble that averages predictions from a sklearn model and a DNN.

    The sklearn model is used for explainability (feature importances,
    SHAP tree explainer) while the DNN provides higher capacity.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        input_dim: int,
        num_classes: int,
        sklearn_weight: float = 0.4,
        dnn_weight: float = 0.6,
    ):
        self.sklearn_model = SklearnClassifier(config)
        self.dnn_model = DNNClassifier(config, input_dim, num_classes)
        self.w_sk = sklearn_weight
        self.w_dnn = dnn_weight
        self.input_dim = input_dim
        self.num_classes = num_classes
        logger.info(
            "HybridClassifier — sklearn weight={:.1f}, DNN weight={:.1f}",
            self.w_sk,
            self.w_dnn,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_sk = self.sklearn_model.predict_proba(X)
        p_dnn = self.dnn_model.predict_proba(X)
        return self.w_sk * p_sk + self.w_dnn * p_dnn

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.sklearn_model.save(os.path.join(dir_path, "sklearn_model.pkl"))
        self.dnn_model.save(os.path.join(dir_path, "dnn_model.pt"))

    def load(self, dir_path: str) -> None:
        self.sklearn_model.load(os.path.join(dir_path, "sklearn_model.pkl"))
        self.dnn_model.load(os.path.join(dir_path, "dnn_model.pt"))
