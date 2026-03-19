"""
Metrics calculator — accuracy, ROC-AUC, F1, confusion matrix.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from loguru import logger


class MetricsCalculator:
    """Compute classification metrics."""

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return a dict of all supported metrics."""
        results: Dict[str, Any] = {}
        results["accuracy"] = self.accuracy(y_true, y_pred)
        results["f1_score"] = self.f1_score(y_true, y_pred)
        results["confusion_matrix"] = self.confusion_matrix(y_true, y_pred)

        if y_proba is not None:
            results["roc_auc"] = self.roc_auc(y_true, y_proba)
        else:
            results["roc_auc"] = 0.0

        return results

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import f1_score as sk_f1
        return float(sk_f1(y_true, y_pred, average="weighted", zero_division=0))

    @staticmethod
    def roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        try:
            if y_proba.ndim == 1:
                return float(roc_auc_score(y_true, y_proba))
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except ValueError:
            logger.warning("ROC-AUC could not be computed (likely single class present).")
            return 0.0

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        from sklearn.metrics import confusion_matrix as sk_cm
        return sk_cm(y_true, y_pred)
