"""
Model evaluator — runs full evaluation pipeline and saves reports.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.evaluation.metrics import MetricsCalculator


class ModelEvaluator:
    """End-to-end evaluation with metric reporting and visualisations.

    Parameters
    ----------
    config : dict
        Full project configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get("evaluation", {}).get("output_dir", "experiments/results")
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics = MetricsCalculator()
        self.class_names = [f"Class_{i}" for i in range(config.get("data", {}).get("num_classes", 4))]

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Run predictions, compute metrics, generate plots, save report."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        report = self.metrics.compute_all(y_test, y_pred, y_proba)

        logger.info("═" * 50)
        logger.info("  EVALUATION REPORT")
        logger.info("═" * 50)
        logger.info("  Accuracy  : {:.4f}", report["accuracy"])
        logger.info("  F1-Score  : {:.4f}", report["f1_score"])
        logger.info("  ROC-AUC   : {:.4f}", report["roc_auc"])
        logger.info("═" * 50)

        # Plots
        self._plot_confusion_matrix(report["confusion_matrix"])
        self._plot_roc_curve(y_test, y_proba)

        # Save JSON report
        serialisable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in report.items()
        }
        report_path = os.path.join(self.output_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Report saved → {}", report_path)

        return report

    # ------------------------------------------------------------------ #
    # Visualisation helpers                                               #
    # ------------------------------------------------------------------ #

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        path = os.path.join(self.output_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved → {}", path)

    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        classes = sorted(np.unique(y_true))
        if len(classes) < 2:
            return

        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(classes):
            if i >= y_proba.shape[1]:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{self.class_names[cls]} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.legend(loc="lower right")
        path = os.path.join(self.output_dir, "roc_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curves saved → {}", path)
