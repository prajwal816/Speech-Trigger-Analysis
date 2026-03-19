"""
Accent variation analysis — measure and visualise how model behaviour
shifts across accent groups.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.evaluation.metrics import MetricsCalculator


class AccentVariationAnalyzer:
    """Analyze model performance stratified by accent.

    Parameters
    ----------
    config : dict
        Full project configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.accents: List[str] = config.get("data", {}).get(
            "accents", ["neutral", "british", "indian", "australian"]
        )
        self.output_dir = config.get("explainability", {}).get(
            "output_dir", "experiments/explainability"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics = MetricsCalculator()

    def analyze(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        accent_ids: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-accent metrics and save visualisation.

        Returns
        -------
        results : {accent_name: {metric: value, …}, …}
        """
        results: Dict[str, Dict[str, float]] = {}

        for acc_idx, accent_name in enumerate(self.accents):
            mask = accent_ids == acc_idx
            if mask.sum() == 0:
                continue

            X_acc = X[mask]
            y_acc = y[mask]
            y_pred = model.predict(X_acc)
            y_proba = model.predict_proba(X_acc)

            report = self.metrics.compute_all(y_acc, y_pred, y_proba)
            results[accent_name] = {
                "accuracy": report["accuracy"],
                "f1_score": report["f1_score"],
                "roc_auc": report["roc_auc"],
                "n_samples": int(mask.sum()),
            }
            logger.info(
                "  {} — acc={:.4f}  f1={:.4f}  auc={:.4f}  (n={})",
                accent_name,
                report["accuracy"],
                report["f1_score"],
                report["roc_auc"],
                mask.sum(),
            )

        self._plot_accent_comparison(results)
        self._plot_accent_confusion(model, X, y, accent_ids)
        self._save_report(results)

        return results

    # ------------------------------------------------------------------ #
    # Feature-level accent analysis                                       #
    # ------------------------------------------------------------------ #

    def feature_distribution_by_accent(
        self,
        X: np.ndarray,
        accent_ids: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 8,
    ) -> None:
        """Plot distributions of top-k features across accents."""
        D = min(top_k, X.shape[1])
        fig, axes = plt.subplots(2, (D + 1) // 2, figsize=(16, 8))
        axes = axes.flatten()

        for d in range(D):
            ax = axes[d]
            for acc_idx, accent_name in enumerate(self.accents):
                mask = accent_ids == acc_idx
                if mask.sum() == 0:
                    continue
                ax.hist(X[mask, d], bins=30, alpha=0.5, label=accent_name, density=True)
            name = feature_names[d] if feature_names else f"Feature {d}"
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=7)

        for d in range(D, len(axes)):
            axes[d].axis("off")

        fig.suptitle("Feature Distributions by Accent", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.output_dir, "accent_feature_distributions.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Accent feature distribution plot saved → {}", path)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _plot_accent_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        accents = list(results.keys())
        metrics_keys = ["accuracy", "f1_score", "roc_auc"]

        x = np.arange(len(accents))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, metric in enumerate(metrics_keys):
            values = [results[a].get(metric, 0) for a in accents]
            ax.bar(x + i * width, values, width, label=metric)

        ax.set_xticks(x + width)
        ax.set_xticklabels(accents)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance by Accent")
        ax.legend()
        path = os.path.join(self.output_dir, "accent_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Accent comparison plot saved → {}", path)

    def _plot_accent_confusion(
        self, model, X, y, accent_ids
    ) -> None:
        """Side-by-side confusion matrices per accent."""
        n_accents = len(self.accents)
        fig, axes = plt.subplots(1, n_accents, figsize=(5 * n_accents, 5))
        if n_accents == 1:
            axes = [axes]

        for acc_idx, (ax, accent_name) in enumerate(zip(axes, self.accents)):
            mask = accent_ids == acc_idx
            if mask.sum() == 0:
                ax.set_title(f"{accent_name}\n(no samples)")
                continue
            y_pred = model.predict(X[mask])
            cm = self.metrics.confusion_matrix(y[mask], y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_title(accent_name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        fig.suptitle("Confusion Matrices by Accent", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(self.output_dir, "accent_confusion_matrices.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Accent confusion matrices saved → {}", path)

    def _save_report(self, results: Dict[str, Dict[str, float]]) -> None:
        path = os.path.join(self.output_dir, "accent_analysis_report.txt")
        with open(path, "w") as f:
            f.write("Accent Variation Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            for accent, metrics in results.items():
                f.write(f"Accent: {accent}\n")
                for k, v in metrics.items():
                    f.write(f"  {k:15s}: {v}\n")
                f.write("\n")
        logger.info("Accent analysis report saved → {}", path)
