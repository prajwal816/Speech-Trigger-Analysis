"""
Feature importance analysis — model-intrinsic and permutation-based.
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


class FeatureImportanceAnalyzer:
    """Analyse and visualise feature importances.

    Supports:
    * Model-intrinsic importances (e.g. tree-based ``feature_importances_``)
    * Permutation importance (model-agnostic)

    Parameters
    ----------
    config : dict
        Full project configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.top_k = config.get("explainability", {}).get("feature_importance", {}).get("top_k", 20)
        self.output_dir = config.get("explainability", {}).get(
            "output_dir", "experiments/explainability"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Intrinsic importance                                                 #
    # ------------------------------------------------------------------ #

    def intrinsic_importance(
        self,
        model,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Extract and plot tree-based feature importances."""
        if hasattr(model, "sklearn_model"):
            imp = model.sklearn_model.feature_importances_
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            logger.warning("Model has no intrinsic feature importances.")
            return np.array([])

        self._plot_importances(imp, feature_names, "intrinsic_importance")
        self._save_report(imp, feature_names, "intrinsic_importance_report.txt")
        return imp

    # ------------------------------------------------------------------ #
    # Permutation importance                                               #
    # ------------------------------------------------------------------ #

    def permutation_importance(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 5,
    ) -> np.ndarray:
        """Compute permutation importance by shuffling one feature at a time.

        Returns
        -------
        importances : (D,) array of mean accuracy drops
        """
        from src.evaluation.metrics import MetricsCalculator

        metrics = MetricsCalculator()
        baseline_acc = metrics.accuracy(y, model.predict(X))

        D = X.shape[1]
        importances = np.zeros(D, dtype=np.float64)

        logger.info("Computing permutation importance ({} features, {} repeats) …", D, n_repeats)
        for d in range(D):
            drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                np.random.shuffle(X_perm[:, d])
                perm_acc = metrics.accuracy(y, model.predict(X_perm))
                drops.append(baseline_acc - perm_acc)
            importances[d] = np.mean(drops)

        self._plot_importances(importances, feature_names, "permutation_importance")
        self._save_report(importances, feature_names, "permutation_importance_report.txt")
        return importances

    # ------------------------------------------------------------------ #
    # Correlation analysis                                                 #
    # ------------------------------------------------------------------ #

    def feature_correlation(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute and plot feature correlation matrix."""
        corr = np.corrcoef(X.T)

        fig, ax = plt.subplots(figsize=(14, 12))
        top_k = min(self.top_k, X.shape[1])
        # Show only top_k features for readability
        sub_corr = corr[:top_k, :top_k]
        labels = (
            feature_names[:top_k]
            if feature_names
            else [f"F{i}" for i in range(top_k)]
        )
        sns.heatmap(sub_corr, xticklabels=labels, yticklabels=labels, cmap="coolwarm",
                     center=0, annot=False, ax=ax)
        ax.set_title(f"Feature Correlation (top {top_k})")
        path = os.path.join(self.output_dir, "feature_correlation.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Feature correlation plot saved → {}", path)
        return corr

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _plot_importances(
        self,
        importances: np.ndarray,
        feature_names: Optional[List[str]],
        name: str,
    ) -> None:
        top_k = min(self.top_k, len(importances))
        indices = np.argsort(importances)[-top_k:]
        vals = importances[indices]
        names = (
            [feature_names[i] for i in indices]
            if feature_names
            else [f"Feature {i}" for i in indices]
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, vals, color="#9b59b6")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-{top_k} {name.replace('_', ' ').title()}")
        path = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("{} plot saved → {}", name, path)

    def _save_report(
        self,
        importances: np.ndarray,
        feature_names: Optional[List[str]],
        filename: str,
    ) -> None:
        path = os.path.join(self.output_dir, filename)
        indices = np.argsort(importances)[::-1]
        with open(path, "w") as f:
            f.write("Rank | Feature | Importance\n")
            f.write("-" * 40 + "\n")
            for rank, idx in enumerate(indices, 1):
                name = feature_names[idx] if feature_names else f"Feature {idx}"
                f.write(f"{rank:4d} | {name:20s} | {importances[idx]:.6f}\n")
        logger.info("Report saved → {}", path)
