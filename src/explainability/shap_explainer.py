"""
SHAP-based model explainability.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger


class SHAPExplainer:
    """Compute and visualise SHAP values for trigger classifiers.

    Parameters
    ----------
    config : dict
        Full project configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.shap_cfg = config.get("explainability", {}).get("shap", {})
        self.max_samples = self.shap_cfg.get("max_samples", 500)
        self.plot_type = self.shap_cfg.get("plot_type", "bar")
        self.output_dir = config.get("explainability", {}).get(
            "output_dir", "experiments/explainability"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def explain(
        self,
        model,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Compute SHAP values and generate summary plots.

        Parameters
        ----------
        model : object with ``predict`` or ``predict_proba``
        X : (N, D) background + explanation samples
        feature_names : optional list of feature names

        Returns
        -------
        shap_values : np.ndarray
        """
        import shap

        sample = X[: self.max_samples]
        logger.info("Computing SHAP values for {} samples …", len(sample))

        # Choose explainer based on model type
        explainer, shap_values = self._create_explainer_and_values(model, sample)

        # ---- Plots ----
        self._summary_plot(shap_values, sample, feature_names)
        self._bar_plot(shap_values, sample, feature_names)

        return shap_values

    def explain_single(
        self,
        model,
        X_background: np.ndarray,
        x_instance: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """Generate a waterfall / force plot for a single instance."""
        import shap

        bg = X_background[: min(100, len(X_background))]
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        sv = explainer.shap_values(x_instance.reshape(1, -1))

        if isinstance(sv, list):
            sv = sv[0]

        fig, ax = plt.subplots(figsize=(12, 4))
        top_k = min(15, x_instance.shape[0])
        indices = np.argsort(np.abs(sv.flatten()))[-top_k:]
        vals = sv.flatten()[indices]
        names = (
            [feature_names[i] for i in indices]
            if feature_names
            else [f"F{i}" for i in indices]
        )

        colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.set_xlabel("SHAP value")
        ax.set_title("Single-Instance SHAP Explanation")
        path = os.path.join(self.output_dir, "shap_single_instance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Single-instance SHAP plot saved → {}", path)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _create_explainer_and_values(self, model, X):
        import shap

        # Try tree explainer first (fast for gradient-boosting / RF)
        try:
            if hasattr(model, "sklearn_model"):
                inner = model.sklearn_model.model
            elif hasattr(model, "model"):
                inner = model.model
            else:
                inner = model

            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(X)
            logger.info("Using TreeExplainer")
            return explainer, shap_values
        except Exception:
            pass

        # Fallback to KernelExplainer
        logger.info("Falling back to KernelExplainer")
        bg = shap.sample(X, min(100, len(X)))
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values

    def _summary_plot(self, shap_values, X, feature_names):
        import shap
        fig = plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
        shap.summary_plot(sv, X, feature_names=feature_names, show=False)
        path = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP summary plot saved → {}", path)

    def _bar_plot(self, shap_values, X, feature_names):
        """Custom bar plot of mean |SHAP| per feature."""
        if isinstance(shap_values, list):
            sv = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            sv = np.abs(shap_values)

        mean_abs = sv.mean(axis=0)
        top_k = min(20, len(mean_abs))
        indices = np.argsort(mean_abs)[-top_k:]

        fig, ax = plt.subplots(figsize=(10, 6))
        names = (
            [feature_names[i] for i in indices]
            if feature_names
            else [f"Feature {i}" for i in indices]
        )
        ax.barh(names, mean_abs[indices], color="#2ecc71")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top Feature Importances (SHAP)")
        path = os.path.join(self.output_dir, "shap_bar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP bar plot saved → {}", path)
