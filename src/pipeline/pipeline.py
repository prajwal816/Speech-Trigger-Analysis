"""
End-to-end pipeline — orchestrates data generation → training → evaluation
→ explainability in a single entry point.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import yaml
from loguru import logger

from src.pipeline.data_generator import DataGenerator
from src.pipeline.data_loader import DataLoaderFactory
from src.models.classifier import HybridClassifier, SklearnClassifier, DNNClassifier
from src.training.trainer import Trainer
from src.evaluation.evaluator import ModelEvaluator
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.feature_importance import FeatureImportanceAnalyzer
from src.explainability.accent_analysis import AccentVariationAnalyzer


class TriggerAnalysisPipeline:
    """Config-driven, end-to-end pipeline.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file.
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path) as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self._setup_logging()
        self.seed = self.config.get("project", {}).get("seed", 42)
        np.random.seed(self.seed)

        logger.info("Pipeline initialised from {}", config_path)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        skip_data_gen: bool = False,
        fast_mode: bool = True,
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the full pipeline.

        Parameters
        ----------
        skip_data_gen : bool
            Skip data generation (use existing ``.npz`` files).
        fast_mode : bool
            If True use the fast analytical data generator (recommended
            for 1M+ samples).
        num_samples : int, optional
            Override the number of samples from config.

        Returns
        -------
        results : dict with evaluation metrics, SHAP values, etc.
        """
        t_start = time.time()
        results: Dict[str, Any] = {}

        # ── Step 1: Data generation ──────────────────────────────────
        if not skip_data_gen:
            gen = DataGenerator(self.config)
            if fast_mode:
                X, y, accent_ids, noise_ids = gen.generate_fast(num_samples)
            else:
                X, y, accent_ids, noise_ids = gen.generate(num_samples)
            gen.save(X, y, accent_ids, noise_ids)
        else:
            logger.info("Skipping data generation — loading from disk")

        # ── Step 2: Load data ────────────────────────────────────────
        loader = DataLoaderFactory(self.config)
        X_train, y_train, acc_train, _ = loader.load_arrays("train")
        X_val, y_val, acc_val, _ = loader.load_arrays("val")
        X_test, y_test, acc_test, noise_test = loader.load_arrays("test")

        feature_dim = X_train.shape[1]
        num_classes = int(self.config.get("data", {}).get("num_classes", 4))

        logger.info(
            "Loaded splits — train={}, val={}, test={}, feature_dim={}",
            len(y_train), len(y_val), len(y_test), feature_dim,
        )

        # ── Step 3: Build model ──────────────────────────────────────
        model_type = self.config.get("model", {}).get("type", "hybrid")
        if model_type == "hybrid":
            model = HybridClassifier(self.config, feature_dim, num_classes)
        elif model_type == "sklearn":
            model = SklearnClassifier(self.config)
        elif model_type == "dnn":
            model = DNNClassifier(self.config, feature_dim, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # ── Step 4: Training ─────────────────────────────────────────
        trainer = Trainer(self.config)
        if model_type == "hybrid":
            trainer.train_hybrid(model, X_train, y_train, X_val, y_val)
        elif model_type == "sklearn":
            trainer.train_sklearn(model, X_train, y_train, X_val, y_val)
        else:
            trainer.train_dnn(model, X_train, y_train, X_val, y_val)

        # ── Step 5: Evaluation ───────────────────────────────────────
        evaluator = ModelEvaluator(self.config)
        eval_report = evaluator.evaluate(model, X_test, y_test)
        results["evaluation"] = eval_report

        # ── Step 6: Explainability ───────────────────────────────────
        feature_names = self._build_feature_names(feature_dim)

        # SHAP
        shap_explainer = SHAPExplainer(self.config)
        shap_model = model.sklearn_model if hasattr(model, "sklearn_model") else model
        shap_values = shap_explainer.explain(shap_model, X_test, feature_names)
        results["shap_values"] = shap_values

        # Feature importance
        fi_analyzer = FeatureImportanceAnalyzer(self.config)
        fi_analyzer.intrinsic_importance(model, feature_names)

        # Accent analysis
        accent_analyzer = AccentVariationAnalyzer(self.config)
        accent_results = accent_analyzer.analyze(model, X_test, y_test, acc_test)
        accent_analyzer.feature_distribution_by_accent(X_test, acc_test, feature_names)
        results["accent_analysis"] = accent_results

        elapsed = time.time() - t_start
        logger.info("Pipeline completed in {:.1f}s", elapsed)
        return results

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _build_feature_names(self, feature_dim: int) -> list[str]:
        """Build descriptive feature names matching the extractor output."""
        from src.features.extractor import FeatureExtractor
        ext = FeatureExtractor(self.config.get("features", {}))

        names: list[str] = []
        # MFCC summary stats
        for stat in ("mean", "std", "min", "max"):
            for i in range(ext.n_mfcc):
                names.append(f"mfcc_{i}_{stat}")
        # Mel summary stats
        for stat in ("mean", "std", "min", "max"):
            for i in range(ext.n_mels):
                names.append(f"mel_{i}_{stat}")
        # Temporal features
        temporal_cfg = self.config.get("features", {}).get("temporal", {})
        if temporal_cfg.get("zero_crossing", True):
            names.append("zero_crossing_rate")
        if temporal_cfg.get("rms_energy", True):
            names.append("rms_energy")
        if temporal_cfg.get("spectral_centroid", True):
            names.append("spectral_centroid")
        if temporal_cfg.get("spectral_rolloff", True):
            names.append("spectral_rolloff")

        # Pad / trim to match actual feature_dim
        if len(names) < feature_dim:
            names += [f"feature_{i}" for i in range(len(names), feature_dim)]
        return names[:feature_dim]

    def _setup_logging(self) -> None:
        log_cfg = self.config.get("logging", {})
        log_level = log_cfg.get("level", "INFO")
        log_file = log_cfg.get("log_file", "experiments/logs/run.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger.remove()
        logger.add(sys.stderr, level=log_level)
        logger.add(log_file, level=log_level, rotation="10 MB")


# ────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Speech Trigger Analysis Pipeline")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override number of samples to generate",
    )
    parser.add_argument(
        "--skip-datagen",
        action="store_true",
        help="Skip data generation (use existing .npz files)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use the slow waveform-based data generator",
    )
    args = parser.parse_args()

    pipeline = TriggerAnalysisPipeline(args.config)
    pipeline.run(
        skip_data_gen=args.skip_datagen,
        fast_mode=not args.slow,
        num_samples=args.samples,
    )


if __name__ == "__main__":
    main()
