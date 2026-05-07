"""
Trained Consistency Predictor for Tool-Calling Prompts

ML-based predictor (GBM) that estimates the probability of inconsistent
LLM outputs from prompt + tool definitions alone — no generation needed.

Trained on 225K+ consistency measurements across 21 LLMs using 120 raw features
(linguistic, schema, schema-interaction, and task-schema mismatch).

Usage:
    from sted import ConsistencyPredictor

    # Load pre-trained model
    predictor = ConsistencyPredictor.load("models/consistency_predictor.joblib")

    # Predict
    result = predictor.predict(prompt="Find me a hotel", tools=[...])
    print(result.inconsistency_prob)   # 0.72
    print(result.risk_level)           # "high"
    print(result.predicted_cmean)      # 0.83
    print(result.top_factors)          # [("schema_total_params", 0.15), ...]

    # Train from scratch
    predictor = ConsistencyPredictor()
    predictor.train(features_csv, consistency_json)
    predictor.save("models/consistency_predictor.joblib")
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None


@dataclass
class PredictionResult:
    """Result of a consistency prediction."""
    predicted_cmean: float          # Predicted c_mean (0-1, higher = more consistent)
    inconsistency_prob: float       # P(c_mean < threshold), from classifier
    risk_level: str                 # "low", "medium", "high"
    top_factors: List[Tuple[str, float]]  # Top contributing features [(name, importance)]
    features: Dict[str, float] = field(default_factory=dict)  # Raw feature values

    def to_dict(self) -> dict:
        return {
            "predicted_cmean": round(self.predicted_cmean, 4),
            "inconsistency_prob": round(self.inconsistency_prob, 4),
            "risk_level": self.risk_level,
            "top_factors": [(n, round(v, 4)) for n, v in self.top_factors],
        }


class ConsistencyPredictor:
    """
    ML-based consistency predictor using GBM trained on empirical data.

    Two models:
      - Regressor: predicts c_mean directly (GradientBoostingRegressor)
      - Classifier: predicts P(inconsistent) via predict_proba (GradientBoostingClassifier)

    Features are extracted from prompt + tool definitions using the
    120-feature linguistic/schema/SI/TSM pipeline from the predictor experiments.
    """

    # Default hyperparameters (from exp4 best config)
    DEFAULT_REG_PARAMS = dict(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    DEFAULT_CLF_PARAMS = dict(
        n_estimators=100, max_depth=3, random_state=42,
    )
    DEFAULT_THRESHOLD = 0.9  # c_mean threshold for binary classification

    def __init__(
        self,
        regressor=None,
        classifier=None,
        scaler=None,
        feature_names: Optional[List[str]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        metadata: Optional[dict] = None,
    ):
        self.regressor = regressor
        self.classifier = classifier
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.threshold = threshold
        self.metadata = metadata or {}
        self._feature_extractor = None

    def _get_feature_extractor(self):
        """Lazy-load the feature extraction pipeline."""
        if self._feature_extractor is None:
            import sys
            features_dir = Path(__file__).parent.parent / "scripts" / "experiments" / "colm_consistency_predictor"
            if str(features_dir) not in sys.path:
                sys.path.insert(0, str(features_dir))
            from features.extract_all_features import extract_all_features
            self._feature_extractor = extract_all_features
        return self._feature_extractor

    def extract_features(self, prompt: str, tools: List[Dict]) -> Dict[str, float]:
        """Extract features from prompt + tools (120 raw features)."""
        extractor = self._get_feature_extractor()
        features = extractor(prompt, tools)
        return features.to_dict()

    def _features_to_array(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to array matching training feature order."""
        if not self.feature_names:
            raise ValueError("No feature_names set — model not trained or loaded")
        vec = [float(feature_dict.get(f, 0.0)) for f in self.feature_names]
        return np.array(vec).reshape(1, -1)

    def predict(self, prompt: str, tools: List[Dict]) -> PredictionResult:
        """
        Predict consistency for a single prompt + tools.

        Returns PredictionResult with predicted_cmean, inconsistency_prob,
        risk_level, and top contributing factors.
        """
        if self.regressor is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        feature_dict = self.extract_features(prompt, tools)
        X = self._features_to_array(feature_dict)

        # Scale
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Regression prediction
        predicted_cmean = float(np.clip(self.regressor.predict(X_scaled)[0], 0, 1))

        # Classification prediction
        if self.classifier is not None:
            # predict_proba returns [P(consistent), P(inconsistent)]
            proba = self.classifier.predict_proba(X_scaled)[0]
            if proba.shape[0] == 2:
                inconsistency_prob = float(proba[0])  # class 0 = inconsistent (c_mean <= threshold)
            else:
                inconsistency_prob = 1.0 - predicted_cmean
        else:
            inconsistency_prob = 1.0 - predicted_cmean

        # Risk level
        if inconsistency_prob >= 0.6:
            risk_level = "high"
        elif inconsistency_prob >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Top factors from feature importance
        top_factors = self._get_top_factors(X_scaled[0])

        return PredictionResult(
            predicted_cmean=predicted_cmean,
            inconsistency_prob=inconsistency_prob,
            risk_level=risk_level,
            top_factors=top_factors,
            features=feature_dict,
        )

    def predict_from_runs(
        self, runs: List[Any], n_probes: Optional[int] = 5,
    ) -> PredictionResult:
        """
        Predict consistency from LLM probe runs using output-based features.

        This is a lightweight, training-free predictor that directly measures
        behavioral variance from a few probe generations. It achieves AUC~0.93
        with 5 probes and generalizes across domains without retraining.

        Args:
            runs: List of generation results (tool call dicts or None for invalid)
            n_probes: Number of probe runs to use (default: 5)

        Returns:
            PredictionResult with predicted_cmean and inconsistency_prob
        """
        import sys
        features_dir = Path(__file__).parent.parent / "scripts" / "experiments" / "colm_consistency_predictor"
        if str(features_dir) not in sys.path:
            sys.path.insert(0, str(features_dir))
        from features.output_features import extract_output_features

        out_feat = extract_output_features(runs, n_probes=n_probes)

        # Primary signal: response length CV (Spearman=-0.84 with c_mean at 5 probes)
        resp_cv = out_feat["out_response_len_cv"]
        arg_cv = out_feat["out_arg_count_cv"]
        tool_var = out_feat["out_has_tool_variation"]

        # Calibrated inconsistency score (higher = more likely inconsistent)
        # Weights derived from correlation analysis on BFCL validation
        incon_score = 0.6 * min(resp_cv / 0.3, 1.0) + \
                      0.3 * min(arg_cv / 0.5, 1.0) + \
                      0.1 * tool_var

        predicted_cmean = float(np.clip(1.0 - resp_cv, 0, 1))
        inconsistency_prob = float(np.clip(incon_score, 0, 1))

        if inconsistency_prob >= 0.5:
            risk_level = "high"
        elif inconsistency_prob >= 0.2:
            risk_level = "medium"
        else:
            risk_level = "low"

        top_factors = [
            ("out_response_len_cv", resp_cv),
            ("out_arg_count_cv", arg_cv),
            ("out_has_tool_variation", tool_var),
        ]

        return PredictionResult(
            predicted_cmean=predicted_cmean,
            inconsistency_prob=inconsistency_prob,
            risk_level=risk_level,
            top_factors=top_factors,
            features=out_feat,
        )

    def predict_batch(
        self, samples: List[Dict[str, Any]],
        prompt_key: str = "query", tools_key: str = "tools",
    ) -> List[PredictionResult]:
        """Predict consistency for a batch of samples."""
        results = []
        for sample in samples:
            prompt = sample.get(prompt_key, "")
            tools = sample.get(tools_key, [])
            results.append(self.predict(prompt, tools))
        return results

    def _get_top_factors(self, x: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top contributing features for a prediction."""
        if not hasattr(self.regressor, 'feature_importances_'):
            return []

        importances = self.regressor.feature_importances_
        # Weight by actual feature value (normalized) to show sample-specific contribution
        contributions = importances * np.abs(x)
        top_idx = np.argsort(contributions)[::-1][:top_k]

        return [(self.feature_names[i], float(importances[i])) for i in top_idx]

    # ================================================================
    # Training
    # ================================================================

    def train(
        self,
        features_csv: str,
        consistency_json: str,
        threshold: Optional[float] = None,
        model_filter: Optional[str] = None,
    ) -> dict:
        """
        Train the predictor from feature CSV and consistency metrics JSON.

        Args:
            features_csv: Path to extracted_features.csv (67 features + sample_idx)
            consistency_json: Path to combined_consistency_metrics_results.json
            threshold: c_mean threshold for binary classification (default: 0.9)
            model_filter: If set, train only on this model's data

        Returns:
            dict with training metrics (R², F1, AUC, etc.)
        """
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import r2_score, f1_score, roc_auc_score
        from scipy.stats import pearsonr

        import pandas as pd

        if threshold is not None:
            self.threshold = threshold

        # Load data
        features_df = pd.read_csv(features_csv)
        with open(consistency_json) as f:
            consistency_data = json.load(f)

        # Feature columns
        feature_cols = [
            c for c in features_df.columns
            if c not in ('sample_idx', 'sample_id')
            and features_df[c].dtype in ('float64', 'int64', 'float32', 'int32')
        ]

        # Remove zero-variance
        variances = features_df[feature_cols].var()
        zero_var = variances[variances == 0].index.tolist()
        feature_cols = [c for c in feature_cols if c not in zero_var]

        self.feature_names = feature_cols

        # Build training data
        X_all, y_all, groups_all = [], [], []

        models_to_use = ([model_filter] if model_filter else
                         sorted(consistency_data.keys()))

        for model_name in models_to_use:
            if model_name not in consistency_data:
                continue
            samples = consistency_data[model_name]
            model_df = pd.DataFrame(samples)
            agg = model_df.groupby('sample_idx').agg({'c_mean': 'mean'}).reset_index()
            agg = agg.rename(columns={'c_mean': 'c_mean_avg'})

            merged = features_df.merge(agg, on='sample_idx', how='inner')
            if len(merged) < 10:
                continue

            X_all.append(merged[feature_cols].values)
            y_all.append(merged['c_mean_avg'].values)
            groups_all.append(merged['sample_idx'].values)

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        groups = np.concatenate(groups_all)

        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(models_to_use)} model(s)")

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train regressor
        self.regressor = GradientBoostingRegressor(**self.DEFAULT_REG_PARAMS)
        self.regressor.fit(X_scaled, y)

        # Train classifier
        y_binary = (y > self.threshold).astype(int)  # 1 = consistent, 0 = inconsistent
        if len(np.unique(y_binary)) == 2:
            self.classifier = GradientBoostingClassifier(**self.DEFAULT_CLF_PARAMS)
            self.classifier.fit(X_scaled, y_binary)

        # Cross-validated metrics
        metrics = self._evaluate_cv(X, y, groups)

        self.metadata = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_models": len(models_to_use),
            "threshold": self.threshold,
            "zero_variance_removed": zero_var,
            "cv_metrics": metrics,
        }

        return metrics

    def train_from_raw(
        self,
        raw_results_json: str,
        threshold: Optional[float] = None,
    ) -> dict:
        """
        Train directly from raw LLM generation results (e.g., BFCL all_results.json).

        Extracts features on-the-fly and uses per-sample c_mean as target.

        Args:
            raw_results_json: Path to all_results.json with query, tools, and
                              consistency_metrics per sample
            threshold: c_mean threshold for binary classification
        """
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        if threshold is not None:
            self.threshold = threshold

        with open(raw_results_json) as f:
            raw = json.load(f)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw

        print(f"Extracting features from {len(results)} samples...")
        extractor = self._get_feature_extractor()

        feature_dicts = []
        y_vals = []
        for i, sample in enumerate(results):
            query = sample.get("query", "")
            tools = sample.get("tools", [])
            c_metrics = sample.get("consistency_metrics", {})
            c_mean = c_metrics.get("c_mean")
            if c_mean is None:
                continue

            features = extractor(query, tools)
            feature_dicts.append(features.to_dict())
            y_vals.append(c_mean)

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(results)}]")

        if not feature_dicts:
            raise ValueError("No samples with consistency_metrics found")

        # Build feature matrix
        self.feature_names = sorted(feature_dicts[0].keys())
        X = np.array([[fd.get(f, 0.0) for f in self.feature_names] for fd in feature_dicts])
        y = np.array(y_vals)

        # Remove zero-variance
        variances = X.var(axis=0)
        keep = variances > 0
        self.feature_names = [f for f, k in zip(self.feature_names, keep) if k]
        X = X[:, keep]

        print(f"Training: {X.shape[0]} samples, {X.shape[1]} features")

        # Scale and train
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.regressor = GradientBoostingRegressor(**self.DEFAULT_REG_PARAMS)
        self.regressor.fit(X_scaled, y)

        y_binary = (y > self.threshold).astype(int)
        if len(np.unique(y_binary)) == 2:
            self.classifier = GradientBoostingClassifier(**self.DEFAULT_CLF_PARAMS)
            self.classifier.fit(X_scaled, y_binary)

        self.metadata = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "threshold": self.threshold,
            "source": str(raw_results_json),
        }

        return {"n_samples": X.shape[0], "n_features": X.shape[1]}

    def _evaluate_cv(self, X, y, groups, n_splits=5) -> dict:
        """Run GroupKFold cross-validation and return metrics."""
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import r2_score, f1_score, roc_auc_score
        from scipy.stats import pearsonr

        gkf = GroupKFold(n_splits=n_splits)
        r2s, pearsons, f1s, aucs = [], [], [], []

        y_binary = (y > self.threshold).astype(int)
        has_both_classes = len(np.unique(y_binary)) == 2

        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Regression
            reg = GradientBoostingRegressor(**self.DEFAULT_REG_PARAMS)
            reg.fit(X_tr_s, y_tr)
            y_pred = np.clip(reg.predict(X_te_s), 0, 1)
            r2s.append(r2_score(y_te, y_pred))
            if np.std(y_pred) > 1e-10:
                pearsons.append(pearsonr(y_te, y_pred)[0])

            # Classification
            if has_both_classes:
                yb_tr, yb_te = y_binary[train_idx], y_binary[test_idx]
                if len(np.unique(yb_tr)) == 2 and len(np.unique(yb_te)) == 2:
                    clf = GradientBoostingClassifier(**self.DEFAULT_CLF_PARAMS)
                    clf.fit(X_tr_s, yb_tr)
                    yb_pred = clf.predict(X_te_s)
                    yb_proba = clf.predict_proba(X_te_s)
                    f1s.append(f1_score(yb_te, yb_pred, average='macro'))
                    if yb_proba.shape[1] == 2:
                        aucs.append(roc_auc_score(yb_te, yb_proba[:, 1]))

        return {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "pearson_mean": float(np.mean(pearsons)) if pearsons else 0.0,
            "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
            "auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        }

    # ================================================================
    # Serialization
    # ================================================================

    def save(self, path: str):
        """Save trained model to disk."""
        if joblib is None:
            raise ImportError("joblib is required to save models: pip install joblib")
        if self.regressor is None:
            raise ValueError("No trained model to save")

        data = {
            "regressor": self.regressor,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ConsistencyPredictor":
        """Load a trained model from disk."""
        if joblib is None:
            raise ImportError("joblib is required to load models: pip install joblib")

        data = joblib.load(path)
        return cls(
            regressor=data["regressor"],
            classifier=data["classifier"],
            scaler=data["scaler"],
            feature_names=data["feature_names"],
            threshold=data["threshold"],
            metadata=data.get("metadata", {}),
        )
