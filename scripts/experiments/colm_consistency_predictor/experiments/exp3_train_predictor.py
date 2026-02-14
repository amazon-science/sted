#!/usr/bin/env python3
"""
Experiment 3: Train Consistency Predictor Models

COLM 2026 Consistency Predictor

This experiment:
1. Loads extracted features and consistency metrics
2. Trains multiple predictor models (Linear, Ridge, RF, XGBoost, MLP)
3. Evaluates with cross-validation and test set
4. Compares feature category ablations
5. Reports per-model and cross-model transfer performance

Key research questions:
- Can we predict consistency from prompt features alone?
- Which model architecture works best?
- Do all feature categories contribute?
- Does the predictor transfer across LLMs?
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Check for XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available, skipping")


def load_data(exp_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load extracted features and consistency metrics."""
    features_df = pd.read_csv(exp_dir / "results" / "exp1_correlations" / "extracted_features.csv")

    metrics_path = PROJECT_ROOT / "results" / "toucan_exact_final" / "combined_consistency_metrics_results.json"
    with open(metrics_path) as f:
        metrics_data = json.load(f)

    rows = []
    for model, samples in metrics_data.items():
        for sample in samples:
            rows.append({
                'model': model,
                'sample_idx': sample['sample_idx'],
                'temperature': sample.get('temperature', 0.5),
                'c_mean': sample.get('c_mean', 0),
                'stability_score': sample.get('stability_score', 0),
            })

    metrics_df = pd.DataFrame(rows)
    return features_df, metrics_df


def prepare_dataset(
    features_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    target_metric: str = 'c_mean',
    aggregate: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare X and y for training.

    Args:
        aggregate: How to aggregate across models ('mean', 'min', 'max', 'std')
    """
    # Aggregate metrics across models
    if aggregate == 'mean':
        agg_metrics = metrics_df.groupby('sample_idx')[target_metric].mean().reset_index()
    elif aggregate == 'min':
        agg_metrics = metrics_df.groupby('sample_idx')[target_metric].min().reset_index()
    elif aggregate == 'std':
        agg_metrics = metrics_df.groupby('sample_idx')[target_metric].std().reset_index()
    else:
        agg_metrics = metrics_df.groupby('sample_idx')[target_metric].mean().reset_index()

    merged = features_df.merge(agg_metrics, on='sample_idx', how='inner')

    # Feature columns
    feature_cols = [c for c in features_df.columns if c not in ['sample_idx', 'sample_id']]

    X = merged[feature_cols].values
    y = merged[target_metric].values

    return X, y, feature_cols


def get_category_features(feature_names: List[str]) -> Dict[str, List[int]]:
    """Get indices for each feature category."""
    categories = {
        'surface': [],
        'semantic': [],
        'pragmatic': [],
        'schema': []
    }

    for i, name in enumerate(feature_names):
        for cat in categories:
            if name.startswith(cat):
                categories[cat].append(i)
                break

    return categories


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    model
) -> Dict[str, float]:
    """Train model and evaluate on test set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Clip predictions to [0, 1]
    y_pred = np.clip(y_pred, 0, 1)

    return {
        'model': model_name,
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'pearson_r': pearsonr(y_test, y_pred)[0],
        'spearman_r': spearmanr(y_test, y_pred)[0],
    }


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_folds: int = 5
) -> Dict[str, float]:
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    r2_scores = []
    pearson_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, 1)

        r2_scores.append(r2_score(y_test, y_pred))
        pearson_scores.append(pearsonr(y_test, y_pred)[0])

    return {
        'cv_r2_mean': np.mean(r2_scores),
        'cv_r2_std': np.std(r2_scores),
        'cv_pearson_mean': np.mean(pearson_scores),
        'cv_pearson_std': np.std(pearson_scores),
    }


def category_ablation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_class,
    model_kwargs: Dict = None
) -> pd.DataFrame:
    """
    Ablation study: train with each category removed.
    """
    if model_kwargs is None:
        model_kwargs = {}

    categories = get_category_features(feature_names)
    all_indices = list(range(len(feature_names)))

    results = []

    # Full model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = model_class(**model_kwargs)
    model.fit(X_train_scaled, y_train)
    y_pred = np.clip(model.predict(X_test_scaled), 0, 1)
    full_r2 = r2_score(y_test, y_pred)

    results.append({
        'ablation': 'full_model',
        'n_features': len(all_indices),
        'r2': full_r2,
        'delta_r2': 0
    })

    # Remove each category
    for cat, indices in categories.items():
        if not indices:
            continue

        keep_indices = [i for i in all_indices if i not in indices]
        X_ablated = X[:, keep_indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X_ablated, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = model_class(**model_kwargs)
        model.fit(X_train_scaled, y_train)
        y_pred = np.clip(model.predict(X_test_scaled), 0, 1)
        ablated_r2 = r2_score(y_test, y_pred)

        results.append({
            'ablation': f'remove_{cat}',
            'n_features': len(keep_indices),
            'r2': ablated_r2,
            'delta_r2': full_r2 - ablated_r2  # Positive = category helped
        })

    # Single category only
    for cat, indices in categories.items():
        if not indices:
            continue

        X_single = X[:, indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X_single, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = model_class(**model_kwargs)
        model.fit(X_train_scaled, y_train)
        y_pred = np.clip(model.predict(X_test_scaled), 0, 1)
        single_r2 = r2_score(y_test, y_pred)

        results.append({
            'ablation': f'only_{cat}',
            'n_features': len(indices),
            'r2': single_r2,
            'delta_r2': single_r2 - full_r2
        })

    return pd.DataFrame(results)


def cross_model_transfer(
    features_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    model_class,
    model_kwargs: Dict = None,
    target_metric: str = 'c_mean'
) -> pd.DataFrame:
    """
    Train on one LLM's data, test on others.
    """
    if model_kwargs is None:
        model_kwargs = {}

    feature_cols = [c for c in features_df.columns if c not in ['sample_idx', 'sample_id']]
    models = metrics_df['model'].unique()

    results = []

    for train_model in models[:5]:  # Limit to 5 models for speed
        # Prepare training data
        train_metrics = metrics_df[metrics_df['model'] == train_model].groupby('sample_idx')[target_metric].mean().reset_index()
        train_merged = features_df.merge(train_metrics, on='sample_idx', how='inner')

        X_train = train_merged[feature_cols].values
        y_train = train_merged[target_metric].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = model_class(**model_kwargs)
        model.fit(X_train_scaled, y_train)

        # Test on other models
        for test_model in models[:5]:
            test_metrics = metrics_df[metrics_df['model'] == test_model].groupby('sample_idx')[target_metric].mean().reset_index()
            test_merged = features_df.merge(test_metrics, on='sample_idx', how='inner')

            X_test = test_merged[feature_cols].values
            y_test = test_merged[target_metric].values

            X_test_scaled = scaler.transform(X_test)
            y_pred = np.clip(model.predict(X_test_scaled), 0, 1)

            results.append({
                'train_model': train_model,
                'test_model': test_model,
                'r2': r2_score(y_test, y_pred),
                'pearson_r': pearsonr(y_test, y_pred)[0],
                'same_model': train_model == test_model
            })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("COLM 2026 Experiment 3: Train Consistency Predictor")
    print("=" * 70)

    # Setup
    exp_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446"
    output_dir = exp_dir / "results" / "exp3_predictor"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/5] Loading data...")
    features_df, metrics_df = load_data(exp_dir)
    X, y, feature_names = prepare_dataset(features_df, metrics_df, 'c_mean')
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)

    # Train and evaluate all models
    print("\n[2/5] Training models...")
    results = []
    for name, model in models.items():
        print(f"  Training {name}...")
        result = train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test, name, model
        )
        results.append(result)

    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 70)
    print(results_df[['model', 'r2', 'rmse', 'mae', 'pearson_r', 'spearman_r']].to_string())

    # Cross-validation for best model
    print("\n[3/5] Cross-validation for top models...")
    cv_results = []
    for name in ['Ridge', 'RandomForest', 'GradientBoosting']:
        model_class = type(models[name])
        model = model_class()
        X_scaled = scaler.fit_transform(X)
        cv = cross_validate_model(X_scaled, y, model, n_folds=5)
        cv['model'] = name
        cv_results.append(cv)
        print(f"  {name}: R² = {cv['cv_r2_mean']:.3f} ± {cv['cv_r2_std']:.3f}")

    cv_df = pd.DataFrame(cv_results)

    # Category ablation
    print("\n[4/5] Feature category ablation...")
    ablation_df = category_ablation(X, y, feature_names, Ridge, {'alpha': 1.0})

    print("\n" + "=" * 70)
    print("FEATURE CATEGORY ABLATION (Ridge)")
    print("=" * 70)
    print(ablation_df.to_string())

    # Cross-model transfer
    print("\n[5/5] Cross-model transfer analysis...")
    transfer_df = cross_model_transfer(features_df, metrics_df, Ridge, {'alpha': 1.0})

    # Summarize transfer
    same_model = transfer_df[transfer_df['same_model']]['r2'].mean()
    diff_model = transfer_df[~transfer_df['same_model']]['r2'].mean()

    print("\n" + "=" * 70)
    print("CROSS-MODEL TRANSFER")
    print("=" * 70)
    print(f"  Same model R²: {same_model:.3f}")
    print(f"  Different model R²: {diff_model:.3f}")
    print(f"  Transfer gap: {same_model - diff_model:.3f}")

    # Feature importance (from Ridge coefficients)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': ridge.coef_,
        'abs_coefficient': np.abs(ridge.coef_)
    }).sort_values('abs_coefficient', ascending=False)

    print("\n" + "=" * 70)
    print("TOP 15 FEATURES BY IMPORTANCE (Ridge |coef|)")
    print("=" * 70)
    print(importance.head(15).to_string())

    # Save results
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    cv_df.to_csv(output_dir / "cross_validation.csv", index=False)
    ablation_df.to_csv(output_dir / "category_ablation.csv", index=False)
    transfer_df.to_csv(output_dir / "cross_model_transfer.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    summary = {
        'best_model': results_df.iloc[0]['model'],
        'best_r2': float(results_df.iloc[0]['r2']),
        'best_rmse': float(results_df.iloc[0]['rmse']),
        'best_pearson': float(results_df.iloc[0]['pearson_r']),
        'cv_r2': float(cv_df[cv_df['model'] == 'Ridge']['cv_r2_mean'].values[0]),
        'transfer_same_model_r2': float(same_model),
        'transfer_diff_model_r2': float(diff_model),
        'top_features': importance.head(10)['feature'].tolist(),
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")

    return results_df, ablation_df


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    results_df, ablation_df = main()
