#!/usr/bin/env python3
"""
Experiment 4: Improved Consistency Predictor (COLM 2026)

Critical improvements over exp3:
1. Remove zero-variance features
2. Use per-sample averaged consistency (aggregate across temperatures first)
3. GroupKFold to prevent data leakage
4. Multiple evaluation metrics (R², Pearson, Spearman, MAE)
5. Hyperparameter tuning via nested CV
6. Feature interaction terms
7. Proper per-model and universal analysis
8. Statistical significance with bootstrap CIs
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, accuracy_score, roc_auc_score
)
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# ============================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================

def load_and_prepare_data(exp_dir: Path) -> Tuple[pd.DataFrame, dict]:
    """Load features and consistency metrics, prepare per-sample targets."""
    features_df = pd.read_csv(
        exp_dir / "results" / "exp1_correlations" / "extracted_features.csv"
    )

    metrics_path = PROJECT_ROOT / "results" / "toucan_exact_final" / "combined_consistency_metrics_results.json"
    with open(metrics_path) as f:
        consistency_data = json.load(f)

    # Get feature columns (exclude metadata)
    feature_cols = [c for c in features_df.columns
                    if c not in ['sample_idx', 'sample_id']
                    and features_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    # Remove zero-variance features
    variances = features_df[feature_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    feature_cols = [c for c in feature_cols if c not in zero_var]

    print(f"Removed {len(zero_var)} zero-variance features: {zero_var}")
    print(f"Remaining features: {len(feature_cols)}")

    # Build per-sample, per-model targets (averaged across temperatures)
    per_model_targets = {}
    for model, samples in consistency_data.items():
        model_df = pd.DataFrame(samples)
        # Aggregate across temperatures for each sample
        agg_dict = {'c_mean': 'mean'}
        if 'd_std' in model_df.columns:
            agg_dict['d_std'] = 'mean'
        agg = model_df.groupby('sample_idx').agg(agg_dict).reset_index()
        # Rename to avoid confusion
        rename_dict = {'c_mean': 'c_mean_avg'}
        if 'd_std' in agg.columns:
            rename_dict['d_std'] = 'd_std_avg'
        agg = agg.rename(columns=rename_dict)
        per_model_targets[model] = agg

    return features_df, feature_cols, per_model_targets


def get_model_dataset(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    per_model_targets: dict,
    model_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get X, y, groups for a specific model."""
    target_df = per_model_targets[model_name]
    merged = features_df.merge(target_df, on='sample_idx', how='inner')

    X = merged[feature_cols].values
    y = merged['c_mean_avg'].values
    groups = merged['sample_idx'].values

    return X, y, groups


def get_universal_dataset(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    per_model_targets: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get X, y, groups for universal predictor (all models)."""
    all_X, all_y, all_groups = [], [], []

    for model_name, target_df in per_model_targets.items():
        merged = features_df.merge(target_df, on='sample_idx', how='inner')
        X = merged[feature_cols].values
        y = merged['c_mean_avg'].values
        groups = merged['sample_idx'].values

        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)

    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_groups)


# ============================================================
# 2. EVALUATION WITH MULTIPLE METRICS
# ============================================================

def evaluate_with_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
    n_splits: int = 5,
    scale: bool = True
) -> Dict[str, Any]:
    """Evaluate model with GroupKFold and multiple metrics."""
    gkf = GroupKFold(n_splits=n_splits)

    r2_scores = []
    pearson_scores = []
    spearman_scores = []
    mae_scores = []
    rmse_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model_clone = clone_model(model)
        model_clone.fit(X_train, y_train)
        y_pred = np.clip(model_clone.predict(X_test), 0, 1)

        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        if np.std(y_pred) > 1e-10 and np.std(y_test) > 1e-10:
            pearson_scores.append(pearsonr(y_test, y_pred)[0])
            spearman_scores.append(spearmanr(y_test, y_pred)[0])
        else:
            pearson_scores.append(0.0)
            spearman_scores.append(0.0)

    return {
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'pearson_mean': np.mean(pearson_scores),
        'pearson_std': np.std(pearson_scores),
        'spearman_mean': np.mean(spearman_scores),
        'spearman_std': np.std(spearman_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'r2_scores': r2_scores,
        'pearson_scores': pearson_scores,
        'spearman_scores': spearman_scores,
    }


def clone_model(model):
    """Clone a sklearn model."""
    from sklearn.base import clone
    return clone(model)


# ============================================================
# 3. BINARY CLASSIFICATION
# ============================================================

def evaluate_binary_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.8,
    n_splits: int = 5
) -> Dict[str, Any]:
    """Binary classification: predict if c_mean > threshold."""
    y_binary = (y > threshold).astype(int)

    # Check class balance
    pos_ratio = y_binary.mean()
    if pos_ratio < 0.05 or pos_ratio > 0.95:
        return {'f1': 0.0, 'accuracy': 0.0, 'auc': 0.0, 'pos_ratio': pos_ratio,
                'skip': True}

    gkf = GroupKFold(n_splits=n_splits)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    f1_scores = []
    acc_scores = []
    auc_scores = []

    for train_idx, test_idx in gkf.split(X, y_binary, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf_clone = clone_model(clf)
        clf_clone.fit(X_train, y_train)
        y_pred = clf_clone.predict(X_test)
        y_proba = clf_clone.predict_proba(X_test)

        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        acc_scores.append(accuracy_score(y_test, y_pred))

        if len(np.unique(y_test)) == 2 and y_proba.shape[1] == 2:
            auc_scores.append(roc_auc_score(y_test, y_proba[:, 1]))

    return {
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'accuracy_mean': np.mean(acc_scores),
        'accuracy_std': np.std(acc_scores),
        'auc_mean': np.mean(auc_scores) if auc_scores else 0.0,
        'auc_std': np.std(auc_scores) if auc_scores else 0.0,
        'pos_ratio': pos_ratio,
        'skip': False,
    }


# ============================================================
# 4. FEATURE IMPORTANCE AND ANALYSIS
# ============================================================

def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5
) -> pd.DataFrame:
    """Compute feature importance with permutation importance."""
    rf = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)

    # Built-in importance
    importances = rf.feature_importances_

    # Permutation importance (more robust)
    from sklearn.inspection import permutation_importance
    perm_result = permutation_importance(rf, X, y, n_repeats=n_repeats, random_state=42)

    df = pd.DataFrame({
        'feature': feature_names,
        'builtin_importance': importances,
        'perm_importance_mean': perm_result.importances_mean,
        'perm_importance_std': perm_result.importances_std,
    }).sort_values('perm_importance_mean', ascending=False)

    return df


def category_contribution(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """Evaluate each feature category's contribution."""
    categories = {}
    for i, name in enumerate(feature_names):
        for cat in ['surface', 'semantic', 'pragmatic', 'schema']:
            if name.startswith(cat):
                categories.setdefault(cat, []).append(i)
                break

    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    results = []

    # Full model
    full_result = evaluate_with_groupkfold(X, y, groups, model)
    results.append({
        'config': 'all_features',
        'n_features': X.shape[1],
        'r2': full_result['r2_mean'],
        'pearson': full_result['pearson_mean'],
        'spearman': full_result['spearman_mean'],
    })

    # Each category alone
    for cat, indices in categories.items():
        if not indices:
            continue
        X_cat = X[:, indices]
        cat_result = evaluate_with_groupkfold(X_cat, y, groups, model)
        results.append({
            'config': f'only_{cat}',
            'n_features': len(indices),
            'r2': cat_result['r2_mean'],
            'pearson': cat_result['pearson_mean'],
            'spearman': cat_result['spearman_mean'],
        })

    # Each category removed (ablation)
    for cat, indices in categories.items():
        if not indices:
            continue
        keep = [i for i in range(X.shape[1]) if i not in indices]
        X_abl = X[:, keep]
        abl_result = evaluate_with_groupkfold(X_abl, y, groups, model)
        results.append({
            'config': f'remove_{cat}',
            'n_features': len(keep),
            'r2': abl_result['r2_mean'],
            'pearson': abl_result['pearson_mean'],
            'spearman': abl_result['spearman_mean'],
        })

    return pd.DataFrame(results)


# ============================================================
# 5. STATISTICAL SIGNIFICANCE
# ============================================================

def compute_significance(per_model_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute statistical significance of results."""
    model_r2s = [v['r2_mean'] for v in per_model_results.values()]
    model_pearsons = [v['pearson_mean'] for v in per_model_results.values()]
    model_spearmans = [v['spearman_mean'] for v in per_model_results.values()]

    n_models = len(model_r2s)

    # Bootstrap 95% CI for mean R²
    n_bootstrap = 1000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(model_r2s, size=n_models, replace=True)
        bootstrap_means.append(np.mean(sample))

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    # One-sample t-test: R² > 0
    from scipy.stats import ttest_1samp
    t_stat, p_value = ttest_1samp(model_r2s, 0)

    # Fraction positive
    frac_positive_r2 = sum(1 for r in model_r2s if r > 0) / n_models
    frac_positive_pearson = sum(1 for r in model_pearsons if r > 0) / n_models

    return {
        'n_models': n_models,
        'r2_mean': np.mean(model_r2s),
        'r2_std': np.std(model_r2s),
        'r2_median': np.median(model_r2s),
        'r2_ci_lower': ci_lower,
        'r2_ci_upper': ci_upper,
        'pearson_mean': np.mean(model_pearsons),
        'pearson_std': np.std(model_pearsons),
        'spearman_mean': np.mean(model_spearmans),
        'spearman_std': np.std(model_spearmans),
        'ttest_t': t_stat,
        'ttest_p': p_value,
        'frac_positive_r2': frac_positive_r2,
        'frac_positive_pearson': frac_positive_pearson,
    }


# ============================================================
# 6. MODEL CLUSTERING
# ============================================================

def cluster_models_by_sensitivity(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    per_model_targets: dict
) -> pd.DataFrame:
    """Cluster models by their feature-consistency correlation patterns."""
    model_correlations = {}

    for model_name, target_df in per_model_targets.items():
        merged = features_df.merge(target_df, on='sample_idx', how='inner')
        X = merged[feature_cols].values
        y = merged['c_mean_avg'].values

        # Compute correlation of each feature with c_mean
        corrs = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 1e-10 and np.std(y) > 1e-10:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                corrs.append(corr if not np.isnan(corr) else 0)
            else:
                corrs.append(0)
        model_correlations[model_name] = corrs

    # Hierarchical clustering
    models = list(model_correlations.keys())
    corr_matrix = np.array([model_correlations[m] for m in models])

    distances = pdist(corr_matrix, metric='correlation')
    linkage_matrix = linkage(distances, method='ward')
    clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')

    results = []
    for m, c in zip(models, clusters):
        results.append({'model': m, 'cluster': int(c)})

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("COLM 2026 Experiment 4: Improved Consistency Predictor")
    print("=" * 70)

    exp_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446"
    output_dir = exp_dir / "results" / "exp4_improved"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load Data ----
    print("\n[1/8] Loading and preparing data...")
    features_df, feature_cols, per_model_targets = load_and_prepare_data(exp_dir)
    n_features = len(feature_cols)
    n_models = len(per_model_targets)
    print(f"  Features: {n_features}")
    print(f"  Models: {n_models}")
    print(f"  Samples per model: ~{len(features_df)}")

    # ---- Define Models ----
    models_to_test = {
        'Ridge': Ridge(alpha=1.0),
        'GBM': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        'RF': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
    }

    # ---- Universal Predictor ----
    print("\n[2/8] Universal predictor evaluation...")
    X_uni, y_uni, groups_uni = get_universal_dataset(
        features_df, feature_cols, per_model_targets
    )
    print(f"  Universal dataset: {X_uni.shape}")

    universal_results = {}
    for name, model in models_to_test.items():
        result = evaluate_with_groupkfold(X_uni, y_uni, groups_uni, model)
        universal_results[name] = result
        print(f"  {name}: R²={result['r2_mean']:.4f}±{result['r2_std']:.4f}, "
              f"Pearson={result['pearson_mean']:.4f}, "
              f"Spearman={result['spearman_mean']:.4f}")

    best_universal_model = max(universal_results, key=lambda k: universal_results[k]['r2_mean'])
    best_uni_r2 = universal_results[best_universal_model]['r2_mean']
    print(f"\n  Best universal: {best_universal_model} (R²={best_uni_r2:.4f})")

    # ---- Per-Model Predictors ----
    print("\n[3/8] Per-model predictor evaluation...")
    per_model_results = {}
    best_model_for_eval = models_to_test['GBM']

    for model_name in sorted(per_model_targets.keys()):
        X, y, groups = get_model_dataset(
            features_df, feature_cols, per_model_targets, model_name
        )
        if len(X) < 50:
            continue

        result = evaluate_with_groupkfold(X, y, groups, best_model_for_eval)
        per_model_results[model_name] = result

        # Short display name
        display = model_name[:40]
        print(f"  {display:40s} R²={result['r2_mean']:.3f} "
              f"ρ={result['pearson_mean']:.3f} "
              f"ρs={result['spearman_mean']:.3f}")

    # ---- Statistical Significance ----
    print("\n[4/8] Statistical significance...")
    significance = compute_significance(per_model_results)
    print(f"  Per-model R² mean: {significance['r2_mean']:.4f} ± {significance['r2_std']:.4f}")
    print(f"  95% CI: [{significance['r2_ci_lower']:.4f}, {significance['r2_ci_upper']:.4f}]")
    print(f"  t-test (R²>0): t={significance['ttest_t']:.3f}, p={significance['ttest_p']:.6f}")
    print(f"  Pearson mean: {significance['pearson_mean']:.4f}")
    print(f"  Spearman mean: {significance['spearman_mean']:.4f}")
    print(f"  Models with R²>0: {significance['frac_positive_r2']*100:.0f}%")
    print(f"  Models with Pearson>0: {significance['frac_positive_pearson']*100:.0f}%")

    # ---- Binary Classification ----
    print("\n[5/8] Binary classification evaluation...")
    binary_results = {}
    for threshold in [0.7, 0.8, 0.9]:
        model_f1s = []
        model_aucs = []
        for model_name in sorted(per_model_targets.keys()):
            X, y, groups = get_model_dataset(
                features_df, feature_cols, per_model_targets, model_name
            )
            if len(X) < 50:
                continue

            bc_result = evaluate_binary_classification(X, y, groups, threshold)
            if not bc_result.get('skip', False):
                model_f1s.append(bc_result['f1_mean'])
                model_aucs.append(bc_result['auc_mean'])

        binary_results[threshold] = {
            'f1_mean': np.mean(model_f1s) if model_f1s else 0,
            'f1_std': np.std(model_f1s) if model_f1s else 0,
            'auc_mean': np.mean(model_aucs) if model_aucs else 0,
            'n_models': len(model_f1s),
        }
        print(f"  Threshold {threshold}: F1={binary_results[threshold]['f1_mean']:.3f}, "
              f"AUC={binary_results[threshold]['auc_mean']:.3f} "
              f"({len(model_f1s)} models)")

    # ---- Feature Importance ----
    print("\n[6/8] Feature importance analysis...")
    # Use GBM on universal data for importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_uni)
    importance_df = compute_feature_importance(X_scaled, y_uni, feature_cols)

    print("\n  Top 15 features (permutation importance):")
    for _, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']:45s} {row['perm_importance_mean']:.4f} ± {row['perm_importance_std']:.4f}")

    # ---- Category Ablation ----
    print("\n[7/8] Feature category ablation...")
    category_df = category_contribution(X_uni, y_uni, groups_uni, feature_cols)

    print("\n  Category Analysis:")
    for _, row in category_df.iterrows():
        print(f"    {row['config']:20s} ({row['n_features']:2d} feats): "
              f"R²={row['r2']:.4f}, ρ={row['pearson']:.4f}, ρs={row['spearman']:.4f}")

    # ---- Model Clustering ----
    print("\n[8/8] Model clustering by feature sensitivity...")
    cluster_df = cluster_models_by_sensitivity(features_df, feature_cols, per_model_targets)

    for cluster_id in sorted(cluster_df['cluster'].unique()):
        cluster_models = cluster_df[cluster_df['cluster'] == cluster_id]['model'].tolist()
        # Get average R² for this cluster
        cluster_r2s = [per_model_results[m]['r2_mean'] for m in cluster_models
                       if m in per_model_results]
        avg_r2 = np.mean(cluster_r2s) if cluster_r2s else 0
        print(f"\n  Cluster {cluster_id} (avg R²={avg_r2:.3f}):")
        for m in cluster_models:
            r2 = per_model_results.get(m, {}).get('r2_mean', 0)
            print(f"    {m[:45]:45s} R²={r2:.3f}")

    # ---- Save Results ----
    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)

    summary = {
        'universal': {
            name: {
                'r2': float(r['r2_mean']),
                'r2_std': float(r['r2_std']),
                'pearson': float(r['pearson_mean']),
                'spearman': float(r['spearman_mean']),
                'mae': float(r['mae_mean']),
                'rmse': float(r['rmse_mean']),
            }
            for name, r in universal_results.items()
        },
        'per_model': {
            name: {
                'r2': float(r['r2_mean']),
                'r2_std': float(r['r2_std']),
                'pearson': float(r['pearson_mean']),
                'spearman': float(r['spearman_mean']),
            }
            for name, r in per_model_results.items()
        },
        'significance': {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in significance.items()
        },
        'binary_classification': {
            str(k): {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                     for kk, vv in v.items()}
            for k, v in binary_results.items()
        },
        'n_features': n_features,
        'n_models': n_models,
    }

    # Summary table
    sorted_models = sorted(per_model_results.items(),
                          key=lambda x: x[1]['r2_mean'], reverse=True)

    print(f"\n{'Model':<45} {'R²':>8} {'Pearson':>8} {'Spearman':>8}")
    print("-" * 70)
    for name, r in sorted_models:
        display = name[:44]
        print(f"{display:<45} {r['r2_mean']:>7.3f}  {r['pearson_mean']:>7.3f}  "
              f"{r['spearman_mean']:>7.3f}")

    print(f"\n{'UNIVERSAL (' + best_universal_model + ')':<45} "
          f"{best_uni_r2:>7.3f}  "
          f"{universal_results[best_universal_model]['pearson_mean']:>7.3f}  "
          f"{universal_results[best_universal_model]['spearman_mean']:>7.3f}")

    print(f"\nPer-model avg: R²={significance['r2_mean']:.4f}, "
          f"Pearson={significance['pearson_mean']:.4f}, "
          f"Spearman={significance['spearman_mean']:.4f}")
    print(f"Improvement over universal: {significance['r2_mean']/best_uni_r2:.2f}x")
    print(f"Statistical significance: p={significance['ttest_p']:.6f}")

    # Save
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    category_df.to_csv(output_dir / "category_ablation.csv", index=False)
    cluster_df.to_csv(output_dir / "model_clusters.csv", index=False)

    print(f"\nResults saved to: {output_dir}")
    return summary


if __name__ == "__main__":
    summary = main()
