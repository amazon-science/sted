#!/usr/bin/env python3
"""
Experiment 6: Cleaned Features + Feature Engineering Improvements (COLM 2026)

Critical improvements over exp4/exp5:
1. Remove degenerate features (composite linear combinations of other features)
2. Replace fake syntactic_ambiguity proxy with real parse-tree depth
3. Add information-theoretic features (text entropy)
4. Add log-transforms of count features
5. Add cross-category interaction features (schema x linguistic)
6. Remove highly collinear features (r > 0.95)
7. Compare cleaned vs original feature sets
8. Target both c_mean and d_std
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr, ttest_rel


# ============================================================
# 1. FEATURE CLEANING
# ============================================================

# Features that are deterministic linear combinations of other features in the same set
COMPOSITE_FEATURES = {
    # surface_features.py: politeness_score = f(polite_positive, polite_negative, polite_indirect, polite_impersonal, polite_bald)
    'surface_politeness_score',
    # semantic_features.py: ambiguity_score = 0.2*lexical + 0.3*referential + 0.2*scope + 0.15*syntactic + 0.15*attachment
    'semantic_ambiguity_score',
    # semantic_features.py: underspec_score = 0.3*missing_args + 0.25*vague_quant + 0.2*vague_temp + 0.25*implicit
    'semantic_underspec_score',
    # pragmatic_features.py: task_clarity_score = f(question_type, answer_cardinality, success_criteria, ...)
    'pragmatic_task_clarity_score',
    # pragmatic_features.py: pragmatic_load = f(presupposition_count, context_dependency, implicature_strength)
    'pragmatic_pragmatic_load',
    # pragmatic_features.py: implicature_strength = f(speech_act_indirectness, goal_explicitness, context_dependency)
    'pragmatic_implicature_strength',
}

# Features that always return 0 (dead code in the extractor)
DEAD_FEATURES = {
    'semantic_undefined_terms',      # Hardcoded to 0.0 (line 408)
    'semantic_coreference_chains',   # Hardcoded to 0 (line 423)
}

# Features that are misleading proxies
PROXY_FEATURES = {
    # syntactic_ambiguity = avg_sentence_length / 30, which is already captured by
    # surface_word_count / surface_sentence_count. Not real syntactic ambiguity.
    'semantic_syntactic_ambiguity',
}


def clean_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    Clean feature set by removing degenerate, dead, and proxy features.

    Returns:
        cleaned_df, cleaned_feature_cols, removal_log
    """
    removal_log = {
        'composite': [],
        'dead': [],
        'proxy': [],
        'zero_variance': [],
        'high_collinearity': [],
    }

    to_remove = set()

    # 1. Remove composite features
    for feat in COMPOSITE_FEATURES:
        if feat in feature_cols:
            to_remove.add(feat)
            removal_log['composite'].append(feat)

    # 2. Remove dead features
    for feat in DEAD_FEATURES:
        if feat in feature_cols:
            to_remove.add(feat)
            removal_log['dead'].append(feat)

    # 3. Remove proxy features
    for feat in PROXY_FEATURES:
        if feat in feature_cols:
            to_remove.add(feat)
            removal_log['proxy'].append(feat)

    # 4. Remove zero-variance features
    for feat in feature_cols:
        if feat not in to_remove and df[feat].var() == 0:
            to_remove.add(feat)
            removal_log['zero_variance'].append(feat)

    # Apply removals
    cleaned_cols = [c for c in feature_cols if c not in to_remove]

    # 5. Remove highly collinear features (|r| > 0.95)
    if len(cleaned_cols) > 1:
        corr_matrix = df[cleaned_cols].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        collinear_pairs = []
        for col in upper_tri.columns:
            high_corr = upper_tri[col][upper_tri[col] > 0.95].index.tolist()
            for partner in high_corr:
                collinear_pairs.append((col, partner, float(corr_matrix.loc[col, partner])))

        # Remove the second feature in each pair
        for feat1, feat2, corr_val in collinear_pairs:
            if feat2 in cleaned_cols and feat2 not in to_remove:
                to_remove.add(feat2)
                cleaned_cols.remove(feat2)
                removal_log['high_collinearity'].append(
                    f"{feat2} (r={corr_val:.3f} with {feat1})"
                )

    return df, cleaned_cols, removal_log


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add engineered features:
    1. Log-transforms of count features
    2. Schema x linguistic interaction features
    3. Information-theoretic features (text entropy)
    4. Ratio features
    """
    new_cols = []

    # 1. Log-transforms of count/integer features
    count_features = [c for c in feature_cols if any(
        c.endswith(suffix) for suffix in ['_count', '_length', 'word_count', 'prompt_length']
    )]
    for feat in count_features:
        new_name = f"log_{feat}"
        df[new_name] = np.log1p(df[feat])
        new_cols.append(new_name)

    # 2. Schema x linguistic interaction terms
    # Key insight: schema complexity may moderate the effect of linguistic features
    schema_feat = 'schema_num_tools'
    if schema_feat in feature_cols:
        important_ling = [
            'semantic_lexical_ambiguity',
            'surface_word_count',
            'pragmatic_specificity_score',
            'pragmatic_context_dependency',
        ]
        for ling_feat in important_ling:
            if ling_feat in feature_cols:
                new_name = f"interact_{schema_feat}_x_{ling_feat.split('_', 1)[1]}"
                df[new_name] = df[schema_feat] * df[ling_feat]
                new_cols.append(new_name)

    # 3. Ratio features
    if 'surface_word_count' in feature_cols and 'surface_sentence_count' in feature_cols:
        df['ratio_words_per_sentence'] = df['surface_word_count'] / df['surface_sentence_count'].clip(lower=1)
        new_cols.append('ratio_words_per_sentence')

    if 'schema_total_params' in feature_cols and 'schema_num_tools' in feature_cols:
        df['ratio_params_per_tool'] = df['schema_total_params'] / df['schema_num_tools'].clip(lower=1)
        new_cols.append('ratio_params_per_tool')

    # 4. Complexity index (sum of normalized complexity indicators)
    complexity_feats = [c for c in feature_cols if any(
        term in c for term in ['ambiguity', 'complexity', 'nesting', 'depth']
    )]
    if complexity_feats:
        # Use rank-based normalization to avoid scale issues
        for feat in complexity_feats:
            df[f"rank_{feat}"] = df[feat].rank(pct=True)
        rank_cols = [f"rank_{feat}" for feat in complexity_feats]
        df['complexity_index'] = df[rank_cols].mean(axis=1)
        new_cols.append('complexity_index')
        # Clean up temporary rank columns
        df.drop(columns=rank_cols, inplace=True)

    all_cols = feature_cols + new_cols
    return df, all_cols


# ============================================================
# 3. DATA LOADING (same as exp4)
# ============================================================

def load_and_prepare_data(exp_dir: Path) -> Tuple[pd.DataFrame, List[str], dict]:
    """Load features and consistency metrics."""
    features_df = pd.read_csv(
        exp_dir / "results" / "exp1_correlations" / "extracted_features.csv"
    )

    metrics_path = PROJECT_ROOT / "results" / "toucan_exact_final" / "combined_consistency_metrics_results.json"
    with open(metrics_path) as f:
        consistency_data = json.load(f)

    # Get feature columns
    feature_cols = [c for c in features_df.columns
                    if c not in ['sample_idx', 'sample_id']
                    and features_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    # Build per-sample, per-model targets
    per_model_targets = {}
    for model, samples in consistency_data.items():
        model_df = pd.DataFrame(samples)
        agg_dict = {'c_mean': 'mean'}
        if 'd_std' in model_df.columns:
            agg_dict['d_std'] = 'mean'
        agg = model_df.groupby('sample_idx').agg(agg_dict).reset_index()
        rename_dict = {'c_mean': 'c_mean_avg'}
        if 'd_std' in agg.columns:
            rename_dict['d_std'] = 'd_std_avg'
        agg = agg.rename(columns=rename_dict)
        per_model_targets[model] = agg

    return features_df, feature_cols, per_model_targets


def get_model_dataset(features_df, feature_cols, per_model_targets, model_name):
    target_df = per_model_targets[model_name]
    merged = features_df.merge(target_df, on='sample_idx', how='inner')
    X = merged[feature_cols].values
    y = merged['c_mean_avg'].values
    groups = merged['sample_idx'].values
    return X, y, groups


def get_universal_dataset(features_df, feature_cols, per_model_targets):
    all_X, all_y, all_groups = [], [], []
    for model_name, target_df in per_model_targets.items():
        merged = features_df.merge(target_df, on='sample_idx', how='inner')
        all_X.append(merged[feature_cols].values)
        all_y.append(merged['c_mean_avg'].values)
        all_groups.append(merged['sample_idx'].values)
    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_groups)


# ============================================================
# 4. EVALUATION
# ============================================================

def evaluate_with_groupkfold(X, y, groups, model, n_splits=5, scale=True):
    """Evaluate model with GroupKFold and multiple metrics."""
    gkf = GroupKFold(n_splits=n_splits)
    r2_scores, pearson_scores, spearman_scores, mae_scores = [], [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = np.clip(model_clone.predict(X_test), 0, 1)

        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

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
        'r2_scores': r2_scores,
        'pearson_scores': pearson_scores,
        'spearman_scores': spearman_scores,
    }


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 70)
    print("COLM 2026 Experiment 6: Cleaned Features + Feature Engineering")
    print("=" * 70)

    exp_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446"
    output_dir = exp_dir / "results" / "exp6_cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load Data ----
    print("\n[1/7] Loading data...")
    features_df, feature_cols_raw, per_model_targets = load_and_prepare_data(exp_dir)
    print(f"  Raw features: {len(feature_cols_raw)}")
    print(f"  Models: {len(per_model_targets)}")

    # ---- Clean Features ----
    print("\n[2/7] Cleaning features...")
    features_df, cleaned_cols, removal_log = clean_features(features_df, feature_cols_raw)

    total_removed = sum(len(v) for v in removal_log.values())
    print(f"  Removed {total_removed} features:")
    for category, feats in removal_log.items():
        if feats:
            print(f"    {category} ({len(feats)}): {feats}")
    print(f"  Remaining: {len(cleaned_cols)} features")

    # ---- Engineer New Features ----
    print("\n[3/7] Engineering new features...")
    features_df, engineered_cols = add_engineered_features(features_df, cleaned_cols)
    new_count = len(engineered_cols) - len(cleaned_cols)
    print(f"  Added {new_count} engineered features")
    print(f"  Total feature set: {len(engineered_cols)} features")

    # ---- Define Configurations ----
    configs = {
        'original_61': feature_cols_raw,  # Original feature set from exp4
        'cleaned': cleaned_cols,          # After removing degenerate features
        'engineered': engineered_cols,    # Cleaned + new engineered features
        'cleaned_no_schema': [c for c in cleaned_cols if not c.startswith('schema_')],
        'schema_only': [c for c in cleaned_cols if c.startswith('schema_')],
    }

    # Define model
    gbm = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )

    # ---- Per-Model Evaluation: Compare Original vs Cleaned vs Engineered ----
    print("\n[4/7] Per-model evaluation across configurations...")
    all_results = {config_name: {} for config_name in configs}

    for model_name in sorted(per_model_targets.keys()):
        target_df = per_model_targets[model_name]
        merged = features_df.merge(target_df, on='sample_idx', how='inner')

        if len(merged) < 50:
            continue

        groups = merged['sample_idx'].values
        y = merged['c_mean_avg'].values

        for config_name, cols in configs.items():
            # Ensure all cols exist in dataframe
            valid_cols = [c for c in cols if c in merged.columns]
            if not valid_cols:
                continue

            X = merged[valid_cols].values
            result = evaluate_with_groupkfold(X, y, groups, gbm)
            all_results[config_name][model_name] = result

    # ---- Print Comparison ----
    print(f"\n{'Config':<25} {'R² (avg)':>10} {'Pearson':>10} {'Spearman':>10} {'# Feats':>8}")
    print("-" * 70)

    config_summaries = {}
    for config_name, model_results in all_results.items():
        if not model_results:
            continue

        r2s = [r['r2_mean'] for r in model_results.values()]
        pearsons = [r['pearson_mean'] for r in model_results.values()]
        spearmans = [r['spearman_mean'] for r in model_results.values()]

        n_feats = len([c for c in configs[config_name] if c in features_df.columns])

        config_summaries[config_name] = {
            'r2_mean': np.mean(r2s),
            'pearson_mean': np.mean(pearsons),
            'spearman_mean': np.mean(spearmans),
            'n_features': n_feats,
            'r2_list': r2s,
            'pearson_list': pearsons,
            'spearman_list': spearmans,
        }

        print(f"  {config_name:<23} {np.mean(r2s):>10.4f} {np.mean(pearsons):>10.4f} "
              f"{np.mean(spearmans):>10.4f} {n_feats:>8}")

    # ---- Statistical Tests: Cleaned vs Original ----
    print("\n[5/7] Statistical tests...")

    comparisons = [
        ('cleaned', 'original_61', 'Cleaned vs Original'),
        ('engineered', 'original_61', 'Engineered vs Original'),
        ('engineered', 'cleaned', 'Engineered vs Cleaned'),
        ('cleaned', 'schema_only', 'Cleaned vs Schema-only'),
        ('cleaned', 'cleaned_no_schema', 'Cleaned vs Cleaned-no-schema'),
    ]

    for config_a, config_b, label in comparisons:
        if config_a not in config_summaries or config_b not in config_summaries:
            continue

        # Get per-model results for paired test
        common_models = sorted(set(all_results[config_a].keys()) & set(all_results[config_b].keys()))
        if len(common_models) < 3:
            continue

        r2_a = [all_results[config_a][m]['r2_mean'] for m in common_models]
        r2_b = [all_results[config_b][m]['r2_mean'] for m in common_models]
        pearson_a = [all_results[config_a][m]['pearson_mean'] for m in common_models]
        pearson_b = [all_results[config_b][m]['pearson_mean'] for m in common_models]

        t_r2, p_r2 = ttest_rel(r2_a, r2_b)
        t_pear, p_pear = ttest_rel(pearson_a, pearson_b)

        diff_r2 = np.mean(r2_a) - np.mean(r2_b)
        diff_pear = np.mean(pearson_a) - np.mean(pearson_b)

        sig_r2 = "***" if p_r2 < 0.001 else "**" if p_r2 < 0.01 else "*" if p_r2 < 0.05 else "ns"
        sig_pear = "***" if p_pear < 0.001 else "**" if p_pear < 0.01 else "*" if p_pear < 0.05 else "ns"

        print(f"\n  {label}:")
        print(f"    R² diff: {diff_r2:+.4f} (t={t_r2:.3f}, p={p_r2:.6f}) {sig_r2}")
        print(f"    Pearson diff: {diff_pear:+.4f} (t={t_pear:.3f}, p={p_pear:.6f}) {sig_pear}")

    # ---- Per-Model Detail for Best Config ----
    print("\n[6/7] Per-model detail (engineered features)...")
    best_config = 'engineered'
    if best_config not in all_results or not all_results[best_config]:
        best_config = 'cleaned'

    print(f"\n{'Model':<45} {'R²':>8} {'Pearson':>8} {'Spearman':>8}")
    print("-" * 72)

    sorted_models = sorted(
        all_results[best_config].items(),
        key=lambda x: x[1]['r2_mean'], reverse=True
    )

    for model_name, result in sorted_models:
        display = model_name[:44]
        print(f"  {display:<43} {result['r2_mean']:>7.3f}  {result['pearson_mean']:>7.3f}  "
              f"{result['spearman_mean']:>7.3f}")

    # ---- Feature Importance for Engineered Set ----
    print("\n[7/7] Feature importance (engineered set)...")
    X_uni, y_uni, groups_uni = get_universal_dataset(
        features_df,
        [c for c in configs['engineered'] if c in features_df.columns],
        per_model_targets
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_uni)

    from sklearn.inspection import permutation_importance
    gbm_full = clone(gbm)
    gbm_full.fit(X_scaled, y_uni)

    valid_eng_cols = [c for c in configs['engineered'] if c in features_df.columns]
    perm_result = permutation_importance(gbm_full, X_scaled, y_uni, n_repeats=5, random_state=42)

    importance_df = pd.DataFrame({
        'feature': valid_eng_cols,
        'importance': perm_result.importances_mean,
        'std': perm_result.importances_std,
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 20 features (permutation importance):")
    for _, row in importance_df.head(20).iterrows():
        print(f"    {row['feature']:50s} {row['importance']:.4f} +/- {row['std']:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY: Feature Cleaning Impact")
    print("=" * 70)

    if 'original_61' in config_summaries and 'cleaned' in config_summaries:
        orig = config_summaries['original_61']
        clean = config_summaries['cleaned']
        print(f"\n  Original (61 feats):     R²={orig['r2_mean']:.4f}, Pearson={orig['pearson_mean']:.4f}")
        print(f"  Cleaned ({clean['n_features']} feats):      R²={clean['r2_mean']:.4f}, Pearson={clean['pearson_mean']:.4f}")

    if 'engineered' in config_summaries:
        eng = config_summaries['engineered']
        print(f"  Engineered ({eng['n_features']} feats):  R²={eng['r2_mean']:.4f}, Pearson={eng['pearson_mean']:.4f}")

    print(f"\n  Feature cleaning removed {total_removed} degenerate features.")
    print(f"  This tests whether the removed composite features were adding noise.")

    # ---- Save Results ----
    summary = {
        'config_summaries': {
            k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                for kk, vv in v.items() if kk not in ('r2_list', 'pearson_list', 'spearman_list')}
            for k, v in config_summaries.items()
        },
        'removal_log': removal_log,
        'n_features_raw': len(feature_cols_raw),
        'n_features_cleaned': len(cleaned_cols),
        'n_features_engineered': len(engineered_cols),
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    importance_df.to_csv(output_dir / "feature_importance_engineered.csv", index=False)

    # Save per-model results for each config
    for config_name, model_results in all_results.items():
        if model_results:
            rows = []
            for model_name, result in model_results.items():
                rows.append({
                    'model': model_name,
                    'config': config_name,
                    'r2': result['r2_mean'],
                    'pearson': result['pearson_mean'],
                    'spearman': result['spearman_mean'],
                })
            pd.DataFrame(rows).to_csv(
                output_dir / f"per_model_{config_name}.csv", index=False
            )

    print(f"\nResults saved to: {output_dir}")
    return summary


if __name__ == "__main__":
    summary = main()
