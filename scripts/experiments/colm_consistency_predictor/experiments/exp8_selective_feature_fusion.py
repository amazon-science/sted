#!/usr/bin/env python3
"""
Experiment 8: Selective Feature Fusion (COLM 2026)

Problem from Exp7: Adding all 36 new features to the 51 original features
increased dimensionality from 51->87 and hurt performance (R² 0.109 -> 0.084).
This despite the new features individually ranking highly in importance.

Solution: Use principled feature selection to find the optimal subset of
new features to add to the original cleaned set.

Methods:
1. Forward selection using POOLED universal dataset (21x faster than per-model)
2. Top-K by importance from exp7
3. Correlation-filtered: only add low-collinearity features
4. Final validation: per-model evaluation of best configs
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, ttest_rel

try:
    import spacy
    _nlp = spacy.load('en_core_web_sm')
    SPACY_OK = True
except:
    SPACY_OK = False
    print("WARNING: spaCy not available, results will be suboptimal")

# Import feature extractor from exp7
from exp7_linguistic_deep_features import extract_all_features

# Import cleaning utilities from exp6
from exp6_cleaned_features import (
    COMPOSITE_FEATURES, DEAD_FEATURES, PROXY_FEATURES,
    clean_features, add_engineered_features
)


# ============================================================
# FAST EVALUATION USING POOLED DATASET
# ============================================================

def build_pooled_dataset(combined_df, feature_cols, per_model_targets):
    """Build a single pooled dataset across all models for fast evaluation."""
    all_X, all_y, all_groups = [], [], []
    for model_name, target_df in per_model_targets.items():
        merged = combined_df.merge(target_df, on='sample_idx', how='inner')
        if len(merged) < 50:
            continue
        valid = [c for c in feature_cols if c in merged.columns]
        if not valid:
            continue
        all_X.append(merged[valid].values)
        all_y.append(merged['c_mean_avg'].values)
        all_groups.append(merged['sample_idx'].values)
    if not all_X:
        return None, None, None
    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_groups)


def fast_eval_pooled(X, y, groups, model, n_splits=5):
    """Fast GroupKFold evaluation on pooled dataset."""
    gkf = GroupKFold(n_splits=n_splits)
    r2s = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        m = clone(model)
        m.fit(X_tr, y_tr)
        yp = np.clip(m.predict(X_te), 0, 1)
        r2s.append(r2_score(y_te, yp))
    return np.mean(r2s)


def fast_forward_select(combined_df, orig_cols, new_cols, per_model_targets, model, max_steps=15):
    """Fast forward selection using pooled dataset."""
    # We need ALL columns (orig + all new) in the pooled dataset
    all_possible_cols = orig_cols + new_cols
    X_full, y, groups = build_pooled_dataset(combined_df, all_possible_cols, per_model_targets)
    if X_full is None:
        return [], 0

    # Map column names to indices
    valid_cols = [c for c in all_possible_cols if c in combined_df.columns]
    col_to_idx = {c: i for i, c in enumerate(valid_cols)}

    # Start with original columns
    current_idx = [col_to_idx[c] for c in orig_cols if c in col_to_idx]
    baseline_r2 = fast_eval_pooled(X_full[:, current_idx], y, groups, model)
    current_r2 = baseline_r2

    remaining = [c for c in new_cols if c in col_to_idx]
    selected = []

    for step in range(min(max_steps, len(remaining))):
        best_feat = None
        best_r2 = current_r2

        for candidate in remaining:
            trial_idx = current_idx + [col_to_idx[candidate]]
            trial_r2 = fast_eval_pooled(X_full[:, trial_idx], y, groups, model)
            if trial_r2 > best_r2:
                best_r2 = trial_r2
                best_feat = candidate

        if best_feat is None:
            print(f"  Step {step+1}: No improvement found, stopping.")
            break

        current_idx.append(col_to_idx[best_feat])
        remaining.remove(best_feat)
        current_r2 = best_r2
        delta = best_r2 - baseline_r2
        selected.append({
            'step': step + 1,
            'feature': best_feat,
            'r2_after': best_r2,
            'delta_from_baseline': delta,
        })
        print(f"  Step {step+1}: +{best_feat:<35s} R²={best_r2:.4f} (Δ={delta:+.4f})")

    return selected, baseline_r2


def evaluate_with_groupkfold(X, y, groups, model, n_splits=5):
    """Full evaluation with all metrics."""
    gkf = GroupKFold(n_splits=n_splits)
    r2s, pearsons, spearmans = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        m = clone(model)
        m.fit(X_tr, y_tr)
        yp = np.clip(m.predict(X_te), 0, 1)
        r2s.append(r2_score(y_te, yp))
        if np.std(yp) > 1e-10 and np.std(y_te) > 1e-10:
            pearsons.append(pearsonr(y_te, yp)[0])
            spearmans.append(spearmanr(y_te, yp)[0])
        else:
            pearsons.append(0.0)
            spearmans.append(0.0)
    return {
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
        'pearson_mean': np.mean(pearsons), 'spearman_mean': np.mean(spearmans),
    }


def per_model_eval(combined_df, feature_cols, per_model_targets, model):
    """Per-model evaluation."""
    results = {}
    for model_name in sorted(per_model_targets.keys()):
        target_df = per_model_targets[model_name]
        merged = combined_df.merge(target_df, on='sample_idx', how='inner')
        if len(merged) < 50:
            continue
        valid = [c for c in feature_cols if c in merged.columns]
        if not valid:
            continue
        X = merged[valid].values
        y = merged['c_mean_avg'].values
        groups = merged['sample_idx'].values
        result = evaluate_with_groupkfold(X, y, groups, model)
        results[model_name] = result
    return results


def quick_per_model_r2(combined_df, feature_cols, per_model_targets, model):
    """Quick per-model R² average."""
    r2s = []
    for model_name, target_df in per_model_targets.items():
        merged = combined_df.merge(target_df, on='sample_idx', how='inner')
        if len(merged) < 50:
            continue
        valid = [c for c in feature_cols if c in merged.columns]
        if not valid:
            continue
        X = merged[valid].values
        y = merged['c_mean_avg'].values
        groups = merged['sample_idx'].values
        result = evaluate_with_groupkfold(X, y, groups, model)
        r2s.append(result['r2_mean'])
    return np.mean(r2s) if r2s else -1.0


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("COLM 2026 Experiment 8: Selective Feature Fusion")
    print("=" * 80)

    exp_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446"
    output_dir = exp_dir / "results" / "exp8_selective_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("\n[1/7] Loading data...")
    original_features = pd.read_csv(
        exp_dir / "results" / "exp1_correlations" / "extracted_features.csv"
    )
    with open(PROJECT_ROOT / "results/toucan_exact_final/combined_consistency_metrics_results.json") as f:
        consistency_data = json.load(f)

    per_model_targets = {}
    for model, samples in consistency_data.items():
        mdf = pd.DataFrame(samples)
        agg = mdf.groupby('sample_idx').agg({'c_mean': 'mean'}).reset_index()
        agg = agg.rename(columns={'c_mean': 'c_mean_avg'})
        per_model_targets[model] = agg

    sample_indices = sorted(original_features['sample_idx'].unique())
    print(f"  Samples: {len(sample_indices)}, Models: {len(per_model_targets)}")

    # ---- Extract new features ----
    print("\n[2/7] Extracting deep linguistic features...")
    data_dir = PROJECT_ROOT / "data" / "toucan" / "samples"
    new_features_df = extract_all_features(data_dir, sample_indices)
    new_feat_cols = [c for c in new_features_df.columns if c != 'sample_idx']

    # Remove zero-variance
    zero_var = [c for c in new_feat_cols if new_features_df[c].var() == 0]
    new_feat_cols = [c for c in new_feat_cols if c not in zero_var]
    if zero_var:
        print(f"  Removed {len(zero_var)} zero-variance: {zero_var}")
    print(f"  New features available: {len(new_feat_cols)}")

    # ---- Get cleaned original features ----
    print("\n[3/7] Preparing original cleaned features...")
    orig_feat_cols = [c for c in original_features.columns
                      if c not in ['sample_idx', 'sample_id']
                      and original_features[c].dtype in ['float64', 'int64', 'float32', 'int32']
                      and c not in COMPOSITE_FEATURES and c not in DEAD_FEATURES and c not in PROXY_FEATURES
                      and original_features[c].var() > 0]

    # Remove high-collinearity among originals
    corr_matrix = original_features[orig_feat_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > 0.95].index.tolist()
        to_drop.update(high_corr)
    orig_feat_cols = [c for c in orig_feat_cols if c not in to_drop]
    print(f"  Original cleaned features: {len(orig_feat_cols)}")

    # ---- Also get engineered features from exp6 ----
    features_eng_df = original_features.copy()
    features_eng_df, eng_cleaned_cols, _ = clean_features(features_eng_df,
        [c for c in original_features.columns
         if c not in ['sample_idx', 'sample_id']
         and original_features[c].dtype in ['float64', 'int64', 'float32', 'int32']])
    features_eng_df, engineered_cols = add_engineered_features(features_eng_df, eng_cleaned_cols)
    print(f"  Engineered feature set: {len(engineered_cols)}")

    # ---- Merge datasets ----
    combined_df = original_features.merge(new_features_df, on='sample_idx', how='inner')
    combined_eng_df = features_eng_df.merge(new_features_df, on='sample_idx', how='inner')

    gbm = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )

    # ---- Method 1: Forward selection using POOLED dataset (FAST) ----
    print("\n[4/7] Forward selection (pooled dataset, ~21x faster)...")
    selected_order, pooled_baseline = fast_forward_select(
        combined_df, orig_feat_cols, new_feat_cols, per_model_targets, gbm, max_steps=15
    )
    forward_new_only = [s['feature'] for s in selected_order]
    forward_feats = orig_feat_cols + forward_new_only
    print(f"\n  Forward-selected: {len(forward_new_only)} new features added")
    print(f"  Pooled R²: {pooled_baseline:.4f} -> {selected_order[-1]['r2_after']:.4f}" if selected_order else "  No features selected")

    # ---- Method 2: Top-K by importance (from exp7) ----
    print("\n[5/7] Top-K new features by permutation importance...")

    imp_path = exp_dir / "results" / "exp7_deep_features" / "feature_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        new_ranked = imp_df[imp_df['is_new'] == 'NEW'].sort_values('importance', ascending=False)
        new_ranked_feats = [f for f in new_ranked['feature'].tolist() if f in new_feat_cols]
    else:
        new_ranked_feats = new_feat_cols

    # Evaluate top-K using pooled dataset for speed
    all_possible_cols = orig_feat_cols + new_feat_cols
    X_pool, y_pool, g_pool = build_pooled_dataset(combined_df, all_possible_cols, per_model_targets)
    valid_cols = [c for c in all_possible_cols if c in combined_df.columns]
    col_to_idx = {c: i for i, c in enumerate(valid_cols)}

    orig_idx = [col_to_idx[c] for c in orig_feat_cols if c in col_to_idx]

    topk_results = {}
    for k in [3, 5, 7, 10, 15, 20, 25, 30, 36]:
        if k > len(new_ranked_feats):
            break
        top_k_idx = orig_idx + [col_to_idx[c] for c in new_ranked_feats[:k] if c in col_to_idx]
        r2_k = fast_eval_pooled(X_pool[:, top_k_idx], y_pool, g_pool, gbm)
        topk_results[k] = r2_k
        delta = r2_k - pooled_baseline
        print(f"  Top-{k:2d}: pooled R²={r2_k:.4f} (Δ={delta:+.4f})")

    best_k = max(topk_results, key=topk_results.get) if topk_results else 5
    topk_feats = orig_feat_cols + new_ranked_feats[:best_k]

    # ---- Method 3: Correlation-filtered ----
    print("\n[6/7] Correlation-filtered new features...")

    corr_filtered = []
    for nf in new_feat_cols:
        if nf not in combined_df.columns:
            continue
        max_corr = 0
        for of in orig_feat_cols:
            if of in combined_df.columns:
                c = abs(combined_df[nf].corr(combined_df[of]))
                max_corr = max(max_corr, c)
        if max_corr < 0.7:
            corr_filtered.append((nf, max_corr))

    corr_filtered.sort(key=lambda x: x[1])
    print(f"  New features with max corr < 0.7 to originals: {len(corr_filtered)}")
    for nf, mc in corr_filtered[:10]:
        print(f"    {nf:<35s} max_corr={mc:.3f}")

    corr_filtered_cols = [nf for nf, _ in corr_filtered]
    corr_feats = orig_feat_cols + corr_filtered_cols
    corr_idx = orig_idx + [col_to_idx[c] for c in corr_filtered_cols if c in col_to_idx]
    corr_r2 = fast_eval_pooled(X_pool[:, corr_idx], y_pool, g_pool, gbm)
    print(f"  Corr-filtered pooled R²: {corr_r2:.4f} (Δ={corr_r2-pooled_baseline:+.4f})")

    # ---- Final validation: per-model eval of best configs ----
    print("\n[7/7] Per-model validation of top configurations...")

    configs = {
        'original_cleaned': (orig_feat_cols, combined_df),
        f'forward_selected_+{len(forward_new_only)}': (forward_feats, combined_df),
        f'topk_{best_k}_importance': (topk_feats, combined_df),
        'corr_filtered': (corr_feats, combined_df),
    }

    # Also add engineered combos
    eng_forward_feats = engineered_cols + forward_new_only
    eng_topk_feats = engineered_cols + new_ranked_feats[:best_k]
    configs[f'eng_+_forward_{len(forward_new_only)}'] = (eng_forward_feats, combined_eng_df)
    configs[f'eng_+_topk_{best_k}'] = (eng_topk_feats, combined_eng_df)

    all_results = {}
    for config_name, (feats, df) in configs.items():
        print(f"\n  Evaluating {config_name}...")
        results = per_model_eval(df, feats, per_model_targets, gbm)
        all_results[config_name] = results

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("SUMMARY: Per-Model Average Across All Configurations")
    print("=" * 80)

    print(f"\n{'Configuration':<40} {'# Feats':>8} {'R²':>8} {'Pearson':>8} {'Spearman':>8} {'Δ R²':>8}")
    print("-" * 82)

    baseline_avg_r2 = None
    config_avg_r2 = {}
    for config_name, results in all_results.items():
        feats, df = configs[config_name]
        n = len([c for c in feats if c in df.columns])
        avg_r2 = np.mean([r['r2_mean'] for r in results.values()])
        avg_p = np.mean([r['pearson_mean'] for r in results.values()])
        avg_s = np.mean([r['spearman_mean'] for r in results.values()])
        config_avg_r2[config_name] = avg_r2

        if config_name == 'original_cleaned':
            baseline_avg_r2 = avg_r2

        delta = avg_r2 - baseline_avg_r2 if baseline_avg_r2 is not None else 0
        print(f"  {config_name:<38} {n:>8} {avg_r2:>8.4f} {avg_p:>8.4f} {avg_s:>8.4f} {delta:>+8.4f}")

    # ---- Best config detail ----
    best_config = max(config_avg_r2, key=config_avg_r2.get)
    print(f"\n\nBest configuration: {best_config}")

    best_results = all_results[best_config]
    orig_results = all_results['original_cleaned']

    print(f"\n{'Model':<42} {'Orig':>8} {'Best':>8} {'Δ':>8}")
    print("-" * 68)

    common = sorted(set(best_results.keys()) & set(orig_results.keys()))
    csv_rows = []
    for m in sorted(common, key=lambda x: best_results[x]['r2_mean'], reverse=True):
        ro = orig_results[m]['r2_mean']
        rb = best_results[m]['r2_mean']
        d = rb - ro
        print(f"  {m[:40]:<40} {ro:>8.3f} {rb:>8.3f} {d:>+8.3f}")
        csv_rows.append({'model': m, 'r2_orig': ro, 'r2_best': rb, 'delta': d})

    avg_orig = np.mean([orig_results[m]['r2_mean'] for m in common])
    avg_best = np.mean([best_results[m]['r2_mean'] for m in common])
    print("-" * 68)
    print(f"  {'AVERAGE':<40} {avg_orig:>8.3f} {avg_best:>8.3f} {avg_best-avg_orig:>+8.3f}")

    # Paired t-test
    r2_orig_list = [orig_results[m]['r2_mean'] for m in common]
    r2_best_list = [best_results[m]['r2_mean'] for m in common]
    if len(common) >= 3:
        t_stat, p_val = ttest_rel(r2_best_list, r2_orig_list)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_val:.6f} {sig}")

    wins = sum(1 for m in common if best_results[m]['r2_mean'] > orig_results[m]['r2_mean'])
    print(f"  Wins: {wins}/{len(common)} ({100*wins/len(common):.0f}%)")

    # Pearson/Spearman
    avg_pearson_best = np.mean([best_results[m]['pearson_mean'] for m in common])
    avg_spearman_best = np.mean([best_results[m]['spearman_mean'] for m in common])
    avg_pearson_orig = np.mean([orig_results[m]['pearson_mean'] for m in common])
    avg_spearman_orig = np.mean([orig_results[m]['spearman_mean'] for m in common])
    print(f"\n  Pearson:  orig={avg_pearson_orig:.4f}, best={avg_pearson_best:.4f} (Δ={avg_pearson_best-avg_pearson_orig:+.4f})")
    print(f"  Spearman: orig={avg_spearman_orig:.4f}, best={avg_spearman_best:.4f} (Δ={avg_spearman_best-avg_spearman_orig:+.4f})")

    # ---- Save results ----
    pd.DataFrame(selected_order).to_csv(output_dir / "forward_selection_order.csv", index=False)
    pd.DataFrame(csv_rows).to_csv(output_dir / "per_model_best_vs_orig.csv", index=False)

    summary = {
        'baseline_r2': float(baseline_avg_r2),
        'forward_selected_features': forward_new_only,
        'forward_n_total': len(forward_feats),
        f'best_topk': best_k,
        f'topk_features': new_ranked_feats[:best_k],
        'corr_filtered_features': corr_filtered_cols,
        'best_config': best_config,
        'best_r2': float(config_avg_r2[best_config]),
        'improvement': float(config_avg_r2[best_config] - baseline_avg_r2),
        'all_config_r2': {k: float(v) for k, v in config_avg_r2.items()},
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return summary


if __name__ == "__main__":
    main()
