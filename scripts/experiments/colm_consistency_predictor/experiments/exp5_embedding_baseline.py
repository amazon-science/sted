#!/usr/bin/env python3
"""
Experiment 5: Embedding Baseline Comparison (COLM 2026)

Critical experiment: Compare hand-crafted 61 linguistic features against
sentence embedding baselines and the combination of both.

Configurations tested:
1. Random baseline (predict mean)
2. Prompt length only (1 feature)
3. Schema features only (4 features)
4. Linguistic features only (61 features)
5. Sentence embeddings only (384-dim from all-MiniLM-L6-v2)
6. Linguistic + Embeddings combined (61 + 384 = 445 features)
7. PCA-reduced embeddings (50-dim)
8. Embeddings + schema (384 + 4 = 388 features)

Evaluation: GroupKFold, R², Pearson, Spearman per model and universal
"""

import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Suppress numpy warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon


def load_data():
    """Load features, embeddings, and consistency data."""
    # Load linguistic features
    exp_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446"
    features_df = pd.read_csv(
        exp_dir / "results" / "exp1_correlations" / "extracted_features.csv"
    )

    feature_cols = [c for c in features_df.columns
                    if c not in ['sample_idx', 'sample_id']
                    and features_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    # Remove zero-variance
    variances = features_df[feature_cols].var()
    feature_cols = [c for c in feature_cols if variances[c] > 0]

    schema_cols = [c for c in feature_cols if c.startswith('schema_')]
    linguistic_cols = [c for c in feature_cols if not c.startswith('schema_')]

    # Load consistency data
    with open(PROJECT_ROOT / "results" / "toucan_exact_final" / "combined_consistency_metrics_results.json") as f:
        consistency_data = json.load(f)

    # Build per-model targets (avg across temps)
    per_model_targets = {}
    for model, samples in consistency_data.items():
        model_df = pd.DataFrame(samples)
        agg = model_df.groupby('sample_idx')['c_mean'].mean().reset_index()
        agg.columns = ['sample_idx', 'c_mean_avg']
        per_model_targets[model] = agg

    return features_df, feature_cols, schema_cols, linguistic_cols, per_model_targets


def generate_embeddings():
    """Generate sentence embeddings for all 1006 prompts."""
    cache_path = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446" / "results" / "prompt_embeddings.npy"
    idx_path = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446" / "results" / "prompt_embedding_indices.json"

    if cache_path.exists() and idx_path.exists():
        print("  Loading cached embeddings...")
        embeddings = np.load(str(cache_path))
        with open(idx_path) as f:
            indices = json.load(f)
        print(f"  Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
        return embeddings, indices

    print("  Generating embeddings with all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer

    # Load Toucan prompts
    with open(PROJECT_ROOT / "data" / "toucan" / "toucan_tool_calls_1006.json") as f:
        toucan_data = json.load(f)

    prompts = [item['question'] for item in toucan_data]
    indices = list(range(len(prompts)))

    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(prompts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), embeddings)
    with open(idx_path, 'w') as f:
        json.dump(indices, f)

    print(f"  Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    return embeddings, indices


def eval_groupkfold(X, y, groups, model_template, n_splits=5):
    """Evaluate with GroupKFold, return multiple metrics."""
    gkf = GroupKFold(n_splits=n_splits)
    r2s, pearsons, spearmans, maes = [], [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        m = clone(model_template)
        m.fit(X_tr, y[train_idx])
        y_pred = np.clip(m.predict(X_te), 0, 1)

        r2s.append(r2_score(y[test_idx], y_pred))
        maes.append(mean_absolute_error(y[test_idx], y_pred))
        if np.std(y_pred) > 1e-10 and np.std(y[test_idx]) > 1e-10:
            pearsons.append(pearsonr(y[test_idx], y_pred)[0])
            spearmans.append(spearmanr(y[test_idx], y_pred)[0])
        else:
            pearsons.append(0.0)
            spearmans.append(0.0)

    return {
        'r2': np.mean(r2s),
        'r2_std': np.std(r2s),
        'pearson': np.mean(pearsons),
        'spearman': np.mean(spearmans),
        'mae': np.mean(maes),
    }


def run_comparison(features_df, feature_cols, schema_cols, linguistic_cols,
                   per_model_targets, embeddings, model_name, target_df):
    """Run all configurations for a single model."""
    merged = features_df.merge(target_df, on='sample_idx', how='inner')

    X_ling = merged[feature_cols].values  # 61 features
    X_schema = merged[schema_cols].values  # 4 features
    X_linguistic_only = merged[linguistic_cols].values  # 57 features
    X_length = merged[['surface_word_count']].values  # 1 feature
    y = merged['c_mean_avg'].values
    groups = merged['sample_idx'].values

    # Get embeddings for matching samples
    sample_indices = merged['sample_idx'].values.astype(int)
    valid_mask = sample_indices < len(embeddings)
    if not np.all(valid_mask):
        # Filter
        X_ling = X_ling[valid_mask]
        X_schema = X_schema[valid_mask]
        X_linguistic_only = X_linguistic_only[valid_mask]
        X_length = X_length[valid_mask]
        y = y[valid_mask]
        groups = groups[valid_mask]
        sample_indices = sample_indices[valid_mask]

    X_emb = embeddings[sample_indices]  # 384 features

    # PCA-reduced embeddings
    pca = PCA(n_components=50, random_state=42)
    X_emb_pca = pca.fit_transform(X_emb)

    # Combined
    X_combined = np.hstack([X_ling, X_emb])  # 61 + 384
    X_emb_schema = np.hstack([X_emb, X_schema])  # 384 + 4
    X_ling_emb_pca = np.hstack([X_ling, X_emb_pca])  # 61 + 50

    gbm = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )

    results = {}

    configs = {
        'length_only': X_length,
        'schema_only': X_schema,
        'linguistic_61': X_ling,
        'embedding_384': X_emb,
        'embedding_pca50': X_emb_pca,
        'emb+schema': X_emb_schema,
        'ling+emb_combined': X_combined,
        'ling+emb_pca': X_ling_emb_pca,
    }

    for config_name, X_config in configs.items():
        try:
            result = eval_groupkfold(X_config, y, groups, gbm)
            results[config_name] = result
        except Exception as e:
            results[config_name] = {'r2': 0, 'pearson': 0, 'spearman': 0, 'mae': 0, 'error': str(e)}

    return results


def main():
    print("=" * 70)
    print("COLM 2026 Experiment 5: Embedding Baseline Comparison")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    features_df, feature_cols, schema_cols, linguistic_cols, per_model_targets = load_data()
    print(f"  Linguistic features: {len(feature_cols)} ({len(linguistic_cols)} non-schema + {len(schema_cols)} schema)")

    # Generate embeddings
    print("\n[2/4] Generating/loading embeddings...")
    embeddings, indices = generate_embeddings()

    # Run per-model comparison
    print("\n[3/4] Running per-model comparison...")
    gbm = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )

    all_results = {}
    config_r2s = {config: [] for config in [
        'length_only', 'schema_only', 'linguistic_61', 'embedding_384',
        'embedding_pca50', 'emb+schema', 'ling+emb_combined', 'ling+emb_pca'
    ]}
    config_pearsons = {config: [] for config in config_r2s}
    config_spearmans = {config: [] for config in config_r2s}

    for model_name in sorted(per_model_targets.keys()):
        target_df = per_model_targets[model_name]
        results = run_comparison(
            features_df, feature_cols, schema_cols, linguistic_cols,
            per_model_targets, embeddings, model_name, target_df
        )
        all_results[model_name] = results

        # Collect per-config scores
        for config in config_r2s:
            if config in results and 'error' not in results[config]:
                config_r2s[config].append(results[config]['r2'])
                config_pearsons[config].append(results[config]['pearson'])
                config_spearmans[config].append(results[config]['spearman'])

        # Print per-model results
        display = model_name[:35]
        ling_r2 = results.get('linguistic_61', {}).get('r2', 0)
        emb_r2 = results.get('embedding_384', {}).get('r2', 0)
        comb_r2 = results.get('ling+emb_combined', {}).get('r2', 0)
        ling_p = results.get('linguistic_61', {}).get('pearson', 0)
        emb_p = results.get('embedding_384', {}).get('pearson', 0)
        comb_p = results.get('ling+emb_combined', {}).get('pearson', 0)
        print(f"  {display:35s} Ling R²={ling_r2:.3f} Emb R²={emb_r2:.3f} "
              f"Comb R²={comb_r2:.3f} | Ling ρ={ling_p:.3f} Emb ρ={emb_p:.3f} Comb ρ={comb_p:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("[4/4] SUMMARY: EMBEDDING BASELINE COMPARISON")
    print("=" * 70)

    print(f"\n{'Configuration':<25} {'R² (avg)':>10} {'Pearson':>10} {'Spearman':>10} {'# Models':>10}")
    print("-" * 65)
    for config in ['length_only', 'schema_only', 'linguistic_61', 'embedding_384',
                    'embedding_pca50', 'emb+schema', 'ling+emb_combined', 'ling+emb_pca']:
        r2_avg = np.mean(config_r2s[config]) if config_r2s[config] else 0
        p_avg = np.mean(config_pearsons[config]) if config_pearsons[config] else 0
        s_avg = np.mean(config_spearmans[config]) if config_spearmans[config] else 0
        n = len(config_r2s[config])
        marker = ""
        if config == 'linguistic_61':
            marker = " <-- OURS"
        elif config == 'embedding_384':
            marker = " <-- BASELINE"
        elif config == 'ling+emb_combined':
            marker = " <-- COMBINED"
        print(f"  {config:<25} {r2_avg:>8.4f}   {p_avg:>8.4f}   {s_avg:>8.4f}   {n:>5}{marker}")

    # Statistical tests: linguistic vs embedding
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    ling_r2_list = config_r2s['linguistic_61']
    emb_r2_list = config_r2s['embedding_384']
    comb_r2_list = config_r2s['ling+emb_combined']
    ling_p_list = config_pearsons['linguistic_61']
    emb_p_list = config_pearsons['embedding_384']
    comb_p_list = config_pearsons['ling+emb_combined']

    min_len = min(len(ling_r2_list), len(emb_r2_list), len(comb_r2_list))

    # Paired t-tests
    print("\nPaired t-tests (R²):")
    t, p = ttest_rel(ling_r2_list[:min_len], emb_r2_list[:min_len])
    print(f"  Linguistic vs Embedding: t={t:.3f}, p={p:.6f} "
          f"{'(Ling wins)' if np.mean(ling_r2_list) > np.mean(emb_r2_list) else '(Emb wins)'}")

    t, p = ttest_rel(comb_r2_list[:min_len], ling_r2_list[:min_len])
    print(f"  Combined vs Linguistic:  t={t:.3f}, p={p:.6f} "
          f"{'(Comb wins)' if np.mean(comb_r2_list) > np.mean(ling_r2_list) else '(Ling wins)'}")

    t, p = ttest_rel(comb_r2_list[:min_len], emb_r2_list[:min_len])
    print(f"  Combined vs Embedding:   t={t:.3f}, p={p:.6f} "
          f"{'(Comb wins)' if np.mean(comb_r2_list) > np.mean(emb_r2_list) else '(Emb wins)'}")

    print("\nPaired t-tests (Pearson):")
    t, p = ttest_rel(ling_p_list[:min_len], emb_p_list[:min_len])
    print(f"  Linguistic vs Embedding: t={t:.3f}, p={p:.6f} "
          f"{'(Ling wins)' if np.mean(ling_p_list) > np.mean(emb_p_list) else '(Emb wins)'}")

    t, p = ttest_rel(comb_p_list[:min_len], ling_p_list[:min_len])
    print(f"  Combined vs Linguistic:  t={t:.3f}, p={p:.6f} "
          f"{'(Comb wins)' if np.mean(comb_p_list) > np.mean(ling_p_list) else '(Ling wins)'}")

    t, p = ttest_rel(comb_p_list[:min_len], emb_p_list[:min_len])
    print(f"  Combined vs Embedding:   t={t:.3f}, p={p:.6f} "
          f"{'(Comb wins)' if np.mean(comb_p_list) > np.mean(emb_p_list) else '(Emb wins)'}")

    # Wilcoxon
    print("\nWilcoxon signed-rank tests (R²):")
    try:
        w, p = wilcoxon(np.array(ling_r2_list[:min_len]) - np.array(emb_r2_list[:min_len]))
        print(f"  Linguistic vs Embedding: W={w:.0f}, p={p:.6f}")
    except:
        print(f"  Linguistic vs Embedding: could not compute")

    try:
        w, p = wilcoxon(np.array(comb_r2_list[:min_len]) - np.array(ling_r2_list[:min_len]))
        print(f"  Combined vs Linguistic:  W={w:.0f}, p={p:.6f}")
    except:
        print(f"  Combined vs Linguistic:  could not compute")

    # Win rates
    print("\nWin rates (R²):")
    ling_wins_emb = sum(1 for l, e in zip(ling_r2_list, emb_r2_list) if l > e)
    comb_wins_ling = sum(1 for c, l in zip(comb_r2_list, ling_r2_list) if c > l)
    comb_wins_emb = sum(1 for c, e in zip(comb_r2_list, emb_r2_list) if c > e)
    n = min_len
    print(f"  Linguistic > Embedding: {ling_wins_emb}/{n} ({100*ling_wins_emb/n:.0f}%)")
    print(f"  Combined > Linguistic:  {comb_wins_ling}/{n} ({100*comb_wins_ling/n:.0f}%)")
    print(f"  Combined > Embedding:   {comb_wins_emb}/{n} ({100*comb_wins_emb/n:.0f}%)")

    # LOMO with embeddings
    print("\n" + "=" * 70)
    print("LOMO GENERALIZATION WITH EMBEDDINGS")
    print("=" * 70)

    models_list = sorted(per_model_targets.keys())

    for config_name, get_X in [
        ('linguistic_61', lambda merged: merged[feature_cols].values),
        ('embedding_384', lambda merged: embeddings[merged['sample_idx'].values.astype(int)]),
        ('ling+emb_combined', lambda merged: np.hstack([
            merged[feature_cols].values,
            embeddings[merged['sample_idx'].values.astype(int)]
        ])),
    ]:
        lomo_pearsons = []
        for hold_out in models_list:
            train_X_list, train_y_list = [], []
            for m in models_list:
                if m == hold_out:
                    continue
                target_df = per_model_targets[m]
                merged = features_df.merge(target_df, on='sample_idx', how='inner')
                valid = merged['sample_idx'].values.astype(int) < len(embeddings)
                merged = merged[valid]
                X = get_X(merged)
                train_X_list.append(X)
                train_y_list.append(merged['c_mean_avg'].values)

            X_train = np.vstack(train_X_list)
            y_train = np.concatenate(train_y_list)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            m_gbm = clone(gbm)
            m_gbm.fit(X_train_s, y_train)

            # Test on held-out
            target_df = per_model_targets[hold_out]
            merged = features_df.merge(target_df, on='sample_idx', how='inner')
            valid = merged['sample_idx'].values.astype(int) < len(embeddings)
            merged = merged[valid]
            X_test = get_X(merged)
            y_test = merged['c_mean_avg'].values

            X_test_s = scaler.transform(X_test)
            y_pred = np.clip(m_gbm.predict(X_test_s), 0, 1)

            if np.std(y_pred) > 1e-10:
                lomo_pearsons.append(pearsonr(y_test, y_pred)[0])

        print(f"  {config_name:25s} LOMO Pearson: {np.mean(lomo_pearsons):.4f} ± {np.std(lomo_pearsons):.4f}")

    # Save results
    output_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446" / "results" / "exp5_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        config: {
            'r2_mean': float(np.mean(config_r2s[config])),
            'r2_std': float(np.std(config_r2s[config])),
            'pearson_mean': float(np.mean(config_pearsons[config])),
            'spearman_mean': float(np.mean(config_spearmans[config])),
            'n_models': len(config_r2s[config]),
        }
        for config in config_r2s if config_r2s[config]
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
