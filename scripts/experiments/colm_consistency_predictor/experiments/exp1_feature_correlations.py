#!/usr/bin/env python3
"""
Experiment 1: Feature-Consistency Correlation Analysis

COLM 2026 Consistency Predictor

This experiment:
1. Loads Toucan dataset prompts and tools
2. Extracts all 67 prompt features
3. Loads pre-computed STED consistency scores
4. Computes correlations between features and consistency metrics
5. Identifies top predictive features

Key research questions:
- Which feature categories best predict consistency?
- Do semantic/pragmatic features outperform surface features?
- What is the correlation between task clarity and consistency?
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from features import extract_all_features, AllFeatures


def load_toucan_prompts(
    base_dir: str = "llm_gen_results/toucan"
) -> Dict[int, Dict[str, Any]]:
    """
    Load all unique prompts from Toucan dataset.

    Returns:
        Dict mapping sample_idx to {prompt, tools}
    """
    prompts = {}
    base_path = PROJECT_ROOT / base_dir

    # Find any run directory to get the prompts (they're the same across runs)
    for model_dir in base_path.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("generations-"):
            for run_dir in model_dir.iterdir():
                if run_dir.is_dir():
                    results_file = run_dir / "intermediate_results.json"
                    if results_file.exists():
                        with open(results_file) as f:
                            data = json.load(f)
                        for idx, item in enumerate(data):
                            if idx not in prompts:
                                prompts[idx] = {
                                    'sample_id': item.get('sample_id', f'sample_{idx}'),
                                    'prompt': item.get('query', ''),
                                    'tools': item.get('tools', [])
                                }
                        if len(prompts) > 0:
                            return prompts

    return prompts


def load_consistency_metrics(
    results_file: str = "results/toucan_exact_final/combined_consistency_metrics_results.json"
) -> pd.DataFrame:
    """
    Load pre-computed consistency metrics.

    Returns:
        DataFrame with sample_idx, model, temperature, and consistency metrics
    """
    results_path = PROJECT_ROOT / results_file

    with open(results_path) as f:
        data = json.load(f)

    rows = []
    for model, samples in data.items():
        for sample in samples:
            rows.append({
                'model': model,
                'sample_idx': sample['sample_idx'],
                'temperature': sample.get('temperature', 0.5),
                'c_mean': sample.get('c_mean', sample.get('mean_similarity', 0)),
                'd_std': sample.get('d_std', 0),
                'stability_score': sample.get('stability_score', 0),
                'validity_rate': sample.get('validity_rate', 1),
                'c_adj': sample.get('c_adj', sample.get('c_mean', 0)),
            })

    return pd.DataFrame(rows)


def extract_features_for_dataset(
    prompts: Dict[int, Dict[str, Any]],
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract all 67 features for each prompt in the dataset.
    """
    rows = []
    n = len(prompts)

    for i, (idx, item) in enumerate(prompts.items()):
        if show_progress and (i + 1) % 50 == 0:
            print(f"Extracting features: {i+1}/{n}...")

        features = extract_all_features(item['prompt'], item.get('tools', []))
        feature_dict = features.to_dict()
        feature_dict['sample_idx'] = idx
        feature_dict['sample_id'] = item.get('sample_id', f'sample_{idx}')

        rows.append(feature_dict)

    return pd.DataFrame(rows)


def compute_correlations(
    features_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    target_metric: str = 'c_mean',
    method: str = 'spearman'
) -> pd.DataFrame:
    """
    Compute correlations between features and consistency metrics.

    Args:
        features_df: DataFrame with feature columns
        metrics_df: DataFrame with consistency metrics
        target_metric: Which metric to correlate with
        method: 'pearson' or 'spearman'

    Returns:
        DataFrame with feature correlations
    """
    # Merge features with metrics (aggregate metrics across models/temps)
    # Use mean across all models and temperatures as baseline
    agg_metrics = metrics_df.groupby('sample_idx').agg({
        target_metric: 'mean',
        'd_std': 'mean',
        'stability_score': 'mean',
        'validity_rate': 'mean'
    }).reset_index()

    merged = features_df.merge(agg_metrics, on='sample_idx', how='inner')

    # Get feature columns (exclude metadata)
    feature_cols = [c for c in features_df.columns
                   if c not in ['sample_idx', 'sample_id']]

    results = []
    corr_func = spearmanr if method == 'spearman' else pearsonr

    for col in feature_cols:
        if merged[col].std() == 0:
            # Skip constant features
            continue

        corr, pval = corr_func(merged[col], merged[target_metric])
        results.append({
            'feature': col,
            'correlation': corr,
            'p_value': pval,
            'significant': pval < 0.05,
            'category': col.split('_')[0]  # surface, semantic, pragmatic, schema
        })

    return pd.DataFrame(results).sort_values('correlation', key=abs, ascending=False)


def compute_model_specific_correlations(
    features_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    target_metric: str = 'c_mean',
    method: str = 'spearman'
) -> Dict[str, pd.DataFrame]:
    """
    Compute correlations for each model separately.
    """
    results = {}
    corr_func = spearmanr if method == 'spearman' else pearsonr

    for model in metrics_df['model'].unique():
        model_metrics = metrics_df[metrics_df['model'] == model].groupby('sample_idx').agg({
            target_metric: 'mean'
        }).reset_index()

        merged = features_df.merge(model_metrics, on='sample_idx', how='inner')

        feature_cols = [c for c in features_df.columns
                       if c not in ['sample_idx', 'sample_id']]

        model_results = []
        for col in feature_cols:
            if merged[col].std() == 0:
                continue
            corr, pval = corr_func(merged[col], merged[target_metric])
            model_results.append({
                'feature': col,
                'correlation': corr,
                'p_value': pval,
            })

        results[model] = pd.DataFrame(model_results).sort_values(
            'correlation', key=abs, ascending=False
        )

    return results


def analyze_category_importance(corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze importance by feature category.
    """
    category_stats = corr_df.groupby('category').agg({
        'correlation': ['mean', 'std', 'max', 'min', 'count'],
        'significant': 'sum'
    }).round(4)

    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
    category_stats = category_stats.rename(columns={
        'correlation_mean': 'avg_corr',
        'correlation_std': 'std_corr',
        'correlation_max': 'max_corr',
        'correlation_min': 'min_corr',
        'correlation_count': 'n_features',
        'significant_sum': 'n_significant'
    })

    # Mean absolute correlation (better measure of predictive power)
    abs_corr = corr_df.groupby('category')['correlation'].apply(
        lambda x: np.abs(x).mean()
    )
    category_stats['avg_abs_corr'] = abs_corr

    return category_stats.sort_values('avg_abs_corr', ascending=False)


def main():
    print("=" * 70)
    print("COLM 2026 Experiment 1: Feature-Consistency Correlations")
    print("=" * 70)

    # Create output directory
    output_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446" / "results" / "exp1_correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load prompts
    print("\n[1/4] Loading Toucan prompts...")
    prompts = load_toucan_prompts()
    print(f"  Loaded {len(prompts)} unique prompts")

    # Step 2: Extract features
    print("\n[2/4] Extracting 67 features from each prompt...")
    features_df = extract_features_for_dataset(prompts)
    print(f"  Features extracted: {features_df.shape}")

    # Save features
    features_df.to_csv(output_dir / "extracted_features.csv", index=False)

    # Step 3: Load consistency metrics
    print("\n[3/4] Loading pre-computed consistency metrics...")
    metrics_df = load_consistency_metrics()
    print(f"  Metrics loaded: {metrics_df.shape}")
    print(f"  Models: {metrics_df['model'].nunique()}")
    print(f"  Unique samples: {metrics_df['sample_idx'].nunique()}")

    # Step 4: Compute correlations
    print("\n[4/4] Computing feature-consistency correlations...")

    # Overall correlations with c_mean
    corr_df = compute_correlations(features_df, metrics_df, 'c_mean', 'spearman')

    print("\n" + "=" * 70)
    print("TOP 20 FEATURES CORRELATED WITH CONSISTENCY (c_mean)")
    print("=" * 70)
    print(corr_df.head(20).to_string())

    # Category analysis
    print("\n" + "=" * 70)
    print("FEATURE CATEGORY IMPORTANCE")
    print("=" * 70)
    category_stats = analyze_category_importance(corr_df)
    print(category_stats.to_string())

    # Model-specific correlations
    print("\n" + "=" * 70)
    print("MODEL-SPECIFIC TOP FEATURES")
    print("=" * 70)
    model_corrs = compute_model_specific_correlations(features_df, metrics_df)
    for model, df in model_corrs.items():
        top3 = df.head(3)['feature'].tolist()
        top_corr = df.head(1)['correlation'].values[0]
        print(f"  {model}: {top3[0]} ({top_corr:.3f})")

    # Correlations with other metrics
    print("\n" + "=" * 70)
    print("CORRELATIONS WITH DIFFERENT METRICS")
    print("=" * 70)
    for metric in ['c_mean', 'd_std', 'stability_score', 'validity_rate']:
        corr_metric = compute_correlations(features_df, metrics_df, metric, 'spearman')
        top_feat = corr_metric.iloc[0]
        print(f"  {metric}: {top_feat['feature']} (r={top_feat['correlation']:.3f})")

    # Save results
    corr_df.to_csv(output_dir / "feature_correlations.csv", index=False)
    category_stats.to_csv(output_dir / "category_importance.csv")

    # Save model-specific results
    model_corr_dict = {model: df.to_dict('records') for model, df in model_corrs.items()}
    with open(output_dir / "model_specific_correlations.json", 'w') as f:
        json.dump(model_corr_dict, f, indent=2)

    # Summary statistics
    summary = {
        'n_prompts': len(prompts),
        'n_features': len([c for c in features_df.columns if c not in ['sample_idx', 'sample_id']]),
        'n_significant_features': int(corr_df['significant'].sum()),
        'top_feature': corr_df.iloc[0]['feature'],
        'top_correlation': float(corr_df.iloc[0]['correlation']),
        'category_ranking': category_stats.index.tolist(),
        'avg_abs_corr_by_category': category_stats['avg_abs_corr'].to_dict()
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  Files: extracted_features.csv, feature_correlations.csv, category_importance.csv")

    return corr_df, category_stats


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    corr_df, category_stats = main()
