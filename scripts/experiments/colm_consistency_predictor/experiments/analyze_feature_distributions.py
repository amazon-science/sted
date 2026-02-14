#!/usr/bin/env python3
"""
Feature Distribution Analysis

Analyzes the distribution of all 67 features:
- Non-zero rate (how many samples have this feature)
- Mean, std, min, max, median
- Percentiles
- Histograms saved as data

Important for understanding feature sparsity and coverage.
"""

import json
import sys
from pathlib import Path
import warnings

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def analyze_feature_distributions(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution statistics for each feature.
    """
    feature_cols = [c for c in features_df.columns if c not in ['sample_idx', 'sample_id']]

    stats = []
    for col in feature_cols:
        values = features_df[col].values
        non_zero = values[values != 0]

        stat = {
            'feature': col,
            'category': col.split('_')[0],
            # Coverage
            'n_samples': len(values),
            'n_nonzero': len(non_zero),
            'nonzero_rate': len(non_zero) / len(values),
            'n_zero': len(values) - len(non_zero),
            'zero_rate': 1 - len(non_zero) / len(values),
            # Basic stats (all values)
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            # Percentiles
            'p25': np.percentile(values, 25),
            'p75': np.percentile(values, 75),
            'p90': np.percentile(values, 90),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            # Non-zero stats
            'nonzero_mean': np.mean(non_zero) if len(non_zero) > 0 else 0,
            'nonzero_std': np.std(non_zero) if len(non_zero) > 0 else 0,
            # Unique values (for discrete features)
            'n_unique': len(np.unique(values)),
        }
        stats.append(stat)

    return pd.DataFrame(stats)


def compute_histograms(features_df: pd.DataFrame, n_bins: int = 20) -> dict:
    """
    Compute histogram data for each feature.
    """
    feature_cols = [c for c in features_df.columns if c not in ['sample_idx', 'sample_id']]

    histograms = {}
    for col in feature_cols:
        values = features_df[col].values

        # Handle binary/discrete features
        unique_vals = np.unique(values)
        if len(unique_vals) <= 10:
            # Discrete: use value counts
            counts = pd.Series(values).value_counts().sort_index()
            histograms[col] = {
                'type': 'discrete',
                'values': counts.index.tolist(),
                'counts': counts.values.tolist(),
            }
        else:
            # Continuous: use histogram
            hist, bin_edges = np.histogram(values, bins=n_bins)
            histograms[col] = {
                'type': 'continuous',
                'bin_edges': bin_edges.tolist(),
                'counts': hist.tolist(),
            }

    return histograms


def summarize_by_category(dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize feature coverage by category.
    """
    summary = dist_df.groupby('category').agg({
        'nonzero_rate': ['mean', 'min', 'max'],
        'mean': 'mean',
        'std': 'mean',
        'n_unique': 'mean',
    }).round(3)

    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    return summary


def main():
    print("=" * 70)
    print("Feature Distribution Analysis")
    print("=" * 70)

    # Load extracted features
    exp_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446"
    features_path = exp_dir / "results" / "exp1_correlations" / "extracted_features.csv"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        print("Run exp1_feature_correlations.py first")
        return

    features_df = pd.read_csv(features_path)
    print(f"\nLoaded {len(features_df)} samples with {len(features_df.columns)-2} features")

    # Compute distributions
    print("\nComputing feature distributions...")
    dist_df = analyze_feature_distributions(features_df)

    # Sort by non-zero rate
    dist_sorted = dist_df.sort_values('nonzero_rate', ascending=True)

    print("\n" + "=" * 70)
    print("FEATURES WITH LOWEST NON-ZERO RATE (Sparse Features)")
    print("=" * 70)
    sparse_features = dist_sorted.head(15)[['feature', 'category', 'nonzero_rate', 'n_nonzero', 'mean', 'max']]
    print(sparse_features.to_string())

    print("\n" + "=" * 70)
    print("FEATURES WITH HIGHEST NON-ZERO RATE (Dense Features)")
    print("=" * 70)
    dense_features = dist_sorted.tail(15)[['feature', 'category', 'nonzero_rate', 'mean', 'std', 'max']]
    print(dense_features.to_string())

    # Category summary
    print("\n" + "=" * 70)
    print("COVERAGE BY CATEGORY")
    print("=" * 70)
    cat_summary = summarize_by_category(dist_df)
    print(cat_summary.to_string())

    # Compute histograms
    print("\nComputing histograms...")
    histograms = compute_histograms(features_df)

    # Save results
    output_dir = exp_dir / "results" / "feature_distributions"
    output_dir.mkdir(parents=True, exist_ok=True)

    dist_df.to_csv(output_dir / "feature_statistics.csv", index=False)
    cat_summary.to_csv(output_dir / "category_summary.csv")

    with open(output_dir / "histograms.json", 'w') as f:
        json.dump(histograms, f, indent=2)

    # Summary JSON
    summary = {
        'n_samples': len(features_df),
        'n_features': len(dist_df),
        'overall_nonzero_rate': float(dist_df['nonzero_rate'].mean()),
        'sparsest_features': dist_sorted.head(10)[['feature', 'nonzero_rate']].to_dict('records'),
        'densest_features': dist_sorted.tail(10)[['feature', 'nonzero_rate']].to_dict('records'),
        'category_coverage': cat_summary['nonzero_rate_mean'].to_dict(),
        'features_with_zero_variance': dist_df[dist_df['std'] == 0]['feature'].tolist(),
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  Files: feature_statistics.csv, histograms.json, summary.json")

    # Print warning about sparse features
    very_sparse = dist_df[dist_df['nonzero_rate'] < 0.1]
    if len(very_sparse) > 0:
        print(f"\n  WARNING: {len(very_sparse)} features have <10% non-zero rate")
        print(f"  Consider removing or transforming: {very_sparse['feature'].tolist()}")

    return dist_df


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    dist_df = main()
