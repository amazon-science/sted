#!/usr/bin/env python
"""
Calculate inconsistent sample rate for each model at each temperature.

This script analyzes consistency metrics results and computes the rate of
inconsistent samples based on configurable thresholds.
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_consistency_results(results_path: str) -> Dict[str, List[Dict]]:
    """Load consistency metrics results JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def calculate_inconsistency_rate(
    results: Dict[str, List[Dict]],
    consistency_threshold: float = 0.8,
    metric: str = "consistency_coefficient",
) -> pd.DataFrame:
    """
    Calculate inconsistent sample rate for each model at each temperature.

    Args:
        results: Dict mapping model name to list of sample results
        consistency_threshold: Samples below this threshold are considered inconsistent
        metric: Which metric to use for determining inconsistency
               Options: consistency_coefficient, mean_similarity, penalized_consistency_coefficient

    Returns:
        DataFrame with columns: model, temperature, total_samples, inconsistent_samples,
                                inconsistency_rate, mean_consistency, std_consistency
    """
    rows = []

    for model_name, samples in results.items():
        # Group samples by temperature
        temp_groups = defaultdict(list)
        for sample in samples:
            temp = sample.get("temperature", 0.0)
            temp_groups[temp].append(sample)

        for temp in sorted(temp_groups.keys()):
            temp_samples = temp_groups[temp]
            total = len(temp_samples)

            # Get metric values (handle None/missing)
            metric_values = []
            for s in temp_samples:
                val = s.get(metric)
                if val is not None:
                    metric_values.append(val)

            if not metric_values:
                continue

            # Count inconsistent samples
            inconsistent = sum(1 for v in metric_values if v < consistency_threshold)

            rows.append({
                "model": model_name,
                "temperature": temp,
                "total_samples": total,
                "valid_samples": len(metric_values),
                "inconsistent_samples": inconsistent,
                "inconsistency_rate": inconsistent / len(metric_values) if metric_values else 0,
                "mean_consistency": np.mean(metric_values),
                "std_consistency": np.std(metric_values),
                "min_consistency": np.min(metric_values),
                "max_consistency": np.max(metric_values),
            })

    return pd.DataFrame(rows)


def calculate_multi_threshold_rates(
    results: Dict[str, List[Dict]],
    thresholds: List[float] = [0.7, 0.8, 0.9, 0.95],
    metric: str = "consistency_coefficient",
) -> pd.DataFrame:
    """
    Calculate inconsistency rates at multiple thresholds.

    Returns DataFrame with columns for each threshold.
    """
    rows = []

    for model_name, samples in results.items():
        temp_groups = defaultdict(list)
        for sample in samples:
            temp = sample.get("temperature", 0.0)
            temp_groups[temp].append(sample)

        for temp in sorted(temp_groups.keys()):
            temp_samples = temp_groups[temp]

            metric_values = [s.get(metric) for s in temp_samples if s.get(metric) is not None]

            if not metric_values:
                continue

            row = {
                "model": model_name,
                "temperature": temp,
                "n_samples": len(metric_values),
                "mean": np.mean(metric_values),
            }

            for thresh in thresholds:
                inconsistent = sum(1 for v in metric_values if v < thresh)
                row[f"rate_below_{thresh}"] = inconsistent / len(metric_values)

            rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame, metric_name: str = "consistency_coefficient"):
    """Print a summary table of inconsistency rates."""
    print("\n" + "=" * 100)
    print(f"INCONSISTENCY RATE SUMMARY (metric: {metric_name})")
    print("=" * 100)

    # Pivot table: models as rows, temperatures as columns
    pivot = df.pivot_table(
        index='model',
        columns='temperature',
        values='inconsistency_rate',
        aggfunc='mean'
    )

    # Format as percentages
    pivot_formatted = pivot.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")

    print("\nInconsistency Rate by Model and Temperature:")
    print(pivot_formatted.to_string())

    # Summary stats per model
    print("\n" + "-" * 80)
    print("Model Summary (averaged across temperatures):")
    print("-" * 80)

    model_summary = df.groupby('model').agg({
        'inconsistency_rate': ['mean', 'std', 'min', 'max'],
        'mean_consistency': 'mean',
    }).round(4)

    print(model_summary.to_string())

    # Summary stats per temperature
    print("\n" + "-" * 80)
    print("Temperature Summary (averaged across models):")
    print("-" * 80)

    temp_summary = df.groupby('temperature').agg({
        'inconsistency_rate': ['mean', 'std'],
        'mean_consistency': 'mean',
        'inconsistent_samples': 'sum',
        'valid_samples': 'sum',
    }).round(4)

    print(temp_summary.to_string())


def export_results(
    df: pd.DataFrame,
    output_path: str,
    format: str = "csv"
):
    """Export results to file."""
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient='records', indent=2)
    elif format == "markdown":
        with open(output_path, 'w') as f:
            f.write(df.to_markdown(index=False))

    print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate inconsistent sample rate for each model at each temperature"
    )
    parser.add_argument(
        "--input",
        default="results/sharegpt_titan512/combined_consistency_metrics_results.json",
        help="Path to consistency metrics results JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (optional)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Consistency threshold below which samples are considered inconsistent (default: 0.8)",
    )
    parser.add_argument(
        "--metric",
        choices=["consistency_coefficient", "mean_similarity", "penalized_consistency_coefficient"],
        default="consistency_coefficient",
        help="Which metric to use for determining inconsistency",
    )
    parser.add_argument(
        "--multi-threshold",
        action="store_true",
        help="Calculate rates at multiple thresholds (0.7, 0.8, 0.9, 0.95)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "markdown"],
        default="csv",
        help="Output format",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading results from: {args.input}")
    results = load_consistency_results(args.input)
    print(f"Loaded {len(results)} models")

    if args.multi_threshold:
        # Calculate at multiple thresholds
        df = calculate_multi_threshold_rates(results, metric=args.metric)
        print("\n" + "=" * 100)
        print(f"MULTI-THRESHOLD INCONSISTENCY RATES (metric: {args.metric})")
        print("=" * 100)

        # Show rate columns for each model/temp
        rate_cols = [c for c in df.columns if c.startswith('rate_below')]
        print("\nInconsistency rates at different thresholds:")

        # Group by model and show mean rates across temperatures
        model_rates = df.groupby('model')[rate_cols].mean()
        model_rates_pct = model_rates.map(lambda x: f"{x*100:.1f}%")
        print(model_rates_pct.to_string())
    else:
        # Calculate at single threshold
        df = calculate_inconsistency_rate(
            results,
            consistency_threshold=args.threshold,
            metric=args.metric,
        )
        print_summary_table(df, args.metric)

    # Export if output path specified
    if args.output:
        export_results(df, args.output, args.format)
    else:
        # Default output path
        input_path = Path(args.input)
        output_path = input_path.parent / f"inconsistency_rate_analysis_{args.metric}.csv"
        export_results(df, str(output_path), "csv")


if __name__ == "__main__":
    main()
