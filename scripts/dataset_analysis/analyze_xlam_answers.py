#!/usr/bin/env python3
"""
Analyze structural complexity metrics for Salesforce xlam-function-calling-60k dataset.

This script analyzes the "answers" field (structured outputs) and generates:
1. Statistical summary of structural complexity metrics
2. Depth distribution visualization
3. Field type distribution
4. Function call distribution

Usage:
    python scripts/dataset_analysis/analyze_xlam_answers.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import os

# Set up paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_PATH = PROJECT_ROOT / "research" / "datasets" / "Salesforce_xlam-function-calling-60k.json"
OUTPUT_DIR = PROJECT_ROOT / "research" / "analysis_results"


def calculate_comprehensive_metrics(data):
    """Calculate structural complexity metrics for a JSON object."""
    metrics = {
        'max_depth': 0,
        'total_fields': 0,
        'nested_objects': 0,
        'arrays': 0,
        'total_nodes': 0,
        'leaf_nodes': 0,
        'string_fields': 0,
        'integer_fields': 0,
        'float_fields': 0,
        'boolean_fields': 0,
        'null_fields': 0,
        'array_fields': 0,
        'object_fields': 0,
    }

    def traverse(obj, depth=0):
        metrics['max_depth'] = max(metrics['max_depth'], depth)
        metrics['total_nodes'] += 1

        if isinstance(obj, dict):
            metrics['object_fields'] += 1
            if depth > 0:
                metrics['nested_objects'] += 1
            for key, value in obj.items():
                metrics['total_fields'] += 1
                traverse(value, depth + 1)
        elif isinstance(obj, list):
            metrics['arrays'] += 1
            metrics['array_fields'] += 1
            for item in obj:
                traverse(item, depth + 1)
        else:
            metrics['leaf_nodes'] += 1
            if isinstance(obj, str):
                metrics['string_fields'] += 1
            elif isinstance(obj, bool):
                metrics['boolean_fields'] += 1
            elif isinstance(obj, int):
                metrics['integer_fields'] += 1
            elif isinstance(obj, float):
                metrics['float_fields'] += 1
            elif obj is None:
                metrics['null_fields'] += 1

    traverse(data)
    return metrics


def load_and_analyze_dataset(dataset_path):
    """Load dataset and extract answers metrics."""
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Total samples: {len(dataset)}")

    answers_metrics = []
    parse_errors = 0

    print("\nProcessing samples...")
    for i, sample in enumerate(dataset):
        if i % 10000 == 0:
            print(f"  Processed {i}/{len(dataset)}...")

        try:
            answers = json.loads(sample['answers'])
            ans_metrics = calculate_comprehensive_metrics(answers)
            ans_metrics['num_function_calls'] = len(answers) if isinstance(answers, list) else 1
            answers_metrics.append(ans_metrics)
        except Exception as e:
            parse_errors += 1

    print(f"\nValid samples: {len(answers_metrics)}")
    print(f"Parse errors: {parse_errors}")

    return answers_metrics


def print_statistical_summary(metrics_list):
    """Print comprehensive statistical summary."""
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY: ANSWERS (STRUCTURED OUTPUTS)")
    print("=" * 80)

    # Core structural metrics
    core_metrics = ['max_depth', 'total_fields', 'total_nodes', 'leaf_nodes',
                    'nested_objects', 'arrays', 'num_function_calls']

    print(f"\n{'─' * 80}")
    print("CORE STRUCTURAL METRICS")
    print(f"{'─' * 80}")
    print(f"{'Metric':<22} {'Mean':>10} {'Std':>10} {'Min':>8} {'Max':>8} {'Median':>10} {'P25':>8} {'P75':>8}")
    print("-" * 80)

    stats = {}
    for metric in core_metrics:
        values = np.array([m[metric] for m in metrics_list])
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'p25': np.percentile(values, 25),
            'p75': np.percentile(values, 75),
            'values': values
        }
        s = stats[metric]
        print(f"{metric:<22} {s['mean']:>10.2f} {s['std']:>10.2f} {s['min']:>8} {s['max']:>8} {s['median']:>10.1f} {s['p25']:>8.1f} {s['p75']:>8.1f}")

    # Field type distribution
    print(f"\n{'─' * 80}")
    print("FIELD TYPE DISTRIBUTION")
    print(f"{'─' * 80}")

    type_fields = ['string_fields', 'object_fields', 'integer_fields', 'array_fields',
                   'boolean_fields', 'float_fields', 'null_fields']

    total_by_type = {}
    for field in type_fields:
        total_by_type[field] = sum(m[field] for m in metrics_list)

    grand_total = sum(total_by_type.values())

    print(f"{'Type':<20} {'Total Count':>15} {'Percentage':>12} {'Avg/Sample':>12}")
    print("-" * 60)

    type_stats = {}
    for field in sorted(type_fields, key=lambda x: total_by_type[x], reverse=True):
        count = total_by_type[field]
        pct = (count / grand_total * 100) if grand_total > 0 else 0
        avg = count / len(metrics_list) if metrics_list else 0
        type_name = field.replace('_fields', '')
        type_stats[type_name] = {'count': count, 'percentage': pct, 'avg': avg}
        print(f"{type_name:<20} {count:>15,} {pct:>11.1f}% {avg:>12.2f}")

    return stats, type_stats


def plot_depth_distribution(metrics_list, output_path=None):
    """Plot depth distribution histogram."""
    depths = [m['max_depth'] for m in metrics_list]
    depth_counts = defaultdict(int)
    for d in depths:
        depth_counts[d] += 1

    # Prepare data for plotting
    all_depths = sorted(depth_counts.keys())
    counts = [depth_counts[d] for d in all_depths]
    percentages = [c / len(metrics_list) * 100 for c in counts]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Bar chart with counts
    ax1 = axes[0]
    bars1 = ax1.bar(all_depths, counts, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Max Depth', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Depth Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.set_xticks(all_depths)
    ax1.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        if count > 0:
            ax1.annotate(f'{count:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    # Plot 2: Bar chart with percentages
    ax2 = axes[1]
    bars2 = ax2.bar(all_depths, percentages, color='coral', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Max Depth', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Depth Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_xticks(all_depths)
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for bar, pct in zip(bars2, percentages):
        if pct > 0.5:  # Only label if > 0.5%
            ax2.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nDepth distribution plot saved to: {output_path}")

    plt.show()

    # Print text summary
    print(f"\n{'─' * 60}")
    print("DEPTH DISTRIBUTION SUMMARY")
    print(f"{'─' * 60}")
    print(f"{'Depth':<10} {'Count':>12} {'Percentage':>12} {'Cumulative':>12}")
    print("-" * 50)

    cumulative = 0
    for depth in all_depths:
        count = depth_counts[depth]
        pct = count / len(metrics_list) * 100
        cumulative += pct
        print(f"{depth:<10} {count:>12,} {pct:>11.1f}% {cumulative:>11.1f}%")

    return depth_counts


def plot_function_call_distribution(metrics_list, output_path=None):
    """Plot function call distribution."""
    calls = [m['num_function_calls'] for m in metrics_list]
    call_counts = defaultdict(int)
    for c in calls:
        call_counts[c] += 1

    # Group 6+ into "6+"
    grouped_counts = defaultdict(int)
    for c, count in call_counts.items():
        if c <= 5:
            grouped_counts[str(c)] = count
        else:
            grouped_counts['6+'] = grouped_counts.get('6+', 0) + count

    # Prepare data
    labels = ['1', '2', '3', '4', '5', '6+']
    counts = [grouped_counts.get(l, 0) for l in labels]
    percentages = [c / len(metrics_list) * 100 for c in counts]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, percentages, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#95a5a6'],
                  edgecolor='black', alpha=0.8)

    ax.set_xlabel('Number of Function Calls', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Function Calls per Sample Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add labels
    for bar, pct, count in zip(bars, percentages, counts):
        ax.annotate(f'{pct:.1f}%\n({count:,})',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFunction call distribution plot saved to: {output_path}")

    plt.show()

    return call_counts


def plot_field_type_distribution(type_stats, output_path=None):
    """Plot field type distribution as pie chart."""
    # Filter out types with 0%
    filtered = {k: v for k, v in type_stats.items() if v['percentage'] > 0.1}

    labels = list(filtered.keys())
    sizes = [filtered[k]['percentage'] for k in labels]
    counts = [filtered[k]['count'] for k in labels]

    # Colors
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#95a5a6']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors[:len(labels)], startangle=90)
    ax1.set_title('Field Type Distribution', fontsize=14, fontweight='bold')

    # Bar chart
    ax2 = axes[1]
    bars = ax2.barh(labels, counts, color=colors[:len(labels)], edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_title('Field Type Counts', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax2.annotate(f'{count:,}',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    ha='left', va='center', fontsize=10, xytext=(5, 0),
                    textcoords='offset points')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nField type distribution plot saved to: {output_path}")

    plt.show()


def main():
    """Main analysis function."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and analyze dataset
    metrics_list = load_and_analyze_dataset(DATASET_PATH)

    if not metrics_list:
        print("Error: No valid samples found!")
        return

    # Print statistical summary
    stats, type_stats = print_statistical_summary(metrics_list)

    # Plot depth distribution
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    depth_output = OUTPUT_DIR / "xlam_depth_distribution.png"
    plot_depth_distribution(metrics_list, depth_output)

    # Plot function call distribution
    calls_output = OUTPUT_DIR / "xlam_function_calls_distribution.png"
    plot_function_call_distribution(metrics_list, calls_output)

    # Plot field type distribution
    types_output = OUTPUT_DIR / "xlam_field_types_distribution.png"
    plot_field_type_distribution(type_stats, types_output)

    # Save summary statistics to JSON
    summary = {
        'dataset': 'Salesforce_xlam-function-calling-60k',
        'total_samples': len(metrics_list),
        'structural_metrics': {
            metric: {
                'mean': float(stats[metric]['mean']),
                'std': float(stats[metric]['std']),
                'min': int(stats[metric]['min']),
                'max': int(stats[metric]['max']),
                'median': float(stats[metric]['median']),
                'p25': float(stats[metric]['p25']),
                'p75': float(stats[metric]['p75']),
            }
            for metric in stats
        },
        'field_type_distribution': type_stats
    }

    summary_output = OUTPUT_DIR / "xlam_answers_summary.json"
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary statistics saved to: {summary_output}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
