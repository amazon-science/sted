#!/usr/bin/env python3
"""
Analyze structural complexity metrics for Jiraya HTML-to-JSON dataset.

This script analyzes the "extracted_output" field (structured outputs) and generates:
1. Statistical summary of structural complexity metrics
2. Depth distribution visualization
3. Field type distribution
4. Items per sample distribution

Usage:
    python scripts/dataset_analysis/analyze_jiraya_dataset.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_PATH = PROJECT_ROOT / "research" / "datasets" / "Jiraya_html_to_json_information_extraction_dataset.json"
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


def main():
    """Main analysis function."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading Jiraya HTML-to-JSON dataset from: {DATASET_PATH}")
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    print(f"Total samples: {len(dataset)}")

    # Analyze extracted_output
    output_metrics = []
    parse_errors = 0
    unique_keys = set()
    all_extracted_items = []

    print("\nProcessing samples...")
    for i, sample in enumerate(dataset):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(dataset)}...")

        try:
            # Parse extracted_output (it's a JSON string)
            extracted = json.loads(sample['extracted_output'])
            metrics = calculate_comprehensive_metrics(extracted)

            # Count items if it's an array
            if isinstance(extracted, list):
                metrics['num_items'] = len(extracted)
                for item in extracted:
                    if isinstance(item, dict):
                        unique_keys.update(item.keys())
                        all_extracted_items.append(item)
            else:
                metrics['num_items'] = 1
                if isinstance(extracted, dict):
                    unique_keys.update(extracted.keys())
                    all_extracted_items.append(extracted)

            output_metrics.append(metrics)
        except Exception as e:
            parse_errors += 1

    print(f"\nValid samples: {len(output_metrics)}")
    print(f"Parse errors: {parse_errors}")

    # Print statistical summary
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY: JIRAYA HTML-TO-JSON (EXTRACTED OUTPUT)")
    print("=" * 80)

    # Core structural metrics
    core_metrics = ['max_depth', 'total_fields', 'total_nodes', 'leaf_nodes',
                    'nested_objects', 'arrays', 'num_items']

    print(f"\n{'─' * 80}")
    print("CORE STRUCTURAL METRICS")
    print(f"{'─' * 80}")
    print(f"{'Metric':<22} {'Mean':>10} {'Std':>10} {'Min':>8} {'Max':>8} {'Median':>10} {'P25':>8} {'P75':>8}")
    print("-" * 80)

    stats = {}
    for metric in core_metrics:
        values = np.array([m[metric] for m in output_metrics if metric in m])
        if len(values) > 0:
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
            print(f"{metric:<22} {s['mean']:>10.2f} {s['std']:>10.2f} {int(s['min']):>8} {int(s['max']):>8} {s['median']:>10.1f} {s['p25']:>8.1f} {s['p75']:>8.1f}")

    # Field type distribution
    print(f"\n{'─' * 80}")
    print("FIELD TYPE DISTRIBUTION")
    print(f"{'─' * 80}")

    type_fields = ['string_fields', 'object_fields', 'integer_fields', 'array_fields',
                   'boolean_fields', 'float_fields', 'null_fields']

    total_by_type = {}
    for field in type_fields:
        total_by_type[field] = sum(m[field] for m in output_metrics)

    grand_total = sum(total_by_type.values())

    print(f"{'Type':<20} {'Total Count':>15} {'Percentage':>12} {'Avg/Sample':>12}")
    print("-" * 60)

    type_stats = {}
    for field in sorted(type_fields, key=lambda x: total_by_type[x], reverse=True):
        count = total_by_type[field]
        pct = (count / grand_total * 100) if grand_total > 0 else 0
        avg = count / len(output_metrics) if output_metrics else 0
        type_name = field.replace('_fields', '')
        type_stats[type_name] = {'count': count, 'percentage': pct, 'avg': avg}
        print(f"{type_name:<20} {count:>15,} {pct:>11.1f}% {avg:>12.2f}")

    # Unique keys analysis
    print(f"\n{'─' * 80}")
    print("SCHEMA ANALYSIS (Unique Keys)")
    print(f"{'─' * 80}")
    print(f"Total unique keys found: {len(unique_keys)}")
    print(f"\nKeys: {sorted(unique_keys)}")

    # Key frequency analysis
    print(f"\n{'─' * 80}")
    print("KEY FREQUENCY ANALYSIS")
    print(f"{'─' * 80}")
    key_counts = defaultdict(int)
    for item in all_extracted_items:
        for key in item.keys():
            key_counts[key] += 1

    print(f"{'Key':<40} {'Count':>10} {'Percentage':>12}")
    print("-" * 65)
    for key, count in sorted(key_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_extracted_items) * 100 if all_extracted_items else 0
        print(f"{key:<40} {count:>10,} {pct:>11.1f}%")

    # Depth distribution
    print(f"\n{'─' * 80}")
    print("DEPTH DISTRIBUTION")
    print(f"{'─' * 80}")
    depths = [m['max_depth'] for m in output_metrics]
    depth_counts = defaultdict(int)
    for d in depths:
        depth_counts[d] += 1

    print(f"{'Depth':<10} {'Count':>12} {'Percentage':>12} {'Cumulative':>12}")
    print("-" * 50)
    cumulative = 0
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        pct = count / len(output_metrics) * 100
        cumulative += pct
        print(f"{depth:<10} {count:>12,} {pct:>11.1f}% {cumulative:>11.1f}%")

    # Items per sample distribution
    print(f"\n{'─' * 80}")
    print("ITEMS PER SAMPLE DISTRIBUTION")
    print(f"{'─' * 80}")
    items = [m['num_items'] for m in output_metrics]
    item_counts = defaultdict(int)
    for i in items:
        if i <= 10:
            item_counts[str(i)] = item_counts.get(str(i), 0) + 1
        elif i <= 20:
            item_counts['11-20'] = item_counts.get('11-20', 0) + 1
        elif i <= 50:
            item_counts['21-50'] = item_counts.get('21-50', 0) + 1
        else:
            item_counts['51+'] = item_counts.get('51+', 0) + 1

    print(f"{'Items':<15} {'Count':>12} {'Percentage':>12}")
    print("-" * 40)
    for label in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11-20', '21-50', '51+']:
        if label in item_counts:
            count = item_counts[label]
            pct = count / len(output_metrics) * 100
            print(f"{label:<15} {count:>12,} {pct:>11.1f}%")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Plot 1: Depth distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_depths = sorted(depth_counts.keys())
    counts = [depth_counts[d] for d in all_depths]
    percentages = [c / len(output_metrics) * 100 for c in counts]

    ax1 = axes[0]
    bars1 = ax1.bar(all_depths, counts, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Max Depth', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Depth Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.set_xticks(all_depths)
    ax1.grid(axis='y', alpha=0.3)
    for bar, count in zip(bars1, counts):
        if count > 0:
            ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    ax2 = axes[1]
    bars2 = ax2.bar(all_depths, percentages, color='coral', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Max Depth', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Depth Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_xticks(all_depths)
    ax2.grid(axis='y', alpha=0.3)
    for bar, pct in zip(bars2, percentages):
        if pct > 0.5:
            ax2.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    depth_output = OUTPUT_DIR / "jiraya_depth_distribution.png"
    plt.savefig(depth_output, dpi=150, bbox_inches='tight')
    print(f"Depth distribution plot saved to: {depth_output}")
    plt.close()

    # Plot 2: Items per sample distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11-20', '21-50', '51+']
    values = [item_counts.get(l, 0) for l in labels]
    pcts = [v / len(output_metrics) * 100 for v in values]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.bar(labels, pcts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Number of Extracted Items', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Extracted Items per Sample Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, pct, val in zip(bars, pcts, values):
        if pct > 0.5:
            ax.annotate(f'{pct:.1f}%\n({val})', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    items_output = OUTPUT_DIR / "jiraya_items_distribution.png"
    plt.savefig(items_output, dpi=150, bbox_inches='tight')
    print(f"Items distribution plot saved to: {items_output}")
    plt.close()

    # Plot 3: Field type distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    filtered = {k: v for k, v in type_stats.items() if v['percentage'] > 0.1}
    labels_type = list(filtered.keys())
    sizes = [filtered[k]['percentage'] for k in labels_type]
    counts_list = [filtered[k]['count'] for k in labels_type]
    colors_pie = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#95a5a6']

    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels_type, autopct='%1.1f%%',
                                        colors=colors_pie[:len(labels_type)], startangle=90)
    ax1.set_title('Field Type Distribution', fontsize=14, fontweight='bold')

    ax2 = axes[1]
    bars = ax2.barh(labels_type, counts_list, color=colors_pie[:len(labels_type)], edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_title('Field Type Counts', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for bar, count in zip(bars, counts_list):
        ax2.annotate(f'{count:,}', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')

    plt.tight_layout()
    types_output = OUTPUT_DIR / "jiraya_field_types_distribution.png"
    plt.savefig(types_output, dpi=150, bbox_inches='tight')
    print(f"Field types distribution plot saved to: {types_output}")
    plt.close()

    # Save summary to JSON
    summary = {
        'dataset': 'Jiraya_html_to_json_information_extraction_dataset',
        'total_samples': len(output_metrics),
        'unique_keys': sorted(list(unique_keys)),
        'structural_metrics': {
            metric: {
                'mean': float(stats[metric]['mean']),
                'std': float(stats[metric]['std']),
                'min': int(stats[metric]['min']),
                'max': int(stats[metric]['max']),
                'median': float(stats[metric]['median']),
            }
            for metric in stats
        },
        'field_type_distribution': type_stats,
        'key_frequency': dict(key_counts)
    }

    summary_output = OUTPUT_DIR / "jiraya_extracted_output_summary.json"
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_output}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
