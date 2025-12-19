#!/usr/bin/env python3
"""
Analyze the Maxscha JSON generation datasets for STED evaluation.
Extracts generated JSON outputs and analyzes their structural complexity.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import argparse


def calculate_depth(obj, current_depth=1):
    """Calculate the maximum depth of a JSON object."""
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(calculate_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(calculate_depth(item, current_depth + 1) for item in obj)
    return current_depth


def count_fields(obj):
    """Count total fields in a JSON object (recursively)."""
    if isinstance(obj, dict):
        count = len(obj)
        for v in obj.values():
            count += count_fields(v)
        return count
    elif isinstance(obj, list):
        return sum(count_fields(item) for item in obj)
    return 0


def count_nodes(obj):
    """Count total nodes (including values) in a JSON object."""
    if isinstance(obj, dict):
        count = 1  # The dict itself
        for v in obj.values():
            count += count_nodes(v)
        return count
    elif isinstance(obj, list):
        count = 1  # The list itself
        for item in obj:
            count += count_nodes(item)
        return count
    return 1  # Leaf value


def count_leaf_nodes(obj):
    """Count leaf nodes (primitive values) in a JSON object."""
    if isinstance(obj, dict):
        if not obj:
            return 0
        return sum(count_leaf_nodes(v) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return 0
        return sum(count_leaf_nodes(item) for item in obj)
    return 1  # This is a leaf


def get_all_keys(obj, prefix=""):
    """Get all unique keys in a JSON object."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            keys.update(get_all_keys(v, full_key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            keys.update(get_all_keys(item, f"{prefix}[]"))
    return keys


def get_type_distribution(obj, type_counts=None):
    """Get the distribution of data types in a JSON object."""
    if type_counts is None:
        type_counts = defaultdict(int)

    if isinstance(obj, dict):
        type_counts['object'] += 1
        for v in obj.values():
            get_type_distribution(v, type_counts)
    elif isinstance(obj, list):
        type_counts['array'] += 1
        for item in obj:
            get_type_distribution(item, type_counts)
    elif isinstance(obj, str):
        type_counts['string'] += 1
    elif isinstance(obj, bool):
        type_counts['boolean'] += 1
    elif isinstance(obj, int):
        type_counts['integer'] += 1
    elif isinstance(obj, float):
        type_counts['float'] += 1
    elif obj is None:
        type_counts['null'] += 1

    return type_counts


def parse_output(output_str: str):
    """Parse the output string to JSON."""
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        return None


def analyze_maxscha_dataset(data_path: str, output_dir: str, dataset_name: str):
    """Analyze the Maxscha dataset and generate reports."""

    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Parse outputs
    valid_outputs = []
    parse_errors = 0

    for item in data:
        output_str = item.get('output', '')
        parsed = parse_output(output_str)
        if parsed is not None:
            valid_outputs.append(parsed)
        else:
            parse_errors += 1

    print(f"Valid outputs: {len(valid_outputs)}")
    print(f"Parse errors: {parse_errors}")

    if not valid_outputs:
        print("No valid outputs found in the dataset.")
        return

    # Calculate metrics for each output
    metrics = {
        'max_depth': [],
        'total_fields': [],
        'total_nodes': [],
        'leaf_nodes': [],
    }

    all_keys = set()
    total_type_counts = defaultdict(int)
    depth_counter = Counter()

    for output in valid_outputs:
        depth = calculate_depth(output)
        fields = count_fields(output)
        nodes = count_nodes(output)
        leaves = count_leaf_nodes(output)

        metrics['max_depth'].append(depth)
        metrics['total_fields'].append(fields)
        metrics['total_nodes'].append(nodes)
        metrics['leaf_nodes'].append(leaves)

        depth_counter[depth] += 1
        all_keys.update(get_all_keys(output))
        get_type_distribution(output, total_type_counts)

    # Calculate statistics
    stats = {}
    for metric_name, values in metrics.items():
        stats[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'median': np.median(values)
        }

    # Print summary
    print("\n" + "="*70)
    print(f"{dataset_name.upper()} DATASET ANALYSIS - JSON GENERATION OUTPUTS")
    print("="*70)

    print(f"\nDataset Overview:")
    print(f"  Total samples: {len(data)}")
    print(f"  Valid outputs: {len(valid_outputs)}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Unique keys: {len(all_keys)}")

    print(f"\nStructural Metrics:")
    for metric_name, stat in stats.items():
        print(f"  {metric_name}:")
        print(f"    Mean: {stat['mean']:.2f} ± {stat['std']:.2f}")
        print(f"    Range: [{stat['min']}, {stat['max']}]")
        print(f"    Median: {stat['median']:.2f}")

    print(f"\nDepth Distribution:")
    for depth in sorted(depth_counter.keys()):
        count = depth_counter[depth]
        pct = count / len(valid_outputs) * 100
        print(f"  Depth {depth}: {count} ({pct:.1f}%)")

    print(f"\nData Type Distribution:")
    total_types = sum(total_type_counts.values())
    for dtype in ['string', 'integer', 'float', 'boolean', 'null', 'object', 'array']:
        count = total_type_counts.get(dtype, 0)
        pct = count / total_types * 100 if total_types > 0 else 0
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    # Calculate leaf type distribution
    leaf_types = {k: v for k, v in total_type_counts.items()
                  if k not in ['object', 'array']}
    total_leaves = sum(leaf_types.values())

    print(f"\nLeaf Value Type Distribution:")
    for dtype in ['string', 'integer', 'float', 'boolean', 'null']:
        count = leaf_types.get(dtype, 0)
        pct = count / total_leaves * 100 if total_leaves > 0 else 0
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename prefix
    file_prefix = dataset_name.lower().replace(' ', '_').replace('-', '_')

    # Generate visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} Dataset Analysis', fontsize=14, fontweight='bold')

    # 1. Depth distribution
    ax1 = axes[0, 0]
    depths = sorted(depth_counter.keys())
    counts = [depth_counter[d] for d in depths]
    ax1.bar(depths, counts, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Count')
    ax1.set_title('JSON Output Depth Distribution')
    # Only show labels for bars with significant count
    for d, c in zip(depths, counts):
        if c > len(valid_outputs) * 0.01:  # Show label if > 1%
            ax1.annotate(str(c), (d, c), ha='center', va='bottom', fontsize=8)

    # 2. Fields distribution
    ax2 = axes[0, 1]
    ax2.hist(metrics['total_fields'], bins=30, color='coral', edgecolor='black')
    ax2.set_xlabel('Number of Fields')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Fields per Output Distribution')
    ax2.axvline(np.mean(metrics['total_fields']), color='red', linestyle='--',
                label=f'Mean: {np.mean(metrics["total_fields"]):.1f}')
    ax2.legend()

    # 3. Type distribution (all types)
    ax3 = axes[1, 0]
    types = ['string', 'integer', 'float', 'boolean', 'null', 'object', 'array']
    type_values = [total_type_counts.get(t, 0) for t in types]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6', '#f39c12', '#1abc9c']
    bars = ax3.bar(types, type_values, color=colors, edgecolor='black')
    ax3.set_xlabel('Data Type')
    ax3.set_ylabel('Count')
    ax3.set_title('Data Type Distribution')
    ax3.set_xticklabels(types, rotation=45, ha='right')
    for bar, val in zip(bars, type_values):
        if val > 0:
            ax3.annotate(f'{val}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)

    # 4. Leaf type pie chart
    ax4 = axes[1, 1]
    leaf_labels = [t for t in ['string', 'integer', 'float', 'boolean', 'null']
                   if leaf_types.get(t, 0) > 0]
    leaf_values = [leaf_types[t] for t in leaf_labels]
    leaf_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6'][:len(leaf_labels)]

    def autopct_filter(pct):
        return f'{pct:.1f}%' if pct > 1 else ''

    if leaf_values:
        wedges, texts, autotexts = ax4.pie(leaf_values, labels=leaf_labels, autopct=autopct_filter,
                                            colors=leaf_colors, startangle=90)
        ax4.set_title('Leaf Value Type Distribution')
    else:
        ax4.text(0.5, 0.5, 'No leaf values', ha='center', va='center')
        ax4.set_title('Leaf Value Type Distribution')

    plt.tight_layout()
    plt.savefig(output_path / f'{file_prefix}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to {output_path / f'{file_prefix}_analysis.png'}")

    # Save summary to JSON
    summary = {
        'name': dataset_name,
        'total_samples': len(data),
        'valid_outputs': len(valid_outputs),
        'parse_errors': parse_errors,
        'unique_keys': len(all_keys),
        'stats': stats,
        'depth_distribution': dict(depth_counter),
        'type_distribution': dict(total_type_counts),
        'leaf_type_distribution': dict(leaf_types),
        'sample_keys': list(all_keys)[:100]  # Save first 100 keys as sample
    }

    with open(output_path / f'{file_prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path / f'{file_prefix}_summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze Maxscha JSON generation dataset')
    parser.add_argument('--variant', choices=['regular', 'large', 'both'], default='both',
                        help='Which variant to analyze')
    parser.add_argument('--output-dir', default='research/analysis_results',
                        help='Output directory for results')
    args = parser.parse_args()

    datasets = []
    if args.variant in ['regular', 'both']:
        datasets.append({
            'path': 'research/datasets/Maxscha_json-instruct-generation.json',
            'name': 'Maxscha'
        })
    if args.variant in ['large', 'both']:
        datasets.append({
            'path': 'research/datasets/Maxscha_json-instruct-generation-large.json',
            'name': 'Maxscha-large'
        })

    summaries = []
    for ds in datasets:
        print("\n" + "="*70)
        summary = analyze_maxscha_dataset(ds['path'], args.output_dir, ds['name'])
        if summary:
            summaries.append(summary)
        print("\n")

    # If both were analyzed, create a comparison
    if len(summaries) == 2:
        print("\n" + "="*70)
        print("COMPARISON: Maxscha vs Maxscha-large")
        print("="*70)

        for metric in ['max_depth', 'total_fields', 'total_nodes', 'leaf_nodes']:
            print(f"\n{metric}:")
            for s in summaries:
                stat = s['stats'][metric]
                print(f"  {s['name']}: {stat['mean']:.2f} ± {stat['std']:.2f} (range: [{stat['min']}, {stat['max']}])")

        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Maxscha vs Maxscha-large Comparison', fontsize=14, fontweight='bold')

        # Metrics comparison
        ax1 = axes[0]
        metrics_names = ['max_depth', 'total_fields', 'total_nodes', 'leaf_nodes']
        x = np.arange(len(metrics_names))
        width = 0.35

        means1 = [summaries[0]['stats'][m]['mean'] for m in metrics_names]
        means2 = [summaries[1]['stats'][m]['mean'] for m in metrics_names]

        ax1.bar(x - width/2, means1, width, label=summaries[0]['name'], color='steelblue')
        ax1.bar(x + width/2, means2, width, label=summaries[1]['name'], color='coral')
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Mean Value')
        ax1.set_title('Structural Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', '\n') for m in metrics_names])
        ax1.legend()

        # Leaf type comparison
        ax2 = axes[1]
        leaf_types = ['string', 'integer', 'float', 'boolean', 'null']
        x = np.arange(len(leaf_types))

        leaf1 = [summaries[0]['leaf_type_distribution'].get(t, 0) for t in leaf_types]
        leaf2 = [summaries[1]['leaf_type_distribution'].get(t, 0) for t in leaf_types]

        # Normalize to percentages
        total1 = sum(leaf1)
        total2 = sum(leaf2)
        leaf1_pct = [v/total1*100 if total1 > 0 else 0 for v in leaf1]
        leaf2_pct = [v/total2*100 if total2 > 0 else 0 for v in leaf2]

        ax2.bar(x - width/2, leaf1_pct, width, label=summaries[0]['name'], color='steelblue')
        ax2.bar(x + width/2, leaf2_pct, width, label=summaries[1]['name'], color='coral')
        ax2.set_xlabel('Data Type')
        ax2.set_ylabel('Percentage')
        ax2.set_title('Leaf Type Distribution Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(leaf_types)
        ax2.legend()

        plt.tight_layout()
        output_path = Path(args.output_dir)
        plt.savefig(output_path / 'maxscha_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nComparison visualization saved to {output_path / 'maxscha_comparison.png'}")


if __name__ == '__main__':
    main()
