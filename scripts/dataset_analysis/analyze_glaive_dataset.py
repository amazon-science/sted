#!/usr/bin/env python3
"""
Analyze the Glaive function-calling dataset for STED evaluation.
Extracts function call arguments and analyzes their structural complexity.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt


def extract_function_calls(chat_text: str) -> list:
    """Extract function call JSON arguments from chat text."""
    function_calls = []

    # Pattern 1: <functioncall> {...}
    pattern1 = r'<functioncall>\s*(\{[^}]+\})'
    matches1 = re.findall(pattern1, chat_text, re.DOTALL)

    # Pattern 2: AI to=function_name: {...}
    pattern2 = r'AI to=\w+:\s*(\{[^}]+\})'
    matches2 = re.findall(pattern2, chat_text, re.DOTALL)

    # Pattern 3: FUNCTION RESPONSE: {...}
    pattern3 = r'FUNCTION RESPONSE:\s*(\{[^}]+\})'
    matches3 = re.findall(pattern3, chat_text, re.DOTALL)

    all_matches = matches1 + matches2 + matches3

    for match in all_matches:
        try:
            # Try to parse as JSON
            parsed = json.loads(match)
            function_calls.append(parsed)
        except json.JSONDecodeError:
            # Try to fix common issues
            try:
                # Handle single quotes
                fixed = match.replace("'", '"')
                parsed = json.loads(fixed)
                function_calls.append(parsed)
            except:
                pass

    return function_calls


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
            keys.update(get_all_keys(item, f"{prefix}[{i}]"))
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


def analyze_glaive_dataset(data_path: str, output_dir: str):
    """Analyze the Glaive dataset and generate reports."""

    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Extract function calls
    all_function_calls = []
    samples_with_calls = 0

    for item in data:
        chat = item.get('chat', '')
        calls = extract_function_calls(chat)
        if calls:
            samples_with_calls += 1
            all_function_calls.extend(calls)

    print(f"Samples with function calls: {samples_with_calls}")
    print(f"Total function calls extracted: {len(all_function_calls)}")

    if not all_function_calls:
        print("No function calls found in the dataset.")
        return

    # Calculate metrics for each function call
    metrics = {
        'max_depth': [],
        'total_fields': [],
        'total_nodes': [],
        'leaf_nodes': [],
    }

    all_keys = set()
    total_type_counts = defaultdict(int)
    depth_counter = Counter()

    for fc in all_function_calls:
        depth = calculate_depth(fc)
        fields = count_fields(fc)
        nodes = count_nodes(fc)
        leaves = count_leaf_nodes(fc)

        metrics['max_depth'].append(depth)
        metrics['total_fields'].append(fields)
        metrics['total_nodes'].append(nodes)
        metrics['leaf_nodes'].append(leaves)

        depth_counter[depth] += 1
        all_keys.update(get_all_keys(fc))
        get_type_distribution(fc, total_type_counts)

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
    print("GLAIVE DATASET ANALYSIS - FUNCTION CALL ARGUMENTS")
    print("="*70)

    print(f"\nDataset Overview:")
    print(f"  Total samples: {len(data)}")
    print(f"  Samples with function calls: {samples_with_calls}")
    print(f"  Total function calls: {len(all_function_calls)}")
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
        pct = count / len(all_function_calls) * 100
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

    # Generate visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Depth distribution
    ax1 = axes[0, 0]
    depths = sorted(depth_counter.keys())
    counts = [depth_counter[d] for d in depths]
    ax1.bar(depths, counts, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Count')
    ax1.set_title('Function Call Depth Distribution')
    ax1.set_xticks(depths)
    for i, (d, c) in enumerate(zip(depths, counts)):
        ax1.annotate(str(c), (d, c), ha='center', va='bottom')

    # 2. Fields distribution
    ax2 = axes[0, 1]
    ax2.hist(metrics['total_fields'], bins=20, color='coral', edgecolor='black')
    ax2.set_xlabel('Number of Fields')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Fields per Function Call Distribution')
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
        ax3.annotate(f'{val}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)

    # 4. Leaf type pie chart
    ax4 = axes[1, 1]
    leaf_labels = [t for t in ['string', 'integer', 'float', 'boolean', 'null']
                   if leaf_types.get(t, 0) > 0]
    leaf_values = [leaf_types[t] for t in leaf_labels]
    leaf_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6'][:len(leaf_labels)]
    wedges, texts, autotexts = ax4.pie(leaf_values, labels=leaf_labels, autopct='%1.1f%%',
                                        colors=leaf_colors, startangle=90)
    ax4.set_title('Leaf Value Type Distribution')

    plt.tight_layout()
    plt.savefig(output_path / 'glaive_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to {output_path / 'glaive_analysis.png'}")

    # Save summary to JSON
    summary = {
        'name': 'Glaive (Function Calling)',
        'total_samples': len(data),
        'samples_with_calls': samples_with_calls,
        'total_function_calls': len(all_function_calls),
        'unique_keys': len(all_keys),
        'stats': stats,
        'depth_distribution': dict(depth_counter),
        'type_distribution': dict(total_type_counts),
        'leaf_type_distribution': dict(leaf_types),
        'sample_keys': list(all_keys)[:50]  # Save first 50 keys as sample
    }

    with open(output_path / 'glaive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path / 'glaive_summary.json'}")


if __name__ == '__main__':
    data_path = 'research/datasets/glaiveai_glaive-function-calling-v2.json'
    output_dir = 'research/analysis_results'
    analyze_glaive_dataset(data_path, output_dir)
