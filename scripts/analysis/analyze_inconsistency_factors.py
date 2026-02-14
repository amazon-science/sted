#!/usr/bin/env python3
"""
Analyze factors contributing to inconsistency in Toucan tool calling results.

This script analyzes the relationship between input characteristics and consistency:
- Query length
- Number of available tools
- Ground truth tool calls
- Inconsistency patterns

Generates visualizations showing correlations between these factors and consistency.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_toucan_metadata():
    """Load Toucan dataset metadata."""
    toucan_path = PROJECT_ROOT / 'toucan_data' / 'toucan_tool_calls_1006.json'
    with open(toucan_path) as f:
        toucan_data = json.load(f)

    return {item['id']: {
        'num_tool_calls': item.get('num_tool_calls', 1),
        'question': item.get('question', ''),
    } for item in toucan_data}


def categorize_inconsistency(runs):
    """Categorize types of inconsistency between runs."""
    result = {
        'tool_selection': False,
        'parameter_value': False,
        'tool_count': False,
        'malformed_output': False,
        'is_consistent': True,
        'num_unique_outputs': 1,
    }

    valid_runs = [r for r in runs if r and isinstance(r, list)]
    if len(valid_runs) < 2:
        return result

    # Check for malformed outputs
    for run in valid_runs:
        for tool in run:
            if isinstance(tool, dict):
                name = tool.get('name', '')
                if '[TOOL_CALLS]' in name:
                    result['malformed_output'] = True
                if '-server-' in name:
                    base = name.split('-server-')[0]
                    if name.count(base) > 1:
                        result['malformed_output'] = True

    # Get tool names per run
    tool_names_per_run = []
    for run in valid_runs:
        names = tuple(t.get('name', '') for t in run if isinstance(t, dict))
        tool_names_per_run.append(names)

    # Check tool selection variation
    unique_tool_sets = set(frozenset(names) for names in tool_names_per_run)
    if len(unique_tool_sets) > 1:
        result['tool_selection'] = True

    # Check tool count variation
    tool_counts = [len(names) for names in tool_names_per_run]
    if len(set(tool_counts)) > 1:
        result['tool_count'] = True

    # Check parameter variation
    if not result['tool_selection'] and not result['tool_count']:
        args_per_run = []
        for run in valid_runs:
            args = tuple(
                json.dumps(t.get('arguments', {}), sort_keys=True)
                for t in run if isinstance(t, dict)
            )
            args_per_run.append(args)
        if len(set(args_per_run)) > 1:
            result['parameter_value'] = True

    # Count unique outputs
    output_strs = [json.dumps(r, sort_keys=True) for r in valid_runs]
    result['num_unique_outputs'] = len(set(output_strs))
    result['is_consistent'] = result['num_unique_outputs'] == 1

    return result


def analyze_model(model_path, model_name, toucan_metadata, temperature=0.5):
    """Analyze a single model's results."""
    temp_str = f"temp_0_{int(temperature * 100):02d}"

    dirs = os.listdir(model_path)
    temp_dirs = [d for d in dirs if temp_str in d]
    if not temp_dirs:
        return None

    results_path = os.path.join(model_path, temp_dirs[0], 'all_results.json')
    if not os.path.exists(results_path):
        return None

    with open(results_path) as f:
        data = json.load(f)

    samples = []
    for sample in data.get('results', []):
        runs = sample.get('generated_runs', [])
        categories = categorize_inconsistency(runs)

        sample_id = sample.get('sample_id', '')
        gt_info = toucan_metadata.get(sample_id, {})

        samples.append({
            'model': model_name,
            'sample_id': sample_id,
            'query_length': len(sample.get('query', '')),
            'num_tools': len(sample.get('tools', [])),
            'gt_tool_calls': gt_info.get('num_tool_calls', 1),
            'is_consistent': categories['is_consistent'],
            'num_unique_outputs': categories['num_unique_outputs'],
            'tool_selection': categories['tool_selection'],
            'parameter_value': categories['parameter_value'],
            'tool_count': categories['tool_count'],
            'malformed_output': categories['malformed_output'],
        })

    return samples


def create_visualizations(all_samples, output_dir):
    """Create visualizations for the analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Query Length vs Consistency
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1a: Query Length Distribution by Consistency
    ax = axes[0, 0]
    consistent = [s['query_length'] for s in all_samples if s['is_consistent']]
    inconsistent = [s['query_length'] for s in all_samples if not s['is_consistent']]

    bins = np.linspace(0, 2000, 30)
    ax.hist(consistent, bins=bins, alpha=0.7, label=f'Consistent (n={len(consistent)})', color='#2ecc71')
    ax.hist(inconsistent, bins=bins, alpha=0.7, label=f'Inconsistent (n={len(inconsistent)})', color='#e74c3c')
    ax.set_xlabel('Query Length (characters)', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Query Length Distribution by Consistency', fontsize=12, fontweight='bold')
    ax.legend()

    c_mean, i_mean = np.mean(consistent), np.mean(inconsistent)
    ax.axvline(c_mean, color='#27ae60', linestyle='--', linewidth=2, label=f'Consistent mean: {c_mean:.0f}')
    ax.axvline(i_mean, color='#c0392b', linestyle='--', linewidth=2, label=f'Inconsistent mean: {i_mean:.0f}')
    ax.legend()

    # 1b: Inconsistency Rate by Query Length Bins
    ax = axes[0, 1]
    query_bins = [(0, 300), (300, 500), (500, 700), (700, 1000), (1000, 2000)]
    bin_labels = ['0-300', '300-500', '500-700', '700-1000', '1000+']
    rates = []
    counts = []
    for low, high in query_bins:
        subset = [s for s in all_samples if low <= s['query_length'] < high]
        if subset:
            rate = sum(1 for s in subset if not s['is_consistent']) / len(subset) * 100
            rates.append(rate)
            counts.append(len(subset))
        else:
            rates.append(0)
            counts.append(0)

    bars = ax.bar(bin_labels, rates, color='#3498db', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Query Length (characters)', fontsize=11)
    ax.set_ylabel('Inconsistency Rate (%)', fontsize=11)
    ax.set_title('Inconsistency Rate by Query Length', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.2 if rates else 50)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    # 1c: Number of Tools vs Consistency
    ax = axes[1, 0]
    tool_bins = [(1, 3), (4, 6), (7, 10), (11, 20), (21, 100)]
    bin_labels = ['1-3', '4-6', '7-10', '11-20', '21+']
    rates = []
    counts = []
    for low, high in tool_bins:
        subset = [s for s in all_samples if low <= s['num_tools'] <= high]
        if subset:
            rate = sum(1 for s in subset if not s['is_consistent']) / len(subset) * 100
            rates.append(rate)
            counts.append(len(subset))
        else:
            rates.append(0)
            counts.append(0)

    bars = ax.bar(bin_labels, rates, color='#9b59b6', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Number of Available Tools', fontsize=11)
    ax.set_ylabel('Inconsistency Rate (%)', fontsize=11)
    ax.set_title('Inconsistency Rate by Available Tools', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.2 if rates else 50)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    # 1d: GT Tool Calls vs Consistency
    ax = axes[1, 1]
    gt_values = sorted(set(s['gt_tool_calls'] for s in all_samples))
    rates = []
    counts = []
    labels = []
    for gt in gt_values:
        subset = [s for s in all_samples if s['gt_tool_calls'] == gt]
        if len(subset) >= 20:  # Only show if enough samples
            rate = sum(1 for s in subset if not s['is_consistent']) / len(subset) * 100
            rates.append(rate)
            counts.append(len(subset))
            labels.append(str(gt))

    bars = ax.bar(labels, rates, color='#e67e22', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Ground Truth Tool Calls', fontsize=11)
    ax.set_ylabel('Inconsistency Rate (%)', fontsize=11)
    ax.set_title('Inconsistency Rate by Required Tool Calls', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.2 if rates else 50)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'inconsistency_factors_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Inconsistency Patterns Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2a: Pattern distribution
    ax = axes[0]
    patterns = ['Tool Selection', 'Parameter Value', 'Tool Count', 'Malformed']
    pattern_keys = ['tool_selection', 'parameter_value', 'tool_count', 'malformed_output']

    pattern_counts = [sum(1 for s in all_samples if s[k]) for k in pattern_keys]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    bars = ax.bar(patterns, pattern_counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Inconsistency Pattern', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Inconsistency Pattern Distribution (All Models)', fontsize=12, fontweight='bold')

    for bar, count in zip(bars, pattern_counts):
        pct = count / len(all_samples) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2b: Patterns by model
    ax = axes[1]
    models = sorted(set(s['model'] for s in all_samples))
    x = np.arange(len(models))
    width = 0.2

    for i, (pattern, key, color) in enumerate(zip(patterns, pattern_keys, colors)):
        rates = []
        for model in models:
            model_samples = [s for s in all_samples if s['model'] == model]
            rate = sum(1 for s in model_samples if s[key]) / len(model_samples) * 100 if model_samples else 0
            rates.append(rate)
        ax.bar(x + i * width, rates, width, label=pattern, color=color, edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Pattern Rate (%)', fontsize=11)
    ax.set_title('Inconsistency Patterns by Model', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=8)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'inconsistency_patterns_by_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Model comparison - Query length effect
    fig, ax = plt.subplots(figsize=(12, 6))

    models = sorted(set(s['model'] for s in all_samples))
    x = np.arange(len(models))
    width = 0.35

    consistent_means = []
    inconsistent_means = []
    for model in models:
        c_lens = [s['query_length'] for s in all_samples if s['model'] == model and s['is_consistent']]
        i_lens = [s['query_length'] for s in all_samples if s['model'] == model and not s['is_consistent']]
        consistent_means.append(np.mean(c_lens) if c_lens else 0)
        inconsistent_means.append(np.mean(i_lens) if i_lens else 0)

    bars1 = ax.bar(x - width/2, consistent_means, width, label='Consistent', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, inconsistent_means, width, label='Inconsistent', color='#e74c3c', edgecolor='black')

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Mean Query Length (characters)', fontsize=11)
    ax.set_title('Mean Query Length: Consistent vs Inconsistent Samples by Model', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=9)
    ax.legend()

    # Add percentage difference annotations
    for i, (c, ic) in enumerate(zip(consistent_means, inconsistent_means)):
        if c > 0:
            diff = (ic - c) / c * 100
            ax.annotate(f'+{diff:.0f}%', xy=(i, max(c, ic) + 20), ha='center', fontsize=9, color='#c0392b')

    plt.tight_layout()
    plt.savefig(output_dir / 'query_length_by_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: Heatmap - Inconsistency rate by query length and tool count
    fig, ax = plt.subplots(figsize=(10, 8))

    query_bins = [(0, 400), (400, 600), (600, 800), (800, 1200), (1200, 3000)]
    tool_bins = [(1, 4), (5, 8), (9, 15), (16, 30), (31, 200)]

    query_labels = ['<400', '400-600', '600-800', '800-1200', '1200+']
    tool_labels = ['1-4', '5-8', '9-15', '16-30', '31+']

    heatmap_data = np.zeros((len(tool_bins), len(query_bins)))
    count_data = np.zeros((len(tool_bins), len(query_bins)))

    for i, (tl, th) in enumerate(tool_bins):
        for j, (ql, qh) in enumerate(query_bins):
            subset = [s for s in all_samples if tl <= s['num_tools'] <= th and ql <= s['query_length'] < qh]
            if len(subset) >= 10:
                heatmap_data[i, j] = sum(1 for s in subset if not s['is_consistent']) / len(subset) * 100
                count_data[i, j] = len(subset)
            else:
                heatmap_data[i, j] = np.nan

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=60)

    ax.set_xticks(np.arange(len(query_labels)))
    ax.set_yticks(np.arange(len(tool_labels)))
    ax.set_xticklabels(query_labels)
    ax.set_yticklabels(tool_labels)
    ax.set_xlabel('Query Length (characters)', fontsize=11)
    ax.set_ylabel('Number of Available Tools', fontsize=11)
    ax.set_title('Inconsistency Rate Heatmap: Query Length × Tool Count', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(tool_bins)):
        for j in range(len(query_bins)):
            if not np.isnan(heatmap_data[i, j]):
                text = f'{heatmap_data[i, j]:.0f}%\n(n={int(count_data[i, j])})'
                color = 'white' if heatmap_data[i, j] > 35 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Inconsistency Rate (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'inconsistency_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")


def print_summary_statistics(all_samples):
    """Print summary statistics."""
    print("=" * 80)
    print("INCONSISTENCY FACTORS ANALYSIS - SUMMARY STATISTICS")
    print("=" * 80)

    total = len(all_samples)
    consistent = sum(1 for s in all_samples if s['is_consistent'])
    inconsistent = total - consistent

    print(f"\nTotal samples analyzed: {total}")
    print(f"Consistent: {consistent} ({consistent/total*100:.1f}%)")
    print(f"Inconsistent: {inconsistent} ({inconsistent/total*100:.1f}%)")

    # Query length comparison
    print("\n" + "-" * 80)
    print("QUERY LENGTH ANALYSIS")
    print("-" * 80)

    c_lens = [s['query_length'] for s in all_samples if s['is_consistent']]
    i_lens = [s['query_length'] for s in all_samples if not s['is_consistent']]

    print(f"\n{'Metric':<25} {'Consistent':<15} {'Inconsistent':<15} {'Diff':<10}")
    print("-" * 65)
    print(f"{'Mean query length':<25} {np.mean(c_lens):<15.1f} {np.mean(i_lens):<15.1f} {((np.mean(i_lens)-np.mean(c_lens))/np.mean(c_lens)*100):+.1f}%")
    print(f"{'Median query length':<25} {np.median(c_lens):<15.1f} {np.median(i_lens):<15.1f}")
    print(f"{'Std query length':<25} {np.std(c_lens):<15.1f} {np.std(i_lens):<15.1f}")

    # Tool count analysis
    print("\n" + "-" * 80)
    print("AVAILABLE TOOLS ANALYSIS")
    print("-" * 80)

    tool_bins = [(1, 3), (4, 6), (7, 10), (11, 20), (21, 100)]
    print(f"\n{'Tool Range':<15} {'Total':<10} {'Inconsistent':<15} {'Rate':<10}")
    print("-" * 50)
    for low, high in tool_bins:
        subset = [s for s in all_samples if low <= s['num_tools'] <= high]
        if subset:
            inc = sum(1 for s in subset if not s['is_consistent'])
            print(f"{low}-{high:<12} {len(subset):<10} {inc:<15} {inc/len(subset)*100:.1f}%")

    # GT tool calls analysis
    print("\n" + "-" * 80)
    print("GROUND TRUTH TOOL CALLS ANALYSIS")
    print("-" * 80)

    print(f"\n{'GT Calls':<15} {'Total':<10} {'Inconsistent':<15} {'Rate':<10}")
    print("-" * 50)
    for gt in sorted(set(s['gt_tool_calls'] for s in all_samples)):
        subset = [s for s in all_samples if s['gt_tool_calls'] == gt]
        if len(subset) >= 20:
            inc = sum(1 for s in subset if not s['is_consistent'])
            print(f"{gt:<15} {len(subset):<10} {inc:<15} {inc/len(subset)*100:.1f}%")

    # Pattern analysis
    print("\n" + "-" * 80)
    print("INCONSISTENCY PATTERNS")
    print("-" * 80)

    patterns = [
        ('Tool Selection', 'tool_selection'),
        ('Parameter Value', 'parameter_value'),
        ('Tool Count', 'tool_count'),
        ('Malformed Output', 'malformed_output'),
    ]

    print(f"\n{'Pattern':<20} {'Count':<10} {'Rate':<10}")
    print("-" * 40)
    for name, key in patterns:
        count = sum(1 for s in all_samples if s[key])
        print(f"{name:<20} {count:<10} {count/total*100:.1f}%")

    # Per-model breakdown
    print("\n" + "-" * 80)
    print("PER-MODEL SUMMARY")
    print("-" * 80)

    models = sorted(set(s['model'] for s in all_samples))
    print(f"\n{'Model':<25} {'Samples':<10} {'Consist.':<12} {'Query Diff':<12}")
    print("-" * 60)
    for model in models:
        model_samples = [s for s in all_samples if s['model'] == model]
        cons_rate = sum(1 for s in model_samples if s['is_consistent']) / len(model_samples) * 100

        c_lens = [s['query_length'] for s in model_samples if s['is_consistent']]
        i_lens = [s['query_length'] for s in model_samples if not s['is_consistent']]
        if c_lens and i_lens:
            diff = (np.mean(i_lens) - np.mean(c_lens)) / np.mean(c_lens) * 100
        else:
            diff = 0

        print(f"{model:<25} {len(model_samples):<10} {cons_rate:<11.1f}% {diff:+.1f}%")


def save_results_csv(all_samples, output_dir):
    """Save detailed results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'inconsistency_factors_detailed.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'model', 'sample_id', 'query_length', 'num_tools', 'gt_tool_calls',
            'is_consistent', 'num_unique_outputs', 'tool_selection',
            'parameter_value', 'tool_count', 'malformed_output'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in all_samples:
            writer.writerow(sample)

    print(f"\nDetailed results saved to: {csv_path}")


def main():
    """Main analysis function."""
    toucan_base = PROJECT_ROOT / 'llm_gen_results' / 'toucan'

    # Model directories
    model_dirs = {
        'Claude-Opus-4': 'generations-claude-opus-4-20251222',
        'Claude-Opus-4.5': 'generations-claude-opus-4.5-20251224',
        'Claude-Sonnet-4': 'generations-claude-sonnet-4-20251223',
        'Claude-3.5-Sonnet': 'generations-claude-3.5-sonnet-20251223',
        'Claude-3.5-Haiku': 'generations-claude-3.5-haiku-20251224',
        'Claude-3.7-Sonnet': 'generations-claude37-sonnet-20251229',
        'Llama-3.3-70B': 'generations-llama-3.3-70b-20251223',
        'Nova-2-Lite': 'generations-nova2-lite-20251222_075929',
        'Qwen3-235B': 'generations-qwen3-235b-a22b-20251229',
        'Qwen3-32B': 'generations-qwen3-32b-20251224',
        'Mistral-Large-675B': 'generations-mistral.mistral-large-3-675b-instruct-20251224_155007',
        'Mimo-V2-Flash': 'generations-mimo-v2-flash-20251229',
    }

    print("Loading Toucan metadata...")
    toucan_metadata = load_toucan_metadata()

    print("Analyzing models...")
    all_samples = []

    for model_name, dir_name in model_dirs.items():
        model_path = toucan_base / dir_name
        if not model_path.exists():
            print(f"  Warning: {model_name} directory not found")
            continue

        samples = analyze_model(str(model_path), model_name, toucan_metadata, temperature=0.5)
        if samples:
            all_samples.extend(samples)
            print(f"  {model_name}: {len(samples)} samples")

    if not all_samples:
        print("No samples found!")
        return

    # Print summary statistics
    print_summary_statistics(all_samples)

    # Create visualizations
    output_dir = toucan_base / 'consistency_results' / 'visualization'
    create_visualizations(all_samples, output_dir)

    # Save detailed CSV
    save_results_csv(all_samples, output_dir)


if __name__ == '__main__':
    main()
