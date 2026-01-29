#!/usr/bin/env python
"""
Visualize best-match selection human validation results for paper.

Generates publication-quality figures:
1. Bar chart with win rates and confidence intervals
2. Comparison chart showing statistical significance
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_win_rates_bar(results: dict, output_path: str):
    """Create bar chart with win rates and confidence intervals."""
    methods = ['STED', 'BERTScore', 'DeepDiff', 'TED']
    method_keys = ['sted', 'bertscore', 'deepdiff', 'ted']

    win_rates = [results['win_rates'][k] * 100 for k in method_keys]
    ci_lower = [results['confidence_intervals_95'][k][0] * 100 for k in method_keys]
    ci_upper = [results['confidence_intervals_95'][k][1] * 100 for k in method_keys]

    # Calculate error bars (asymmetric)
    yerr_lower = [win_rates[i] - ci_lower[i] for i in range(len(methods))]
    yerr_upper = [ci_upper[i] - win_rates[i] for i in range(len(methods))]

    # Colors: highlight STED
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    fig, ax = plt.subplots(figsize=(5, 4))

    x = np.arange(len(methods))
    bars = ax.bar(x, win_rates, color=colors, edgecolor='black', linewidth=0.8, width=0.6)

    # Add error bars
    ax.errorbar(x, win_rates, yerr=[yerr_lower, yerr_upper],
                fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)

    # Add significance markers
    p_values = results.get('p_values_vs_sted', {})
    for i, method_key in enumerate(method_keys):
        if method_key == 'sted':
            continue
        p_val = p_values.get(method_key, 1.0)
        if p_val < 0.01:
            sig_marker = '**'
        elif p_val < 0.05:
            sig_marker = '*'
        else:
            sig_marker = ''

        if sig_marker:
            ax.text(i, ci_upper[i] + 3, sig_marker, ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 8,
                f'{rate:.1f}%', ha='center', va='top', color='white', fontweight='bold', fontsize=10)

    ax.set_ylabel('Human Agreement Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='Random baseline')

    # Add sample size annotation
    n_samples = results['metadata']['total_samples']
    ax.text(0.98, 0.02, f'n={n_samples}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, style='italic')

    # Add legend for significance
    ax.text(0.98, 0.98, '* p<0.05 vs STED', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, style='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png')
    print(f"Saved: {output_path}")
    plt.close()


def plot_win_rates_horizontal(results: dict, output_path: str):
    """Create horizontal bar chart (alternative layout for paper)."""
    methods = ['TED', 'DeepDiff', 'BERTScore', 'STED']  # Bottom to top
    method_keys = ['ted', 'deepdiff', 'bertscore', 'sted']

    win_rates = [results['win_rates'][k] * 100 for k in method_keys]
    ci_lower = [results['confidence_intervals_95'][k][0] * 100 for k in method_keys]
    ci_upper = [results['confidence_intervals_95'][k][1] * 100 for k in method_keys]

    xerr_lower = [win_rates[i] - ci_lower[i] for i in range(len(methods))]
    xerr_upper = [ci_upper[i] - win_rates[i] for i in range(len(methods))]

    # Colors
    colors = ['#C73E1D', '#F18F01', '#A23B72', '#2E86AB']

    fig, ax = plt.subplots(figsize=(5, 3))

    y = np.arange(len(methods))
    bars = ax.barh(y, win_rates, color=colors, edgecolor='black', linewidth=0.8, height=0.6)

    ax.errorbar(win_rates, y, xerr=[xerr_lower, xerr_upper],
                fmt='none', color='black', capsize=3, capthick=1.2, linewidth=1.2)

    # Add significance markers
    p_values = results.get('p_values_vs_sted', {})
    for i, method_key in enumerate(method_keys):
        if method_key == 'sted':
            continue
        p_val = p_values.get(method_key, 1.0)
        if p_val < 0.05:
            ax.text(ci_upper[i] + 2, i, '*', ha='left', va='center', fontsize=14, fontweight='bold')

    # Value labels
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        ax.text(rate - 3, i, f'{rate:.1f}%', ha='right', va='center',
                color='white', fontweight='bold', fontsize=9)

    ax.set_xlabel('Human Agreement Rate (%)')
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(0, 110)
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    n_samples = results['metadata']['total_samples']
    ax.text(0.98, 0.02, f'n={n_samples}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, style='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_comparison(results: dict, output_path: str):
    """Create combined figure showing win rates with statistical annotations."""
    methods = ['STED', 'BERTScore', 'DeepDiff', 'TED']
    method_keys = ['sted', 'bertscore', 'deepdiff', 'ted']

    win_rates = [results['win_rates'][k] * 100 for k in method_keys]
    ci_lower = [results['confidence_intervals_95'][k][0] * 100 for k in method_keys]
    ci_upper = [results['confidence_intervals_95'][k][1] * 100 for k in method_keys]
    win_counts = [results['win_counts'][k] for k in method_keys]
    n_samples = results['metadata']['total_samples']

    yerr_lower = [win_rates[i] - ci_lower[i] for i in range(len(methods))]
    yerr_upper = [ci_upper[i] - win_rates[i] for i in range(len(methods))]

    # Use a colorblind-friendly palette
    colors = ['#0077BB', '#EE7733', '#009988', '#CC3311']

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    x = np.arange(len(methods))
    bars = ax.bar(x, win_rates, color=colors, edgecolor='black', linewidth=0.5, width=0.65)

    ax.errorbar(x, win_rates, yerr=[yerr_lower, yerr_upper],
                fmt='none', color='black', capsize=4, capthick=1.2, linewidth=1.2)

    # Add win count labels inside bars
    for i, (bar, rate, count) in enumerate(zip(bars, win_rates, win_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{count}/{n_samples}', ha='center', va='center',
                color='white', fontweight='bold', fontsize=9)

    # Add p-value annotations with brackets for STED vs TED
    p_values = results.get('p_values_vs_sted', {})
    ted_p = p_values.get('ted', 1.0)
    if ted_p < 0.05:
        # Draw bracket between STED and TED
        bracket_y = max(ci_upper[0], ci_upper[3]) + 8
        ax.plot([0, 0, 3, 3], [bracket_y-2, bracket_y, bracket_y, bracket_y-2],
                color='black', linewidth=1)
        sig_text = f'p={ted_p:.3f}' if ted_p >= 0.001 else 'p<0.001'
        ax.text(1.5, bracket_y + 1, sig_text, ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Human Agreement Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 115)

    # Random baseline
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(3.4, 51, 'chance', fontsize=8, color='gray', va='bottom')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize human validation results')
    parser.add_argument('--results', type=str, default='best_match_report_results.json',
                        help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results)

    print("Generating figures...")

    # Generate all figure variants
    plot_win_rates_bar(results, os.path.join(args.output_dir, 'human_validation_bar.pdf'))
    plot_win_rates_horizontal(results, os.path.join(args.output_dir, 'human_validation_horizontal.pdf'))
    plot_combined_comparison(results, os.path.join(args.output_dir, 'human_validation_combined.pdf'))

    print("\nAll figures generated successfully!")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
