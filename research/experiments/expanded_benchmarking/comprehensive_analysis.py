#!/usr/bin/env python3
"""
Comprehensive Analysis for Main Conference Paper

This script performs rigorous statistical analysis on LLM consistency benchmarking data.
Designed to produce publication-ready results with:
- Statistical significance tests (p-values, confidence intervals)
- Effect sizes (Cohen's d, d-prime)
- Temperature discrimination analysis
- Model ranking analysis
- Ablation study (STED vs TED vs BERTScore vs DeepDiff)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict
import re
from itertools import combinations

# Paths
STED_PROJECT = Path("/Users/guanghu/Documents/genai/projects/sted")
LLM_RESULTS_DIR = STED_PROJECT / "llm_gen_results"
OUTPUT_DIR = Path(__file__).parent

# Model directories
MODEL_DIRS = {
    'Claude-3-Haiku': 'generations-claude-3-haiku',
    'Claude-3.5-Haiku': 'generations-claude3-5-haiku',
    'Claude-3.7-Sonnet': 'generations-claude3-7-sonnet',
    'DeepSeek-V3': 'generations-deepseek.v3-v1',
    'Gemini-2.5-Flash-Lite': 'generations-gemini-2.5-flash-lite',
    'GPT-4.1-Mini': 'generations-gpt-4.1-mini',
    'Llama-3.3-70B': 'generations-llama3-3-70b',
    'Nova-Pro-v1': 'generations-nova-pro-v1',
    'Qwen3-32B': 'generations-qwen3-32b-v1',
    'Qwen3-235B': 'generations-qwen3-235b-a22b-2507',
}

METRICS = ['ted', 'bertscore', 'deepdiff']


def extract_temperature_from_dir(dirname: str) -> float:
    """Extract temperature value from directory name."""
    match = re.search(r'temp_(\d+)_(\d+)', dirname)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None


def load_metric_results(model_dir: Path, metric: str) -> Dict[float, List[Dict]]:
    """Load results for a specific metric across all temperatures."""
    results = {}

    if not model_dir.exists():
        return results

    for subdir in model_dir.iterdir():
        if not subdir.is_dir():
            continue

        temp = extract_temperature_from_dir(subdir.name)
        if temp is None:
            continue

        result_file = subdir / f"results_{metric}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                results[temp] = data.get('results', [])

    return results


def load_all_data() -> Dict[str, Dict[str, Dict[float, List[Dict]]]]:
    """Load all available data."""
    all_data = {}

    for model_name, dirname in MODEL_DIRS.items():
        model_dir = LLM_RESULTS_DIR / dirname
        if not model_dir.exists():
            continue

        model_data = {}
        for metric in METRICS:
            metric_results = load_metric_results(model_dir, metric)
            if metric_results:
                model_data[metric] = metric_results

        if model_data:
            all_data[model_name] = model_data

    return all_data


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval."""
    if len(data) < 2:
        return (np.nan, np.nan)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (lower, upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_temperature_effect(all_data: Dict) -> Dict:
    """
    Analyze how each metric captures temperature effect on consistency.

    Statistical tests:
    - Spearman correlation (monotonic relationship)
    - Mann-Whitney U (low vs high temp groups)
    - Kruskal-Wallis (across all temp levels)
    """
    results = {}

    for metric in METRICS:
        temp_scores = defaultdict(list)

        for model_name, model_data in all_data.items():
            if metric not in model_data:
                continue

            for temp, samples in model_data[metric].items():
                for sample in samples:
                    temp_scores[temp].append(sample['mean'])

        temps = sorted(temp_scores.keys())
        if len(temps) < 2:
            continue

        # Flatten for correlation
        all_temps = []
        all_scores = []
        for t in temps:
            all_temps.extend([t] * len(temp_scores[t]))
            all_scores.extend(temp_scores[t])

        all_temps = np.array(all_temps)
        all_scores = np.array(all_scores)

        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(all_temps, all_scores)

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(all_temps, all_scores)

        # Group into low (T <= 0.2) and high (T >= 0.7)
        low_mask = all_temps <= 0.2
        high_mask = all_temps >= 0.7

        low_scores = all_scores[low_mask]
        high_scores = all_scores[high_mask]

        # Mann-Whitney U test
        if len(low_scores) > 0 and len(high_scores) > 0:
            mw_stat, mw_p = stats.mannwhitneyu(low_scores, high_scores, alternative='greater')
            effect_size = cohens_d(low_scores, high_scores)

            # ROC AUC
            labels = np.concatenate([np.ones(len(low_scores)), np.zeros(len(high_scores))])
            scores_combined = np.concatenate([low_scores, high_scores])
            fpr, tpr, _ = roc_curve(labels, scores_combined)
            roc_auc_val = auc(fpr, tpr)

            # Bootstrap CI for mean difference
            diff = np.mean(low_scores) - np.mean(high_scores)
            low_ci = bootstrap_ci(low_scores)
            high_ci = bootstrap_ci(high_scores)
        else:
            mw_stat, mw_p = np.nan, np.nan
            effect_size = np.nan
            roc_auc_val = np.nan
            diff = np.nan
            low_ci, high_ci = (np.nan, np.nan), (np.nan, np.nan)

        # Kruskal-Wallis test
        temp_groups = [np.array(temp_scores[t]) for t in temps if len(temp_scores[t]) > 0]
        if len(temp_groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*temp_groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        # Summary by temperature
        temp_summary = {}
        for t in temps:
            scores = np.array(temp_scores[t])
            ci = bootstrap_ci(scores)
            temp_summary[float(t)] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'n': len(scores),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1])
            }

        results[metric] = {
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'mannwhitney_stat': float(mw_stat) if not np.isnan(mw_stat) else None,
            'mannwhitney_p': float(mw_p) if not np.isnan(mw_p) else None,
            'kruskal_stat': float(kw_stat) if not np.isnan(kw_stat) else None,
            'kruskal_p': float(kw_p) if not np.isnan(kw_p) else None,
            'cohens_d': float(effect_size) if not np.isnan(effect_size) else None,
            'roc_auc': float(roc_auc_val) if not np.isnan(roc_auc_val) else None,
            'mean_diff_low_high': float(diff) if not np.isnan(diff) else None,
            'n_low': int(len(low_scores)),
            'n_high': int(len(high_scores)),
            'n_total': len(all_scores),
            'n_temps': len(temps),
            'temp_summary': temp_summary
        }

    return results


def analyze_model_ranking(all_data: Dict) -> Dict:
    """
    Analyze model ranking consistency across metrics.

    Tests:
    - Kendall's tau (ranking correlation between metrics)
    - Score spread analysis
    """
    results = {}

    # Get scores by model and metric
    model_scores = {metric: {} for metric in METRICS}

    for model_name, model_data in all_data.items():
        for metric in METRICS:
            if metric not in model_data:
                continue

            all_means = []
            for temp, samples in model_data[metric].items():
                for sample in samples:
                    all_means.append(sample['mean'])

            if all_means:
                model_scores[metric][model_name] = {
                    'mean': float(np.mean(all_means)),
                    'std': float(np.std(all_means)),
                    'n': len(all_means)
                }

    # Rankings
    rankings = {}
    for metric in METRICS:
        if model_scores[metric]:
            sorted_models = sorted(model_scores[metric].keys(),
                                   key=lambda m: model_scores[metric][m]['mean'],
                                   reverse=True)
            rankings[metric] = sorted_models

    # Kendall tau between metric rankings
    kendall_results = {}
    metric_pairs = list(combinations(METRICS, 2))

    for m1, m2 in metric_pairs:
        if m1 in rankings and m2 in rankings:
            # Get common models
            common = set(rankings[m1]) & set(rankings[m2])
            if len(common) >= 2:
                ranks1 = [rankings[m1].index(m) for m in common]
                ranks2 = [rankings[m2].index(m) for m in common]
                tau, p = stats.kendalltau(ranks1, ranks2)
                kendall_results[f'{m1}_vs_{m2}'] = {
                    'tau': float(tau),
                    'p_value': float(p),
                    'n_models': len(common)
                }

    results['model_scores'] = model_scores
    results['rankings'] = rankings
    results['kendall_tau'] = kendall_results

    return results


def analyze_metric_correlations(all_data: Dict) -> Dict:
    """
    Analyze correlations between metrics at sample level.
    """
    # Collect paired observations (same model, temp, sample)
    paired_data = []

    for model_name, model_data in all_data.items():
        # Get common temperatures
        common_temps = set()
        for metric in METRICS:
            if metric in model_data:
                if not common_temps:
                    common_temps = set(model_data[metric].keys())
                else:
                    common_temps &= set(model_data[metric].keys())

        for temp in common_temps:
            # Get minimum sample count
            min_samples = min(len(model_data[m][temp]) for m in METRICS if m in model_data)

            for i in range(min_samples):
                row = {'model': model_name, 'temp': temp}
                for metric in METRICS:
                    if metric in model_data:
                        row[metric] = model_data[metric][temp][i]['mean']
                if all(m in row for m in METRICS):
                    paired_data.append(row)

    if not paired_data:
        return {}

    # Calculate correlations
    correlations = {}
    for m1, m2 in combinations(METRICS, 2):
        vals1 = np.array([d[m1] for d in paired_data])
        vals2 = np.array([d[m2] for d in paired_data])

        pearson_r, pearson_p = stats.pearsonr(vals1, vals2)
        spearman_r, spearman_p = stats.spearmanr(vals1, vals2)

        correlations[f'{m1}_vs_{m2}'] = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'n_samples': len(paired_data)
        }

    return correlations


def create_publication_figures(temp_results: Dict, model_results: Dict,
                               corr_results: Dict, output_dir: Path):
    """Create publication-ready figures."""

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Temperature effect comparison
    ax1 = fig.add_subplot(2, 3, 1)
    metrics = list(temp_results.keys())
    x = np.arange(len(metrics))

    spearman_vals = [abs(temp_results[m]['spearman_r']) for m in metrics]
    ax1.bar(x, spearman_vals, color=['steelblue', 'coral', 'green'][:len(metrics)])
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in metrics])
    ax1.set_ylabel('|Spearman \u03c1|')
    ax1.set_title('Temperature Discrimination\n(higher = better)')
    ax1.set_ylim([0, 0.35])

    # Add significance stars
    for i, m in enumerate(metrics):
        p = temp_results[m]['spearman_p']
        if p < 0.001:
            ax1.text(i, spearman_vals[i] + 0.01, '***', ha='center')
        elif p < 0.01:
            ax1.text(i, spearman_vals[i] + 0.01, '**', ha='center')
        elif p < 0.05:
            ax1.text(i, spearman_vals[i] + 0.01, '*', ha='center')

    # Plot 2: Effect sizes
    ax2 = fig.add_subplot(2, 3, 2)
    effect_sizes = [temp_results[m]['cohens_d'] or 0 for m in metrics]
    bars = ax2.bar(x, effect_sizes, color=['steelblue', 'coral', 'green'][:len(metrics)])
    ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax2.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium effect')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in metrics])
    ax2.set_ylabel("Cohen's d")
    ax2.set_title("Effect Size (Low vs High Temp)\n(higher = better)")
    ax2.legend(fontsize=8)

    # Plot 3: ROC AUC
    ax3 = fig.add_subplot(2, 3, 3)
    aucs = [temp_results[m]['roc_auc'] or 0.5 for m in metrics]
    ax3.bar(x, aucs, color=['steelblue', 'coral', 'green'][:len(metrics)])
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Random')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.upper() for m in metrics])
    ax3.set_ylabel('ROC AUC')
    ax3.set_title('Binary Classification\n(T\u22640.2 vs T\u22650.7)')
    ax3.set_ylim([0.3, 0.8])
    ax3.legend()

    # Plot 4: Consistency by temperature for each metric
    ax4 = fig.add_subplot(2, 3, 4)
    colors = {'ted': 'steelblue', 'bertscore': 'coral', 'deepdiff': 'green'}

    for metric in metrics:
        temps = sorted(temp_results[metric]['temp_summary'].keys())
        means = [temp_results[metric]['temp_summary'][t]['mean'] for t in temps]
        ci_low = [temp_results[metric]['temp_summary'][t]['ci_lower'] for t in temps]
        ci_high = [temp_results[metric]['temp_summary'][t]['ci_upper'] for t in temps]

        ax4.errorbar(temps, means,
                     yerr=[np.array(means) - np.array(ci_low),
                           np.array(ci_high) - np.array(means)],
                     label=metric.upper(), color=colors.get(metric, 'gray'),
                     capsize=3, marker='o')

    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Mean Consistency Score')
    ax4.set_title('Consistency vs Temperature\n(with 95% CI)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Metric correlations heatmap
    ax5 = fig.add_subplot(2, 3, 5)
    if corr_results:
        corr_matrix = np.eye(len(metrics))
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if i != j:
                    key = f'{m1}_vs_{m2}' if f'{m1}_vs_{m2}' in corr_results else f'{m2}_vs_{m1}'
                    if key in corr_results:
                        corr_matrix[i, j] = corr_results[key]['pearson_r']

        im = ax5.imshow(corr_matrix, cmap='RdYlBu', vmin=0, vmax=1)
        ax5.set_xticks(range(len(metrics)))
        ax5.set_yticks(range(len(metrics)))
        ax5.set_xticklabels([m.upper() for m in metrics])
        ax5.set_yticklabels([m.upper() for m in metrics])

        for i in range(len(metrics)):
            for j in range(len(metrics)):
                ax5.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=10)

        plt.colorbar(im, ax=ax5, label='Pearson r')
        ax5.set_title('Metric Correlations')

    # Plot 6: Sample sizes
    ax6 = fig.add_subplot(2, 3, 6)
    n_totals = [temp_results[m]['n_total'] for m in metrics]
    n_temps = [temp_results[m]['n_temps'] for m in metrics]

    ax6.bar(x - 0.2, n_totals, 0.4, label='Total Samples', color='steelblue', alpha=0.7)
    ax6_twin = ax6.twinx()
    ax6_twin.bar(x + 0.2, n_temps, 0.4, label='Temp Settings', color='coral', alpha=0.7)

    ax6.set_xticks(x)
    ax6.set_xticklabels([m.upper() for m in metrics])
    ax6.set_ylabel('Total Samples')
    ax6_twin.set_ylabel('Temperature Settings')
    ax6.set_title('Data Coverage')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(output_dir / 'publication_figures.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Publication figures saved to: {output_dir / 'publication_figures.png'}")


def print_latex_tables(temp_results: Dict, model_results: Dict, corr_results: Dict):
    """Generate LaTeX tables for paper."""

    print("\n" + "="*80)
    print("LATEX TABLES FOR PAPER")
    print("="*80)

    # Table 1: Temperature discrimination
    print("\n% Table 1: Temperature Discrimination Results")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Temperature Discrimination Analysis}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Metric & Spearman $\\rho$ & p-value & Cohen's d & ROC AUC & n \\\\")
    print("\\hline")

    for metric in temp_results.keys():
        r = temp_results[metric]
        p_str = f"{r['spearman_p']:.2e}" if r['spearman_p'] < 0.01 else f"{r['spearman_p']:.3f}"
        print(f"{metric.upper()} & {r['spearman_r']:.3f} & {p_str} & "
              f"{r['cohens_d']:.3f} & {r['roc_auc']:.3f} & {r['n_total']} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:temp_discrimination}")
    print("\\end{table}")

    # Table 2: Metric correlations
    if corr_results:
        print("\n% Table 2: Inter-Metric Correlations")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Inter-Metric Correlations (Pearson r)}")
        print("\\begin{tabular}{lccc}")
        print("\\hline")
        print("Comparison & Pearson r & p-value & n \\\\")
        print("\\hline")

        for key, val in corr_results.items():
            p_str = f"{val['pearson_p']:.2e}" if val['pearson_p'] < 0.01 else f"{val['pearson_p']:.3f}"
            print(f"{key.replace('_', ' ').upper()} & {val['pearson_r']:.3f} & {p_str} & {val['n_samples']} \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\label{tab:metric_correlations}")
        print("\\end{table}")


def main():
    """Run comprehensive analysis."""
    print("="*80)
    print("COMPREHENSIVE ANALYSIS FOR MAIN CONFERENCE PAPER")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    all_data = load_all_data()

    print(f"   Loaded {len(all_data)} models:")
    for model, data in all_data.items():
        metrics = list(data.keys())
        n_temps = len(next(iter(data.values())))
        print(f"   - {model}: {metrics}, {n_temps} temperature settings")

    if len(all_data) < 2:
        print("\n   WARNING: Insufficient data for comprehensive analysis.")
        print("   Need to compute metrics for more models first.")
        print("   Run: python compute_all_metrics.py")
        return

    # Temperature effect analysis
    print("\n2. Analyzing temperature effect...")
    temp_results = analyze_temperature_effect(all_data)

    print("\n   Temperature Discrimination Results:")
    print(f"   {'Metric':<12} {'Spearman r':>12} {'p-value':>12} {'Cohen d':>10} {'ROC AUC':>10}")
    print("   " + "-"*58)
    for metric, res in temp_results.items():
        print(f"   {metric:<12} {res['spearman_r']:>12.4f} {res['spearman_p']:>12.2e} "
              f"{res['cohens_d'] or 0:>10.3f} {res['roc_auc'] or 0:>10.3f}")

    # Model ranking analysis
    print("\n3. Analyzing model rankings...")
    model_results = analyze_model_ranking(all_data)

    print("\n   Model Rankings:")
    for metric, ranking in model_results['rankings'].items():
        print(f"   {metric.upper()}: {' > '.join(ranking)}")

    # Metric correlations
    print("\n4. Analyzing metric correlations...")
    corr_results = analyze_metric_correlations(all_data)

    if corr_results:
        print("\n   Inter-Metric Correlations:")
        for key, val in corr_results.items():
            print(f"   {key}: r={val['pearson_r']:.3f}, p={val['pearson_p']:.2e}")

    # Create figures
    print("\n5. Creating publication figures...")
    create_publication_figures(temp_results, model_results, corr_results, OUTPUT_DIR)

    # Print LaTeX tables
    print_latex_tables(temp_results, model_results, corr_results)

    # Save results
    results = {
        'temperature_effect': temp_results,
        'model_ranking': model_results,
        'metric_correlations': corr_results,
        'summary': {
            'n_models': len(all_data),
            'models': list(all_data.keys()),
            'metrics': list(temp_results.keys())
        }
    }

    results_path = OUTPUT_DIR / 'comprehensive_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved to: {results_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Find best metric
    best_spearman = max(temp_results.keys(), key=lambda m: abs(temp_results[m]['spearman_r']))
    best_effect = max(temp_results.keys(), key=lambda m: temp_results[m]['cohens_d'] or 0)
    best_auc = max(temp_results.keys(), key=lambda m: temp_results[m]['roc_auc'] or 0)

    print(f"""
   KEY FINDINGS (n={len(all_data)} models):

   1. Temperature Discrimination:
      - Best correlation: {best_spearman.upper()} (r={temp_results[best_spearman]['spearman_r']:.3f})
      - Best effect size: {best_effect.upper()} (d={temp_results[best_effect]['cohens_d']:.3f})
      - Best ROC AUC: {best_auc.upper()} (AUC={temp_results[best_auc]['roc_auc']:.3f})

   2. Statistical Significance:
      - TED: p={temp_results['ted']['spearman_p']:.2e} {'***' if temp_results['ted']['spearman_p'] < 0.001 else '**' if temp_results['ted']['spearman_p'] < 0.01 else '*' if temp_results['ted']['spearman_p'] < 0.05 else 'ns'}
      - BERTScore: p={temp_results['bertscore']['spearman_p']:.2e} {'***' if temp_results['bertscore']['spearman_p'] < 0.001 else '**' if temp_results['bertscore']['spearman_p'] < 0.01 else '*' if temp_results['bertscore']['spearman_p'] < 0.05 else 'ns'}
      - DeepDiff: p={temp_results['deepdiff']['spearman_p']:.2e} {'***' if temp_results['deepdiff']['spearman_p'] < 0.001 else '**' if temp_results['deepdiff']['spearman_p'] < 0.01 else '*' if temp_results['deepdiff']['spearman_p'] < 0.05 else 'ns'}

   3. Model Ranking Consistency:
      - Rankings are {'consistent' if len(set(tuple(r) for r in model_results['rankings'].values())) == 1 else 'variable'} across metrics
    """)

    # Recommendations
    print("""
   RECOMMENDATIONS FOR PAPER:

   1. Expand model coverage to 10 models (run compute_all_metrics.py)
   2. Current data shows {status} temperature discrimination
   3. Consider adding STED comparison once metrics are computed
   4. Bootstrap CIs provide robust uncertainty estimates

   Next steps:
   - Run: python compute_all_metrics.py  (computes all metrics for all models)
   - Re-run this analysis after metrics are computed
    """.format(status='moderate' if max(temp_results[m]['roc_auc'] or 0 for m in temp_results) > 0.6 else 'limited'))

    return results


if __name__ == "__main__":
    results = main()
