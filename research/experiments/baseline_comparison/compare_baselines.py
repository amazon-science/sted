"""
Comprehensive Baseline Comparison for STED

Compare STED against stable algorithmic baselines:
1. TED (Tree Edit Distance) - structural comparison
2. BERTScore - semantic text similarity
3. DeepDiff - dictionary comparison

Evaluation criteria:
- Discrimination: Can the metric distinguish between temperature levels?
- Stability: Are model rankings consistent?
- Sensitivity: Does the metric detect variations?
- Correlation: How do metrics correlate with each other?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple
import json
import os
from collections import defaultdict
from pathlib import Path
import re

# Paths
STED_PROJECT = Path("/Users/guanghu/Documents/genai/projects/sted")
LLM_RESULTS_DIR = STED_PROJECT / "llm_gen_results"
OUTPUT_DIR = Path(__file__).parent
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    """
    Load all data for all models and metrics.

    Returns:
        {model: {metric: {temp: [samples]}}}
    """
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


def compute_consistency_score(samples: List[Dict], use_mean: bool = True) -> float:
    """
    Compute consistency score from sample results.

    For each sample, we have mean similarity across 10 runs.
    Higher mean + lower std = higher consistency.
    """
    if not samples:
        return 0.0

    if use_mean:
        # Simple: use mean of mean similarities
        means = [s['mean'] for s in samples]
        return float(np.mean(means))
    else:
        # Penalize high variance
        means = [s['mean'] for s in samples]
        stds = [s['std'] for s in samples]
        # Consistency = mean - penalty * std
        mean_sim = np.mean(means)
        mean_std = np.mean(stds)
        return float(mean_sim * (1 - min(mean_std * 10, 1.0)))


def analyze_temperature_discrimination(all_data: Dict) -> Dict:
    """
    Analyze how well each metric discriminates between temperature levels.
    """
    results = {}

    for metric in METRICS:
        temp_scores = defaultdict(list)

        for model_name, model_data in all_data.items():
            if metric not in model_data:
                continue

            for temp, samples in model_data[metric].items():
                score = compute_consistency_score(samples)
                temp_scores[temp].append(score)

        # Compute statistics
        temps = sorted(temp_scores.keys())
        if len(temps) < 2:
            continue

        mean_by_temp = {t: np.mean(temp_scores[t]) for t in temps}
        std_by_temp = {t: np.std(temp_scores[t]) for t in temps}

        # Spearman correlation (should be negative: higher temp = lower consistency)
        all_temps = []
        all_scores = []
        for t in temps:
            all_temps.extend([t] * len(temp_scores[t]))
            all_scores.extend(temp_scores[t])

        spearman_corr, spearman_p = stats.spearmanr(all_temps, all_scores)

        # Score range
        score_range = max(mean_by_temp.values()) - min(mean_by_temp.values())

        # Effect size (Cohen's d between low and high temp)
        low_temps = [t for t in temps if t <= 0.3]
        high_temps = [t for t in temps if t >= 0.7]

        low_scores = []
        high_scores = []
        for t in low_temps:
            low_scores.extend(temp_scores[t])
        for t in high_temps:
            high_scores.extend(temp_scores[t])

        if low_scores and high_scores:
            pooled_std = np.sqrt((np.var(low_scores) + np.var(high_scores)) / 2)
            if pooled_std > 1e-10:
                cohens_d = (np.mean(low_scores) - np.mean(high_scores)) / pooled_std
            else:
                cohens_d = 0
        else:
            cohens_d = 0

        results[metric] = {
            'mean_by_temp': mean_by_temp,
            'std_by_temp': std_by_temp,
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'score_range': float(score_range),
            'cohens_d': float(cohens_d),
            'n_temps': len(temps),
            'n_samples': len(all_scores)
        }

    return results


def analyze_model_ranking(all_data: Dict) -> Dict:
    """
    Analyze model rankings by each metric.
    """
    results = {}

    for metric in METRICS:
        model_scores = {}

        for model_name, model_data in all_data.items():
            if metric not in model_data:
                continue

            all_samples = []
            for temp, samples in model_data[metric].items():
                all_samples.extend(samples)

            if all_samples:
                score = compute_consistency_score(all_samples)
                model_scores[model_name] = score

        if model_scores:
            # Rank models
            rankings = sorted(model_scores.keys(),
                            key=lambda m: model_scores[m],
                            reverse=True)

            results[metric] = {
                'model_scores': model_scores,
                'rankings': rankings,
                'score_range': max(model_scores.values()) - min(model_scores.values())
            }

    return results


def analyze_metric_correlations(all_data: Dict) -> Dict:
    """
    Analyze correlations between different metrics.
    """
    # Collect paired observations
    metric_values = {m: [] for m in METRICS}

    for model_name, model_data in all_data.items():
        # Get all temperatures that have all metrics
        all_temps = set()
        for metric in METRICS:
            if metric in model_data:
                all_temps.update(model_data[metric].keys())

        for temp in all_temps:
            values = {}
            for metric in METRICS:
                if metric in model_data and temp in model_data[metric]:
                    values[metric] = compute_consistency_score(model_data[metric][temp])

            if len(values) == len(METRICS):
                for metric in METRICS:
                    metric_values[metric].append(values[metric])

    # Compute correlations
    correlations = {}
    for i, m1 in enumerate(METRICS):
        for m2 in METRICS[i+1:]:
            if metric_values[m1] and metric_values[m2]:
                corr, p_val = stats.pearsonr(metric_values[m1], metric_values[m2])
                correlations[f'{m1}_vs_{m2}'] = {
                    'pearson_r': float(corr),
                    'p_value': float(p_val),
                    'n_samples': len(metric_values[m1])
                }

    return correlations


def analyze_sensitivity(all_data: Dict) -> Dict:
    """
    Analyze sensitivity of each metric to variations.

    A good metric should detect variations (non-zero std across runs).
    """
    results = {}

    for metric in METRICS:
        all_stds = []
        zero_std_count = 0
        total_count = 0

        for model_name, model_data in all_data.items():
            if metric not in model_data:
                continue

            for temp, samples in model_data[metric].items():
                for sample in samples:
                    std = sample.get('std', 0)
                    all_stds.append(std)
                    total_count += 1
                    if std < 1e-10:
                        zero_std_count += 1

        if all_stds:
            results[metric] = {
                'mean_std': float(np.mean(all_stds)),
                'median_std': float(np.median(all_stds)),
                'max_std': float(np.max(all_stds)),
                'zero_std_ratio': zero_std_count / total_count if total_count > 0 else 0,
                'n_samples': total_count
            }

    return results


def analyze_binary_classification(all_data: Dict,
                                   low_threshold: float = 0.2,
                                   high_threshold: float = 0.7) -> Dict:
    """
    Analyze binary classification performance (low vs high temperature).
    """
    results = {}

    for metric in METRICS:
        low_scores = []
        high_scores = []

        for model_name, model_data in all_data.items():
            if metric not in model_data:
                continue

            for temp, samples in model_data[metric].items():
                score = compute_consistency_score(samples)
                if temp <= low_threshold:
                    low_scores.append(score)
                elif temp >= high_threshold:
                    high_scores.append(score)

        if not low_scores or not high_scores:
            continue

        # ROC AUC
        all_scores = low_scores + high_scores
        all_labels = [1] * len(low_scores) + [0] * len(high_scores)

        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)

        # d-prime
        pooled_std = np.sqrt((np.var(low_scores) + np.var(high_scores)) / 2)
        if pooled_std > 1e-10:
            dprime = (np.mean(low_scores) - np.mean(high_scores)) / pooled_std
        else:
            dprime = 0

        results[metric] = {
            'roc_auc': float(roc_auc),
            'dprime': float(dprime),
            'mean_low': float(np.mean(low_scores)),
            'mean_high': float(np.mean(high_scores)),
            'n_low': len(low_scores),
            'n_high': len(high_scores),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
        }

    return results


def create_visualizations(temp_discrimination: Dict,
                          model_ranking: Dict,
                          correlations: Dict,
                          sensitivity: Dict,
                          binary_classification: Dict,
                          output_dir: Path):
    """Generate comprehensive visualizations."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metrics = list(temp_discrimination.keys())
    colors = {'ted': 'steelblue', 'bertscore': 'coral', 'deepdiff': 'green'}

    # 1. Temperature discrimination - Spearman correlation
    ax = axes[0, 0]
    correlations_vals = [temp_discrimination[m]['spearman_correlation'] for m in metrics]
    bars = ax.bar(metrics, correlations_vals,
                  color=[colors.get(m, 'gray') for m in metrics])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Spearman ρ (temp vs score)')
    ax.set_title('Temperature Discrimination\n(more negative = better)')
    for bar, val in zip(bars, correlations_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom')

    # 2. Score range
    ax = axes[0, 1]
    ranges = [temp_discrimination[m]['score_range'] for m in metrics]
    bars = ax.bar(metrics, ranges,
                  color=[colors.get(m, 'gray') for m in metrics])
    ax.set_ylabel('Score Range')
    ax.set_title('Score Range (Low to High Temp)\n(larger = better)')
    for bar, val in zip(bars, ranges):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')

    # 3. Cohen's d
    ax = axes[0, 2]
    cohens = [temp_discrimination[m]['cohens_d'] for m in metrics]
    bars = ax.bar(metrics, cohens,
                  color=[colors.get(m, 'gray') for m in metrics])
    ax.axhline(y=0.8, color='green', linestyle='--', label='Large effect')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='Medium effect')
    ax.set_ylabel("Cohen's d")
    ax.set_title('Effect Size (Low vs High Temp)\n(larger = better)')
    ax.legend()
    for bar, val in zip(bars, cohens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom')

    # 4. Sensitivity (zero_std_ratio)
    ax = axes[1, 0]
    zero_ratios = [sensitivity[m]['zero_std_ratio'] * 100 for m in metrics]
    bars = ax.bar(metrics, zero_ratios,
                  color=[colors.get(m, 'gray') for m in metrics])
    ax.set_ylabel('% Samples with Zero Std')
    ax.set_title('Sensitivity to Variations\n(lower = more sensitive)')
    for bar, val in zip(bars, zero_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom')

    # 5. ROC AUC
    ax = axes[1, 1]
    aucs = [binary_classification[m]['roc_auc'] for m in metrics if m in binary_classification]
    metrics_with_auc = [m for m in metrics if m in binary_classification]
    bars = ax.bar(metrics_with_auc, aucs,
                  color=[colors.get(m, 'gray') for m in metrics_with_auc])
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Binary Classification\n(T≤0.2 vs T≥0.7)')
    ax.set_ylim([0.4, 1.0])
    ax.legend()
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom')

    # 6. ROC curves
    ax = axes[1, 2]
    for metric in metrics:
        if metric in binary_classification:
            fpr = binary_classification[metric]['fpr']
            tpr = binary_classification[metric]['tpr']
            auc_val = binary_classification[metric]['roc_auc']
            ax.plot(fpr, tpr, color=colors.get(metric, 'gray'),
                    label=f'{metric} (AUC={auc_val:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / 'baseline_comparison.png', dpi=150)
    plt.close(fig)

    # Figure 2: Score by temperature for each metric
    fig2, axes2 = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))

    for i, metric in enumerate(metrics):
        ax = axes2[i] if len(metrics) > 1 else axes2
        data = temp_discrimination[metric]
        temps = sorted(data['mean_by_temp'].keys())
        means = [data['mean_by_temp'][t] for t in temps]
        stds = [data['std_by_temp'][t] for t in temps]

        ax.errorbar(temps, means, yerr=stds, fmt='o-',
                    color=colors.get(metric, 'gray'), capsize=3)
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Mean Consistency Score')
        ax.set_title(f'{metric.upper()}\nρ={data["spearman_correlation"]:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(output_dir / 'score_by_temperature.png', dpi=150)
    plt.close(fig2)

    print("Visualizations saved.")


def run_baseline_comparison():
    """Run complete baseline comparison."""

    print("=" * 70)
    print("BASELINE COMPARISON: STED vs TED vs BERTScore vs DeepDiff")
    print("=" * 70)

    # Load all data
    print("\n1. Loading data...")
    all_data = load_all_data()
    print(f"   Loaded {len(all_data)} models")
    for model, data in all_data.items():
        metrics_available = list(data.keys())
        print(f"   - {model}: {metrics_available}")

    # Temperature discrimination
    print("\n2. Analyzing temperature discrimination...")
    temp_discrimination = analyze_temperature_discrimination(all_data)

    print("\n   Metric     | Spearman ρ | Score Range | Cohen's d | Best?")
    print("   " + "-" * 60)
    best_metric = max(temp_discrimination.keys(),
                     key=lambda m: abs(temp_discrimination[m]['spearman_correlation']))
    for metric in METRICS:
        if metric in temp_discrimination:
            d = temp_discrimination[metric]
            is_best = " <-- BEST" if metric == best_metric else ""
            print(f"   {metric:10s} | {d['spearman_correlation']:10.3f} | {d['score_range']:11.3f} | {d['cohens_d']:9.3f} |{is_best}")

    # Model ranking
    print("\n3. Analyzing model rankings...")
    model_ranking = analyze_model_ranking(all_data)

    print("\n   Model Rankings by Metric:")
    for metric in METRICS:
        if metric in model_ranking:
            rankings = model_ranking[metric]['rankings']
            print(f"\n   {metric.upper()}:")
            for i, model in enumerate(rankings[:5], 1):
                score = model_ranking[metric]['model_scores'][model]
                print(f"      {i}. {model}: {score:.4f}")

    # Metric correlations
    print("\n4. Analyzing metric correlations...")
    correlations = analyze_metric_correlations(all_data)

    print("\n   Metric Pair         | Pearson r | p-value")
    print("   " + "-" * 45)
    for pair, data in correlations.items():
        print(f"   {pair:20s} | {data['pearson_r']:9.3f} | {data['p_value']:.2e}")

    # Sensitivity
    print("\n5. Analyzing sensitivity...")
    sensitivity = analyze_sensitivity(all_data)

    print("\n   Metric     | Mean Std | Zero Std % | Interpretation")
    print("   " + "-" * 55)
    for metric in METRICS:
        if metric in sensitivity:
            d = sensitivity[metric]
            interp = "LOW" if d['zero_std_ratio'] > 0.5 else "HIGH" if d['zero_std_ratio'] < 0.1 else "MEDIUM"
            print(f"   {metric:10s} | {d['mean_std']:8.4f} | {d['zero_std_ratio']*100:9.1f}% | {interp} sensitivity")

    # Binary classification
    print("\n6. Analyzing binary classification...")
    binary_classification = analyze_binary_classification(all_data)

    print("\n   Metric     | ROC AUC | d-prime | Mean(Low) | Mean(High)")
    print("   " + "-" * 60)
    for metric in METRICS:
        if metric in binary_classification:
            d = binary_classification[metric]
            print(f"   {metric:10s} | {d['roc_auc']:7.3f} | {d['dprime']:7.3f} | {d['mean_low']:9.3f} | {d['mean_high']:10.3f}")

    # Generate visualizations
    print("\n7. Generating visualizations...")
    create_visualizations(temp_discrimination, model_ranking, correlations,
                          sensitivity, binary_classification, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BASELINE COMPARISON")
    print("=" * 70)

    print("""
    METRIC COMPARISON:
    """)

    # Determine best metric for each criterion
    best_temp_disc = max(temp_discrimination.keys(),
                        key=lambda m: abs(temp_discrimination[m]['spearman_correlation']))
    best_effect = max(temp_discrimination.keys(),
                     key=lambda m: temp_discrimination[m]['cohens_d'])
    best_auc = max(binary_classification.keys(),
                  key=lambda m: binary_classification[m]['roc_auc'])
    best_sensitivity = min(sensitivity.keys(),
                          key=lambda m: sensitivity[m]['zero_std_ratio'])

    print(f"""
    | Criterion              | Best Metric   | Value |
    |------------------------|---------------|-------|
    | Temperature Corr. (|ρ|)| {best_temp_disc:13s} | {abs(temp_discrimination[best_temp_disc]['spearman_correlation']):.3f} |
    | Effect Size (d')       | {best_effect:13s} | {temp_discrimination[best_effect]['cohens_d']:.3f} |
    | ROC AUC                | {best_auc:13s} | {binary_classification[best_auc]['roc_auc']:.3f} |
    | Sensitivity            | {best_sensitivity:13s} | {(1-sensitivity[best_sensitivity]['zero_std_ratio'])*100:.1f}% |
    """)

    # STED comparison note
    print("""
    NOTE: STED (Semantic Tree Edit Distance) combines:
    - TED's structural comparison
    - BERTScore-like semantic similarity
    - Weighted field-level matching

    For paper, compare STED against:
    1. TED - pure structural baseline
    2. BERTScore - pure semantic baseline
    3. DeepDiff - dictionary comparison baseline

    Key advantage of STED: Balances structure AND semantics
    """)

    # Save results
    results = {
        'temperature_discrimination': {
            k: {kk: vv for kk, vv in v.items() if kk not in ['mean_by_temp', 'std_by_temp']}
            for k, v in temp_discrimination.items()
        },
        'model_ranking': {
            k: {'rankings': v['rankings'][:5], 'score_range': v['score_range']}
            for k, v in model_ranking.items()
        },
        'correlations': correlations,
        'sensitivity': sensitivity,
        'binary_classification': {
            k: {kk: vv for kk, vv in v.items() if kk not in ['fpr', 'tpr']}
            for k, v in binary_classification.items()
        },
        'best_metrics': {
            'temperature_discrimination': best_temp_disc,
            'effect_size': best_effect,
            'roc_auc': best_auc,
            'sensitivity': best_sensitivity
        }
    }

    results_path = OUTPUT_DIR / 'baseline_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = run_baseline_comparison()
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON COMPLETE")
    print("=" * 70)
