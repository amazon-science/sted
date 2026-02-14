"""
Validate Theoretical Power Transformation Analysis on Real LLM Data

This script validates the theoretical findings from our analysis using
actual LLM generation results from the temperature experiments.

Key validations:
1. Real dispersion distribution matches theoretical assumptions
2. β=20 provides appropriate discrimination on real data
3. Alternative β values comparison on real data
4. Model ranking stability across β values
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

# Paths
STED_PROJECT = Path("/Users/guanghu/Documents/genai/projects/sted")
METRICS_PATH = STED_PROJECT / "results/temp_experiment_metrics/combined_consistency_metrics_results.json"
OUTPUT_DIR = Path(__file__).parent
LLM_RESULTS_DIR = STED_PROJECT / "llm_gen_results"


def power_transform(sigma: np.ndarray, alpha: float = 2.0, beta: float = 20.0) -> np.ndarray:
    """PDC power transformation"""
    return (1.0 / (1.0 + alpha * sigma)) ** beta


def load_consistency_metrics() -> Dict:
    """Load all consistency metrics from experiments."""
    with open(METRICS_PATH, 'r') as f:
        data = json.load(f)
    return data


def extract_dispersion_by_model_temp(data: Dict) -> Dict[str, Dict[float, List[float]]]:
    """
    Extract dispersion values (normalized_cv) by model and temperature.

    Returns:
        {model_name: {temperature: [dispersion_values]}}
    """
    result = defaultdict(lambda: defaultdict(list))

    for model_name, samples in data.items():
        for sample in samples:
            temp = sample['temperature']
            # Use normalized_cv as proxy for dispersion
            # normalized_cv = cv * 20, so cv = normalized_cv / 20
            # std = cv * mean, so std ≈ (normalized_cv / 20) * mean_similarity
            mean_sim = sample.get('mean_similarity', 0.9)
            normalized_cv = sample.get('normalized_cv', 0)

            # Approximate dispersion (std deviation of similarities)
            dispersion = (normalized_cv / 20.0) * mean_sim

            result[model_name][temp].append(dispersion)

    return dict(result)


def analyze_real_dispersion_distribution(dispersion_data: Dict) -> Dict:
    """
    Analyze the actual distribution of dispersion values.
    Compare with theoretical assumptions.
    """
    all_dispersions = []
    by_temperature = defaultdict(list)

    for model, temp_data in dispersion_data.items():
        for temp, values in temp_data.items():
            all_dispersions.extend(values)
            by_temperature[temp].extend(values)

    all_dispersions = np.array(all_dispersions)

    # Fit distributions
    beta_params = stats.beta.fit(np.clip(all_dispersions / 0.5, 0.001, 0.999))

    results = {
        'overall': {
            'mean': float(np.mean(all_dispersions)),
            'std': float(np.std(all_dispersions)),
            'median': float(np.median(all_dispersions)),
            'min': float(np.min(all_dispersions)),
            'max': float(np.max(all_dispersions)),
            'percentiles': {
                '25': float(np.percentile(all_dispersions, 25)),
                '50': float(np.percentile(all_dispersions, 50)),
                '75': float(np.percentile(all_dispersions, 75)),
                '90': float(np.percentile(all_dispersions, 90)),
                '95': float(np.percentile(all_dispersions, 95)),
            },
            'beta_fit_params': beta_params,
            'n_samples': len(all_dispersions)
        },
        'by_temperature': {}
    }

    for temp in sorted(by_temperature.keys()):
        values = np.array(by_temperature[temp])
        results['by_temperature'][temp] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'n_samples': len(values)
        }

    return results


def validate_beta_discrimination(dispersion_data: Dict,
                                  beta_values: List[float] = [3, 5, 10, 15, 20, 30]) -> Dict:
    """
    Validate how different β values discriminate between temperature levels.

    For each β:
    - Compute PDC scores from real dispersions
    - Measure separation between temperature groups
    - Compute ranking stability
    """
    alpha = 2.0
    results = {}

    # Aggregate by temperature across all models
    temp_dispersions = defaultdict(list)
    for model, temp_data in dispersion_data.items():
        for temp, values in temp_data.items():
            temp_dispersions[temp].extend(values)

    temps = sorted(temp_dispersions.keys())

    for beta in beta_values:
        # Compute PDC scores for each temperature
        temp_scores = {}
        for temp in temps:
            dispersions = np.array(temp_dispersions[temp])
            scores = power_transform(dispersions, alpha, beta)
            temp_scores[temp] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
            }

        # Compute separation metrics
        # Cohen's d between adjacent temperatures
        cohens_d = []
        for i in range(len(temps) - 1):
            t1, t2 = temps[i], temps[i+1]
            scores1 = power_transform(np.array(temp_dispersions[t1]), alpha, beta)
            scores2 = power_transform(np.array(temp_dispersions[t2]), alpha, beta)

            pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
            if pooled_std > 1e-10:
                d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
            else:
                d = 0
            cohens_d.append(d)

        # Spearman correlation between temperature and score
        all_temps_expanded = []
        all_scores = []
        for temp in temps:
            dispersions = np.array(temp_dispersions[temp])
            scores = power_transform(dispersions, alpha, beta)
            all_temps_expanded.extend([temp] * len(scores))
            all_scores.extend(scores)

        spearman_corr, _ = stats.spearmanr(all_temps_expanded, all_scores)

        results[beta] = {
            'temp_scores': temp_scores,
            'cohens_d': cohens_d,
            'mean_cohens_d': float(np.mean(cohens_d)),
            'min_cohens_d': float(np.min(cohens_d)),
            'spearman_correlation': float(spearman_corr),
            'score_range': float(temp_scores[temps[0]]['mean'] - temp_scores[temps[-1]]['mean'])
        }

    return results


def validate_model_ranking_stability(dispersion_data: Dict,
                                     beta_values: List[float] = [3, 5, 10, 15, 20, 30]) -> Dict:
    """
    Check if model rankings are stable across different β values.

    A good β should:
    - Preserve relative model ordering
    - Provide meaningful score differences
    """
    alpha = 2.0
    model_rankings = {}
    model_scores = defaultdict(dict)

    for beta in beta_values:
        scores = {}
        for model, temp_data in dispersion_data.items():
            all_dispersions = []
            for temp, values in temp_data.items():
                all_dispersions.extend(values)

            all_dispersions = np.array(all_dispersions)
            pdc_scores = power_transform(all_dispersions, alpha, beta)
            scores[model] = float(np.mean(pdc_scores))
            model_scores[model][beta] = float(np.mean(pdc_scores))

        # Rank models
        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        model_rankings[beta] = {model: rank + 1 for rank, model in enumerate(ranked)}

    # Compute ranking stability (Kendall's tau between β=20 and other β values)
    reference_ranking = model_rankings[20]
    stability = {}

    for beta in beta_values:
        if beta == 20:
            stability[beta] = 1.0
            continue

        models = list(reference_ranking.keys())
        ranks_ref = [reference_ranking[m] for m in models]
        ranks_curr = [model_rankings[beta][m] for m in models]

        tau, _ = stats.kendalltau(ranks_ref, ranks_curr)
        stability[beta] = float(tau)

    return {
        'rankings': model_rankings,
        'scores': dict(model_scores),
        'ranking_stability_vs_beta20': stability
    }


def validate_binary_classification(dispersion_data: Dict,
                                   beta_values: List[float] = [3, 5, 10, 15, 20, 30],
                                   low_temp_threshold: float = 0.3,
                                   high_temp_threshold: float = 0.7) -> Dict:
    """
    Validate binary classification performance (low temp = consistent, high temp = inconsistent).
    """
    alpha = 2.0

    # Collect dispersions for low and high temperature
    low_temp_dispersions = []
    high_temp_dispersions = []

    for model, temp_data in dispersion_data.items():
        for temp, values in temp_data.items():
            if temp <= low_temp_threshold:
                low_temp_dispersions.extend(values)
            elif temp >= high_temp_threshold:
                high_temp_dispersions.extend(values)

    low_temp_dispersions = np.array(low_temp_dispersions)
    high_temp_dispersions = np.array(high_temp_dispersions)

    results = {}

    for beta in beta_values:
        # Transform
        scores_low = power_transform(low_temp_dispersions, alpha, beta)
        scores_high = power_transform(high_temp_dispersions, alpha, beta)

        # Create classification labels
        all_scores = np.concatenate([scores_low, scores_high])
        all_labels = np.concatenate([np.ones(len(scores_low)), np.zeros(len(scores_high))])

        # Compute ROC AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)

        # Compute d-prime
        pooled_std = np.sqrt((np.var(scores_low) + np.var(scores_high)) / 2)
        if pooled_std > 1e-10:
            dprime = (np.mean(scores_low) - np.mean(scores_high)) / pooled_std
        else:
            dprime = float('inf')

        # Optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)

        results[beta] = {
            'roc_auc': float(roc_auc),
            'dprime': float(dprime),
            'mean_score_low_temp': float(np.mean(scores_low)),
            'mean_score_high_temp': float(np.mean(scores_high)),
            'optimal_threshold': float(thresholds[optimal_idx]),
            'sensitivity': float(tpr[optimal_idx]),
            'specificity': float(1 - fpr[optimal_idx]),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
        }

    return results


def create_validation_visualizations(dispersion_analysis: Dict,
                                     beta_discrimination: Dict,
                                     binary_classification: Dict,
                                     model_ranking: Dict,
                                     output_dir: Path):
    """Generate validation visualizations."""

    # Figure 1: Real dispersion distribution
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1a: Overall distribution
    ax = axes[0]
    # Recreate distribution for plotting
    all_dispersions = []
    for model, temp_data in dispersion_data.items():
        for temp, values in temp_data.items():
            all_dispersions.extend(values)
    all_dispersions = np.array(all_dispersions)

    ax.hist(all_dispersions, bins=50, density=True, alpha=0.7, color='steelblue')
    ax.axvline(dispersion_analysis['overall']['mean'], color='red',
               linestyle='--', label=f"Mean: {dispersion_analysis['overall']['mean']:.3f}")
    ax.axvline(dispersion_analysis['overall']['median'], color='green',
               linestyle='--', label=f"Median: {dispersion_analysis['overall']['median']:.3f}")
    ax.set_xlabel('Dispersion (σ)')
    ax.set_ylabel('Density')
    ax.set_title('Real LLM Output Dispersion Distribution')
    ax.legend()

    # 1b: Distribution by temperature
    ax = axes[1]
    temps = sorted(dispersion_analysis['by_temperature'].keys())
    means = [dispersion_analysis['by_temperature'][t]['mean'] for t in temps]
    stds = [dispersion_analysis['by_temperature'][t]['std'] for t in temps]

    ax.errorbar(temps, means, yerr=stds, fmt='o-', capsize=5, color='steelblue')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mean Dispersion (σ)')
    ax.set_title('Dispersion vs Temperature')
    ax.grid(True, alpha=0.3)

    # 1c: Comparison with theoretical assumption
    ax = axes[2]
    sigma_theoretical = np.linspace(0, 0.3, 100)
    # Our theoretical assumption was Beta(2, 30) * 0.15 for high consistency
    theoretical_high = stats.beta.pdf(sigma_theoretical / 0.3, 2, 30) * 0.3
    ax.plot(sigma_theoretical, theoretical_high / theoretical_high.max(),
            'r--', label='Theoretical (high cons.)', linewidth=2)

    # Real distribution
    hist, bins = np.histogram(all_dispersions, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(bin_centers, hist / hist.max(), 'b-', label='Real Data', linewidth=2)

    ax.set_xlabel('Dispersion (σ)')
    ax.set_ylabel('Normalized Density')
    ax.set_title('Real vs Theoretical Distribution')
    ax.legend()

    plt.tight_layout()
    fig1.savefig(output_dir / 'real_data_dispersion_distribution.png', dpi=150)
    plt.close(fig1)

    # Figure 2: Beta discrimination validation
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))

    beta_values = sorted(beta_discrimination.keys())

    # 2a: Mean Cohen's d by beta
    ax = axes[0, 0]
    mean_cohens = [beta_discrimination[b]['mean_cohens_d'] for b in beta_values]
    min_cohens = [beta_discrimination[b]['min_cohens_d'] for b in beta_values]

    x = range(len(beta_values))
    width = 0.35
    ax.bar([i - width/2 for i in x], mean_cohens, width, label='Mean Cohen\'s d', color='steelblue')
    ax.bar([i + width/2 for i in x], min_cohens, width, label='Min Cohen\'s d', color='coral')
    ax.axhline(y=0.8, color='green', linestyle='--', label='Large effect (0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='Medium effect (0.5)')

    # Highlight β=20
    idx_20 = beta_values.index(20)
    ax.axvspan(idx_20 - 0.5, idx_20 + 0.5, alpha=0.2, color='yellow')

    ax.set_xticks(x)
    ax.set_xticklabels([f'β={b}' for b in beta_values])
    ax.set_ylabel("Cohen's d")
    ax.set_title("Real Data: Temperature Level Discrimination")
    ax.legend()

    # 2b: Score range by beta
    ax = axes[0, 1]
    score_ranges = [beta_discrimination[b]['score_range'] for b in beta_values]
    correlations = [abs(beta_discrimination[b]['spearman_correlation']) for b in beta_values]

    ax.bar([i - width/2 for i in x], score_ranges, width, label='Score Range', color='steelblue')
    ax.bar([i + width/2 for i in x], correlations, width, label='|Spearman ρ|', color='lightgreen')
    ax.axvspan(idx_20 - 0.5, idx_20 + 0.5, alpha=0.2, color='yellow')

    ax.set_xticks(x)
    ax.set_xticklabels([f'β={b}' for b in beta_values])
    ax.set_ylabel("Value")
    ax.set_title("Score Range & Correlation with Temperature")
    ax.legend()

    # 2c: ROC curves
    ax = axes[1, 0]
    for beta in [5, 10, 20, 30]:
        fpr = binary_classification[beta]['fpr']
        tpr = binary_classification[beta]['tpr']
        auc_val = binary_classification[beta]['roc_auc']
        linewidth = 3 if beta == 20 else 1.5
        ax.plot(fpr, tpr, label=f'β={beta} (AUC={auc_val:.3f})', linewidth=linewidth)

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves on Real Data (Low vs High Temp)')
    ax.legend()

    # 2d: d-prime on real data
    ax = axes[1, 1]
    dprimes = [binary_classification[b]['dprime'] for b in beta_values]
    aucs = [binary_classification[b]['roc_auc'] for b in beta_values]

    ax.bar([i - width/2 for i in x], dprimes, width, label="d'", color='steelblue')
    ax.bar([i + width/2 for i in x], [a * 5 for a in aucs], width, label='AUC × 5', color='coral')
    ax.axvspan(idx_20 - 0.5, idx_20 + 0.5, alpha=0.2, color='yellow')
    ax.axhline(y=3, color='green', linestyle='--', label="d' = 3 (excellent)")

    ax.set_xticks(x)
    ax.set_xticklabels([f'β={b}' for b in beta_values])
    ax.set_ylabel("Value")
    ax.set_title("Discrimination Metrics on Real Data")
    ax.legend()

    plt.tight_layout()
    fig2.savefig(output_dir / 'real_data_beta_validation.png', dpi=150)
    plt.close(fig2)

    # Figure 3: Model ranking stability
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 3a: Model scores across beta values
    ax = axes[0]
    models = list(model_ranking['scores'].keys())
    for model in models:
        scores = [model_ranking['scores'][model][b] for b in beta_values]
        ax.plot(beta_values, scores, 'o-', label=model, markersize=4)

    ax.axvline(x=20, color='red', linestyle='--', linewidth=2, label='Current β=20')
    ax.set_xlabel('β')
    ax.set_ylabel('Mean PDC Score')
    ax.set_title('Model Scores Across β Values')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3b: Ranking stability
    ax = axes[1]
    stability = model_ranking['ranking_stability_vs_beta20']
    ax.bar(range(len(beta_values)), [stability[b] for b in beta_values], color='steelblue')
    ax.axhline(y=1.0, color='green', linestyle='--', label='Perfect agreement')
    ax.axhline(y=0.8, color='orange', linestyle='--', label='Strong agreement')
    ax.set_xticks(range(len(beta_values)))
    ax.set_xticklabels([f'β={b}' for b in beta_values])
    ax.set_ylabel("Kendall's τ (vs β=20)")
    ax.set_title("Model Ranking Stability")
    ax.legend()

    plt.tight_layout()
    fig3.savefig(output_dir / 'real_data_model_ranking.png', dpi=150)
    plt.close(fig3)

    print("Visualizations saved.")


def run_validation():
    """Run complete validation on real LLM data."""

    print("=" * 70)
    print("VALIDATION ON REAL LLM DATA")
    print("=" * 70)

    # Load data
    print("\n1. Loading consistency metrics...")
    data = load_consistency_metrics()
    print(f"   Loaded {len(data)} models")

    # Extract dispersion values
    print("\n2. Extracting dispersion values...")
    global dispersion_data  # For use in visualization
    dispersion_data = extract_dispersion_by_model_temp(data)

    total_samples = sum(
        len(v) for model in dispersion_data.values() for v in model.values()
    )
    print(f"   Total samples: {total_samples}")

    # Analyze distribution
    print("\n3. Analyzing real dispersion distribution...")
    dispersion_analysis = analyze_real_dispersion_distribution(dispersion_data)

    print(f"""
   Overall Statistics:
   - Mean dispersion: {dispersion_analysis['overall']['mean']:.4f}
   - Median dispersion: {dispersion_analysis['overall']['median']:.4f}
   - Std deviation: {dispersion_analysis['overall']['std']:.4f}
   - Range: [{dispersion_analysis['overall']['min']:.4f}, {dispersion_analysis['overall']['max']:.4f}]
   - 90th percentile: {dispersion_analysis['overall']['percentiles']['90']:.4f}
   - 95th percentile: {dispersion_analysis['overall']['percentiles']['95']:.4f}
    """)

    # Validate beta discrimination
    print("4. Validating β discrimination on real data...")
    beta_values = [3, 5, 10, 15, 20, 30]
    beta_discrimination = validate_beta_discrimination(dispersion_data, beta_values)

    print("\n   β    | Mean d' | Min d' | Score Range | Spearman ρ")
    print("   " + "-" * 55)
    for beta in beta_values:
        r = beta_discrimination[beta]
        marker = " <-- CURRENT" if beta == 20 else ""
        print(f"   {beta:3d}  | {r['mean_cohens_d']:7.3f} | {r['min_cohens_d']:6.3f} | {r['score_range']:11.3f} | {r['spearman_correlation']:10.3f}{marker}")

    # Binary classification validation
    print("\n5. Validating binary classification (T≤0.3 vs T≥0.7)...")
    binary_classification = validate_binary_classification(dispersion_data, beta_values)

    print("\n   β    | ROC AUC | d-prime | Sensitivity | Specificity")
    print("   " + "-" * 55)
    for beta in beta_values:
        r = binary_classification[beta]
        marker = " <-- CURRENT" if beta == 20 else ""
        print(f"   {beta:3d}  | {r['roc_auc']:7.3f} | {r['dprime']:7.3f} | {r['sensitivity']:11.3f} | {r['specificity']:11.3f}{marker}")

    # Model ranking stability
    print("\n6. Validating model ranking stability...")
    model_ranking = validate_model_ranking_stability(dispersion_data, beta_values)

    print("\n   Ranking Stability (Kendall's τ vs β=20):")
    for beta in beta_values:
        tau = model_ranking['ranking_stability_vs_beta20'][beta]
        print(f"   β={beta:2d}: τ = {tau:.3f}")

    # Generate visualizations
    print("\n7. Generating visualizations...")
    create_validation_visualizations(
        dispersion_analysis, beta_discrimination,
        binary_classification, model_ranking, OUTPUT_DIR
    )

    # Summary and conclusions
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 70)

    conclusions = []

    # Check if β=20 provides good discrimination
    if beta_discrimination[20]['min_cohens_d'] > 0.5:
        conclusions.append(f"β=20 provides MEDIUM+ effect discrimination (d' = {beta_discrimination[20]['min_cohens_d']:.2f})")
    else:
        conclusions.append(f"β=20 provides SMALL effect discrimination (d' = {beta_discrimination[20]['min_cohens_d']:.2f})")

    # Check ROC AUC
    if binary_classification[20]['roc_auc'] > 0.9:
        conclusions.append(f"β=20 achieves EXCELLENT classification (AUC = {binary_classification[20]['roc_auc']:.3f})")
    elif binary_classification[20]['roc_auc'] > 0.8:
        conclusions.append(f"β=20 achieves GOOD classification (AUC = {binary_classification[20]['roc_auc']:.3f})")

    # Check ranking stability
    min_stability = min(model_ranking['ranking_stability_vs_beta20'].values())
    if min_stability > 0.8:
        conclusions.append("Model rankings are STABLE across all β values")
    else:
        conclusions.append(f"Model rankings vary across β (min τ = {min_stability:.2f})")

    # Find best β for real data
    best_beta_disc = max(beta_values, key=lambda b: beta_discrimination[b]['min_cohens_d'])
    best_beta_auc = max(beta_values, key=lambda b: binary_classification[b]['roc_auc'])

    conclusions.append(f"Best β for discrimination: {best_beta_disc}")
    conclusions.append(f"Best β for classification: {best_beta_auc}")

    # Compare with theoretical predictions
    print("\n   VALIDATION CONCLUSIONS:")
    for i, c in enumerate(conclusions, 1):
        print(f"   {i}. {c}")

    # Comparison with theoretical predictions
    print("\n   COMPARISON WITH THEORETICAL PREDICTIONS:")
    print(f"""
   | Metric              | Theoretical | Real Data | Match? |
   |---------------------|-------------|-----------|--------|
   | Best β (discrim.)   | 3-10        | {best_beta_disc}         | {'YES' if best_beta_disc <= 10 else 'PARTIAL'} |
   | β=20 d-prime        | 0.76        | {beta_discrimination[20]['min_cohens_d']:.2f}       | {'YES' if abs(beta_discrimination[20]['min_cohens_d'] - 0.76) < 0.3 else 'NO'} |
   | β=20 AUC            | 1.00        | {binary_classification[20]['roc_auc']:.2f}       | {'YES' if binary_classification[20]['roc_auc'] > 0.9 else 'NO'} |
   | Dispersion range    | [0, 0.4]    | [{dispersion_analysis['overall']['min']:.2f}, {dispersion_analysis['overall']['max']:.2f}] | {'YES' if dispersion_analysis['overall']['max'] < 0.5 else 'NO'} |
    """)

    # Save results
    results = {
        'dispersion_analysis': dispersion_analysis,
        'beta_discrimination': {str(k): v for k, v in beta_discrimination.items()},
        'binary_classification': {str(k): {kk: vv for kk, vv in v.items() if kk not in ['fpr', 'tpr']}
                                  for k, v in binary_classification.items()},
        'model_ranking': {
            'rankings': {str(k): v for k, v in model_ranking['rankings'].items()},
            'ranking_stability': model_ranking['ranking_stability_vs_beta20']
        },
        'conclusions': conclusions
    }

    results_path = OUTPUT_DIR / 'real_data_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = run_validation()
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
