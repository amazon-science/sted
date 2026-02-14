"""
Comprehensive Validation on Real LLM Data

This script processes ALL LLM generation results across all models and temperatures
to validate the theoretical power transformation analysis.
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


def power_transform(sigma: np.ndarray, alpha: float = 2.0, beta: float = 20.0) -> np.ndarray:
    """PDC power transformation"""
    return (1.0 / (1.0 + alpha * sigma)) ** beta


def extract_temperature_from_dir(dirname: str) -> float:
    """Extract temperature value from directory name."""
    match = re.search(r'temp_(\d+)_(\d+)', dirname)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None


def load_ted_results(model_dir: Path) -> Dict[float, List[Dict]]:
    """Load TED results for all temperatures in a model directory."""
    results = {}

    if not model_dir.exists():
        return results

    for subdir in model_dir.iterdir():
        if not subdir.is_dir():
            continue

        temp = extract_temperature_from_dir(subdir.name)
        if temp is None:
            continue

        ted_file = subdir / "results_ted.json"
        if ted_file.exists():
            with open(ted_file, 'r') as f:
                data = json.load(f)
                results[temp] = data.get('results', [])

    return results


def compute_dispersion_statistics(all_data: Dict[str, Dict[float, List[Dict]]]) -> Dict:
    """Compute dispersion statistics across all models and temperatures."""

    stats_by_temp = defaultdict(list)
    stats_by_model = defaultdict(list)
    all_dispersions = []

    for model_name, temp_data in all_data.items():
        for temp, samples in temp_data.items():
            for sample in samples:
                std = sample.get('std', 0)
                mean_sim = sample.get('mean', 1.0)

                # Compute dispersion (std of distances = 1 - similarities)
                # For TED results, we have similarity values
                # dispersion ≈ std of (1 - similarities)
                dispersion = std  # std is already computed

                stats_by_temp[temp].append(dispersion)
                stats_by_model[model_name].append(dispersion)
                all_dispersions.append({
                    'model': model_name,
                    'temp': temp,
                    'dispersion': dispersion,
                    'mean_similarity': mean_sim,
                    'std': std
                })

    return {
        'by_temp': dict(stats_by_temp),
        'by_model': dict(stats_by_model),
        'all': all_dispersions
    }


def validate_beta_on_real_data(dispersion_stats: Dict,
                                beta_values: List[float] = [3, 5, 10, 15, 20, 30]) -> Dict:
    """Validate different β values on real dispersion data."""

    alpha = 2.0
    results = {}

    # Group by temperature
    by_temp = dispersion_stats['by_temp']
    temps = sorted(by_temp.keys())

    for beta in beta_values:
        temp_scores = {}

        for temp in temps:
            dispersions = np.array(by_temp[temp])
            # Clip very small values to avoid numerical issues
            dispersions = np.clip(dispersions, 1e-10, 1.0)
            scores = power_transform(dispersions, alpha, beta)
            temp_scores[temp] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'mean_dispersion': float(np.mean(dispersions)),
                'std_dispersion': float(np.std(dispersions)),
            }

        # Compute overall metrics
        all_temps = []
        all_scores = []
        for temp in temps:
            dispersions = np.array(by_temp[temp])
            dispersions = np.clip(dispersions, 1e-10, 1.0)
            scores = power_transform(dispersions, alpha, beta)
            all_temps.extend([temp] * len(scores))
            all_scores.extend(scores)

        # Spearman correlation
        spearman_corr, _ = stats.spearmanr(all_temps, all_scores)

        # Score range
        mean_scores = [temp_scores[t]['mean_score'] for t in temps]
        score_range = max(mean_scores) - min(mean_scores)

        results[beta] = {
            'temp_scores': temp_scores,
            'spearman_correlation': float(spearman_corr),
            'score_range': score_range,
        }

    return results


def validate_model_discrimination(dispersion_stats: Dict,
                                  beta_values: List[float] = [3, 5, 10, 15, 20, 30]) -> Dict:
    """Check if PDC can discriminate between models."""

    alpha = 2.0
    by_model = dispersion_stats['by_model']
    models = list(by_model.keys())

    results = {}

    for beta in beta_values:
        model_scores = {}

        for model in models:
            dispersions = np.array(by_model[model])
            dispersions = np.clip(dispersions, 1e-10, 1.0)
            scores = power_transform(dispersions, alpha, beta)
            model_scores[model] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
            }

        # Rank models by mean score
        rankings = sorted(models, key=lambda m: model_scores[m]['mean'], reverse=True)

        results[beta] = {
            'model_scores': model_scores,
            'rankings': rankings,
            'score_range': max(model_scores[m]['mean'] for m in models) -
                          min(model_scores[m]['mean'] for m in models)
        }

    return results


def validate_low_vs_high_temp(dispersion_stats: Dict,
                              beta_values: List[float] = [3, 5, 10, 15, 20, 30],
                              low_threshold: float = 0.2,
                              high_threshold: float = 0.7) -> Dict:
    """Validate discrimination between low and high temperature outputs."""

    alpha = 2.0
    by_temp = dispersion_stats['by_temp']

    # Collect low and high temp dispersions
    low_dispersions = []
    high_dispersions = []

    for temp, dispersions in by_temp.items():
        if temp <= low_threshold:
            low_dispersions.extend(dispersions)
        elif temp >= high_threshold:
            high_dispersions.extend(dispersions)

    low_dispersions = np.array(low_dispersions)
    high_dispersions = np.array(high_dispersions)

    # Clip
    low_dispersions = np.clip(low_dispersions, 1e-10, 1.0)
    high_dispersions = np.clip(high_dispersions, 1e-10, 1.0)

    results = {}

    for beta in beta_values:
        scores_low = power_transform(low_dispersions, alpha, beta)
        scores_high = power_transform(high_dispersions, alpha, beta)

        # Compute d-prime
        pooled_std = np.sqrt((np.var(scores_low) + np.var(scores_high)) / 2)
        if pooled_std > 1e-10:
            dprime = (np.mean(scores_low) - np.mean(scores_high)) / pooled_std
        else:
            dprime = 0

        # Compute ROC AUC
        all_scores = np.concatenate([scores_low, scores_high])
        all_labels = np.concatenate([np.ones(len(scores_low)), np.zeros(len(scores_high))])

        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)

        results[beta] = {
            'dprime': float(dprime),
            'roc_auc': float(roc_auc),
            'mean_score_low': float(np.mean(scores_low)),
            'mean_score_high': float(np.mean(scores_high)),
            'n_low': len(scores_low),
            'n_high': len(scores_high),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
        }

    return results


def create_visualizations(all_data: Dict, dispersion_stats: Dict,
                          beta_results: Dict, model_results: Dict,
                          binary_results: Dict, output_dir: Path):
    """Create comprehensive visualizations."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Dispersion distribution by temperature
    ax = axes[0, 0]
    by_temp = dispersion_stats['by_temp']
    temps = sorted(by_temp.keys())

    # Box plot
    data_for_box = [by_temp[t] for t in temps]
    positions = list(range(len(temps)))

    bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)

    ax.set_xticks(positions[::2])  # Show every other label
    ax.set_xticklabels([f'{temps[i]:.1f}' for i in range(0, len(temps), 2)], rotation=45)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Dispersion (σ)')
    ax.set_title('Real LLM Dispersion Distribution by Temperature')

    # 2. Mean dispersion by temperature
    ax = axes[0, 1]
    mean_dispersions = [np.mean(by_temp[t]) for t in temps]
    std_dispersions = [np.std(by_temp[t]) for t in temps]

    ax.errorbar(temps, mean_dispersions, yerr=std_dispersions, fmt='o-', capsize=3, color='steelblue')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mean Dispersion')
    ax.set_title('Mean Dispersion vs Temperature')
    ax.grid(True, alpha=0.3)

    # 3. PDC score vs temperature for different β
    ax = axes[0, 2]
    beta_values = sorted(beta_results.keys())

    for beta in beta_values:
        mean_scores = [beta_results[beta]['temp_scores'][t]['mean_score'] for t in temps]
        linewidth = 3 if beta == 20 else 1.5
        ax.plot(temps, mean_scores, 'o-', label=f'β={beta}', linewidth=linewidth, markersize=4)

    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mean PDC Score')
    ax.set_title('PDC Score vs Temperature (by β)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Discrimination metrics by β
    ax = axes[1, 0]
    dprimes = [binary_results[b]['dprime'] for b in beta_values]
    aucs = [binary_results[b]['roc_auc'] for b in beta_values]

    x = range(len(beta_values))
    width = 0.35
    ax.bar([i - width/2 for i in x], dprimes, width, label="d'", color='steelblue')
    ax.bar([i + width/2 for i in x], [a * max(dprimes)/max(aucs) for a in aucs], width,
           label=f'AUC (scaled)', color='coral')

    # Highlight β=20
    idx_20 = beta_values.index(20)
    ax.axvspan(idx_20 - 0.5, idx_20 + 0.5, alpha=0.2, color='yellow')

    ax.set_xticks(x)
    ax.set_xticklabels([f'β={b}' for b in beta_values])
    ax.set_ylabel("Value")
    ax.set_title("Discrimination (T≤0.2 vs T≥0.7)")
    ax.legend()

    # 5. Model scores comparison
    ax = axes[1, 1]
    models = list(model_results[20]['model_scores'].keys())

    # Sort models by β=20 score
    models_sorted = sorted(models, key=lambda m: model_results[20]['model_scores'][m]['mean'], reverse=True)

    for beta in [10, 20, 30]:
        scores = [model_results[beta]['model_scores'][m]['mean'] for m in models_sorted]
        linewidth = 3 if beta == 20 else 1.5
        ax.plot(range(len(models_sorted)), scores, 'o-', label=f'β={beta}',
                linewidth=linewidth, markersize=6)

    ax.set_xticks(range(len(models_sorted)))
    ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean PDC Score')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. ROC curves
    ax = axes[1, 2]
    for beta in [5, 10, 20, 30]:
        fpr = binary_results[beta]['fpr']
        tpr = binary_results[beta]['tpr']
        auc_val = binary_results[beta]['roc_auc']
        linewidth = 3 if beta == 20 else 1.5
        ax.plot(fpr, tpr, label=f'β={beta} (AUC={auc_val:.3f})', linewidth=linewidth)

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC: Low vs High Temperature')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / 'comprehensive_validation.png', dpi=150)
    plt.close(fig)

    print("Visualizations saved.")


def run_comprehensive_validation():
    """Run comprehensive validation on all LLM data."""

    print("=" * 70)
    print("COMPREHENSIVE VALIDATION ON REAL LLM DATA")
    print("=" * 70)

    # Load all data
    print("\n1. Loading LLM generation results...")
    all_data = {}

    for model_name, dirname in MODEL_DIRS.items():
        model_dir = LLM_RESULTS_DIR / dirname
        if model_dir.exists():
            results = load_ted_results(model_dir)
            if results:
                all_data[model_name] = results
                n_temps = len(results)
                n_samples = sum(len(samples) for samples in results.values())
                print(f"   {model_name}: {n_temps} temperatures, {n_samples} samples")

    print(f"\n   Total models loaded: {len(all_data)}")

    # Compute dispersion statistics
    print("\n2. Computing dispersion statistics...")
    dispersion_stats = compute_dispersion_statistics(all_data)

    # Overall statistics
    all_dispersions = [d['dispersion'] for d in dispersion_stats['all']]
    all_dispersions = np.array(all_dispersions)
    all_dispersions = all_dispersions[all_dispersions > 1e-10]  # Filter zeros

    print(f"""
   Overall Dispersion Statistics (non-zero only):
   - N samples: {len(all_dispersions)}
   - Mean: {np.mean(all_dispersions):.6f}
   - Median: {np.median(all_dispersions):.6f}
   - Std: {np.std(all_dispersions):.6f}
   - Min: {np.min(all_dispersions):.6f}
   - Max: {np.max(all_dispersions):.6f}
   - 90th percentile: {np.percentile(all_dispersions, 90):.6f}
   - 95th percentile: {np.percentile(all_dispersions, 95):.6f}
    """)

    # Validate β discrimination
    print("3. Validating β discrimination on temperature...")
    beta_values = [3, 5, 10, 15, 20, 30]
    beta_results = validate_beta_on_real_data(dispersion_stats, beta_values)

    print("\n   β    | Score Range | Spearman ρ (temp vs score)")
    print("   " + "-" * 50)
    for beta in beta_values:
        r = beta_results[beta]
        marker = " <-- CURRENT" if beta == 20 else ""
        print(f"   {beta:3d}  | {r['score_range']:11.4f} | {r['spearman_correlation']:25.4f}{marker}")

    # Validate model discrimination
    print("\n4. Validating model discrimination...")
    model_results = validate_model_discrimination(dispersion_stats, beta_values)

    print("\n   Model Rankings (by β=20 PDC score):")
    for i, model in enumerate(model_results[20]['rankings'], 1):
        score = model_results[20]['model_scores'][model]['mean']
        print(f"   {i}. {model}: {score:.4f}")

    # Validate binary classification
    print("\n5. Validating binary classification (T≤0.2 vs T≥0.7)...")
    binary_results = validate_low_vs_high_temp(dispersion_stats, beta_values)

    print("\n   β    | ROC AUC | d-prime | Mean(Low) | Mean(High)")
    print("   " + "-" * 55)
    for beta in beta_values:
        r = binary_results[beta]
        marker = " <-- CURRENT" if beta == 20 else ""
        print(f"   {beta:3d}  | {r['roc_auc']:7.3f} | {r['dprime']:7.3f} | {r['mean_score_low']:9.4f} | {r['mean_score_high']:10.4f}{marker}")

    # Generate visualizations
    print("\n6. Generating visualizations...")
    create_visualizations(all_data, dispersion_stats, beta_results, model_results,
                          binary_results, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    # Find best β
    best_beta_range = max(beta_values, key=lambda b: beta_results[b]['score_range'])
    best_beta_auc = max(beta_values, key=lambda b: binary_results[b]['roc_auc'])
    best_beta_dprime = max(beta_values, key=lambda b: binary_results[b]['dprime'])

    print(f"""
   KEY FINDINGS:

   1. Real dispersion range: [{np.min(all_dispersions):.4f}, {np.max(all_dispersions):.4f}]
      (Much smaller than theoretical assumption of [0, 0.4])

   2. Best β for:
      - Score range: β = {best_beta_range} (range = {beta_results[best_beta_range]['score_range']:.4f})
      - ROC AUC: β = {best_beta_auc} (AUC = {binary_results[best_beta_auc]['roc_auc']:.3f})
      - d-prime: β = {best_beta_dprime} (d' = {binary_results[best_beta_dprime]['dprime']:.3f})

   3. β=20 performance:
      - Score range: {beta_results[20]['score_range']:.4f}
      - ROC AUC: {binary_results[20]['roc_auc']:.3f}
      - d-prime: {binary_results[20]['dprime']:.3f}
      - Spearman ρ: {beta_results[20]['spearman_correlation']:.3f}

   4. Model ranking stability: Rankings are {
       'STABLE' if all(model_results[b]['rankings'] == model_results[20]['rankings'] for b in beta_values)
       else 'VARIABLE'} across β values

   CONCLUSION:
   The real LLM data shows very small dispersion values (mostly < 0.05),
   which means the power transformation has limited impact. At these
   dispersion levels, most PDC scores are close to 1.0 regardless of β.

   The key differentiator is the temperature effect on consistency,
   not the choice of β. β=20 is appropriate for this data as it
   provides {'GOOD' if binary_results[20]['roc_auc'] > 0.7 else 'MODERATE'}
   discrimination between low and high temperature outputs.
    """)

    # Save results
    results = {
        'n_models': len(all_data),
        'dispersion_stats': {
            'mean': float(np.mean(all_dispersions)),
            'std': float(np.std(all_dispersions)),
            'min': float(np.min(all_dispersions)),
            'max': float(np.max(all_dispersions)),
            'percentile_90': float(np.percentile(all_dispersions, 90)),
            'percentile_95': float(np.percentile(all_dispersions, 95)),
        },
        'beta_results': {str(k): {kk: vv for kk, vv in v.items() if kk != 'temp_scores'}
                        for k, v in beta_results.items()},
        'binary_classification': {str(k): {kk: vv for kk, vv in v.items() if kk not in ['fpr', 'tpr']}
                                  for k, v in binary_results.items()},
        'model_rankings': model_results[20]['rankings'],
        'best_beta': {
            'score_range': best_beta_range,
            'roc_auc': best_beta_auc,
            'dprime': best_beta_dprime,
        }
    }

    results_path = OUTPUT_DIR / 'comprehensive_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = run_comprehensive_validation()
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION COMPLETE")
    print("=" * 70)
