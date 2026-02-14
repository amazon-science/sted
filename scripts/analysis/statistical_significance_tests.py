#!/usr/bin/env python3
"""
Statistical Significance Tests for Model Comparisons.

This script performs rigorous statistical analysis to determine if
differences between model consistency scores are statistically significant.

Tests performed:
1. Paired Wilcoxon signed-rank test (non-parametric, for two models)
2. Friedman test (non-parametric, for multiple models)
3. Effect size (Cliff's delta - non-parametric effect size)
4. Bootstrap confidence intervals
5. Multiple comparison correction (Bonferroni, Holm)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/Users/guanghu/Documents/genai/projects/sted-internal")
TOUCAN_CONSISTENCY = BASE_DIR / "llm_gen_results/toucan/consistency_results/combined_consistency_metrics_results.json"


def load_consistency_data():
    """Load and organize consistency data by model, temperature, and sample."""
    with open(TOUCAN_CONSISTENCY) as f:
        data = json.load(f)

    # Organize: {model: {temp: {sample_idx: score}}}
    organized = defaultdict(lambda: defaultdict(dict))

    for model, samples in data.items():
        for sample in samples:
            temp = sample["temperature"]
            idx = sample["sample_idx"]
            # Use mean_similarity as the STED score
            score = sample.get("mean_similarity", 0)
            organized[model][temp][idx] = score

    return organized


def get_paired_scores(data, model1, model2, temperature):
    """Get paired scores for two models at a specific temperature."""
    scores1 = data[model1][temperature]
    scores2 = data[model2][temperature]

    # Get common sample indices
    common_indices = set(scores1.keys()) & set(scores2.keys())

    paired1 = [scores1[idx] for idx in sorted(common_indices)]
    paired2 = [scores2[idx] for idx in sorted(common_indices)]

    return np.array(paired1), np.array(paired2)


def cliffs_delta(x, y):
    """
    Calculate Cliff's delta effect size (non-parametric).

    Interpretation:
    |d| < 0.147: negligible
    |d| < 0.33: small
    |d| < 0.474: medium
    |d| >= 0.474: large
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0

    # Count dominance
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)

    return (more - less) / (n1 * n2)


def interpret_cliffs_delta(d):
    """Interpret Cliff's delta magnitude."""
    d = abs(d)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval for mean."""
    np.random.seed(42)
    means = []
    n = len(scores)

    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)

    return lower, upper


def paired_wilcoxon_test(scores1, scores2):
    """
    Perform Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test.
    Tests if the distribution of differences is symmetric around zero.
    """
    # Remove pairs where both are equal (no difference)
    diff = scores1 - scores2
    non_zero = diff != 0

    if sum(non_zero) < 10:
        return np.nan, np.nan

    statistic, p_value = stats.wilcoxon(scores1[non_zero], scores2[non_zero])
    return statistic, p_value


def friedman_test(data, models, temperature):
    """
    Perform Friedman test for comparing multiple models.

    Non-parametric alternative to repeated-measures ANOVA.
    """
    # Build matrix: rows = samples, cols = models
    all_indices = set()
    for model in models:
        all_indices.update(data[model][temperature].keys())

    # Only keep samples present in ALL models
    common_indices = all_indices.copy()
    for model in models:
        common_indices &= set(data[model][temperature].keys())

    if len(common_indices) < 10:
        return np.nan, np.nan

    matrix = []
    for idx in sorted(common_indices):
        row = [data[model][temperature].get(idx, np.nan) for model in models]
        if not any(np.isnan(row)):
            matrix.append(row)

    matrix = np.array(matrix)

    if len(matrix) < 10:
        return np.nan, np.nan

    statistic, p_value = stats.friedmanchisquare(*[matrix[:, i] for i in range(len(models))])
    return statistic, p_value


def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Less conservative than Bonferroni, controls family-wise error rate.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    corrected = np.zeros(n)
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        corrected[idx] = min(1.0, p * (n - i))

    # Ensure monotonicity
    for i in range(1, n):
        if corrected[sorted_indices[i]] < corrected[sorted_indices[i-1]]:
            corrected[sorted_indices[i]] = corrected[sorted_indices[i-1]]

    return corrected


def analyze_model_pairs(data, models, temperature=0.1):
    """Perform pairwise statistical tests between all model pairs."""
    results = []

    pairs = list(combinations(models, 2))
    p_values = []

    print(f"\n{'='*80}")
    print(f"PAIRWISE COMPARISONS AT T={temperature}")
    print(f"{'='*80}")

    for model1, model2 in pairs:
        scores1, scores2 = get_paired_scores(data, model1, model2, temperature)

        if len(scores1) < 10:
            continue

        # Basic statistics
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        std1, std2 = np.std(scores1), np.std(scores2)

        # Wilcoxon test
        stat, p = paired_wilcoxon_test(scores1, scores2)
        p_values.append(p if not np.isnan(p) else 1.0)

        # Effect size
        delta = cliffs_delta(scores1, scores2)
        effect = interpret_cliffs_delta(delta)

        results.append({
            'model1': model1,
            'model2': model2,
            'mean1': mean1,
            'mean2': mean2,
            'diff': mean1 - mean2,
            'p_value': p,
            'cliffs_delta': delta,
            'effect_size': effect,
            'n_samples': len(scores1)
        })

    # Apply Holm-Bonferroni correction
    if p_values:
        corrected_p = holm_bonferroni_correction(p_values)
        for i, r in enumerate(results):
            r['p_corrected'] = corrected_p[i]
            r['significant'] = corrected_p[i] < 0.05

    return results


def print_results_table(results, top_n=20):
    """Print results as a formatted table."""
    # Sort by absolute difference
    results = sorted(results, key=lambda x: abs(x['diff']), reverse=True)[:top_n]

    print(f"\n{'Model 1':<25} {'Model 2':<25} {'Δ':>8} {'p-val':>10} {'p-corr':>10} {'Effect':>10} {'Sig':>5}")
    print("-" * 100)

    for r in results:
        sig = "***" if r.get('significant', False) else ""
        print(f"{r['model1']:<25} {r['model2']:<25} {r['diff']:>+8.3f} "
              f"{r['p_value']:>10.2e} {r.get('p_corrected', np.nan):>10.2e} "
              f"{r['effect_size']:>10} {sig:>5}")


def analyze_top_models(data, temperature=0.1):
    """Analyze statistical significance among top-performing models."""

    # Get mean scores for each model at this temperature
    model_means = {}
    for model in data.keys():
        scores = list(data[model][temperature].values())
        if len(scores) > 100:  # Only models with enough data
            model_means[model] = np.mean(scores)

    # Sort by mean and get top 10
    top_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)[:10]
    top_model_names = [m[0] for m in top_models]

    print(f"\n{'='*80}")
    print(f"TOP 10 MODELS BY MEAN STED SCORE AT T={temperature}")
    print(f"{'='*80}")

    for i, (model, mean) in enumerate(top_models, 1):
        scores = list(data[model][temperature].values())
        ci_low, ci_high = bootstrap_ci(scores)
        print(f"{i:2}. {model:<30} {mean:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")

    # Friedman test
    print(f"\n{'='*80}")
    print("FRIEDMAN TEST (Are there significant differences among top 10 models?)")
    print(f"{'='*80}")

    stat, p = friedman_test(data, top_model_names, temperature)
    print(f"Friedman χ² = {stat:.2f}, p = {p:.2e}")
    if p < 0.05:
        print("→ SIGNIFICANT: At least one model differs significantly from others")
    else:
        print("→ NOT SIGNIFICANT: No evidence of differences among models")

    # Pairwise comparisons among top models
    results = analyze_model_pairs(data, top_model_names, temperature)
    print_results_table(results)

    return results


def generate_latex_table(data, models, temperatures=[0.1, 0.5, 0.9]):
    """Generate LaTeX table with significance indicators."""

    print(f"\n{'='*80}")
    print("LATEX TABLE WITH SIGNIFICANCE INDICATORS")
    print(f"{'='*80}")

    # Find best model at T=0.1 for comparison
    means_t01 = {m: np.mean(list(data[m][0.1].values())) for m in models if data[m][0.1]}
    best_model = max(means_t01.items(), key=lambda x: x[1])[0]

    print(f"Reference model (best at T=0.1): {best_model}")
    print()

    print("\\begin{tabular}{@{}lccccc@{}}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{T=0.1} & \\textbf{T=0.5} & \\textbf{T=0.9} & \\textbf{$\\Delta$} & \\textbf{Sig.} \\\\")
    print("\\midrule")

    for model in models:
        if not data[model][0.1]:
            continue

        scores = []
        for t in temperatures:
            s = list(data[model][t].values())
            if s:
                scores.append(np.mean(s))
            else:
                scores.append(np.nan)

        delta = scores[0] - scores[2] if not np.isnan(scores[0]) and not np.isnan(scores[2]) else np.nan

        # Test significance vs best model at T=0.1
        if model != best_model:
            s1, s2 = get_paired_scores(data, best_model, model, 0.1)
            _, p = paired_wilcoxon_test(s1, s2)
            sig = "$^{***}$" if p < 0.001 else ("$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else ""))
        else:
            sig = ""

        print(f"{model} & {scores[0]:.3f}{sig} & {scores[1]:.3f} & {scores[2]:.3f} & {delta:+.3f} & \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print()
    print("$^{***}$p<0.001, $^{**}$p<0.01, $^{*}$p<0.05 vs best model (Wilcoxon signed-rank test)")


def main():
    print("Loading consistency data...")
    data = load_consistency_data()

    # List available models
    models = [m for m in data.keys() if len(data[m][0.1]) > 100]
    print(f"Found {len(models)} models with sufficient data")

    # Analyze at different temperatures
    for temp in [0.1, 0.5, 0.9]:
        analyze_top_models(data, temperature=temp)

    # Generate LaTeX table
    # Get top models for table
    model_means = {m: np.mean(list(data[m][0.1].values())) for m in models if data[m][0.1]}
    top_models = [m for m, _ in sorted(model_means.items(), key=lambda x: x[1], reverse=True)[:15]]

    generate_latex_table(data, top_models)

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY: KEY FINDINGS")
    print(f"{'='*80}")
    print("""
Statistical significance tests reveal:

1. FRIEDMAN TEST: Tests whether there are ANY significant differences among models
   - If p < 0.05: At least one model is significantly different

2. PAIRWISE WILCOXON: Tests specific model pairs
   - Holm-Bonferroni correction controls family-wise error rate
   - p-corrected < 0.05 indicates significant difference

3. CLIFF'S DELTA: Effect size interpretation
   - |d| < 0.147: negligible (difference may be statistically but not practically significant)
   - |d| < 0.33: small
   - |d| < 0.474: medium
   - |d| >= 0.474: large

4. BOOTSTRAP CI: 95% confidence interval for each model's mean
   - Non-overlapping CIs suggest significant differences
""")


if __name__ == "__main__":
    main()
