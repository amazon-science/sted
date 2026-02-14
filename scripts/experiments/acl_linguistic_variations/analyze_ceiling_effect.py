#!/usr/bin/env python3
"""
ACL 2026 Paper: Ceiling Effect Analysis

Analyzes ceiling effects from Phase 2 intervention data.
Tests KDD hypothesis: interventions help difficult prompts but harm easy prompts.

Statistical tests:
1. Stratified effect sizes (difficult vs medium vs easy)
2. Correlation between baseline and intervention effect
3. Bootstrap confidence intervals
4. Cross-model ceiling effect consistency

Usage:
    python analyze_ceiling_effect.py --results-dir results/acl_linguistic/phase2_interventions
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import statistics

import numpy as np
from scipy import stats


def load_all_results(results_dir: str) -> List[Dict]:
    """Load all Phase 2 intervention results."""
    results_path = Path(results_dir)
    all_results = []

    for json_file in results_path.glob("*_interventions.json"):
        with open(json_file) as f:
            data = json.load(f)
            all_results.append(data)
        print(f"Loaded: {json_file.name}")

    return all_results


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    if len(data) < 2:
        return (data[0] if data else 0.0, 0.0, 0.0)

    data = np.array(data)
    bootstrap_means = [np.mean(np.random.choice(data, size=len(data), replace=True))
                       for _ in range(n_bootstrap)]

    mean = np.mean(data)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    return (mean, lower, upper)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_ceiling_effect(all_results: List[Dict]) -> Dict:
    """
    Main ceiling effect analysis.

    Tests:
    1. Stratified effects by baseline difficulty
    2. Baseline-effect correlation
    3. Cross-model consistency
    """
    print("=" * 70)
    print("CEILING EFFECT ANALYSIS")
    print("=" * 70)

    # Aggregate data across all models
    all_deltas = defaultdict(lambda: defaultdict(list))  # variation -> difficulty -> deltas
    baseline_effect_pairs = defaultdict(list)  # variation -> [(baseline, delta)]

    for model_data in all_results:
        model_name = model_data['metadata']['display_name']
        print(f"\nProcessing: {model_name}")

        for result in model_data.get('results', []):
            if 'metrics' not in result or 'delta_consistency' not in result:
                continue

            variation = result['variation']
            difficulty = result['difficulty_stratum']
            delta = result['delta_consistency']
            baseline = result['baseline_consistency']

            all_deltas[variation][difficulty].append(delta)
            baseline_effect_pairs[variation].append((baseline, delta))

    # ==========================================================================
    # 1. Stratified Effect Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. STRATIFIED EFFECTS BY BASELINE DIFFICULTY")
    print("=" * 70)

    ceiling_summary = {}
    print("\n{:<15} {:>12} {:>12} {:>12} {:>10}".format(
        "Variation", "Difficult", "Medium", "Easy", "Ceiling?"
    ))
    print("-" * 65)

    for variation in sorted(all_deltas.keys()):
        diff_deltas = all_deltas[variation]['difficult']
        med_deltas = all_deltas[variation]['medium']
        easy_deltas = all_deltas[variation]['easy']

        diff_mean = np.mean(diff_deltas) if diff_deltas else 0
        med_mean = np.mean(med_deltas) if med_deltas else 0
        easy_mean = np.mean(easy_deltas) if easy_deltas else 0

        # Ceiling effect: helps difficult, harms easy
        has_ceiling = diff_mean > 0 and easy_mean < 0

        print("{:<15} {:>+11.1%} {:>+11.1%} {:>+11.1%} {:>10}".format(
            variation[:15], diff_mean, med_mean, easy_mean,
            "YES" if has_ceiling else "no"
        ))

        ceiling_summary[variation] = {
            'difficult_delta': diff_mean,
            'medium_delta': med_mean,
            'easy_delta': easy_mean,
            'has_ceiling_effect': has_ceiling,
            'n_difficult': len(diff_deltas),
            'n_medium': len(med_deltas),
            'n_easy': len(easy_deltas)
        }

    # ==========================================================================
    # 2. Baseline-Effect Correlation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. BASELINE-EFFECT CORRELATION (Regression to Mean Test)")
    print("=" * 70)

    correlation_results = {}
    print("\n{:<15} {:>10} {:>10} {:>15}".format(
        "Variation", "r", "p-value", "Interpretation"
    ))
    print("-" * 55)

    for variation in sorted(baseline_effect_pairs.keys()):
        pairs = baseline_effect_pairs[variation]
        if len(pairs) < 5:
            continue

        baselines = [p[0] for p in pairs]
        deltas = [p[1] for p in pairs]

        r, p = stats.pearsonr(baselines, deltas)

        # Negative correlation supports ceiling effect
        if r < -0.3 and p < 0.05:
            interp = "Strong ceiling"
        elif r < 0 and p < 0.05:
            interp = "Weak ceiling"
        else:
            interp = "No ceiling"

        print("{:<15} {:>+9.3f} {:>10.4f} {:>15}".format(
            variation[:15], r, p, interp
        ))

        correlation_results[variation] = {
            'r': r,
            'p_value': p,
            'interpretation': interp,
            'n': len(pairs)
        }

    # ==========================================================================
    # 3. Cross-Model Ceiling Consistency
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. CROSS-MODEL CEILING EFFECT CONSISTENCY")
    print("=" * 70)

    # Check if ceiling effect is consistent across models
    model_ceiling_effects = defaultdict(lambda: defaultdict(bool))

    for model_data in all_results:
        model_name = model_data['metadata']['display_name']
        ceiling_effects = model_data.get('ceiling_effects', {})

        for var_name, var_data in ceiling_effects.items():
            if isinstance(var_data, dict):
                has_ceiling = var_data.get('has_ceiling_effect', False)
                model_ceiling_effects[var_name][model_name] = has_ceiling

    print("\n{:<15} {:>10} {:>10} {:>15}".format(
        "Variation", "Models w/", "Total", "Consistency"
    ))
    print("-" * 55)

    consistency_summary = {}
    for variation in sorted(model_ceiling_effects.keys()):
        model_effects = model_ceiling_effects[variation]
        n_with_ceiling = sum(1 for v in model_effects.values() if v)
        n_total = len(model_effects)
        consistency = n_with_ceiling / n_total if n_total > 0 else 0

        print("{:<15} {:>10} {:>10} {:>14.0%}".format(
            variation[:15], n_with_ceiling, n_total, consistency
        ))

        consistency_summary[variation] = {
            'models_with_ceiling': n_with_ceiling,
            'total_models': n_total,
            'consistency': consistency
        }

    # ==========================================================================
    # 4. Statistical Tests
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. STATISTICAL TESTS (Difficult vs Easy)")
    print("=" * 70)

    stat_tests = {}
    print("\n{:<15} {:>10} {:>10} {:>10} {:>10}".format(
        "Variation", "t-stat", "p-value", "Cohen's d", "Sig?"
    ))
    print("-" * 60)

    for variation in sorted(all_deltas.keys()):
        difficult = all_deltas[variation]['difficult']
        easy = all_deltas[variation]['easy']

        if len(difficult) >= 3 and len(easy) >= 3:
            t_stat, p_value = stats.ttest_ind(difficult, easy)
            d = cohens_d(difficult, easy)
            sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))

            print("{:<15} {:>+9.2f} {:>10.4f} {:>+9.2f} {:>10}".format(
                variation[:15], t_stat, p_value, d, sig
            ))

            stat_tests[variation] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': d,
                'significant': p_value < 0.05
            }

    # ==========================================================================
    # 5. Bootstrap CIs for Key Variations
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. BOOTSTRAP 95% CIs FOR STRATIFIED EFFECTS")
    print("=" * 70)

    bootstrap_results = {}
    key_variations = ['polite_please', 'modal_might', 'hedge_conditional']

    for variation in key_variations:
        if variation not in all_deltas:
            continue

        print(f"\n{variation}:")

        for difficulty in ['difficult', 'medium', 'easy']:
            deltas = all_deltas[variation][difficulty]
            if len(deltas) >= 5:
                mean, lower, upper = bootstrap_ci(deltas)
                print(f"  {difficulty:10s}: {mean:+.3f} [{lower:+.3f}, {upper:+.3f}]")

                if variation not in bootstrap_results:
                    bootstrap_results[variation] = {}
                bootstrap_results[variation][difficulty] = {
                    'mean': mean,
                    'ci_lower': lower,
                    'ci_upper': upper,
                    'n': len(deltas)
                }

    # Compile summary
    summary = {
        'ceiling_summary': ceiling_summary,
        'correlation_results': correlation_results,
        'consistency_summary': consistency_summary,
        'statistical_tests': stat_tests,
        'bootstrap_results': bootstrap_results
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ceiling effects from Phase 2 intervention results'
    )
    parser.add_argument('--results-dir', type=str,
                        default='results/acl_linguistic/phase2_interventions',
                        help='Directory containing intervention results')
    parser.add_argument('--output', type=str,
                        default='results/acl_linguistic/analysis/ceiling_effect_analysis.json',
                        help='Output file for analysis results')

    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        print("Run phase2_linguistic_interventions.py first.")
        sys.exit(1)

    # Load all results
    all_results = load_all_results(args.results_dir)

    if not all_results:
        print("No results found to analyze.")
        sys.exit(1)

    # Run analysis
    summary = analyze_ceiling_effect(all_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSaved analysis results to {output_path}")


if __name__ == '__main__':
    main()
