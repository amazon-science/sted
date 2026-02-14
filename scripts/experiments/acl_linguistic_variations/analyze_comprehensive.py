#!/usr/bin/env python3
"""
ACL 2026 Paper: Comprehensive Results Analysis

Analyzes results from multi-model linguistic experiments and generates
tables and statistics for the ACL paper.

Usage:
    python analyze_comprehensive.py --results-dir DIR [--output FILE]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats


def load_all_results(results_dir: str) -> Dict[str, Dict]:
    """Load all result files from directory."""
    results = {}
    results_path = Path(results_dir)

    for f in results_path.glob("*_results.json"):
        with open(f) as fp:
            data = json.load(fp)
            model_name = data['metadata']['display_name']
            results[model_name] = data

    return results


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
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


def compute_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Spearman correlation with p-value."""
    if len(x) < 3 or len(y) < 3:
        return (0.0, 1.0)
    try:
        rho, p = stats.spearmanr(x, y)
        return (rho, p)
    except:
        return (0.0, 1.0)


def analyze_modal_effect(all_results: Dict[str, Dict]) -> Dict:
    """Analyze effect of modal verb strength on consistency."""
    modal_strength_order = {
        'must': 0.95,
        'need_to': 0.85,
        'should': 0.70,
        'would': 0.55,
        'could': 0.40,
        'might': 0.25
    }

    modal_results = defaultdict(lambda: defaultdict(list))

    for model_name, data in all_results.items():
        for r in data.get('results', []):
            if 'metrics' not in r:
                continue
            features = r.get('linguistic_features', {})
            if 'modal' in features and r.get('temperature') == 1.0:
                modal = features['modal']
                modal_results[model_name][modal].append(r['metrics']['c_mean'])

    analysis = {'by_model': {}, 'aggregate': {}}

    # By model
    for model_name, modals in modal_results.items():
        model_analysis = {}
        strengths = []
        consistencies = []

        for modal, values in modals.items():
            if modal in modal_strength_order and values:
                mean, ci_low, ci_high = bootstrap_ci(values)
                model_analysis[modal] = {
                    'mean': mean,
                    'ci_95': (ci_low, ci_high),
                    'n': len(values),
                    'strength': modal_strength_order[modal]
                }
                strengths.append(modal_strength_order[modal])
                consistencies.append(mean)

        if len(strengths) >= 3:
            rho, p = compute_correlation(strengths, consistencies)
            model_analysis['correlation'] = {'rho': rho, 'p_value': p}

        analysis['by_model'][model_name] = model_analysis

    # Aggregate across models
    aggregate_modals = defaultdict(list)
    for model_name, modals in modal_results.items():
        for modal, values in modals.items():
            aggregate_modals[modal].extend(values)

    all_strengths = []
    all_consistencies = []

    for modal, values in aggregate_modals.items():
        if modal in modal_strength_order and values:
            mean, ci_low, ci_high = bootstrap_ci(values)
            analysis['aggregate'][modal] = {
                'mean': mean,
                'ci_95': (ci_low, ci_high),
                'n': len(values),
                'strength': modal_strength_order[modal]
            }
            all_strengths.append(modal_strength_order[modal])
            all_consistencies.append(mean)

    if len(all_strengths) >= 3:
        rho, p = compute_correlation(all_strengths, all_consistencies)
        analysis['aggregate']['correlation'] = {'rho': rho, 'p_value': p}

    return analysis


def analyze_politeness_effect(all_results: Dict[str, Dict]) -> Dict:
    """Analyze effect of politeness strategies on consistency."""
    face_threat_order = {
        'bald': 1.0,
        'please': 0.7,
        'can_you': 0.5,
        'could_you': 0.4,
        'grateful': 0.3,
        'would_mind': 0.2
    }

    politeness_results = defaultdict(lambda: defaultdict(list))

    for model_name, data in all_results.items():
        for r in data.get('results', []):
            if 'metrics' not in r:
                continue
            features = r.get('linguistic_features', {})
            if 'politeness' in features and r.get('temperature') == 1.0:
                pol = features['politeness']
                politeness_results[model_name][pol].append(r['metrics']['c_mean'])

    analysis = {'by_model': {}, 'aggregate': {}}

    # By model
    for model_name, pols in politeness_results.items():
        model_analysis = {}
        face_threats = []
        consistencies = []

        for pol, values in pols.items():
            if pol in face_threat_order and values:
                mean, ci_low, ci_high = bootstrap_ci(values)
                model_analysis[pol] = {
                    'mean': mean,
                    'ci_95': (ci_low, ci_high),
                    'n': len(values),
                    'face_threat': face_threat_order[pol]
                }
                face_threats.append(face_threat_order[pol])
                consistencies.append(mean)

        if len(face_threats) >= 3:
            rho, p = compute_correlation(face_threats, consistencies)
            model_analysis['correlation'] = {'rho': rho, 'p_value': p}

        analysis['by_model'][model_name] = model_analysis

    # Aggregate
    aggregate_pols = defaultdict(list)
    for model_name, pols in politeness_results.items():
        for pol, values in pols.items():
            aggregate_pols[pol].extend(values)

    all_threats = []
    all_consistencies = []

    for pol, values in aggregate_pols.items():
        if pol in face_threat_order and values:
            mean, ci_low, ci_high = bootstrap_ci(values)
            analysis['aggregate'][pol] = {
                'mean': mean,
                'ci_95': (ci_low, ci_high),
                'n': len(values),
                'face_threat': face_threat_order[pol]
            }
            all_threats.append(face_threat_order[pol])
            all_consistencies.append(mean)

    if len(all_threats) >= 3:
        rho, p = compute_correlation(all_threats, all_consistencies)
        analysis['aggregate']['correlation'] = {'rho': rho, 'p_value': p}

    return analysis


def analyze_category_comparison(all_results: Dict[str, Dict]) -> Dict:
    """Compare consistency across linguistic categories."""
    category_results = defaultdict(lambda: defaultdict(list))

    for model_name, data in all_results.items():
        for r in data.get('results', []):
            if 'metrics' not in r:
                continue
            category = r.get('category', 'unknown')
            if r.get('temperature') == 1.0:
                category_results[model_name][category].append(r['metrics']['c_mean'])

    analysis = {'by_model': {}, 'aggregate': {}}

    # By model
    for model_name, cats in category_results.items():
        model_analysis = {}
        for cat, values in cats.items():
            if values:
                mean, ci_low, ci_high = bootstrap_ci(values)
                model_analysis[cat] = {
                    'mean': mean,
                    'ci_95': (ci_low, ci_high),
                    'n': len(values)
                }
        analysis['by_model'][model_name] = model_analysis

    # Aggregate
    aggregate_cats = defaultdict(list)
    for model_name, cats in category_results.items():
        for cat, values in cats.items():
            aggregate_cats[cat].extend(values)

    for cat, values in aggregate_cats.items():
        if values:
            mean, ci_low, ci_high = bootstrap_ci(values)
            analysis['aggregate'][cat] = {
                'mean': mean,
                'ci_95': (ci_low, ci_high),
                'n': len(values)
            }

    return analysis


def analyze_temperature_interaction(all_results: Dict[str, Dict]) -> Dict:
    """Analyze interaction between temperature and linguistic features."""
    temp_variation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model_name, data in all_results.items():
        for r in data.get('results', []):
            if 'metrics' not in r:
                continue
            var_name = r.get('variation_name', 'unknown')
            temp = r.get('temperature', 0)
            temp_variation_results[model_name][var_name][temp].append(r['metrics']['c_mean'])

    analysis = {'by_model': {}, 'aggregate': {}}

    # Aggregate by variation and temperature
    agg_var_temp = defaultdict(lambda: defaultdict(list))

    for model_name, variations in temp_variation_results.items():
        for var_name, temps in variations.items():
            for temp, values in temps.items():
                agg_var_temp[var_name][temp].extend(values)

    for var_name, temps in agg_var_temp.items():
        analysis['aggregate'][var_name] = {}
        for temp, values in temps.items():
            if values:
                mean, ci_low, ci_high = bootstrap_ci(values)
                analysis['aggregate'][var_name][str(temp)] = {
                    'mean': mean,
                    'ci_95': (ci_low, ci_high),
                    'n': len(values)
                }

    return analysis


def generate_latex_tables(analysis: Dict) -> str:
    """Generate LaTeX tables for the ACL paper."""
    latex = []

    # Table 1: Modal Strength Effect
    latex.append("% Table: Modal Strength Effect on Consistency")
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{llccc}")
    latex.append("\\toprule")
    latex.append("Strength & Modal & $c_{\\text{mean}}$ & 95\\% CI & $n$ \\\\")
    latex.append("\\midrule")

    modal_order = ['must', 'need_to', 'should', 'would', 'could', 'might']
    strength_labels = {'must': 'Strong', 'need_to': 'Strong', 'should': 'Medium',
                       'would': 'Medium', 'could': 'Weak', 'might': 'Weakest'}

    modal_analysis = analysis.get('modal', {}).get('aggregate', {})
    for modal in modal_order:
        if modal in modal_analysis:
            m = modal_analysis[modal]
            ci = m.get('ci_95', (0, 0))
            latex.append(f"{strength_labels.get(modal, '')} & {modal} & "
                        f"{m['mean']:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & {m.get('n', 0)} \\\\")

    if 'correlation' in modal_analysis:
        corr = modal_analysis['correlation']
        latex.append("\\midrule")
        latex.append(f"\\multicolumn{{5}}{{l}}{{Spearman $\\rho$ = {corr['rho']:.3f}, "
                    f"$p$ = {corr['p_value']:.4f}}} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Modal verb strength inversely correlates with consistency.}")
    latex.append("\\label{tab:modal}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 2: Politeness Effect
    latex.append("% Table: Politeness Strategy Effect")
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{llccc}")
    latex.append("\\toprule")
    latex.append("Strategy & Marker & $c_{\\text{mean}}$ & 95\\% CI & $n$ \\\\")
    latex.append("\\midrule")

    pol_order = ['would_mind', 'grateful', 'could_you', 'can_you', 'please', 'bald']
    strategy_labels = {'would_mind': 'Neg. Strong', 'grateful': 'Pos. Strong',
                       'could_you': 'Neg. Medium', 'can_you': 'Neg. Weak',
                       'please': 'Positive', 'bald': 'Bald'}

    pol_analysis = analysis.get('politeness', {}).get('aggregate', {})
    for pol in pol_order:
        if pol in pol_analysis:
            p = pol_analysis[pol]
            ci = p.get('ci_95', (0, 0))
            latex.append(f"{strategy_labels.get(pol, '')} & {pol.replace('_', '\\_')} & "
                        f"{p['mean']:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & {p.get('n', 0)} \\\\")

    if 'correlation' in pol_analysis:
        corr = pol_analysis['correlation']
        latex.append("\\midrule")
        latex.append(f"\\multicolumn{{5}}{{l}}{{Spearman $\\rho$ = {corr['rho']:.3f}, "
                    f"$p$ = {corr['p_value']:.4f}}} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Politeness markers reduce face threat and improve consistency.}")
    latex.append("\\label{tab:politeness}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="Analyze ACL linguistic experiment results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with result files")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")

    args = parser.parse_args()

    print("=" * 70)
    print("ACL Linguistic Experiment Analysis")
    print("=" * 70)

    # Load results
    all_results = load_all_results(args.results_dir)
    print(f"Loaded results for {len(all_results)} models: {list(all_results.keys())}")

    # Run analyses
    analysis = {
        'modal': analyze_modal_effect(all_results),
        'politeness': analyze_politeness_effect(all_results),
        'category': analyze_category_comparison(all_results),
        'temperature': analyze_temperature_interaction(all_results),
        'models': list(all_results.keys())
    }

    # Print summary
    print("\n" + "=" * 70)
    print("MODAL STRENGTH ANALYSIS (Aggregate)")
    print("=" * 70)

    modal_agg = analysis['modal'].get('aggregate', {})
    for modal in ['must', 'need_to', 'should', 'would', 'could', 'might']:
        if modal in modal_agg:
            m = modal_agg[modal]
            print(f"  {modal:10s}: mean={m['mean']:.3f}, 95% CI=[{m['ci_95'][0]:.3f}, {m['ci_95'][1]:.3f}], n={m.get('n', 0)}")

    if 'correlation' in modal_agg:
        c = modal_agg['correlation']
        print(f"\n  Correlation (strength vs consistency): rho={c['rho']:.3f}, p={c['p_value']:.4f}")

    print("\n" + "=" * 70)
    print("POLITENESS ANALYSIS (Aggregate)")
    print("=" * 70)

    pol_agg = analysis['politeness'].get('aggregate', {})
    for pol in ['would_mind', 'grateful', 'could_you', 'can_you', 'please', 'bald']:
        if pol in pol_agg:
            p = pol_agg[pol]
            print(f"  {pol:12s}: mean={p['mean']:.3f}, 95% CI=[{p['ci_95'][0]:.3f}, {p['ci_95'][1]:.3f}], n={p.get('n', 0)}")

    if 'correlation' in pol_agg:
        c = pol_agg['correlation']
        print(f"\n  Correlation (face threat vs consistency): rho={c['rho']:.3f}, p={c['p_value']:.4f}")

    print("\n" + "=" * 70)
    print("CATEGORY COMPARISON (Aggregate at T=1.0)")
    print("=" * 70)

    cat_agg = analysis['category'].get('aggregate', {})
    for cat in sorted(cat_agg.keys(), key=lambda x: -cat_agg[x]['mean']):
        c = cat_agg[cat]
        print(f"  {cat:12s}: mean={c['mean']:.3f}, 95% CI=[{c['ci_95'][0]:.3f}, {c['ci_95'][1]:.3f}], n={c.get('n', 0)}")

    # Save results
    if args.output:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, tuple):
                return [convert_numpy(i) for i in obj]
            return obj

        with open(args.output, 'w') as f:
            json.dump(convert_numpy(analysis), f, indent=2)
        print(f"\nSaved analysis to {args.output}")

    # Generate LaTeX
    if args.latex:
        latex = generate_latex_tables(analysis)
        latex_path = Path(args.results_dir) / "latex_tables.tex"
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"\nGenerated LaTeX tables: {latex_path}")


if __name__ == '__main__':
    main()
