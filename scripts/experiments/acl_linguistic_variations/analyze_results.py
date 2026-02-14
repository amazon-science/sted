"""
ACL Paper: Results Analysis

Analyzes linguistic variation evaluation results and generates tables/figures for the paper.

Usage:
    python analyze_results.py --input results/acl_linguistic/eval_results.json \
                              --output results/acl_linguistic/analysis/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import statistics
from dataclasses import dataclass

# Try importing visualization libraries
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class EffectSize:
    """Effect size statistics."""
    cohens_d: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool


class LinguisticAnalyzer:
    """Analyzes linguistic variation effects on consistency."""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.df = self._to_dataframe() if HAS_PANDAS else None

    def _to_dataframe(self) -> 'pd.DataFrame':
        """Convert results to pandas DataFrame."""
        rows = []
        for r in self.results:
            row = {
                'variation_id': r['variation_id'],
                'base_id': r['base_id'],
                'variation_type': r['variation_type'],
                'variation_subtype': r['variation_subtype'],
                'model': r['model'],
                'temperature': r['temperature'],
                'c_mean': r['consistency_metrics']['c_mean'],
                's_alpha': r['consistency_metrics']['s_alpha'],
                'validity_rate': r['validity_rate'],
            }
            # Add linguistic features
            row.update(r['linguistic_features'])
            rows.append(row)
        return pd.DataFrame(rows)

    def compute_speech_act_effects(self) -> Dict[str, Dict]:
        """Compute effects of different speech act types."""
        effects = defaultdict(list)

        for r in self.results:
            if r['variation_type'] == 'speech_act':
                force = r['linguistic_features'].get('illocutionary_force', 'unknown')
                effects[force].append(r['consistency_metrics']['c_mean'])

        summary = {}
        for force, values in effects.items():
            summary[force] = {
                'mean': statistics.mean(values) if values else 0,
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'n': len(values)
            }

        return summary

    def compute_modal_effects(self) -> Dict[str, Dict]:
        """Compute effects of different modal verbs."""
        by_strength = defaultdict(list)
        by_type = defaultdict(list)

        for r in self.results:
            if r['variation_type'] == 'modal':
                strength = r['linguistic_features'].get('modal_strength', 'unknown')
                modal_type = r['linguistic_features'].get('modal_type', 'unknown')
                c_mean = r['consistency_metrics']['c_mean']

                by_strength[strength].append(c_mean)
                by_type[modal_type].append(c_mean)

        summary = {
            'by_strength': {},
            'by_type': {}
        }

        for key, values in by_strength.items():
            summary['by_strength'][key] = {
                'mean': statistics.mean(values) if values else 0,
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'n': len(values)
            }

        for key, values in by_type.items():
            summary['by_type'][key] = {
                'mean': statistics.mean(values) if values else 0,
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'n': len(values)
            }

        return summary

    def compute_politeness_effects(self) -> Dict[str, Dict]:
        """Compute effects of different politeness strategies."""
        by_strategy = defaultdict(list)
        by_face_threat = defaultdict(list)

        for r in self.results:
            if r['variation_type'] == 'politeness':
                strategy = r['linguistic_features'].get('politeness_strategy', 'unknown')
                face_threat = r['linguistic_features'].get('face_threat', 'unknown')
                c_mean = r['consistency_metrics']['c_mean']

                by_strategy[strategy].append(c_mean)
                by_face_threat[face_threat].append(c_mean)

        summary = {
            'by_strategy': {},
            'by_face_threat': {}
        }

        for key, values in by_strategy.items():
            summary['by_strategy'][key] = {
                'mean': statistics.mean(values) if values else 0,
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'n': len(values)
            }

        for key, values in by_face_threat.items():
            summary['by_face_threat'][key] = {
                'mean': statistics.mean(values) if values else 0,
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'n': len(values)
            }

        return summary

    def compute_syntax_effects(self) -> Dict[str, Dict]:
        """Compute effects of different syntactic structures."""
        by_complexity = defaultdict(list)

        for r in self.results:
            if r['variation_type'] == 'syntax':
                complexity = r['linguistic_features'].get('syntactic_complexity', 'unknown')
                c_mean = r['consistency_metrics']['c_mean']
                by_complexity[complexity].append(c_mean)

        summary = {}
        for key, values in by_complexity.items():
            summary[key] = {
                'mean': statistics.mean(values) if values else 0,
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'n': len(values)
            }

        return summary

    def compute_effect_size(self, group1: List[float], group2: List[float]) -> EffectSize:
        """Compute Cohen's d effect size and significance."""
        if not group1 or not group2:
            return EffectSize(0, 0, 0, 0, 1.0, False)

        mean1 = statistics.mean(group1)
        mean2 = statistics.mean(group2)
        mean_diff = mean1 - mean2

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1 = statistics.variance(group1) if n1 > 1 else 0
        var2 = statistics.variance(group2) if n2 > 1 else 0
        pooled_std = ((var1 * (n1-1) + var2 * (n2-1)) / (n1 + n2 - 2)) ** 0.5 if (n1 + n2 > 2) else 1

        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # T-test for significance
        if HAS_SCIPY:
            t_stat, p_value = scipy_stats.ttest_ind(group1, group2)
        else:
            p_value = 0.5  # Placeholder

        # Bootstrap CI (simplified)
        ci_lower = mean_diff - 1.96 * pooled_std / (n1 + n2) ** 0.5
        ci_upper = mean_diff + 1.96 * pooled_std / (n1 + n2) ** 0.5

        return EffectSize(
            cohens_d=cohens_d,
            mean_diff=mean_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            significant=p_value < 0.05
        )

    def compute_all_contrasts(self) -> Dict[str, EffectSize]:
        """Compute key contrasts for hypothesis testing."""
        contrasts = {}

        # Group data by features
        speech_act_data = defaultdict(list)
        modal_data = defaultdict(list)
        politeness_data = defaultdict(list)
        syntax_data = defaultdict(list)

        for r in self.results:
            c_mean = r['consistency_metrics']['c_mean']
            vtype = r['variation_type']
            features = r['linguistic_features']

            if vtype == 'speech_act':
                speech_act_data[features.get('illocutionary_force', '')].append(c_mean)
            elif vtype == 'modal':
                modal_data[features.get('modal_strength', '')].append(c_mean)
            elif vtype == 'politeness':
                politeness_data[features.get('face_threat', '')].append(c_mean)
            elif vtype == 'syntax':
                syntax_data[features.get('syntactic_complexity', '')].append(c_mean)

        # H1: Direct > Indirect speech acts
        contrasts['H1_direct_vs_indirect'] = self.compute_effect_size(
            speech_act_data['directive'],
            speech_act_data['indirect']
        )

        # H2: Bald on-record > Mitigated (negative politeness)
        contrasts['H2_bald_vs_negative'] = self.compute_effect_size(
            politeness_data['high'],  # bald = high face threat
            politeness_data['low']    # negative politeness = low face threat
        )

        # H3: Strong modal > Weak modal
        contrasts['H3_strong_vs_weak'] = self.compute_effect_size(
            modal_data['strong'],
            modal_data['weak']
        )

        # H4: Simple syntax > Complex syntax
        contrasts['H4_simple_vs_embedded'] = self.compute_effect_size(
            syntax_data['simple'],
            syntax_data['embedded']
        )

        return contrasts

    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for the paper."""
        tables = {}

        # Table 1: Speech Act Effects
        speech_effects = self.compute_speech_act_effects()
        rows = []
        for force, stats in sorted(speech_effects.items()):
            rows.append(f"    {force.capitalize()} & {stats['mean']:.3f} & {stats['std']:.3f} & {stats['n']} \\\\")

        tables['speech_acts'] = f"""\\begin{{table}}[t]
\\centering
\\caption{{Consistency by Speech Act Type}}
\\label{{tab:speech_acts}}
\\small
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
\\textbf{{Illocutionary Force}} & \\textbf{{$C_{{mean}}$}} & \\textbf{{Std}} & \\textbf{{n}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

        # Table 2: Modal Verb Effects
        modal_effects = self.compute_modal_effects()
        rows = []
        for strength in ['strong', 'medium', 'weak', 'none']:
            if strength in modal_effects['by_strength']:
                stats = modal_effects['by_strength'][strength]
                rows.append(f"    {strength.capitalize()} & {stats['mean']:.3f} & {stats['std']:.3f} & {stats['n']} \\\\")

        tables['modal_verbs'] = f"""\\begin{{table}}[t]
\\centering
\\caption{{Consistency by Modal Verb Strength}}
\\label{{tab:modal_verbs}}
\\small
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
\\textbf{{Modal Strength}} & \\textbf{{$C_{{mean}}$}} & \\textbf{{Std}} & \\textbf{{n}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

        # Table 3: Hypothesis Test Results
        contrasts = self.compute_all_contrasts()
        rows = []
        hypotheses = {
            'H1_direct_vs_indirect': 'Direct > Indirect',
            'H2_bald_vs_negative': 'Bald > Mitigated',
            'H3_strong_vs_weak': 'Strong modal > Weak modal',
            'H4_simple_vs_embedded': 'Simple > Embedded'
        }
        for key, label in hypotheses.items():
            if key in contrasts:
                es = contrasts[key]
                sig = '***' if es.p_value < 0.001 else '**' if es.p_value < 0.01 else '*' if es.p_value < 0.05 else ''
                rows.append(f"    {label} & {es.cohens_d:.3f} & [{es.ci_lower:.3f}, {es.ci_upper:.3f}] & {es.p_value:.4f}{sig} \\\\")

        tables['hypotheses'] = f"""\\begin{{table}}[t]
\\centering
\\caption{{Hypothesis Test Results}}
\\label{{tab:hypotheses}}
\\small
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
\\textbf{{Hypothesis}} & \\textbf{{Cohen's $d$}} & \\textbf{{95\\% CI}} & \\textbf{{$p$-value}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\vspace{{0.3em}}
\\raggedright\\footnotesize $^{{***}}p<0.001$, $^{{**}}p<0.01$, $^{{*}}p<0.05$
\\end{{table}}"""

        return tables

    def generate_summary_report(self) -> str:
        """Generate text summary report."""
        lines = []
        lines.append("=" * 60)
        lines.append("ACL Linguistic Variation Analysis Report")
        lines.append("=" * 60)

        # Overall stats
        lines.append(f"\nTotal results: {len(self.results)}")
        unique_bases = len(set(r['base_id'] for r in self.results))
        lines.append(f"Unique base prompts: {unique_bases}")

        # By variation type
        lines.append("\n" + "-" * 40)
        lines.append("Results by Variation Type")
        lines.append("-" * 40)

        by_type = defaultdict(list)
        for r in self.results:
            by_type[r['variation_type']].append(r['consistency_metrics']['c_mean'])

        for vtype, values in sorted(by_type.items()):
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            lines.append(f"  {vtype}: mean={mean:.3f}, std={std:.3f}, n={len(values)}")

        # Speech Act Effects
        lines.append("\n" + "-" * 40)
        lines.append("Speech Act Effects (RQ1)")
        lines.append("-" * 40)
        for force, stats in self.compute_speech_act_effects().items():
            lines.append(f"  {force}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={stats['n']}")

        # Modal Effects
        lines.append("\n" + "-" * 40)
        lines.append("Modal Verb Effects (RQ3)")
        lines.append("-" * 40)
        modal = self.compute_modal_effects()
        lines.append("  By strength:")
        for strength, stats in modal['by_strength'].items():
            lines.append(f"    {strength}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={stats['n']}")

        # Politeness Effects
        lines.append("\n" + "-" * 40)
        lines.append("Politeness Effects (RQ2)")
        lines.append("-" * 40)
        pol = self.compute_politeness_effects()
        lines.append("  By face threat level:")
        for level, stats in pol['by_face_threat'].items():
            lines.append(f"    {level}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={stats['n']}")

        # Hypothesis Tests
        lines.append("\n" + "-" * 40)
        lines.append("Hypothesis Test Results")
        lines.append("-" * 40)
        contrasts = self.compute_all_contrasts()
        for key, es in contrasts.items():
            sig = "SIGNIFICANT" if es.significant else "not significant"
            lines.append(f"  {key}:")
            lines.append(f"    Cohen's d = {es.cohens_d:.3f}")
            lines.append(f"    Mean diff = {es.mean_diff:.3f}")
            lines.append(f"    p-value = {es.p_value:.4f} ({sig})")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze linguistic variation results')
    parser.add_argument('--input', type=str,
                        default='results/acl_linguistic/eval_results.json',
                        help='Input results file')
    parser.add_argument('--output', type=str,
                        default='results/acl_linguistic/analysis/',
                        help='Output directory')

    args = parser.parse_args()

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run run_evaluation.py first to generate results.")
        return

    with open(input_path) as f:
        data = json.load(f)

    results = data['results']
    print(f"Loaded {len(results)} results from {input_path}")

    # Analyze
    analyzer = LinguisticAnalyzer(results)

    # Generate report
    report = analyzer.generate_summary_report()
    print(report)

    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")

    # Save LaTeX tables
    tables = analyzer.generate_latex_tables()
    tables_path = output_dir / 'latex_tables.tex'
    with open(tables_path, 'w') as f:
        f.write("% Auto-generated LaTeX tables for ACL paper\n\n")
        for name, table in tables.items():
            f.write(f"% Table: {name}\n")
            f.write(table)
            f.write("\n\n")
    print(f"Saved LaTeX tables to {tables_path}")

    # Save JSON summary
    summary = {
        'speech_acts': analyzer.compute_speech_act_effects(),
        'modal_verbs': analyzer.compute_modal_effects(),
        'politeness': analyzer.compute_politeness_effects(),
        'syntax': analyzer.compute_syntax_effects(),
        'contrasts': {k: {
            'cohens_d': v.cohens_d,
            'mean_diff': v.mean_diff,
            'p_value': v.p_value,
            'significant': v.significant
        } for k, v in analyzer.compute_all_contrasts().items()}
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
