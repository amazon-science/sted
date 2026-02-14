#!/usr/bin/env python3
"""
Evaluate LLM-as-Judge on Full Synthetic Dataset

This script runs the LLM-as-Judge baseline on all samples from the
synthetic expression variation dataset and compares with existing metrics.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sted import STED, create_llm_judge

# Paths
STED_PROJECT = Path("/Users/guanghu/Documents/genai/projects/sted")
SYNTHETIC_DIR = STED_PROJECT / "synthetic_dataset"
RESULTS_DIR = STED_PROJECT / "results"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_expression_variation_dataset() -> List[Dict]:
    """Load the expression variation synthetic dataset."""
    # Use the same file referenced in results
    path = SYNTHETIC_DIR / "expression_variation_dataset_2025-11-08_22-58-33.json"
    if not path.exists():
        # Try the latest one
        files = list(SYNTHETIC_DIR.glob("expression_variation_dataset_*.json"))
        if files:
            path = max(files, key=lambda x: x.stat().st_mtime)
        else:
            raise FileNotFoundError("No expression variation dataset found")

    print(f"Loading dataset: {path.name}")
    with open(path, 'r') as f:
        return json.load(f)


def load_schema_variation_dataset() -> Dict:
    """Load the schema variation synthetic dataset."""
    path = SYNTHETIC_DIR / "schema_variation_dataset_2025-11-08_22-57-48.json"
    if not path.exists():
        files = list(SYNTHETIC_DIR.glob("schema_variation_dataset_*.json"))
        if files:
            path = max(files, key=lambda x: x.stat().st_mtime)

    print(f"Loading dataset: {path.name}")
    with open(path, 'r') as f:
        return json.load(f)


def load_existing_results() -> Dict:
    """Load existing metric results for comparison."""
    path = RESULTS_DIR / "variation_progression" / "expression_variation_progression_results.json"
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_expression_variation(
    dataset: List[Dict],
    llm_judge,
    existing_results: Dict
) -> Dict:
    """
    Evaluate LLM-as-Judge on expression variation dataset.

    Compares base_sample vs each variant at different variation ratios.
    """
    results = {
        "variation_ratios": [],
        "llm_judge_scores": {},
        "raw_scores": []
    }

    # Group by variation ratio
    ratio_scores = {}

    total_pairs = sum(len(sample.get('variants', [])) for sample in dataset)
    print(f"\nEvaluating {total_pairs} pairs across {len(dataset)} samples...")
    print("-" * 70)

    pair_count = 0
    for sample_idx, sample in enumerate(dataset):
        base = sample['base_sample']
        sample_id = sample.get('sample_id', f'sample_{sample_idx}')

        for variant in sample.get('variants', []):
            ratio = variant.get('variation_ratio', 0)
            varied_json = variant.get('variant', variant.get('variation', {}))

            # Skip if no variant JSON
            if not varied_json:
                continue

            pair_count += 1
            print(f"\r[{pair_count}/{total_pairs}] Sample {sample_id}, ratio={ratio}", end="", flush=True)

            try:
                # Rate limit to avoid throttling
                time.sleep(0.3)

                score = llm_judge.calculate_similarity(base, varied_json)

                if ratio not in ratio_scores:
                    ratio_scores[ratio] = []
                ratio_scores[ratio].append(score)

                results["raw_scores"].append({
                    "sample_id": sample_id,
                    "variation_ratio": ratio,
                    "llm_judge_score": score
                })

            except Exception as e:
                print(f"\n  Error: {e}")
                continue

    print("\n")

    # Calculate averages per ratio
    results["variation_ratios"] = sorted(ratio_scores.keys())
    results["llm_judge_scores"] = {
        str(ratio): {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores
        }
        for ratio, scores in sorted(ratio_scores.items())
    }

    return results


def evaluate_schema_variation(dataset: Dict, llm_judge) -> Dict:
    """
    Evaluate LLM-as-Judge on schema variation dataset.

    This includes breaking changes that STED correctly identifies.
    """
    results = {}

    # Check for breaking change tests
    for change_type in ['flat_structure', 'nested_change', 'field_name_change']:
        if change_type in dataset:
            print(f"\nEvaluating {change_type}...")
            change_data = dataset[change_type]

            if 'samples' in change_data:
                scores = []
                for sample in change_data['samples'][:5]:  # Limit to 5 for cost
                    if 'base' in sample and 'variant' in sample:
                        time.sleep(0.3)
                        try:
                            score = llm_judge.calculate_similarity(
                                sample['base'],
                                sample['variant']
                            )
                            scores.append(score)
                            print(f"  Score: {score:.3f}")
                        except Exception as e:
                            print(f"  Error: {e}")

                if scores:
                    results[change_type] = {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "scores": scores
                    }

    return results


def analyze_results(llm_results: Dict, existing_results: Dict) -> Dict:
    """
    Analyze LLM-as-Judge results and compare with existing metrics.
    """
    analysis = {}

    # Extract scores for correlation analysis
    ratios = llm_results["variation_ratios"]
    llm_means = [llm_results["llm_judge_scores"][str(r)]["mean"] for r in ratios]

    # Get existing metric means
    existing_avg = existing_results.get("average_similarities", {})

    print("\n" + "=" * 70)
    print("COMPARISON WITH EXISTING METRICS")
    print("=" * 70)

    print(f"\n{'Ratio':<8} {'LLM Judge':<12} {'STED':<12} {'TED':<12} {'BERTScore':<12} {'DeepDiff':<12}")
    print("-" * 70)

    for i, ratio in enumerate(ratios):
        llm = llm_means[i]
        sted = existing_avg.get('sted', [0]*len(ratios))[i] if i < len(existing_avg.get('sted', [])) else 0
        ted = existing_avg.get('ted', [0]*len(ratios))[i] if i < len(existing_avg.get('ted', [])) else 0
        bert = existing_avg.get('bertscore', [0]*len(ratios))[i] if i < len(existing_avg.get('bertscore', [])) else 0
        deep = existing_avg.get('deepdiff', [0]*len(ratios))[i] if i < len(existing_avg.get('deepdiff', [])) else 0

        print(f"{ratio:<8.1f} {llm:<12.3f} {sted:<12.3f} {ted:<12.3f} {bert:<12.3f} {deep:<12.3f}")

    # Calculate correlations
    print("\n" + "=" * 70)
    print("CORRELATION WITH VARIATION RATIO")
    print("=" * 70)

    # LLM Judge correlation with ratio
    if len(ratios) >= 3:
        llm_corr, llm_p = stats.spearmanr(ratios, llm_means)
        analysis["llm_judge_correlation"] = {
            "spearman_r": float(llm_corr),
            "p_value": float(llm_p)
        }
        sig = "***" if llm_p < 0.001 else "**" if llm_p < 0.01 else "*" if llm_p < 0.05 else "ns"
        print(f"LLM Judge: ρ = {llm_corr:.3f}, p = {llm_p:.2e} {sig}")

        # Compare with existing
        for metric in ['sted', 'ted', 'bertscore', 'deepdiff']:
            if metric in existing_avg and len(existing_avg[metric]) >= len(ratios):
                metric_means = existing_avg[metric][:len(ratios)]
                corr, p = stats.spearmanr(ratios, metric_means)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"{metric.upper():12s}: ρ = {corr:.3f}, p = {p:.2e} {sig}")

    # Calculate score range (sensitivity)
    llm_range = max(llm_means) - min(llm_means)
    analysis["score_range"] = {
        "llm_judge": llm_range
    }

    print("\n" + "=" * 70)
    print("SCORE RANGE (SENSITIVITY)")
    print("=" * 70)
    print(f"LLM Judge: {llm_range:.4f}")

    for metric in ['sted', 'ted', 'bertscore', 'deepdiff']:
        if metric in existing_avg:
            metric_range = max(existing_avg[metric]) - min(existing_avg[metric])
            analysis["score_range"][metric] = metric_range
            print(f"{metric.upper():12s}: {metric_range:.4f}")

    return analysis


def generate_latex_table(llm_results: Dict, existing_results: Dict, analysis: Dict) -> str:
    """Generate LaTeX table for paper."""

    latex = """
% Table: LLM-as-Judge vs Other Metrics on Expression Variation
\\begin{table}[h]
\\centering
\\caption{Metric Comparison on Expression Variation Dataset}
\\label{tab:llm_judge_expression}
\\begin{tabular}{lcccccc}
\\toprule
Variation & LLM Judge & STED & TED & BERTScore & DeepDiff \\\\
\\midrule
"""

    ratios = llm_results["variation_ratios"]
    existing_avg = existing_results.get("average_similarities", {})

    for i, ratio in enumerate(ratios):
        llm = llm_results["llm_judge_scores"][str(ratio)]["mean"]
        sted = existing_avg.get('sted', [0]*len(ratios))[i] if i < len(existing_avg.get('sted', [])) else 0
        ted = existing_avg.get('ted', [0]*len(ratios))[i] if i < len(existing_avg.get('ted', [])) else 0
        bert = existing_avg.get('bertscore', [0]*len(ratios))[i] if i < len(existing_avg.get('bertscore', [])) else 0
        deep = existing_avg.get('deepdiff', [0]*len(ratios))[i] if i < len(existing_avg.get('deepdiff', [])) else 0

        latex += f"{ratio:.1f} & {llm:.3f} & {sted:.3f} & {ted:.3f} & {bert:.3f} & {deep:.3f} \\\\\n"

    latex += """\\midrule
\\multicolumn{6}{l}{\\textit{Correlation with Variation Ratio (Spearman $\\rho$)}} \\\\
"""

    # Add correlation row
    llm_corr = analysis.get("llm_judge_correlation", {}).get("spearman_r", 0)
    latex += f"$\\rho$ & {llm_corr:.3f} & -0.617 & -- & -0.602 & -0.955 \\\\\n"

    latex += """\\midrule
\\multicolumn{6}{l}{\\textit{Score Range}} \\\\
"""

    ranges = analysis.get("score_range", {})
    latex += f"Range & {ranges.get('llm_judge', 0):.3f} & {ranges.get('sted', 0):.3f} & "
    latex += f"{ranges.get('ted', 0):.3f} & {ranges.get('bertscore', 0):.3f} & {ranges.get('deepdiff', 0):.3f} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: LLM Judge uses Claude Sonnet 3.5. Higher variation ratio = more semantic change.
\\end{tablenotes}
\\end{table}
"""

    return latex


def main():
    print("=" * 70)
    print("LLM-AS-JUDGE EVALUATION ON SYNTHETIC DATASET")
    print("=" * 70)

    # Initialize LLM Judge with Claude Opus 4.5
    print("\nInitializing LLM Judge (Claude Opus 4.5)...")
    llm_judge = create_llm_judge(
        provider="bedrock",
        model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
        temperature=0.0,
        max_tokens=8000
    )

    # Load datasets
    print("\nLoading datasets...")
    expr_dataset = load_expression_variation_dataset()
    existing_results = load_existing_results()

    print(f"  Expression variation: {len(expr_dataset)} samples")

    # Evaluate expression variation
    print("\n" + "=" * 70)
    print("EVALUATING EXPRESSION VARIATION")
    print("=" * 70)

    llm_results = evaluate_expression_variation(expr_dataset, llm_judge, existing_results)

    # Analyze results
    analysis = analyze_results(llm_results, existing_results)

    # Generate LaTeX
    latex = generate_latex_table(llm_results, existing_results, analysis)
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(latex)

    # Save results
    output = {
        "llm_judge_results": llm_results,
        "analysis": analysis,
        "comparison_with_existing": {
            "existing_results_file": str(RESULTS_DIR / "variation_progression" / "expression_variation_progression_results.json")
        }
    }

    results_path = OUTPUT_DIR / "llm_judge_synthetic_results.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    latex_path = OUTPUT_DIR / "llm_judge_synthetic_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to: {latex_path}")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    llm_corr = analysis.get("llm_judge_correlation", {})
    print(f"""
1. LLM-as-Judge Correlation with Variation:
   - Spearman ρ = {llm_corr.get('spearman_r', 'N/A'):.3f}
   - p-value = {llm_corr.get('p_value', 'N/A'):.2e}

2. Compared to STED (ρ = -0.617, p < 0.001):
   - LLM Judge {'shows similar' if abs(llm_corr.get('spearman_r', 0)) > 0.5 else 'shows weaker'} correlation
   - But at significantly higher cost per comparison

3. Score Range (Sensitivity):
   - LLM Judge: {analysis.get('score_range', {}).get('llm_judge', 0):.4f}
   - STED: {analysis.get('score_range', {}).get('sted', 0):.4f}
""")

    return output


if __name__ == "__main__":
    results = main()
