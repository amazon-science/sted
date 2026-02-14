#!/usr/bin/env python3
"""
Evaluate Consistency of Multi-LLM Benchmark Results

This script computes STED and baseline metrics on the generated outputs
from run_multi_llm_experiment.py to evaluate cross-model and within-model consistency.

Usage:
    python evaluate_consistency.py --results-dir ./results/run_YYYYMMDD_HHMMSS
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from itertools import combinations
from datetime import datetime
import numpy as np
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sted import STED

# ============================================================================
# Consistency Metrics
# ============================================================================

def compute_pairwise_sted(outputs: List[Dict], sted_calculator: STED) -> Dict[str, float]:
    """
    Compute pairwise STED scores for a list of outputs.

    Empty/invalid responses are penalized by:
    1. Computing STED only on valid outputs
    2. Multiplying by validity_rate (valid_count / total_count)

    This ensures that models producing many empty responses get lower consistency scores.
    """
    total_count = len(outputs)
    if total_count < 2:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0,
                "valid_count": 0, "total_count": total_count, "validity_rate": 0.0}

    # Filter out empty outputs (check for non-empty dicts)
    valid_outputs = [o for o in outputs if o and len(o) > 0]
    valid_count = len(valid_outputs)
    validity_rate = valid_count / total_count

    # If less than 2 valid outputs, consistency is 0 (penalized)
    if valid_count < 2:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0,
                "valid_count": valid_count, "total_count": total_count, "validity_rate": validity_rate}

    scores = []
    for o1, o2 in combinations(valid_outputs, 2):
        try:
            # calculate_tree_edit_distance returns similarity directly (0 = different, 1 = identical)
            similarity = sted_calculator.calculate_tree_edit_distance(o1, o2)
            # Clamp to [0, 1] range
            similarity = max(0.0, min(1.0, similarity))
            scores.append(similarity)
        except Exception as e:
            # Log the error for debugging
            print(f"  Warning: STED calculation failed: {e}")
            continue

    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0,
                "valid_count": valid_count, "total_count": total_count, "validity_rate": validity_rate}

    # Raw STED scores (among valid outputs only)
    raw_mean = float(np.mean(scores))

    # Penalized score: multiply by validity rate
    # If all outputs are valid (validity_rate=1), no penalty
    # If half are empty (validity_rate=0.5), score is halved
    penalized_mean = raw_mean * validity_rate

    return {
        "mean": penalized_mean,  # Use penalized mean as the primary metric
        "raw_mean": raw_mean,    # Raw STED among valid outputs only
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "count": len(scores),
        "valid_count": valid_count,
        "total_count": total_count,
        "validity_rate": validity_rate
    }


def compute_cross_model_sted(outputs_by_model: Dict[str, List[Dict]], sted_calculator: STED) -> Dict[str, Any]:
    """Compute STED scores between outputs from different models."""
    model_names = list(outputs_by_model.keys())
    if len(model_names) < 2:
        return {}

    cross_scores = {}
    for m1, m2 in combinations(model_names, 2):
        outputs1 = [o for o in outputs_by_model[m1] if o and len(o) > 0]
        outputs2 = [o for o in outputs_by_model[m2] if o and len(o) > 0]

        if not outputs1 or not outputs2:
            continue

        scores = []
        # Compare each output from m1 with each from m2
        for o1 in outputs1[:3]:  # Limit for efficiency
            for o2 in outputs2[:3]:
                try:
                    # calculate_tree_edit_distance returns similarity directly
                    similarity = sted_calculator.calculate_tree_edit_distance(o1, o2)
                    similarity = max(0.0, min(1.0, similarity))
                    scores.append(similarity)
                except Exception as e:
                    continue

        if scores:
            cross_scores[f"{m1}_vs_{m2}"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "count": len(scores)
            }

    return cross_scores


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_within_model_consistency(results: Dict, sted_calculator: STED) -> Dict:
    """Analyze within-model consistency (same model, multiple runs)."""
    analysis = {}

    for sample in tqdm(results.get("by_sample", []), desc="Within-model analysis"):
        sample_id = sample["sample_id"]

        for model_name, model_data in sample.get("outputs_by_model", {}).items():
            if "outputs" not in model_data:
                continue

            outputs = model_data["outputs"]
            sted_scores = compute_pairwise_sted(outputs, sted_calculator)

            if model_name not in analysis:
                analysis[model_name] = {
                    "sample_scores": [],
                    "all_means": [],
                    "all_raw_means": [],
                    "all_validity_rates": []
                }

            analysis[model_name]["sample_scores"].append({
                "sample_id": sample_id,
                "sted": sted_scores
            })

            # Collect scores even for samples with 0 consistency (penalized)
            analysis[model_name]["all_means"].append(sted_scores["mean"])
            if "raw_mean" in sted_scores:
                analysis[model_name]["all_raw_means"].append(sted_scores["raw_mean"])
            analysis[model_name]["all_validity_rates"].append(sted_scores.get("validity_rate", 0.0))

    # Compute summary statistics per model
    for model_name in analysis:
        means = analysis[model_name]["all_means"]
        raw_means = analysis[model_name]["all_raw_means"]
        validity_rates = analysis[model_name]["all_validity_rates"]

        if means:
            analysis[model_name]["summary"] = {
                "mean_consistency": float(np.mean(means)),  # Penalized
                "std_consistency": float(np.std(means)),
                "min_consistency": float(np.min(means)),
                "max_consistency": float(np.max(means)),
                "raw_mean_consistency": float(np.mean(raw_means)) if raw_means else 0.0,  # Raw (valid only)
                "mean_validity_rate": float(np.mean(validity_rates)),  # Average validity rate
                "num_samples": len(means)
            }
        del analysis[model_name]["all_means"]
        del analysis[model_name]["all_raw_means"]
        del analysis[model_name]["all_validity_rates"]

    return analysis


def analyze_cross_model_consistency(results: Dict, sted_calculator: STED) -> Dict:
    """Analyze cross-model consistency (different models, same prompt)."""
    analysis = {
        "by_sample": [],
        "summary": {}
    }

    cross_model_scores = {}

    for sample in tqdm(results.get("by_sample", []), desc="Cross-model analysis"):
        sample_id = sample["sample_id"]
        outputs_by_model = {}

        for model_name, model_data in sample.get("outputs_by_model", {}).items():
            if "outputs" in model_data:
                outputs_by_model[model_name] = model_data["outputs"]

        if len(outputs_by_model) < 2:
            continue

        cross_scores = compute_cross_model_sted(outputs_by_model, sted_calculator)

        analysis["by_sample"].append({
            "sample_id": sample_id,
            "cross_model_sted": cross_scores
        })

        # Aggregate by model pair
        for pair_name, scores in cross_scores.items():
            if pair_name not in cross_model_scores:
                cross_model_scores[pair_name] = []
            cross_model_scores[pair_name].append(scores["mean"])

    # Compute summary per model pair
    for pair_name, means in cross_model_scores.items():
        if means:
            analysis["summary"][pair_name] = {
                "mean_consistency": float(np.mean(means)),
                "std_consistency": float(np.std(means)),
                "num_samples": len(means)
            }

    return analysis


def generate_latex_tables(within_model: Dict, cross_model: Dict) -> str:
    """Generate LaTeX tables for the paper."""
    latex = []

    # Within-model consistency table with validity rate
    latex.append("% Within-Model Consistency (with validity penalty)")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Within-Model JSON Output Consistency (STED)}")
    latex.append("\\label{tab:within_model_consistency}")
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\toprule")
    latex.append("Model & Penalized & Raw & Validity & Min & Max \\\\")
    latex.append("\\midrule")

    for model_name, data in within_model.items():
        if "summary" in data:
            s = data["summary"]
            validity_pct = s.get('mean_validity_rate', 1.0) * 100
            raw_mean = s.get('raw_mean_consistency', s['mean_consistency'])
            latex.append(f"{model_name} & {s['mean_consistency']:.3f} & {raw_mean:.3f} & {validity_pct:.0f}\\% & {s['min_consistency']:.3f} & {s['max_consistency']:.3f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate consistency of multi-LLM benchmark results")

    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory containing experiment_results.json")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as results-dir)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_file = results_dir / "experiment_results.json"

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    print("="*60)
    print("Multi-LLM Consistency Evaluation")
    print("="*60)
    print(f"Loading results from: {results_file}")

    with open(results_file) as f:
        results = json.load(f)

    print(f"Loaded {len(results.get('by_sample', []))} samples")
    print(f"Models: {list(results.get('by_model', {}).keys())}")

    # Initialize STED calculator
    print("\nInitializing STED calculator...")
    sted_calculator = STED()

    # Analyze within-model consistency
    print("\nAnalyzing within-model consistency...")
    within_model = analyze_within_model_consistency(results, sted_calculator)

    # Analyze cross-model consistency
    print("\nAnalyzing cross-model consistency...")
    cross_model = analyze_cross_model_consistency(results, sted_calculator)

    # Generate LaTeX tables
    latex_tables = generate_latex_tables(within_model, cross_model)

    # Save results
    evaluation_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source_results": str(results_file)
        },
        "within_model_consistency": within_model,
        "cross_model_consistency": cross_model
    }

    eval_path = output_dir / "consistency_evaluation.json"
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)

    latex_path = output_dir / "consistency_tables.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_tables)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\n--- Within-Model Consistency (STED) ---")
    print("  (Penalized score = raw_score * validity_rate)")
    print()
    for model_name, data in within_model.items():
        if "summary" in data:
            s = data["summary"]
            print(f"  {model_name}:")
            print(f"    Penalized:     {s['mean_consistency']:.3f} (+/- {s['std_consistency']:.3f})")
            print(f"    Raw (valid):   {s['raw_mean_consistency']:.3f}")
            print(f"    Validity rate: {s['mean_validity_rate']:.1%}")
            print(f"    Samples:       {s['num_samples']}")

    print(f"\nResults saved to:")
    print(f"  - {eval_path}")
    print(f"  - {latex_path}")


if __name__ == "__main__":
    main()
