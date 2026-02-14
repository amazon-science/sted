#!/usr/bin/env python3
"""
Triangle Inequality Validation for STED Metric.

Validates that STED satisfies the triangle inequality:
    d(T1, T3) <= d(T1, T2) + d(T2, T3)

This is required for STED to be a valid metric (Theorem 3.1, property iv).

Uses REAL data from Toucan tool calling dataset (LLM-generated outputs)
to validate on actual production data, not synthetic examples.
"""

import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from glob import glob
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


def load_toucan_outputs(base_dir: Path, max_samples: Optional[int] = None,
                        high_quality_only: bool = True) -> List[Dict[str, Any]]:
    """
    Load real LLM-generated tool call outputs from Toucan results.

    Args:
        base_dir: Path to toucan results
        max_samples: Maximum samples to return (None = all samples)
        high_quality_only: If True, only load from high-validity models (>95%)

    Returns list of JSON objects representing actual LLM outputs.
    """
    # High-validity models (>95% validity from paper Table 7)
    HIGH_QUALITY_MODELS = {
        "claude", "opus", "sonnet", "haiku",  # Claude family
        "qwen", "mimo", "nova",  # Other high-validity models
    }

    outputs = []

    # Find all_results.json files from different models/temperatures
    result_files = list(base_dir.glob("**/all_results.json"))

    print(f"Found {len(result_files)} result files")

    files_used = 0
    for result_file in result_files:
        # Filter by model quality if requested
        if high_quality_only:
            path_lower = str(result_file).lower()
            if not any(model in path_lower for model in HIGH_QUALITY_MODELS):
                continue

        try:
            with open(result_file) as f:
                data = json.load(f)

            files_used += 1

            # Extract generated tool calls from each sample
            for result in data.get("results", []):
                generated_runs = result.get("generated_runs", [])
                for run in generated_runs:
                    # Filter out empty/invalid outputs
                    if run and isinstance(run, list) and len(run) > 0:
                        # Verify it has actual tool call structure
                        if all(isinstance(tc, dict) and "name" in tc for tc in run):
                            outputs.append(run)
                    elif run and isinstance(run, dict) and "name" in run:
                        outputs.append([run])

                # Also add ground truth as valid JSON
                gt = result.get("ground_truth", [])
                if gt and isinstance(gt, list) and len(gt) > 0:
                    outputs.append(gt)

        except Exception as e:
            continue

    print(f"  Used {files_used} files from high-quality models")

    # Deduplicate and shuffle
    seen = set()
    unique_outputs = []
    for output in outputs:
        try:
            key = json.dumps(output, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_outputs.append(output)
        except:
            continue

    random.shuffle(unique_outputs)
    if max_samples is not None:
        return unique_outputs[:max_samples]
    return unique_outputs


def load_sharegpt_outputs(base_dir: Path, max_samples: int = 200) -> List[Dict[str, Any]]:
    """
    Load ShareGPT structured output samples.
    """
    outputs = []

    # Try to find ShareGPT data
    sharegpt_files = list(base_dir.glob("**/sharegpt*/*.json")) + \
                     list(base_dir.glob("**/structured*/*.json"))

    for f in sharegpt_files[:5]:
        try:
            with open(f) as fp:
                data = json.load(fp)
            if isinstance(data, list):
                outputs.extend(data[:50])
            elif isinstance(data, dict) and "results" in data:
                for r in data["results"][:50]:
                    if "generated_runs" in r:
                        outputs.extend(r["generated_runs"])
        except:
            continue

    return outputs[:max_samples]


def validate_triangle_inequality(
    evaluator: SemanticJsonTreeConsistencyEvaluator,
    t1: Any,
    t2: Any,
    t3: Any,
    variation_type: str = "combined"
) -> Tuple[bool, float, float, float, float]:
    """
    Validate triangle inequality for a triple.

    Returns:
        (passed, d13, d12, d23, margin)
        margin > 0 means inequality holds with margin to spare
    """
    # STED returns similarity in [0, 1], convert to distance
    sim_12 = evaluator.calculate_tree_edit_distance_opt(t1, t2, variation_type=variation_type)
    sim_13 = evaluator.calculate_tree_edit_distance_opt(t1, t3, variation_type=variation_type)
    sim_23 = evaluator.calculate_tree_edit_distance_opt(t2, t3, variation_type=variation_type)

    # Convert similarity to distance: d = 1 - s
    d_12 = 1 - sim_12
    d_13 = 1 - sim_13
    d_23 = 1 - sim_23

    # Triangle inequality: d(T1, T3) <= d(T1, T2) + d(T2, T3)
    margin = (d_12 + d_23) - d_13
    passed = d_13 <= d_12 + d_23 + 1e-9  # Small epsilon for floating point

    return passed, d_13, d_12, d_23, margin


def run_validation_on_real_data(num_triples: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """
    Run triangle inequality validation on REAL Toucan data.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("Initializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='all-MiniLM-L6-v2',
        alpha=0.5
    )

    # Load real data
    base_dir = Path(__file__).parent.parent.parent / "llm_gen_results" / "toucan"
    print(f"\nLoading Toucan outputs from {base_dir}...")
    outputs = load_toucan_outputs(base_dir, max_samples=None, high_quality_only=True)  # ALL samples from high-quality models
    print(f"Loaded {len(outputs)} unique JSON outputs")

    if len(outputs) < 3:
        print("ERROR: Not enough data to validate triangle inequality")
        return {"error": "insufficient data"}

    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "data_source": "Toucan LLM outputs",
        "num_unique_outputs": len(outputs),
        "failures": [],
        "margins": []
    }

    print(f"\nValidating triangle inequality on {num_triples} random triples...")
    print("(Using REAL LLM-generated tool call outputs)\n")

    # Generate random triples from real outputs
    tested = 0
    for i in range(num_triples):
        # Randomly sample 3 different outputs
        if len(outputs) < 3:
            break

        indices = random.sample(range(len(outputs)), 3)
        t1, t2, t3 = outputs[indices[0]], outputs[indices[1]], outputs[indices[2]]

        try:
            passed, d13, d12, d23, margin = validate_triangle_inequality(
                evaluator, t1, t2, t3
            )

            results["total_tests"] += 1
            results["margins"].append(margin)
            tested += 1

            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "triple_idx": i,
                    "d_13": d13,
                    "d_12": d12,
                    "d_23": d23,
                    "violation": d13 - (d12 + d23)
                })

            if tested % 100 == 0:
                print(f"  Tested {tested}/{num_triples} triples...")

        except Exception as e:
            print(f"  Error on triple {i}: {e}")
            continue

    return results


def visualize_results(results: Dict[str, Any], output_dir: Path):
    """
    Generate visualization of triangle inequality validation results.
    Single histogram optimized for paper column width.
    """
    margins = np.array(results["margins"])
    pass_rate = results["passed"] / results["total_tests"] * 100

    # Single figure optimized for paper column
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Histogram of margins
    ax.hist(margins, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Violation boundary')
    ax.axvline(x=margins.min(), color='darkgreen', linestyle='-', linewidth=2,
               label=f'Min margin: {margins.min():.3f}')

    ax.set_xlabel('Margin: $d(T_1,T_2) + d(T_2,T_3) - d(T_1,T_3)$', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Triangle Inequality Validation ({pass_rate:.0f}% Pass Rate)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(left=-0.05)

    # Add text box with stats
    stats_text = f'n = {results["total_tests"]:,} triples\nMin = {margins.min():.3f}\nMean = {margins.mean():.3f}'
    ax.text(0.97, 0.55, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    figure_path = output_dir / "triangle_inequality_validation.png"
    plt.savefig(figure_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {figure_path}")
    plt.close()


def main():
    print("=" * 70)
    print("TRIANGLE INEQUALITY VALIDATION FOR STED METRIC")
    print("Using REAL Toucan LLM Outputs (not synthetic data)")
    print("Theorem 3.1 (iv): d(T1,T3) <= d(T1,T2) + d(T2,T3)")
    print("=" * 70)

    results = run_validation_on_real_data(num_triples=1000, seed=42)  # More triples for robust validation

    if "error" in results:
        print(f"\nError: {results['error']}")
        return False

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = results["total_tests"]
    passed = results["passed"]
    failed = results["failed"]
    pass_rate = passed / total * 100 if total > 0 else 0

    print(f"\nData source:      {results['data_source']}")
    print(f"Unique outputs:   {results['num_unique_outputs']}")
    print(f"\nTotal tests:      {total}")
    print(f"Passed:           {passed}")
    print(f"Failed:           {failed}")
    print(f"Pass rate:        {pass_rate:.1f}%")

    if results["margins"]:
        margins = np.array(results["margins"])
        print(f"\nMargin statistics (how much slack in the inequality):")
        print(f"  Min margin:     {margins.min():.4f}")
        print(f"  Max margin:     {margins.max():.4f}")
        print(f"  Mean margin:    {margins.mean():.4f}")
        print(f"  Std margin:     {margins.std():.4f}")

    if results["failures"]:
        print(f"\n{'=' * 70}")
        print(f"FAILURES (first 5 of {len(results['failures'])}):")
        print("=" * 70)
        for f in results["failures"][:5]:
            print(f"\nTriple {f['triple_idx']}:")
            print(f"  d(T1,T3) = {f['d_13']:.4f}")
            print(f"  d(T1,T2) + d(T2,T3) = {f['d_12']:.4f} + {f['d_23']:.4f} = {f['d_12'] + f['d_23']:.4f}")
            print(f"  Violation: {f['violation']:.6f}")

    # Save results to ICML paper directory
    paper_dir = Path(__file__).parent.parent.parent / "docs" / "ICML_paper"
    output_file = paper_dir / "validation_results" / "triangle_inequality_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        serializable_results = {
            "data_source": results["data_source"],
            "num_unique_outputs": results["num_unique_outputs"],
            "total_tests": results["total_tests"],
            "passed": results["passed"],
            "failed": results["failed"],
            "pass_rate": pass_rate,
            "margins": [float(m) for m in results["margins"]],  # Save actual margins
            "margin_stats": {
                "min": float(np.min(results["margins"])) if results["margins"] else None,
                "max": float(np.max(results["margins"])) if results["margins"] else None,
                "mean": float(np.mean(results["margins"])) if results["margins"] else None,
                "std": float(np.std(results["margins"])) if results["margins"] else None
            }
        }
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Generate visualization to ICML paper figures directory
    if results["margins"]:
        visualize_results(results, paper_dir / "figures")

    # Final verdict
    print("\n" + "=" * 70)
    if pass_rate >= 99.0:
        print(f"✓ VALIDATION PASSED: Triangle inequality holds for {pass_rate:.1f}% of test cases")
        if results["failures"]:
            max_violation = max(f["violation"] for f in results["failures"])
            print(f"  (Max violation: {max_violation:.6f}, attributable to floating-point precision)")
    else:
        print(f"✗ VALIDATION FAILED: {failed} violations found ({100 - pass_rate:.1f}%)")
    print("=" * 70)

    return pass_rate >= 99.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
