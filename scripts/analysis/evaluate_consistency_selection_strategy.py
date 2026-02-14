"""
Evaluate Consistency as a Selection Strategy for Improving Accuracy

Experiment Design:
- For each (model, sample) with 10 generated runs:
  1. Baseline: Random selection → accuracy = mean accuracy across runs
  2. Consistency Strategy: Select the run most similar to all others (centroid) → accuracy
  3. Compare: Does selecting the most consistent output improve accuracy?

This tests whether consistency can be used as a SELECTION criterion to improve accuracy.
"""

import json
import os
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sted import SemanticJsonTreeConsistencyEvaluator
from sted.model_config import FINAL_MODELS


def load_generation_results(results_dir: Path, temperature: float) -> dict:
    """Load generation results for a specific temperature."""
    all_results = {}

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('generations-'):
            continue

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue

            match = re.match(r'run_(.+?)_temp_(\d+)_(\d+)_', run_dir.name)
            if not match:
                continue

            model_name = match.group(1)
            temp_major = int(match.group(2))
            temp_minor = int(match.group(3))
            run_temp = temp_major + temp_minor / 100

            if abs(run_temp - temperature) > 0.01:
                continue

            all_results_file = run_dir / "all_results.json"
            if all_results_file.exists():
                try:
                    with open(all_results_file) as f:
                        data = json.load(f)

                    if model_name not in all_results:
                        all_results[model_name] = {}

                    for idx, result in enumerate(data.get('results', [])):
                        all_results[model_name][idx] = {
                            'ground_truth': result.get('ground_truth', []),
                            'generated_runs': result.get('generated_runs', []),
                        }
                    break
                except Exception as e:
                    continue

    return all_results


def is_final_model(model_name):
    """Check if model is in FINAL_MODELS."""
    for fm in FINAL_MODELS:
        if fm.lower() in model_name.lower() or model_name.lower() in fm.lower():
            return True
    return False


def compute_accuracy(evaluator, ground_truth, run):
    """Compute accuracy (STED similarity) between a single run and ground truth."""
    if not ground_truth or not run:
        return None
    try:
        gt_wrapped = {"tool_calls": ground_truth}
        run_wrapped = {"tool_calls": run}
        return evaluator.calculate_tree_edit_distance_fast(gt_wrapped, run_wrapped, variation_type='combined')
    except:
        return None


def compute_pairwise_similarity(evaluator, run1, run2):
    """Compute STED similarity between two runs."""
    if not run1 or not run2:
        return None
    try:
        r1_wrapped = {"tool_calls": run1}
        r2_wrapped = {"tool_calls": run2}
        return evaluator.calculate_tree_edit_distance_fast(r1_wrapped, r2_wrapped, variation_type='combined')
    except:
        return None


def find_centroid_run(evaluator, runs):
    """
    Find the run that is most similar to all other runs (centroid).
    Returns the index of the centroid run.
    """
    valid_runs = [(i, r) for i, r in enumerate(runs) if r]
    if len(valid_runs) < 2:
        return valid_runs[0][0] if valid_runs else None

    # Compute average similarity of each run to all others
    avg_similarities = []
    for i, run_i in valid_runs:
        similarities = []
        for j, run_j in valid_runs:
            if i != j:
                sim = compute_pairwise_similarity(evaluator, run_i, run_j)
                if sim is not None:
                    similarities.append(sim)
        if similarities:
            avg_similarities.append((i, np.mean(similarities)))
        else:
            avg_similarities.append((i, 0))

    # Return index of run with highest average similarity (most consistent)
    if avg_similarities:
        return max(avg_similarities, key=lambda x: x[1])[0]
    return None


def main():
    print("=" * 70)
    print("CONSISTENCY AS SELECTION STRATEGY EXPERIMENT")
    print("=" * 70)
    print("\nQuestion: Can we use consistency to SELECT better outputs?")
    print("\nExperiment:")
    print("  - Baseline: Random selection (mean accuracy across all runs)")
    print("  - Strategy: Select most consistent run (centroid)")
    print("  - Compare: Does consistency-based selection improve accuracy?")
    print("=" * 70)

    # Initialize evaluator
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")

    # Load results at T=1.0 (where there IS variation to select from)
    temperature = 1.0
    print(f"\nLoading generation results at T={temperature}...")
    toucan_dir = project_root / "llm_gen_results" / "toucan"
    gen_results = load_generation_results(toucan_dir, temperature)

    print(f"Found {len(gen_results)} models")

    # Results storage
    model_results = {}

    for model_name in tqdm(gen_results.keys(), desc="Models"):
        if not is_final_model(model_name):
            continue

        model_samples = gen_results[model_name]

        baseline_accuracies = []  # Mean accuracy (random selection)
        centroid_accuracies = []  # Accuracy of centroid run

        for sample_idx, sample_data in model_samples.items():
            gt = sample_data['ground_truth']
            runs = sample_data['generated_runs']

            if not gt or not runs:
                continue

            # Filter valid runs
            valid_runs = [(i, r) for i, r in enumerate(runs) if r]
            if len(valid_runs) < 2:
                continue

            # Compute accuracy for each run
            run_accuracies = []
            for i, run in valid_runs:
                acc = compute_accuracy(evaluator, gt, run)
                if acc is not None:
                    run_accuracies.append((i, acc))

            if not run_accuracies:
                continue

            # Baseline: Mean accuracy (equivalent to random selection)
            baseline_acc = np.mean([acc for _, acc in run_accuracies])
            baseline_accuracies.append(baseline_acc)

            # Consistency Strategy: Find centroid and get its accuracy
            centroid_idx = find_centroid_run(evaluator, runs)
            if centroid_idx is not None:
                centroid_acc = None
                for i, acc in run_accuracies:
                    if i == centroid_idx:
                        centroid_acc = acc
                        break
                if centroid_acc is not None:
                    centroid_accuracies.append(centroid_acc)

        if baseline_accuracies and centroid_accuracies:
            baseline_mean = np.mean(baseline_accuracies)
            centroid_mean = np.mean(centroid_accuracies)
            improvement = centroid_mean - baseline_mean
            pct_improvement = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0

            model_results[model_name] = {
                'baseline_accuracy': baseline_mean,
                'centroid_accuracy': centroid_mean,
                'improvement': improvement,
                'pct_improvement': pct_improvement,
                'n_samples': len(baseline_accuracies)
            }

    # Print results
    print("\n" + "=" * 90)
    print("RESULTS: Does Consistency-Based Selection Improve Accuracy?")
    print("=" * 90)
    print(f"\n{'Model':<30} {'Baseline':>12} {'Centroid':>12} {'Improvement':>12} {'% Change':>10}")
    print("-" * 90)

    sorted_results = sorted(model_results.items(), key=lambda x: x[1]['improvement'], reverse=True)

    for model_name, metrics in sorted_results:
        print(f"{model_name:<30} {metrics['baseline_accuracy']:>12.4f} {metrics['centroid_accuracy']:>12.4f} {metrics['improvement']:>+12.4f} {metrics['pct_improvement']:>+9.1f}%")

    # Summary
    if model_results:
        avg_baseline = np.mean([m['baseline_accuracy'] for m in model_results.values()])
        avg_centroid = np.mean([m['centroid_accuracy'] for m in model_results.values()])
        avg_improvement = avg_centroid - avg_baseline
        avg_pct = (avg_improvement / avg_baseline) * 100 if avg_baseline > 0 else 0

        print("-" * 90)
        print(f"{'AVERAGE':<30} {avg_baseline:>12.4f} {avg_centroid:>12.4f} {avg_improvement:>+12.4f} {avg_pct:>+9.1f}%")

        # Count improvements
        n_improved = sum(1 for m in model_results.values() if m['improvement'] > 0)
        n_total = len(model_results)

        print(f"\n{n_improved}/{n_total} models show improvement with consistency-based selection")

        # Save results
        output_dir = project_root / "results" / "accuracy_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            'experiment': 'consistency_selection_strategy',
            'description': 'Does selecting the most consistent run (centroid) improve accuracy vs random selection?',
            'temperature': temperature,
            'model_results': model_results,
            'summary': {
                'avg_baseline_accuracy': avg_baseline,
                'avg_centroid_accuracy': avg_centroid,
                'avg_improvement': avg_improvement,
                'avg_pct_improvement': avg_pct,
                'n_models_improved': n_improved,
                'n_models_total': n_total
            }
        }

        output_file = output_dir / "consistency_selection_strategy.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
