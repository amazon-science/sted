"""
Evaluate Centroid Selection Strategy at Multiple Temperatures

Experiment Design:
- For each temperature (T=0.0 and T=1.0):
  - Baseline: Random selection (mean accuracy across all runs)
  - Strategy: Centroid selection (pick the run most similar to all others)
  - Compare: Delta accuracy from applying centroid selection

This properly isolates the consistency strategy (centroid selection) from temperature.
Temperature is NOT the strategy - it's a control variable.
"""

import json
import argparse
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
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer


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


def compute_accuracy(evaluator, ground_truth, run, variation_type='combined'):
    """Compute accuracy (STED similarity) between a single run and ground truth."""
    if not ground_truth or not run:
        return None
    try:
        gt_wrapped = {"tool_calls": ground_truth}
        run_wrapped = {"tool_calls": run}
        return evaluator.calculate_tree_edit_distance_fast(gt_wrapped, run_wrapped, variation_type=variation_type)
    except:
        return None


def compute_pairwise_similarity(evaluator, run1, run2, variation_type='combined'):
    """Compute STED similarity between two runs."""
    if not run1 or not run2:
        return None
    try:
        r1_wrapped = {"tool_calls": run1}
        r2_wrapped = {"tool_calls": run2}
        return evaluator.calculate_tree_edit_distance_fast(r1_wrapped, r2_wrapped, variation_type=variation_type)
    except:
        return None


def find_centroid_run(evaluator, runs, variation_type='combined'):
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
                sim = compute_pairwise_similarity(evaluator, run_i, run_j, variation_type)
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


def compute_consistency_metrics(evaluator, analyzer, runs, variation_type='combined'):
    """Compute consistency metrics for a set of runs."""
    valid_runs_wrapped = []
    for run in runs:
        if run:
            valid_runs_wrapped.append({"tool_calls": run})

    if len(valid_runs_wrapped) < 2:
        return {'c_mean': 1.0, 'stability_score': 1.0, 'ranking_score': 1.0}

    try:
        consistency_result = analyzer.evaluate_structural_consistency(
            valid_runs_wrapped,
            variation_type=variation_type
        )
        metrics = consistency_result.get('consistency_metrics', {})
        return {
            'c_mean': metrics.get('c_mean', 0.0),
            'stability_score': metrics.get('stability_score', 0.0),
            'ranking_score': metrics.get('ranking_score', 0.0),
        }
    except:
        return {'c_mean': 0.0, 'stability_score': 0.0, 'ranking_score': 0.0}


def evaluate_at_temperature(evaluator, analyzer, gen_results, max_samples, variation_type='combined'):
    """Evaluate centroid selection strategy at a specific temperature."""
    model_results = {}

    for model_name in tqdm(gen_results.keys(), desc="Models"):
        model_samples = gen_results[model_name]

        baseline_accuracies = []  # Mean accuracy (random selection)
        centroid_accuracies = []  # Accuracy of centroid run
        consistency_metrics_list = []

        sample_list = list(model_samples.keys())[:max_samples]

        for sample_idx in sample_list:
            sample_data = model_samples[sample_idx]
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
                acc = compute_accuracy(evaluator, gt, run, variation_type)
                if acc is not None:
                    run_accuracies.append((i, acc))

            if not run_accuracies:
                continue

            # Baseline: Mean accuracy (equivalent to random selection)
            baseline_acc = np.mean([acc for _, acc in run_accuracies])
            baseline_accuracies.append(baseline_acc)

            # Consistency Strategy: Find centroid and get its accuracy
            centroid_idx = find_centroid_run(evaluator, runs, variation_type)
            if centroid_idx is not None:
                centroid_acc = None
                for i, acc in run_accuracies:
                    if i == centroid_idx:
                        centroid_acc = acc
                        break
                if centroid_acc is not None:
                    centroid_accuracies.append(centroid_acc)

            # Compute consistency metrics for this sample
            cons_metrics = compute_consistency_metrics(evaluator, analyzer, runs, variation_type)
            consistency_metrics_list.append(cons_metrics)

        if baseline_accuracies and centroid_accuracies:
            baseline_mean = np.mean(baseline_accuracies)
            centroid_mean = np.mean(centroid_accuracies)
            improvement = centroid_mean - baseline_mean
            pct_improvement = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0

            avg_c_mean = np.mean([m['c_mean'] for m in consistency_metrics_list]) if consistency_metrics_list else 0
            avg_ranking = np.mean([m['ranking_score'] for m in consistency_metrics_list]) if consistency_metrics_list else 0

            model_results[model_name] = {
                'baseline_accuracy': float(baseline_mean),
                'centroid_accuracy': float(centroid_mean),
                'improvement': float(improvement),
                'pct_improvement': float(pct_improvement),
                'c_mean': float(avg_c_mean),
                'ranking_score': float(avg_ranking),
                'n_samples': len(baseline_accuracies)
            }

    return model_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples per model')
    parser.add_argument('--dataset', default='toucan', choices=['toucan', 'sharegpt'], help='Dataset to use')
    parser.add_argument('--variation-type', default='all', choices=['structural', 'content', 'combined', 'all'],
                        help='Variation type to use (default: all)')
    args = parser.parse_args()

    print("=" * 90)
    print("CENTROID SELECTION STRATEGY AT MULTIPLE TEMPERATURES")
    print("=" * 90)
    print("\nStrategy: Select the run most similar to all others (centroid)")
    print("Control Variable: Temperature (T=0.0 and T=1.0)")
    print("\nFor each temperature, compare:")
    print("  - Baseline: Random selection (mean accuracy)")
    print("  - Strategy: Centroid selection")
    print("=" * 90)

    # Initialize evaluators
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")
    analyzer = StructuralConsistencyAnalyzer(evaluator)

    results_dir = project_root / "llm_gen_results" / args.dataset
    temperatures = [0.0, 1.0]

    # Determine which variation types to run
    if args.variation_type == 'all':
        variation_types = ['structural', 'content', 'combined']
    else:
        variation_types = [args.variation_type]

    # Store all results by variation type
    all_variation_results = {}

    for variation_type in variation_types:
        print(f"\n{'#'*90}")
        print(f"VARIATION TYPE: {variation_type.upper()}")
        print(f"{'#'*90}")

        all_results = {}

        for temp in temperatures:
            print(f"\n{'='*60}")
            print(f"TEMPERATURE = {temp} | VARIATION = {variation_type}")
            print(f"{'='*60}")

            print(f"\nLoading generation results at T={temp}...")
            gen_results = load_generation_results(results_dir, temp)
            print(f"Found {len(gen_results)} models")

            model_results = evaluate_at_temperature(evaluator, analyzer, gen_results, args.max_samples, variation_type)
            all_results[f"T={temp}"] = model_results

            # Print results for this temperature
            print(f"\n{'Model':<30} {'Baseline':>10} {'Centroid':>10} {'Delta':>10} {'c_mean':>8}")
            print("-" * 75)

            sorted_results = sorted(model_results.items(), key=lambda x: x[1]['improvement'], reverse=True)
            for model_name, metrics in sorted_results:
                print(f"{model_name:<30} {metrics['baseline_accuracy']:>10.4f} {metrics['centroid_accuracy']:>10.4f} "
                      f"{metrics['improvement']:>+10.4f} {metrics['c_mean']:>8.4f}")

            if model_results:
                avg_baseline = np.mean([m['baseline_accuracy'] for m in model_results.values()])
                avg_centroid = np.mean([m['centroid_accuracy'] for m in model_results.values()])
                avg_improvement = avg_centroid - avg_baseline
                avg_c_mean = np.mean([m['c_mean'] for m in model_results.values()])

                print("-" * 75)
                print(f"{'AVERAGE':<30} {avg_baseline:>10.4f} {avg_centroid:>10.4f} "
                      f"{avg_improvement:>+10.4f} {avg_c_mean:>8.4f}")

                n_improved = sum(1 for m in model_results.values() if m['improvement'] > 0)
                print(f"\n{n_improved}/{len(model_results)} models improved with centroid selection at T={temp}")

        # Cross-temperature comparison for this variation type
        print("\n" + "=" * 90)
        print(f"CROSS-TEMPERATURE COMPARISON ({variation_type.upper()})")
        print("=" * 90)

        t0_results = all_results.get("T=0.0", {})
        t1_results = all_results.get("T=1.0", {})
        common_models = set(t0_results.keys()) & set(t1_results.keys())

        print(f"\n{'Model':<25} | {'T=1.0 (High Var)':^30} | {'T=0.0 (Low Var)':^30}")
        print(f"{'':25} | {'Base':>8} {'Cent':>8} {'Delta':>8} | {'Base':>8} {'Cent':>8} {'Delta':>8}")
        print("-" * 90)

        for model_name in sorted(common_models):
            t0 = t0_results[model_name]
            t1 = t1_results[model_name]
            print(f"{model_name:<25} | {t1['baseline_accuracy']:>8.4f} {t1['centroid_accuracy']:>8.4f} {t1['improvement']:>+8.4f} | "
                  f"{t0['baseline_accuracy']:>8.4f} {t0['centroid_accuracy']:>8.4f} {t0['improvement']:>+8.4f}")

        # Summary
        if t0_results and t1_results:
            t0_avg_delta = np.mean([m['improvement'] for m in t0_results.values()])
            t1_avg_delta = np.mean([m['improvement'] for m in t1_results.values()])

            print("-" * 90)
            print(f"\nAverage improvement from centroid selection ({variation_type}):")
            print(f"  At T=1.0 (high variation): {t1_avg_delta:+.4f}")
            print(f"  At T=0.0 (low variation):  {t0_avg_delta:+.4f}")
            print(f"\nDifference: {t1_avg_delta - t0_avg_delta:+.4f}")
            print("(Positive means centroid selection helps MORE at high temperature)")

        # Store results for this variation type
        all_variation_results[variation_type] = {
            'results_by_temperature': all_results,
            'summary': {
                'T=1.0': {
                    'avg_improvement': float(np.mean([m['improvement'] for m in t1_results.values()])) if t1_results else 0,
                    'n_improved': sum(1 for m in t1_results.values() if m['improvement'] > 0) if t1_results else 0,
                    'n_total': len(t1_results)
                },
                'T=0.0': {
                    'avg_improvement': float(np.mean([m['improvement'] for m in t0_results.values()])) if t0_results else 0,
                    'n_improved': sum(1 for m in t0_results.values() if m['improvement'] > 0) if t0_results else 0,
                    'n_total': len(t0_results)
                }
            }
        }

    # Print final summary comparing all variation types
    if len(variation_types) > 1:
        print("\n" + "=" * 90)
        print("SUMMARY BY VARIATION TYPE")
        print("=" * 90)
        print(f"\n{'Variation Type':<15} | {'T=1.0 Avg Delta':>15} {'Improved':>12} | {'T=0.0 Avg Delta':>15} {'Improved':>12}")
        print("-" * 75)
        for vt in variation_types:
            s = all_variation_results[vt]['summary']
            print(f"{vt:<15} | {s['T=1.0']['avg_improvement']:>+15.4f} {s['T=1.0']['n_improved']}/{s['T=1.0']['n_total']:>8} | "
                  f"{s['T=0.0']['avg_improvement']:>+15.4f} {s['T=0.0']['n_improved']}/{s['T=0.0']['n_total']:>8}")

    # Save results
    output_dir = project_root / "results" / "accuracy_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'centroid_selection_multi_temp',
        'description': 'Centroid selection strategy evaluated at T=0.0 and T=1.0 for each variation type',
        'dataset': args.dataset,
        'max_samples': args.max_samples,
        'results_by_variation_type': all_variation_results
    }

    output_file = output_dir / f"centroid_selection_multi_temp_{args.dataset}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
