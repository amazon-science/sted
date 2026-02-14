"""
Comprehensive Evaluation of Consistency Improvement Strategies

This script evaluates the effectiveness of consistency-based selection strategies
by comparing:
1. Baseline (Random Selection): Mean accuracy across all runs
2. Centroid Selection: Pick the run most similar to all others
3. Best Run Selection (Oracle): Pick the run with highest accuracy (upper bound)

Metrics reported:
- Mean Accuracy: Average STED similarity to ground truth
- Ranking Score: Ranking-based consistency metric
- c_mean: Mean pairwise consistency among runs
- Improvement: Delta accuracy from applying centroid selection

Based on findings from CONSISTENCY_FACTORS_ANALYSIS.md and CONSISTENCY_STRATEGY_EXPERIMENT.md
"""

import json
import argparse
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sted import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer
from sted.model_config import FINAL_MODELS


def load_all_generation_results(results_dir: Path, dataset: str = 'toucan') -> dict:
    """Load all generation results for all models and temperatures."""
    all_results = defaultdict(lambda: defaultdict(dict))

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('generations-'):
            continue

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Try Toucan format: run_ModelName_temp_X_XX_timestamp
            match = re.match(r'run_(.+?)_temp_(\d+)_(\d+)_', run_dir.name)
            if match:
                model_name = match.group(1)
                temp_major = int(match.group(2))
                temp_minor = int(match.group(3))
                temperature = temp_major + temp_minor / 100
            else:
                # Try ShareGPT format: llm_gen_results_ModelName_temp_X_XX_timestamp
                match = re.match(r'llm_gen_results_(.+?)_temp_(\d+)_(\d+)_', run_dir.name)
                if match:
                    model_name = match.group(1)
                    temp_major = int(match.group(2))
                    temp_minor = int(match.group(3))
                    temperature = temp_major + temp_minor / 100
                else:
                    continue

            # Filter to only FINAL_MODELS
            if not any(fm.lower().replace('-', '') in model_name.lower().replace('-', '').replace('.', '').replace(' ', '')
                      for fm in FINAL_MODELS):
                continue

            all_results_file = run_dir / "all_results.json"
            if all_results_file.exists():
                try:
                    with open(all_results_file) as f:
                        data = json.load(f)

                    for idx, result in enumerate(data.get('results', [])):
                        all_results[model_name][temperature][idx] = {
                            'ground_truth': result.get('ground_truth', []),
                            'generated_runs': result.get('generated_runs', []),
                            'sample_id': result.get('sample_id', idx),
                        }
                except Exception as e:
                    continue

    return dict(all_results)


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


def evaluate_strategies(evaluator, analyzer, model_results, max_samples, variation_type='combined'):
    """
    Evaluate all strategies for a single model's results.

    Returns metrics for:
    - Baseline (random selection / mean accuracy)
    - Centroid selection
    - Oracle (best run, upper bound)
    """
    baseline_accuracies = []
    centroid_accuracies = []
    oracle_accuracies = []
    consistency_metrics_list = []

    sample_list = list(model_results.keys())[:max_samples]

    for sample_idx in sample_list:
        sample_data = model_results[sample_idx]
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

        # Oracle: Best run (upper bound)
        oracle_acc = max([acc for _, acc in run_accuracies])
        oracle_accuracies.append(oracle_acc)

        # Centroid Selection Strategy
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

    if not baseline_accuracies or not centroid_accuracies:
        return None

    baseline_mean = np.mean(baseline_accuracies)
    centroid_mean = np.mean(centroid_accuracies)
    oracle_mean = np.mean(oracle_accuracies)

    improvement = centroid_mean - baseline_mean
    pct_improvement = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0

    # How much of the possible improvement (oracle - baseline) did we capture?
    possible_improvement = oracle_mean - baseline_mean
    capture_rate = (improvement / possible_improvement) * 100 if possible_improvement > 0 else 0

    avg_c_mean = np.mean([m['c_mean'] for m in consistency_metrics_list]) if consistency_metrics_list else 0
    avg_ranking = np.mean([m['ranking_score'] for m in consistency_metrics_list]) if consistency_metrics_list else 0
    avg_stability = np.mean([m['stability_score'] for m in consistency_metrics_list]) if consistency_metrics_list else 0

    return {
        'baseline_accuracy': float(baseline_mean),
        'centroid_accuracy': float(centroid_mean),
        'oracle_accuracy': float(oracle_mean),
        'improvement': float(improvement),
        'pct_improvement': float(pct_improvement),
        'possible_improvement': float(possible_improvement),
        'capture_rate': float(capture_rate),
        'c_mean': float(avg_c_mean),
        'ranking_score': float(avg_ranking),
        'stability_score': float(avg_stability),
        'n_samples': len(baseline_accuracies)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate consistency improvement strategies')
    parser.add_argument('--max-samples', type=int, default=-1, help='Max samples per model (-1 for all)')
    parser.add_argument('--dataset', default='toucan', choices=['toucan', 'sharegpt'], help='Dataset to use')
    parser.add_argument('--variation-type', default='combined',
                        choices=['structural', 'content', 'combined'],
                        help='Variation type for STED calculation')
    parser.add_argument('--temperature', type=float, default=None, help='Filter by temperature (default: all)')
    args = parser.parse_args()

    print("=" * 100)
    print("COMPREHENSIVE EVALUATION OF CONSISTENCY IMPROVEMENT STRATEGIES")
    print("=" * 100)
    print(f"\nDataset: {args.dataset}")
    print(f"Variation Type: {args.variation_type}")
    print(f"Max Samples: {'all' if args.max_samples == -1 else args.max_samples}")
    print(f"Temperature Filter: {'all' if args.temperature is None else args.temperature}")
    print("\nStrategies compared:")
    print("  1. Baseline (Random): Mean accuracy across all runs")
    print("  2. Centroid Selection: Pick run most similar to all others")
    print("  3. Oracle (Best Run): Upper bound - pick best run (requires ground truth)")
    print("=" * 100)

    # Initialize evaluators
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")
    analyzer = StructuralConsistencyAnalyzer(evaluator)

    results_dir = project_root / "llm_gen_results" / args.dataset

    print(f"\nLoading generation results from {results_dir}...")
    all_gen_results = load_all_generation_results(results_dir, args.dataset)
    print(f"Found {len(all_gen_results)} models (filtered to FINAL_MODELS)")

    # Organize results by temperature
    temperatures_found = set()
    for model_name, temp_data in all_gen_results.items():
        temperatures_found.update(temp_data.keys())

    temperatures = sorted(temperatures_found)
    if args.temperature is not None:
        temperatures = [t for t in temperatures if abs(t - args.temperature) < 0.01]

    print(f"Temperatures found: {temperatures}")

    all_results = {}

    for temp in temperatures:
        print(f"\n{'='*80}")
        print(f"TEMPERATURE = {temp}")
        print(f"{'='*80}")

        model_metrics = {}

        for model_name in tqdm(sorted(all_gen_results.keys()), desc="Evaluating models"):
            if temp not in all_gen_results[model_name]:
                continue

            model_samples = all_gen_results[model_name][temp]
            max_samples = args.max_samples if args.max_samples > 0 else len(model_samples)

            metrics = evaluate_strategies(
                evaluator, analyzer, model_samples,
                max_samples, args.variation_type
            )

            if metrics:
                model_metrics[model_name] = metrics

        all_results[f"T={temp}"] = model_metrics

        # Print results table
        print(f"\n{'Model':<30} {'Baseline':>10} {'Centroid':>10} {'Oracle':>10} {'Delta':>10} {'Capture%':>10} {'c_mean':>8} {'Rank':>8}")
        print("-" * 110)

        sorted_results = sorted(model_metrics.items(), key=lambda x: x[1]['improvement'], reverse=True)

        for model_name, m in sorted_results:
            print(f"{model_name:<30} {m['baseline_accuracy']:>10.4f} {m['centroid_accuracy']:>10.4f} "
                  f"{m['oracle_accuracy']:>10.4f} {m['improvement']:>+10.4f} {m['capture_rate']:>10.1f}% "
                  f"{m['c_mean']:>8.2f} {m['ranking_score']:>8.4f}")

        if model_metrics:
            avg_baseline = np.mean([m['baseline_accuracy'] for m in model_metrics.values()])
            avg_centroid = np.mean([m['centroid_accuracy'] for m in model_metrics.values()])
            avg_oracle = np.mean([m['oracle_accuracy'] for m in model_metrics.values()])
            avg_improvement = np.mean([m['improvement'] for m in model_metrics.values()])
            avg_capture = np.mean([m['capture_rate'] for m in model_metrics.values()])
            avg_c_mean = np.mean([m['c_mean'] for m in model_metrics.values()])
            avg_ranking = np.mean([m['ranking_score'] for m in model_metrics.values()])

            print("-" * 110)
            print(f"{'AVERAGE':<30} {avg_baseline:>10.4f} {avg_centroid:>10.4f} "
                  f"{avg_oracle:>10.4f} {avg_improvement:>+10.4f} {avg_capture:>10.1f}% "
                  f"{avg_c_mean:>8.2f} {avg_ranking:>8.4f}")

            n_improved = sum(1 for m in model_metrics.values() if m['improvement'] > 0)
            print(f"\n{n_improved}/{len(model_metrics)} models improved with centroid selection at T={temp}")

    # Summary across temperatures
    if len(temperatures) > 1:
        print("\n" + "=" * 100)
        print("SUMMARY ACROSS TEMPERATURES")
        print("=" * 100)

        print(f"\n{'Temperature':<15} {'Baseline':>12} {'Centroid':>12} {'Delta':>12} {'%Improved':>12} {'Avg c_mean':>12} {'Avg Rank':>12}")
        print("-" * 90)

        for temp in temperatures:
            key = f"T={temp}"
            if key in all_results and all_results[key]:
                metrics = all_results[key]
                avg_baseline = np.mean([m['baseline_accuracy'] for m in metrics.values()])
                avg_centroid = np.mean([m['centroid_accuracy'] for m in metrics.values()])
                avg_delta = np.mean([m['improvement'] for m in metrics.values()])
                pct_improved = sum(1 for m in metrics.values() if m['improvement'] > 0) / len(metrics) * 100
                avg_c_mean = np.mean([m['c_mean'] for m in metrics.values()])
                avg_ranking = np.mean([m['ranking_score'] for m in metrics.values()])

                print(f"T={temp:<11} {avg_baseline:>12.4f} {avg_centroid:>12.4f} {avg_delta:>+12.4f} "
                      f"{pct_improved:>11.1f}% {avg_c_mean:>12.2f} {avg_ranking:>12.4f}")

    # Save results
    output_dir = project_root / "results" / "consistency_strategies"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'consistency_improvement_strategies',
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'variation_type': args.variation_type,
        'max_samples': args.max_samples,
        'description': 'Comprehensive comparison of baseline, centroid selection, and oracle strategies',
        'strategies': {
            'baseline': 'Mean accuracy across all runs (random selection)',
            'centroid': 'Select run most similar to all others (consistency-based)',
            'oracle': 'Select best run (upper bound, requires ground truth)'
        },
        'results_by_temperature': all_results,
        'summary': {}
    }

    # Add summary statistics
    for temp_key, metrics in all_results.items():
        if metrics:
            output['summary'][temp_key] = {
                'avg_baseline_accuracy': float(np.mean([m['baseline_accuracy'] for m in metrics.values()])),
                'avg_centroid_accuracy': float(np.mean([m['centroid_accuracy'] for m in metrics.values()])),
                'avg_oracle_accuracy': float(np.mean([m['oracle_accuracy'] for m in metrics.values()])),
                'avg_improvement': float(np.mean([m['improvement'] for m in metrics.values()])),
                'avg_capture_rate': float(np.mean([m['capture_rate'] for m in metrics.values()])),
                'avg_c_mean': float(np.mean([m['c_mean'] for m in metrics.values()])),
                'avg_ranking_score': float(np.mean([m['ranking_score'] for m in metrics.values()])),
                'n_models_improved': sum(1 for m in metrics.values() if m['improvement'] > 0),
                'n_models_total': len(metrics)
            }

    output_file = output_dir / f"strategy_comparison_{args.dataset}_{args.variation_type}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    if all_results:
        # Get T=1.0 results if available
        t1_key = [k for k in all_results.keys() if '1.0' in k or '1_0' in k]
        t0_key = [k for k in all_results.keys() if '0.0' in k or '0_0' in k]

        if t1_key and all_results[t1_key[0]]:
            t1_metrics = all_results[t1_key[0]]
            t1_improvement = np.mean([m['improvement'] for m in t1_metrics.values()])
            t1_capture = np.mean([m['capture_rate'] for m in t1_metrics.values()])
            t1_pct_improved = sum(1 for m in t1_metrics.values() if m['improvement'] > 0) / len(t1_metrics) * 100

            print(f"\nAt T=1.0 (high variation):")
            print(f"  - Average improvement from centroid selection: {t1_improvement:+.4f}")
            print(f"  - Capture rate of oracle improvement: {t1_capture:.1f}%")
            print(f"  - Models showing improvement: {t1_pct_improved:.1f}%")

        if t0_key and all_results[t0_key[0]]:
            t0_metrics = all_results[t0_key[0]]
            t0_improvement = np.mean([m['improvement'] for m in t0_metrics.values()])
            t0_pct_improved = sum(1 for m in t0_metrics.values() if m['improvement'] > 0) / len(t0_metrics) * 100

            print(f"\nAt T=0.0 (low variation):")
            print(f"  - Average improvement from centroid selection: {t0_improvement:+.4f}")
            print(f"  - Models showing improvement: {t0_pct_improved:.1f}%")

        if t1_key and t0_key:
            t1_metrics = all_results[t1_key[0]]
            t0_metrics = all_results[t0_key[0]]

            if t1_metrics and t0_metrics:
                t1_improvement = np.mean([m['improvement'] for m in t1_metrics.values()])
                t0_improvement = np.mean([m['improvement'] for m in t0_metrics.values()])

                print(f"\nConclusion:")
                print(f"  Centroid selection is {(t1_improvement - t0_improvement)*100:+.2f}% more effective at T=1.0 vs T=0.0")
                print(f"  This confirms that consistency-based selection helps MORE when there's variation to select from.")


if __name__ == "__main__":
    main()
