"""
Compare Accuracy and Ranking Score Before/After Consistency Strategy

Experiment Design:
- Baseline (T=1.0): High temperature = low consistency
  - Calculate mean accuracy (STED similarity to ground truth)
  - Calculate ranking_score = r_v * c_mean * stability_score

- With Consistency Strategy (T=0.0): Low temperature = high consistency
  - Calculate mean accuracy
  - Calculate ranking_score

- Compare: Delta accuracy and delta ranking_score

This uses EXISTING generation data at different temperatures.
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
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer


# Claude models to focus on initially
CLAUDE_MODELS = [
    "Claude-Sonnet-4",
    "Claude-Sonnet-4.5",
    "Claude-Haiku-4.5",
    "Claude-Opus-4.5",
    "Claude-3.5-Sonnet",
    "Claude-3.5-Haiku",
    "Claude-3.7-Sonnet",
]


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


def compute_sample_metrics(evaluator, analyzer, ground_truth, generated_runs):
    """
    Compute accuracy and consistency metrics for a single sample.

    Returns:
        dict with accuracy_mean, c_mean, stability_score, ranking_score, validity_rate
    """
    if not generated_runs:
        return None

    # Wrap ground truth for STED calculation
    gt_wrapped = {"tool_calls": ground_truth} if ground_truth else None

    # Calculate accuracy for each run (STED similarity to ground truth)
    accuracies = []
    valid_runs_wrapped = []

    for run in generated_runs:
        if run:
            run_wrapped = {"tool_calls": run}
            valid_runs_wrapped.append(run_wrapped)

            if gt_wrapped:
                try:
                    sim = evaluator.calculate_tree_edit_distance_fast(
                        gt_wrapped, run_wrapped, variation_type='combined'
                    )
                    accuracies.append(sim)
                except:
                    pass

    if not valid_runs_wrapped:
        return None

    # Calculate consistency metrics using the analyzer
    consistency_result = analyzer.evaluate_structural_consistency(
        valid_runs_wrapped,
        variation_type='combined'
    )

    # Extract metrics from nested consistency_metrics dict
    metrics = consistency_result.get('consistency_metrics', {})

    return {
        'accuracy_mean': float(np.mean(accuracies)) if accuracies else 0.0,
        'accuracy_std': float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
        'c_mean': metrics.get('c_mean', 0.0),
        'stability_score': metrics.get('stability_score', 0.0),
        'ranking_score': metrics.get('ranking_score', 0.0),
        'validity_rate': metrics.get('r_v', 0.0),
        'num_valid_runs': len(valid_runs_wrapped),
        'num_accuracies': len(accuracies)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples per model')
    parser.add_argument('--all-models', action='store_true', help='Run on all models')
    parser.add_argument('--dataset', default='toucan', choices=['toucan', 'sharegpt'], help='Dataset to use')
    args = parser.parse_args()

    print("=" * 80)
    print("CONSISTENCY STRATEGY COMPARISON: T=1.0 (Baseline) vs T=0.0 (Consistency)")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Max samples per model: {args.max_samples}")
    print(f"Models: {'All' if args.all_models else 'Claude only'}")

    # Initialize evaluators
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")
    analyzer = StructuralConsistencyAnalyzer(evaluator)

    # Load data at both temperatures
    results_dir = project_root / "llm_gen_results" / args.dataset

    print(f"\nLoading T=1.0 (baseline) data...")
    baseline_data = load_generation_results(results_dir, temperature=1.0)
    print(f"Found {len(baseline_data)} models at T=1.0")

    print(f"\nLoading T=0.0 (consistency strategy) data...")
    strategy_data = load_generation_results(results_dir, temperature=0.0)
    print(f"Found {len(strategy_data)} models at T=0.0")

    # Find common models
    common_models = set(baseline_data.keys()) & set(strategy_data.keys())
    if not args.all_models:
        common_models = {m for m in common_models if any(c.lower() in m.lower() for c in CLAUDE_MODELS)}

    print(f"\nProcessing {len(common_models)} models: {sorted(common_models)}")

    # Process each model
    results = {}

    for model_name in tqdm(sorted(common_models), desc="Models"):
        baseline_samples = baseline_data[model_name]
        strategy_samples = strategy_data[model_name]

        # Find common sample indices
        common_indices = set(baseline_samples.keys()) & set(strategy_samples.keys())
        sample_list = list(common_indices)[:args.max_samples]

        baseline_metrics = []
        strategy_metrics = []

        for sample_idx in tqdm(sample_list, desc=f"{model_name}", leave=False):
            # Baseline (T=1.0)
            bl_gt = baseline_samples[sample_idx]['ground_truth']
            bl_runs = baseline_samples[sample_idx]['generated_runs']
            bl_result = compute_sample_metrics(evaluator, analyzer, bl_gt, bl_runs)
            if bl_result:
                baseline_metrics.append(bl_result)

            # Strategy (T=0.0)
            st_gt = strategy_samples[sample_idx]['ground_truth']
            st_runs = strategy_samples[sample_idx]['generated_runs']
            st_result = compute_sample_metrics(evaluator, analyzer, st_gt, st_runs)
            if st_result:
                strategy_metrics.append(st_result)

        if baseline_metrics and strategy_metrics:
            results[model_name] = {
                'baseline': {
                    'accuracy_mean': float(np.mean([m['accuracy_mean'] for m in baseline_metrics])),
                    'accuracy_std': float(np.mean([m['accuracy_std'] for m in baseline_metrics])),
                    'c_mean': float(np.mean([m['c_mean'] for m in baseline_metrics])),
                    'stability_score': float(np.mean([m['stability_score'] for m in baseline_metrics])),
                    'ranking_score': float(np.mean([m['ranking_score'] for m in baseline_metrics])),
                    'validity_rate': float(np.mean([m['validity_rate'] for m in baseline_metrics])),
                    'n_samples': len(baseline_metrics)
                },
                'strategy': {
                    'accuracy_mean': float(np.mean([m['accuracy_mean'] for m in strategy_metrics])),
                    'accuracy_std': float(np.mean([m['accuracy_std'] for m in strategy_metrics])),
                    'c_mean': float(np.mean([m['c_mean'] for m in strategy_metrics])),
                    'stability_score': float(np.mean([m['stability_score'] for m in strategy_metrics])),
                    'ranking_score': float(np.mean([m['ranking_score'] for m in strategy_metrics])),
                    'validity_rate': float(np.mean([m['validity_rate'] for m in strategy_metrics])),
                    'n_samples': len(strategy_metrics)
                }
            }

            # Calculate deltas
            results[model_name]['delta'] = {
                'accuracy': results[model_name]['strategy']['accuracy_mean'] - results[model_name]['baseline']['accuracy_mean'],
                'c_mean': results[model_name]['strategy']['c_mean'] - results[model_name]['baseline']['c_mean'],
                'ranking_score': results[model_name]['strategy']['ranking_score'] - results[model_name]['baseline']['ranking_score'],
            }

    # Print results
    print("\n" + "=" * 120)
    print("RESULTS: Baseline (T=1.0) vs Consistency Strategy (T=0.0)")
    print("=" * 120)

    print(f"\n{'Model':<25} | {'Baseline T=1.0':^35} | {'Strategy T=0.0':^35} | {'Delta':^20}")
    print(f"{'':25} | {'Acc':>10} {'c_mean':>10} {'R_score':>10} | {'Acc':>10} {'c_mean':>10} {'R_score':>10} | {'Δ Acc':>8} {'Δ R':>8}")
    print("-" * 120)

    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['delta']['accuracy'], reverse=True):
        bl = metrics['baseline']
        st = metrics['strategy']
        delta = metrics['delta']

        print(f"{model_name:<25} | {bl['accuracy_mean']:>10.4f} {bl['c_mean']:>10.4f} {bl['ranking_score']:>10.4f} | "
              f"{st['accuracy_mean']:>10.4f} {st['c_mean']:>10.4f} {st['ranking_score']:>10.4f} | "
              f"{delta['accuracy']:>+8.4f} {delta['ranking_score']:>+8.4f}")

    # Summary
    if results:
        avg_delta_acc = np.mean([r['delta']['accuracy'] for r in results.values()])
        avg_delta_r = np.mean([r['delta']['ranking_score'] for r in results.values()])

        print("-" * 120)
        print(f"{'AVERAGE':<25} | {'':<35} | {'':<35} | {avg_delta_acc:>+8.4f} {avg_delta_r:>+8.4f}")

        n_improved_acc = sum(1 for r in results.values() if r['delta']['accuracy'] > 0)
        n_improved_r = sum(1 for r in results.values() if r['delta']['ranking_score'] > 0)
        n_total = len(results)

        print(f"\nAccuracy improved: {n_improved_acc}/{n_total} models")
        print(f"Ranking score improved: {n_improved_r}/{n_total} models")

        # Save results
        output_dir = project_root / "results" / "accuracy_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            'experiment': 'consistency_strategy_comparison',
            'description': 'Compare Baseline (T=1.0) vs Consistency Strategy (T=0.0)',
            'dataset': args.dataset,
            'max_samples': args.max_samples,
            'model_results': results,
            'summary': {
                'avg_delta_accuracy': float(avg_delta_acc),
                'avg_delta_ranking_score': float(avg_delta_r),
                'n_models_improved_accuracy': n_improved_acc,
                'n_models_improved_ranking': n_improved_r,
                'n_models_total': n_total
            }
        }

        output_file = output_dir / f"consistency_strategy_comparison_{args.dataset}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
