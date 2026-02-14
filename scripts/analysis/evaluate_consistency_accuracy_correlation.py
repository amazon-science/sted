"""
Evaluate Correlation Between Consistency and Accuracy

Research Question: Does higher consistency correlate with higher accuracy?

For each sample:
- Compute consistency (c_mean) across N runs
- Compute accuracy (mean STED similarity to ground truth)
- Calculate correlation between consistency and accuracy

This directly tests whether STED consistency is a useful signal for quality.
"""

import json
import argparse
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scipy import stats

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


def compute_sample_metrics(evaluator, analyzer, ground_truth, generated_runs, variation_type='combined'):
    """
    Compute consistency and accuracy for a single sample.

    Returns:
        dict with c_mean (consistency) and accuracy (mean similarity to ground truth)
    """
    if not generated_runs or not ground_truth:
        return None

    gt_wrapped = {"tool_calls": ground_truth}

    # Get valid runs and compute accuracy for each
    accuracies = []
    valid_runs_wrapped = []

    for run in generated_runs:
        if run:
            run_wrapped = {"tool_calls": run}
            valid_runs_wrapped.append(run_wrapped)

            try:
                sim = evaluator.calculate_tree_edit_distance_fast(
                    gt_wrapped, run_wrapped, variation_type=variation_type
                )
                accuracies.append(sim)
            except:
                pass

    if len(valid_runs_wrapped) < 2 or not accuracies:
        return None

    # Compute consistency (c_mean) across runs
    try:
        consistency_result = analyzer.evaluate_structural_consistency(
            valid_runs_wrapped,
            variation_type=variation_type
        )
        metrics = consistency_result.get('consistency_metrics', {})
        c_mean = metrics.get('c_mean', 0.0)
    except:
        return None

    return {
        'c_mean': float(c_mean),
        'accuracy': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
        'n_runs': len(valid_runs_wrapped)
    }


def evaluate_correlation(evaluator, analyzer, gen_results, max_samples, variation_type='combined'):
    """Evaluate correlation between consistency and accuracy for all models."""
    model_results = {}

    for model_name in tqdm(gen_results.keys(), desc="Models"):
        model_samples = gen_results[model_name]

        consistencies = []
        accuracies = []
        sample_metrics = []

        sample_list = list(model_samples.keys())[:max_samples]

        for sample_idx in sample_list:
            sample_data = model_samples[sample_idx]
            gt = sample_data['ground_truth']
            runs = sample_data['generated_runs']

            result = compute_sample_metrics(evaluator, analyzer, gt, runs, variation_type)
            if result:
                consistencies.append(result['c_mean'])
                accuracies.append(result['accuracy'])
                sample_metrics.append(result)

        if len(consistencies) >= 10:  # Need enough samples for meaningful correlation
            # Compute Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(consistencies, accuracies)
            # Compute Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = stats.spearmanr(consistencies, accuracies)

            model_results[model_name] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'mean_consistency': float(np.mean(consistencies)),
                'mean_accuracy': float(np.mean(accuracies)),
                'std_consistency': float(np.std(consistencies)),
                'std_accuracy': float(np.std(accuracies)),
                'n_samples': len(consistencies)
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
    print("CONSISTENCY-ACCURACY CORRELATION ANALYSIS")
    print("=" * 90)
    print("\nResearch Question: Does higher consistency correlate with higher accuracy?")
    print("For each sample: compute c_mean and accuracy, then calculate correlation.")
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

    all_results = {}

    for variation_type in variation_types:
        print(f"\n{'#'*90}")
        print(f"VARIATION TYPE: {variation_type.upper()}")
        print(f"{'#'*90}")

        variation_results = {}

        for temp in temperatures:
            print(f"\n{'='*60}")
            print(f"TEMPERATURE = {temp} | VARIATION = {variation_type}")
            print(f"{'='*60}")

            print(f"\nLoading generation results at T={temp}...")
            gen_results = load_generation_results(results_dir, temp)
            print(f"Found {len(gen_results)} models")

            model_results = evaluate_correlation(evaluator, analyzer, gen_results, args.max_samples, variation_type)
            variation_results[f"T={temp}"] = model_results

            # Print results
            print(f"\n{'Model':<30} {'Pearson r':>12} {'p-value':>12} {'Spearman r':>12} {'p-value':>12}")
            print("-" * 80)

            sorted_results = sorted(model_results.items(), key=lambda x: x[1]['pearson_r'], reverse=True)
            for model_name, metrics in sorted_results:
                sig_p = "*" if metrics['pearson_p'] < 0.05 else ""
                sig_s = "*" if metrics['spearman_p'] < 0.05 else ""
                print(f"{model_name:<30} {metrics['pearson_r']:>11.4f}{sig_p} {metrics['pearson_p']:>12.4f} "
                      f"{metrics['spearman_r']:>11.4f}{sig_s} {metrics['spearman_p']:>12.4f}")

            if model_results:
                avg_pearson = np.mean([m['pearson_r'] for m in model_results.values()])
                avg_spearman = np.mean([m['spearman_r'] for m in model_results.values()])
                n_sig_pearson = sum(1 for m in model_results.values() if m['pearson_p'] < 0.05)
                n_sig_spearman = sum(1 for m in model_results.values() if m['spearman_p'] < 0.05)

                print("-" * 80)
                print(f"{'AVERAGE':<30} {avg_pearson:>12.4f} {'':>12} {avg_spearman:>12.4f}")
                print(f"\nSignificant correlations (p<0.05): Pearson={n_sig_pearson}/{len(model_results)}, "
                      f"Spearman={n_sig_spearman}/{len(model_results)}")

        all_results[variation_type] = variation_results

    # Summary across all variation types
    print("\n" + "=" * 90)
    print("SUMMARY: CONSISTENCY-ACCURACY CORRELATION")
    print("=" * 90)

    print(f"\n{'Variation':<15} {'Temp':>8} {'Avg Pearson r':>15} {'Avg Spearman r':>15} {'Sig. (p<0.05)':>15}")
    print("-" * 70)

    for vt in variation_types:
        for temp_key in ['T=0.0', 'T=1.0']:
            if temp_key in all_results[vt]:
                results = all_results[vt][temp_key]
                if results:
                    avg_p = np.mean([m['pearson_r'] for m in results.values()])
                    avg_s = np.mean([m['spearman_r'] for m in results.values()])
                    n_sig = sum(1 for m in results.values() if m['pearson_p'] < 0.05)
                    print(f"{vt:<15} {temp_key:>8} {avg_p:>15.4f} {avg_s:>15.4f} {n_sig}/{len(results):>14}")

    # Save results
    output_dir = project_root / "results" / "accuracy_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'consistency_accuracy_correlation',
        'description': 'Correlation between consistency (c_mean) and accuracy across samples',
        'dataset': args.dataset,
        'max_samples': args.max_samples,
        'results_by_variation_type': all_results
    }

    output_file = output_dir / f"consistency_accuracy_correlation_{args.dataset}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
