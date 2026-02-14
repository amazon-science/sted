"""
Evaluate Accuracy vs Consistency at Multiple Temperatures

This script measures REAL accuracy (STED similarity to ground truth) at different
temperatures to verify that adopting consistency strategies (lower T) improves accuracy.

Temperatures: 0.0, 0.3, 0.6, 1.0
"""

import json
import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sted import SemanticJsonTreeConsistencyEvaluator
from sted.model_config import FINAL_MODELS, INVALID_SHAREGPT_SAMPLES


def load_generation_results(results_dir: Path, temperature: float, dataset_name: str = "") -> dict:
    """Load generation results for a specific temperature."""
    all_results = {}

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('generations-'):
            continue

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Extract model name and temperature from directory name
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
                            'dataset': dataset_name
                        }
                    break
                except Exception as e:
                    continue

    return all_results


def compute_accuracy(evaluator, ground_truth, generated_runs) -> dict:
    """Compute accuracy metrics."""
    if not ground_truth or not generated_runs:
        return {'accuracy_mean': 0.0, 'num_valid_runs': 0}

    gt_wrapped = {"tool_calls": ground_truth}
    accuracies = []

    for run in generated_runs:
        if run:
            try:
                run_wrapped = {"tool_calls": run}
                sim = evaluator.calculate_tree_edit_distance_fast(gt_wrapped, run_wrapped, variation_type='combined')
                accuracies.append(sim)
            except:
                continue

    if not accuracies:
        return {'accuracy_mean': 0.0, 'num_valid_runs': 0}

    return {
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'num_valid_runs': len(accuracies)
    }


def is_final_model(model_name):
    """Check if model is in FINAL_MODELS."""
    for fm in FINAL_MODELS:
        if fm.lower() in model_name.lower() or model_name.lower() in fm.lower():
            return True
    return False


def main():
    print("=" * 70)
    print("ACCURACY AT MULTIPLE TEMPERATURES")
    print("Measuring REAL accuracy change when adopting consistency strategy")
    print("=" * 70)

    # Initialize evaluator
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")

    # Temperatures to analyze (just T=0.0 vs T=1.0 extremes)
    temperatures = [0.0, 1.0]

    # Use all samples on EC2 (fast GPU)
    MAX_SAMPLES = None  # No limit

    # Store results per temperature
    all_temp_results = {}

    for temp in temperatures:
        print(f"\n{'='*50}")
        print(f"Processing Temperature = {temp}")
        print(f"{'='*50}")

        # Load Toucan results (main dataset with 1006 samples)
        toucan_dir = project_root / "llm_gen_results" / "toucan"
        gen_results = load_generation_results(toucan_dir, temp, "toucan")

        print(f"Found {len(gen_results)} models at T={temp}")

        # Compute accuracy for each model
        model_metrics = {}

        for model_name in tqdm(gen_results.keys(), desc=f"T={temp}"):
            if not is_final_model(model_name):
                continue

            model_samples = gen_results[model_name]
            accuracies = []
            valid_count = 0

            # Limit samples for faster processing (if MAX_SAMPLES is set)
            sample_items = list(model_samples.items())
            if MAX_SAMPLES:
                sample_items = sample_items[:MAX_SAMPLES]
            for sample_idx, sample_data in sample_items:
                gt = sample_data['ground_truth']
                runs = sample_data['generated_runs']

                if not gt or not runs:
                    continue

                acc_metrics = compute_accuracy(evaluator, gt, runs)
                if acc_metrics['num_valid_runs'] > 0:
                    accuracies.append(acc_metrics['accuracy_mean'])
                    valid_count += 1

            if accuracies:
                model_metrics[model_name] = {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'samples': valid_count
                }

        all_temp_results[temp] = model_metrics

        # Print summary for this temperature
        print(f"\nT={temp} Summary:")
        for model, metrics in sorted(model_metrics.items(), key=lambda x: x[1]['accuracy_mean'], reverse=True):
            print(f"  {model}: Accuracy={metrics['accuracy_mean']:.4f} (n={metrics['samples']})")

    # Compare temperatures
    print("\n" + "=" * 70)
    print("ACCURACY CHANGE: T=1.0 (LOW consistency) vs T=0.0 (HIGH consistency)")
    print("=" * 70)

    comparison_data = []

    for model_name in set(all_temp_results.get(0.0, {}).keys()) & set(all_temp_results.get(1.0, {}).keys()):
        t0_acc = all_temp_results[0.0][model_name]['accuracy_mean']
        t1_acc = all_temp_results[1.0][model_name]['accuracy_mean']
        change = t0_acc - t1_acc
        pct_change = (change / t1_acc) * 100 if t1_acc > 0 else 0

        comparison_data.append({
            'model': model_name,
            'acc_t0': t0_acc,
            'acc_t1': t1_acc,
            'acc_change': change,
            'pct_change': pct_change
        })

    # Sort by improvement
    comparison_data.sort(key=lambda x: x['acc_change'], reverse=True)

    print(f"\n{'Model':<30} {'T=1.0 Acc':>12} {'T=0.0 Acc':>12} {'Change':>12} {'% Change':>12}")
    print("-" * 80)

    for row in comparison_data:
        print(f"{row['model']:<30} {row['acc_t1']:>12.4f} {row['acc_t0']:>12.4f} {row['acc_change']:>+12.4f} {row['pct_change']:>+11.1f}%")

    # Calculate averages
    avg_t0 = np.mean([r['acc_t0'] for r in comparison_data])
    avg_t1 = np.mean([r['acc_t1'] for r in comparison_data])
    avg_change = avg_t0 - avg_t1
    avg_pct = (avg_change / avg_t1) * 100 if avg_t1 > 0 else 0

    print("-" * 80)
    print(f"{'AVERAGE':<30} {avg_t1:>12.4f} {avg_t0:>12.4f} {avg_change:>+12.4f} {avg_pct:>+11.1f}%")

    # Save results
    output_dir = project_root / "results" / "accuracy_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'temperatures': temperatures,
        'results_by_temp': {str(t): all_temp_results.get(t, {}) for t in temperatures},
        'comparison_t0_vs_t1': comparison_data,
        'summary': {
            'avg_accuracy_t0': float(avg_t0),
            'avg_accuracy_t1': float(avg_t1),
            'avg_accuracy_change': float(avg_change),
            'avg_pct_improvement': float(avg_pct)
        }
    }

    output_file = output_dir / "accuracy_multi_temp.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
