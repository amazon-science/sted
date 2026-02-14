#!/usr/bin/env python3
"""
Apply Centroid Selection Strategy to LLM Generation Results

This script:
1. Loads existing generation results from llm_gen_results/
2. Applies centroid selection strategy to pick the most consistent run
3. Saves results in the same format as original generation scripts

Output format matches generate_tool_calls.py and generate_structured_outputs.py:
{
    "metadata": {...},
    "results": [
        {
            "sample_id": ...,
            "ground_truth": [...],
            "generated_runs": [...],          # Original 10 runs
            "centroid_selected_run": [...],   # The centroid-selected run
            "centroid_run_idx": 3,            # Index of the selected run
            "consistency_metrics": {...},     # c_mean, ranking_score, etc.
        }
    ],
    "summary": {...}
}

Usage:
    python apply_centroid_selection.py --dataset toucan --temperature 1.0
    python apply_centroid_selection.py --dataset sharegpt --temperature 0.0 --all-temps
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


def load_generation_results(results_dir: Path, dataset: str = 'toucan') -> dict:
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

                    all_results[model_name][temperature] = {
                        'source_file': str(all_results_file),
                        'source_dir': str(run_dir),
                        'metadata': data.get('metadata', {}),
                        'results': data.get('results', []),
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {all_results_file}: {e}")
                    continue

    return dict(all_results)


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
    Returns the index of the centroid run and average similarity scores.
    """
    valid_runs = [(i, r) for i, r in enumerate(runs) if r]
    if len(valid_runs) < 2:
        return (valid_runs[0][0], 1.0, [1.0]) if valid_runs else (None, 0.0, [])

    # Compute average similarity of each run to all others
    avg_similarities = []
    all_sims = []
    for i, run_i in valid_runs:
        similarities = []
        for j, run_j in valid_runs:
            if i != j:
                sim = compute_pairwise_similarity(evaluator, run_i, run_j, variation_type)
                if sim is not None:
                    similarities.append(sim)
        if similarities:
            avg_similarities.append((i, np.mean(similarities)))
            all_sims.append(similarities)
        else:
            avg_similarities.append((i, 0))
            all_sims.append([])

    # Return index of run with highest average similarity (most consistent)
    if avg_similarities:
        best_idx = max(range(len(avg_similarities)), key=lambda x: avg_similarities[x][1])
        return (avg_similarities[best_idx][0], avg_similarities[best_idx][1],
                [s[1] for s in avg_similarities])
    return (None, 0.0, [])


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
            'c_mean': float(metrics.get('c_mean', 0.0)),
            'stability_score': float(metrics.get('stability_score', 0.0)),
            'ranking_score': float(metrics.get('ranking_score', 0.0)),
        }
    except:
        return {'c_mean': 0.0, 'stability_score': 0.0, 'ranking_score': 0.0}


def apply_centroid_selection(evaluator, analyzer, model_data, variation_type='combined', max_samples=-1):
    """
    Apply centroid selection to all samples for a model.
    Returns new results list with centroid_selected_run added.
    """
    results = model_data.get('results', [])
    if max_samples > 0:
        results = results[:max_samples]

    new_results = []

    for result in results:
        sample_id = result.get('sample_id', '')
        ground_truth = result.get('ground_truth', [])
        generated_runs = result.get('generated_runs', [])

        if not generated_runs:
            new_results.append(result)
            continue

        # Find centroid run
        centroid_idx, centroid_avg_sim, all_avg_sims = find_centroid_run(
            evaluator, generated_runs, variation_type
        )

        # Compute consistency metrics
        cons_metrics = compute_consistency_metrics(
            evaluator, analyzer, generated_runs, variation_type
        )

        # Build new result
        new_result = {
            'sample_id': sample_id,
            'query': result.get('query', ''),
            'tools': result.get('tools', []),
            'ground_truth': ground_truth,
            'generated_runs': generated_runs,
            'centroid_selected_run': generated_runs[centroid_idx] if centroid_idx is not None else [],
            'centroid_run_idx': centroid_idx,
            'centroid_avg_similarity': float(centroid_avg_sim) if centroid_avg_sim else 0.0,
            'consistency_metrics': cons_metrics,
            'num_valid_runs': sum(1 for r in generated_runs if r),
        }
        new_results.append(new_result)

    return new_results


def main():
    parser = argparse.ArgumentParser(description='Apply centroid selection strategy to LLM results')
    parser.add_argument('--dataset', default='toucan', choices=['toucan', 'sharegpt'], help='Dataset to process')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature to process (default: all)')
    parser.add_argument('--variation-type', default='combined',
                        choices=['structural', 'content', 'combined'],
                        help='Variation type for STED calculation')
    parser.add_argument('--max-samples', type=int, default=-1, help='Max samples per model (-1 for all)')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: llm_gen_results_centroid/<dataset>)')
    args = parser.parse_args()

    print("=" * 80)
    print("APPLY CENTROID SELECTION TO LLM GENERATION RESULTS")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Variation Type: {args.variation_type}")
    print(f"Temperature Filter: {'all' if args.temperature is None else args.temperature}")
    print(f"Max Samples: {'all' if args.max_samples == -1 else args.max_samples}")
    print("=" * 80)

    # Initialize evaluators
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")
    analyzer = StructuralConsistencyAnalyzer(evaluator)

    # Set directories
    results_dir = project_root / "llm_gen_results" / args.dataset
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = project_root / "llm_gen_results_centroid" / args.dataset

    print(f"\nLoading generation results from {results_dir}...")
    all_gen_results = load_generation_results(results_dir, args.dataset)
    print(f"Found {len(all_gen_results)} models (filtered to FINAL_MODELS)")

    # Collect all temperatures
    temperatures_found = set()
    for model_name, temp_data in all_gen_results.items():
        temperatures_found.update(temp_data.keys())

    temperatures = sorted(temperatures_found)
    if args.temperature is not None:
        temperatures = [t for t in temperatures if abs(t - args.temperature) < 0.01]

    print(f"Temperatures found: {temperatures}")

    # Process each model and temperature
    summary_stats = []

    for model_name in tqdm(sorted(all_gen_results.keys()), desc="Processing models"):
        for temp in temperatures:
            if temp not in all_gen_results[model_name]:
                continue

            model_data = all_gen_results[model_name][temp]
            source_dir = Path(model_data['source_dir'])

            print(f"\n  Processing {model_name} at T={temp}...")

            # Apply centroid selection
            new_results = apply_centroid_selection(
                evaluator, analyzer, model_data,
                args.variation_type, args.max_samples
            )

            # Compute summary stats
            n_samples = len(new_results)
            avg_c_mean = np.mean([r['consistency_metrics']['c_mean'] for r in new_results]) if new_results else 0
            avg_ranking = np.mean([r['consistency_metrics']['ranking_score'] for r in new_results]) if new_results else 0

            summary_stats.append({
                'model': model_name,
                'temperature': temp,
                'n_samples': n_samples,
                'avg_c_mean': float(avg_c_mean),
                'avg_ranking_score': float(avg_ranking),
            })

            # Create output structure
            output_dir = output_base_dir / f"generations-{model_name}"
            temp_str = f"temp_{temp:.2f}".replace(".", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_output_dir = output_dir / f"run_{model_name}_{temp_str}_{timestamp}_centroid"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            # Build output in same format as generate_tool_calls.py
            final_output = {
                "metadata": {
                    **model_data.get('metadata', {}),
                    "centroid_selection": True,
                    "variation_type": args.variation_type,
                    "selection_timestamp": timestamp,
                    "source_file": model_data.get('source_file', ''),
                },
                "results": new_results,
                "summary": {
                    "total_samples": n_samples,
                    "avg_c_mean": float(avg_c_mean),
                    "avg_ranking_score": float(avg_ranking),
                    "samples_with_centroid": sum(1 for r in new_results if r.get('centroid_run_idx') is not None),
                }
            }

            # Save results
            output_file = run_output_dir / "all_results.json"
            with open(output_file, 'w') as f:
                json.dump(final_output, f, indent=2)

            print(f"    Saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<30} {'Temp':>6} {'Samples':>8} {'c_mean':>8} {'Rank':>8}")
    print("-" * 70)

    for stat in sorted(summary_stats, key=lambda x: (x['model'], x['temperature'])):
        print(f"{stat['model']:<30} {stat['temperature']:>6.1f} {stat['n_samples']:>8} "
              f"{stat['avg_c_mean']:>8.4f} {stat['avg_ranking_score']:>8.4f}")

    # Save overall summary
    summary_file = output_base_dir / f"centroid_selection_summary_{args.variation_type}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset': args.dataset,
            'variation_type': args.variation_type,
            'stats': summary_stats,
        }, f, indent=2)

    print(f"\nOverall summary saved to {summary_file}")
    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
