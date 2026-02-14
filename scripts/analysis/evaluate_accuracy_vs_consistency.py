"""
Evaluate Accuracy vs Consistency Relationship

This script computes accuracy (STED similarity to ground truth) for LLM outputs
and analyzes the relationship with consistency metrics.

Key distinction:
- VALIDITY: Whether response was parseable/valid (infrastructure-related)
- ACCURACY: How close generated output is to ground truth (measured via STED)

Focus on T=0.0 since inconsistency at higher temperatures is expected.
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

from sted import SemanticJsonTreeConsistencyEvaluator, StructuralConsistencyAnalyzer


def load_generation_results(results_dir: Path, temperature: float = 0.0, dataset_name: str = "") -> dict:
    """
    Load generation results from llm_gen_results directory.

    Args:
        results_dir: Path to the dataset directory (e.g., llm_gen_results/toucan)
        temperature: Temperature to filter by
        dataset_name: Name of dataset for logging (e.g., "toucan", "sharegpt")

    Returns:
        dict: {model_name: {sample_idx: {"ground_truth": [...], "generated_runs": [[...], ...]}}}
    """
    all_results = {}

    # Find all model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('generations-'):
            continue

        # Find the run directory matching the temperature
        temp_str = f"temp_{temperature:.0f}_{int((temperature % 1) * 100):02d}"  # e.g., temp_0_00

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Extract model name from run directory name
            # Format: run_{model_name}_temp_{temp}_{timestamp}
            match = re.match(r'run_(.+?)_temp_(\d+)_(\d+)_', run_dir.name)
            if not match:
                continue

            model_name = match.group(1)
            temp_major = int(match.group(2))
            temp_minor = int(match.group(3))
            run_temp = temp_major + temp_minor / 100

            # Check if temperature matches
            if abs(run_temp - temperature) > 0.01:
                continue

            all_results_file = run_dir / "all_results.json"
            if all_results_file.exists():
                try:
                    with open(all_results_file) as f:
                        data = json.load(f)

                    # Verify temperature from metadata
                    meta_temp = data.get('metadata', {}).get('temperature', None)
                    if meta_temp is not None and abs(float(meta_temp) - temperature) > 0.01:
                        continue

                    # Convert to per-sample format using position as sample_idx
                    if model_name not in all_results:
                        all_results[model_name] = {}

                    for idx, result in enumerate(data.get('results', [])):
                        all_results[model_name][idx] = {
                            'ground_truth': result.get('ground_truth', []),
                            'generated_runs': result.get('generated_runs', []),
                            'query': result.get('query', ''),
                            'dataset': dataset_name
                        }

                    print(f"  [{dataset_name}] Loaded {model_name}: {len(all_results[model_name])} samples")
                    break  # Found the right temperature, move to next model

                except Exception as e:
                    print(f"  Error loading {all_results_file}: {e}")

    return all_results


def load_all_generation_results(project_root: Path, temperature: float = 0.0,
                                 invalid_sharegpt_samples: list = None) -> dict:
    """
    Load generation results from both ShareGPT (71 valid samples) and Toucan (1006 samples) datasets.

    Args:
        project_root: Path to the project root
        temperature: Temperature to filter by
        invalid_sharegpt_samples: List of ShareGPT sample indices to exclude (Invalid JSON errors)

    Returns:
        dict: {model_name: {sample_idx: {"ground_truth": [...], "generated_runs": [[...], ...], "dataset": str}}}
    """
    if invalid_sharegpt_samples is None:
        invalid_sharegpt_samples = []

    all_results = {}

    # Load ShareGPT dataset (71 valid samples after excluding 9 parsing issues)
    sharegpt_dir = project_root / "llm_gen_results" / "sharegpt"
    if sharegpt_dir.exists():
        print(f"\nLoading ShareGPT results from {sharegpt_dir}...")
        sharegpt_results = load_generation_results(sharegpt_dir, temperature, dataset_name="sharegpt")

        # Add to all_results with offset for sample indices, excluding invalid samples
        for model_name, samples in sharegpt_results.items():
            if model_name not in all_results:
                all_results[model_name] = {}
            # Use "sg_{idx}" prefix to distinguish ShareGPT samples
            for idx, sample_data in samples.items():
                # Skip invalid samples
                if idx in invalid_sharegpt_samples:
                    continue
                all_results[model_name][f"sg_{idx}"] = sample_data

    # Load Toucan dataset (1006 samples)
    toucan_dir = project_root / "llm_gen_results" / "toucan"
    if toucan_dir.exists():
        print(f"\nLoading Toucan results from {toucan_dir}...")
        toucan_results = load_generation_results(toucan_dir, temperature, dataset_name="toucan")

        # Add to all_results with offset for sample indices
        for model_name, samples in toucan_results.items():
            if model_name not in all_results:
                all_results[model_name] = {}
            # Use "tc_{idx}" prefix to distinguish Toucan samples
            for idx, sample_data in samples.items():
                all_results[model_name][f"tc_{idx}"] = sample_data

    return all_results


def compute_accuracy_metrics(evaluator, ground_truth, generated_runs, debug=False) -> dict:
    """
    Compute accuracy metrics by comparing each generated run to ground truth.

    Args:
        evaluator: SemanticJsonTreeConsistencyEvaluator instance
        ground_truth: List of ground truth tool calls
        generated_runs: List of generated runs, each a list of tool calls
        debug: Whether to print debug info

    Returns:
        dict with accuracy metrics
    """
    if not ground_truth or not generated_runs:
        return {
            'accuracy_mean': 0.0,
            'accuracy_std': 0.0,
            'accuracy_min': 0.0,
            'accuracy_max': 0.0,
            'num_valid_runs': 0
        }

    # Wrap lists in dict for STED (expects Dict[str, Any])
    # Format: {"tool_calls": [list of tool calls]}
    gt_wrapped = {"tool_calls": ground_truth}

    # Compute similarity between each generated run and ground truth
    accuracies = []
    errors = []
    for run in generated_runs:
        if run:  # Only process non-empty runs
            try:
                # Wrap run in dict
                run_wrapped = {"tool_calls": run}
                # Use STED to compute similarity (calculate_tree_edit_distance_fast returns similarity)
                sim = evaluator.calculate_tree_edit_distance_fast(gt_wrapped, run_wrapped, variation_type='combined')
                accuracies.append(sim)
            except Exception as e:
                errors.append(str(e))
                continue

    if debug and errors:
        print(f"    Errors: {errors[:2]}...")  # Print first 2 errors

    if not accuracies:
        return {
            'accuracy_mean': 0.0,
            'accuracy_std': 0.0,
            'accuracy_min': 0.0,
            'accuracy_max': 0.0,
            'num_valid_runs': 0,
            'errors': errors[:3] if errors else []
        }

    return {
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'accuracy_min': float(np.min(accuracies)),
        'accuracy_max': float(np.max(accuracies)),
        'num_valid_runs': len(accuracies),
        'raw_accuracies': accuracies
    }


def main():
    print("=" * 70)
    print("ACCURACY VS CONSISTENCY ANALYSIS")
    print("Focus: T=0.0 (deterministic setting)")
    print("Accuracy = STED similarity to ground truth (NOT validity)")
    print("Dataset: 71 ShareGPT (valid) + 1006 Toucan = 1077 samples per model")
    print("=" * 70)

    # Import FINAL_MODELS and INVALID_SHAREGPT_SAMPLES from model_config
    from sted.model_config import FINAL_MODELS, INVALID_SHAREGPT_SAMPLES
    print(f"\nUsing {len(FINAL_MODELS)} final models: {FINAL_MODELS}")
    print(f"Excluding {len(INVALID_SHAREGPT_SAMPLES)} invalid ShareGPT samples: {INVALID_SHAREGPT_SAMPLES}")

    # Initialize evaluator
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id="all-MiniLM-L6-v2"
    )

    # Load generation results from BOTH datasets
    print("\nLoading generation results from ShareGPT (71 valid) + Toucan (1006)...")
    gen_results = load_all_generation_results(
        project_root,
        temperature=0.0,
        invalid_sharegpt_samples=INVALID_SHAREGPT_SAMPLES
    )

    # Count samples by dataset
    for model, samples in gen_results.items():
        sg_count = sum(1 for k in samples.keys() if str(k).startswith('sg_'))
        tc_count = sum(1 for k in samples.keys() if str(k).startswith('tc_'))
        print(f"  - {model}: {sg_count} ShareGPT + {tc_count} Toucan = {len(samples)} total")

    print(f"\nFound {len(gen_results)} models with T=0.0 results")

    # Load existing consistency results from BOTH datasets
    print("\nLoading consistency results from both datasets...")

    consistency_lookup = defaultdict(dict)

    # Load ShareGPT consistency results
    sharegpt_consistency_file = project_root / "results" / "sharegpt" / "minilm-ec2" / "combined_consistency_metrics_results.json"
    if sharegpt_consistency_file.exists():
        with open(sharegpt_consistency_file) as f:
            sharegpt_consistency = json.load(f)
        print(f"  ShareGPT: {len(sharegpt_consistency)} models")

        for model_name, model_results in sharegpt_consistency.items():
            for r in model_results:
                if abs(float(r.get('temperature', 0)) - 0.0) < 0.01:
                    sample_idx = f"sg_{int(r['sample_idx'])}"
                    consistency_lookup[model_name][sample_idx] = {
                        'stability_score': r.get('stability_score', 0),
                        'ranking_score': r.get('ranking_score', 0),
                        'c_mean': r.get('c_mean', 0),
                        'd_std': r.get('d_std', 0),
                        'validity_rate': r.get('validity_rate', 0),
                        'dataset': 'sharegpt'
                    }

    # Load Toucan consistency results
    toucan_consistency_file = project_root / "results" / "toucan" / "minilm-ec2" / "combined_consistency_metrics_results.json"
    if toucan_consistency_file.exists():
        with open(toucan_consistency_file) as f:
            toucan_consistency = json.load(f)
        print(f"  Toucan: {len(toucan_consistency)} models")

        for model_name, model_results in toucan_consistency.items():
            for r in model_results:
                if abs(float(r.get('temperature', 0)) - 0.0) < 0.01:
                    sample_idx = f"tc_{int(r['sample_idx'])}"
                    consistency_lookup[model_name][sample_idx] = {
                        'stability_score': r.get('stability_score', 0),
                        'ranking_score': r.get('ranking_score', 0),
                        'c_mean': r.get('c_mean', 0),
                        'd_std': r.get('d_std', 0),
                        'validity_rate': r.get('validity_rate', 0),
                        'dataset': 'toucan'
                    }

    print(f"Combined consistency models: {list(consistency_lookup.keys())}")

    # Build model name mapping (generation results -> consistency results)
    model_name_mapping = {
        'Claude-3.5-Haiku': 'Claude-3.5-Haiku',
        'Claude-3.5-Sonnet': 'Claude-3.5-Sonnet',
        'Claude-Haiku-4.5': 'Claude-Haiku-4.5',
        'Claude-Sonnet-4': 'Claude-Sonnet-4',
        'Claude-Sonnet-4.5': 'Claude-Sonnet-4.5',
        'Claude-3.7-Sonnet': 'Claude-3.7-Sonnet',
        'us.anthropic.claude-opus-4-20250514-v1': 'us.anthropic.claude-opus-4-20250514-v1',
        'Claude-Opus-4.5': 'Claude-Opus-4.5',
        'GPT-4.1-Mini': 'GPT-4.1-Mini',
        'Qwen3-32B': 'Qwen3-32B',
        'Qwen3-235B-A22B': 'Qwen3-235B-A22B',
        'Llama-3.3-70B': 'Llama-3.3-70B',
        'Gemini-2.5-Flash-Lite': 'Gemini-2.5-Flash-Lite',
        'Grok-4.1-Fast': 'Grok-4.1-Fast',
        'Mistral-Large-3-675B': 'Mistral-Large-3-675B',
        'Nova-2-Lite': 'Nova-2-Lite',
        'Minimax-M2': 'Minimax-M2',
        'Mimo-V2-Flash': 'Mimo-V2-Flash:free',
        'NemoTron-Nano': 'NemoTron-3-Nano-30B-A3B:free',
        'GPT-OSS-120B': 'GPT-OSS-120B',
    }

    # Debug: Check sample indices in consistency lookup
    for model_name in list(consistency_lookup.keys())[:2]:
        sample_indices = list(consistency_lookup[model_name].keys())[:5]
        print(f"  {model_name} sample indices (first 5): {sample_indices}")

    # Helper function to check if model is in FINAL_MODELS
    def is_final_model(model_name):
        for fm in FINAL_MODELS:
            if fm.lower() in model_name.lower() or model_name.lower() in fm.lower():
                return True
        return False

    # Compute accuracy for each (model, sample)
    print("\nComputing accuracy metrics (filtering to FINAL_MODELS)...")
    all_metrics = []
    models_matched = set()
    models_unmatched = set()
    models_skipped_not_final = set()
    debug_counts = {'no_gt': 0, 'no_runs': 0, 'accuracy_failed': 0, 'no_consistency': 0, 'success': 0}

    for model_name in tqdm(gen_results.keys(), desc="Models"):
        # Skip models not in FINAL_MODELS
        if not is_final_model(model_name):
            models_skipped_not_final.add(model_name)
            continue

        model_samples = gen_results[model_name]

        # Find matching model in consistency results
        matching_model = model_name_mapping.get(model_name)
        if not matching_model or matching_model not in consistency_lookup:
            # Try lowercase/case-insensitive matching
            for cm in consistency_lookup.keys():
                if model_name.lower().replace('-', '').replace('_', '') in cm.lower().replace('-', '').replace('_', ''):
                    matching_model = cm
                    break
                if cm.lower().replace('-', '').replace('_', '') in model_name.lower().replace('-', '').replace('_', ''):
                    matching_model = cm
                    break

        if not matching_model or matching_model not in consistency_lookup:
            models_unmatched.add(model_name)
            continue

        models_matched.add(model_name)

        # Debug: Check first sample
        if model_name == list(gen_results.keys())[0]:
            first_idx = list(model_samples.keys())[0]
            first_sample = model_samples[first_idx]
            print(f"\n  DEBUG {model_name}:")
            print(f"    First sample_idx: {first_idx} (type: {type(first_idx)})")
            print(f"    GT type: {type(first_sample['ground_truth'])}, len: {len(first_sample['ground_truth']) if first_sample['ground_truth'] else 0}")
            print(f"    Runs type: {type(first_sample['generated_runs'])}, len: {len(first_sample['generated_runs']) if first_sample['generated_runs'] else 0}")
            if first_sample['generated_runs']:
                print(f"    First run type: {type(first_sample['generated_runs'][0])}, len: {len(first_sample['generated_runs'][0]) if first_sample['generated_runs'][0] else 0}")
            print(f"    Consistency sample indices for {matching_model}: {list(consistency_lookup[matching_model].keys())[:5]}")

        is_first_model = model_name == list(gen_results.keys())[0]
        for i, (sample_idx, sample_data) in enumerate(model_samples.items()):
            gt = sample_data['ground_truth']
            runs = sample_data['generated_runs']

            if not gt:
                debug_counts['no_gt'] += 1
                continue
            if not runs:
                debug_counts['no_runs'] += 1
                continue

            # Compute accuracy
            debug_this = (is_first_model and i == 0)  # Debug first sample of first model
            accuracy_metrics = compute_accuracy_metrics(evaluator, gt, runs, debug=debug_this)

            if accuracy_metrics['num_valid_runs'] == 0:
                debug_counts['accuracy_failed'] += 1
                if debug_this:
                    print(f"    First sample FAILED:")
                    print(f"      GT: {gt[:1]}...")  # First tool call
                    print(f"      Run[0]: {runs[0][:1] if runs[0] else 'empty'}...")
                    print(f"      Errors: {accuracy_metrics.get('errors', [])}")
                continue

            # Get consistency metrics
            consistency_metrics = consistency_lookup[matching_model].get(sample_idx, {})

            if not consistency_metrics:
                debug_counts['no_consistency'] += 1
                continue

            debug_counts['success'] += 1
            all_metrics.append({
                'model': model_name,
                'sample_idx': sample_idx,
                'dataset': sample_data.get('dataset', 'unknown'),
                'accuracy_mean': accuracy_metrics['accuracy_mean'],
                'accuracy_std': accuracy_metrics['accuracy_std'],
                'accuracy_min': accuracy_metrics['accuracy_min'],
                'accuracy_max': accuracy_metrics['accuracy_max'],
                'num_valid_runs': accuracy_metrics['num_valid_runs'],
                'stability_score': consistency_metrics.get('stability_score', 0),
                'ranking_score': consistency_metrics.get('ranking_score', 0),
                'c_mean': consistency_metrics.get('c_mean', 0),
                'd_std': consistency_metrics.get('d_std', 0),
                'validity_rate': consistency_metrics.get('validity_rate', 0)
            })

    print(f"\nDebug counts: {debug_counts}")

    print(f"\nModels matched (FINAL_MODELS): {models_matched}")
    print(f"Models unmatched: {models_unmatched}")
    print(f"Models skipped (not in FINAL_MODELS): {models_skipped_not_final}")

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    print(f"\nTotal samples analyzed: {len(df)}")
    print(f"Models analyzed: {df['model'].nunique()}")

    if len(df) == 0:
        print("No data to analyze. Check if generation results are available.")
        return

    # Analyze relationship between accuracy and consistency
    print("\n" + "=" * 70)
    print("ACCURACY VS CONSISTENCY CORRELATION")
    print("(Accuracy = STED similarity to ground truth)")
    print("=" * 70)

    # Correlation analysis
    correlations = {
        'accuracy_vs_stability': df['accuracy_mean'].corr(df['stability_score']),
        'accuracy_vs_ranking': df['accuracy_mean'].corr(df['ranking_score']),
        'accuracy_vs_c_mean': df['accuracy_mean'].corr(df['c_mean']),
        'accuracy_vs_validity': df['accuracy_mean'].corr(df['validity_rate']),
    }

    print("\nCorrelations (Pearson r):")
    for metric, corr in correlations.items():
        print(f"  {metric}: r = {corr:.4f}")

    # Statistical significance
    from scipy import stats
    r, p = stats.pearsonr(df['accuracy_mean'], df['stability_score'])
    print(f"\nAccuracy vs Stability Score:")
    print(f"  r = {r:.4f}, p = {p:.2e}")

    r, p = stats.pearsonr(df['accuracy_mean'], df['validity_rate'])
    print(f"\nAccuracy vs Validity Rate:")
    print(f"  r = {r:.4f}, p = {p:.2e}")

    # Per-model analysis
    print("\n" + "=" * 70)
    print("PER-MODEL ACCURACY vs CONSISTENCY")
    print("=" * 70)

    model_summary = df.groupby('model').agg({
        'accuracy_mean': ['mean', 'std'],
        'stability_score': ['mean', 'std'],
        'c_mean': 'mean',
        'validity_rate': 'mean'
    }).round(4)

    model_summary.columns = ['acc_mean', 'acc_std', 'stab_mean', 'stab_std', 'c_mean', 'validity']
    model_summary = model_summary.sort_values('acc_mean', ascending=False)

    print(f"\n{'Model':<35} {'Acc Mean':>10} {'Acc Std':>10} {'Stab Mean':>10} {'Validity':>10}")
    print("-" * 80)
    for model, row in model_summary.iterrows():
        print(f"{model[:35]:<35} {row['acc_mean']:>10.4f} {row['acc_std']:>10.4f} {row['stab_mean']:>10.4f} {row['validity']:>10.4f}")

    # Key finding: Is there a trade-off?
    print("\n" + "=" * 70)
    print("KEY FINDING: ACCURACY-CONSISTENCY TRADE-OFF?")
    print("=" * 70)

    overall_corr = df['accuracy_mean'].corr(df['stability_score'])
    if overall_corr > 0.1:
        print(f"\nNO TRADE-OFF DETECTED (r = {overall_corr:.4f})")
        print("Higher consistency is associated with HIGHER accuracy.")
        print("Practitioners can improve consistency without sacrificing accuracy.")
    elif overall_corr < -0.1:
        print(f"\nTRADE-OFF DETECTED (r = {overall_corr:.4f})")
        print("Higher consistency is associated with LOWER accuracy.")
        print("Caution: Consistency improvements may hurt accuracy.")
    else:
        print(f"\nNO SIGNIFICANT RELATIONSHIP (r = {overall_corr:.4f})")
        print("Accuracy and consistency appear independent.")

    # Save results
    output_dir = project_root / "results" / "accuracy_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count samples per dataset
    n_sharegpt = len(df[df['dataset'] == 'sharegpt']) if 'dataset' in df.columns else 0
    n_toucan = len(df[df['dataset'] == 'toucan']) if 'dataset' in df.columns else 0

    output = {
        'correlations': {k: float(v) for k, v in correlations.items()},
        'model_summary': model_summary.to_dict(),
        'overall_correlation': float(overall_corr),
        'n_samples': len(df),
        'n_sharegpt': n_sharegpt,
        'n_toucan': n_toucan,
        'n_models': df['model'].nunique(),
        'final_models': list(models_matched),
        'temperature': 0.0,
        'note': 'Accuracy = STED similarity to ground truth (not validity). Dataset: 71 ShareGPT (valid) + 1006 Toucan = 1077 samples per model.'
    }

    output_file = output_dir / "accuracy_vs_consistency_t0.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Also save detailed per-sample data
    df.to_csv(output_dir / "accuracy_vs_consistency_detailed.csv", index=False)
    print(f"Detailed data saved to {output_dir / 'accuracy_vs_consistency_detailed.csv'}")


if __name__ == "__main__":
    main()
