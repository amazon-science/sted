#!/usr/bin/env python3
"""
Compute TED, BERTScore, and DeepDiff metrics for all LLM models.

This script processes all model generation results and computes consistency metrics
using three baseline methods for comparison with STED.
"""

import json
import numpy as np
import os
import sys
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import re

# Add STED to path
STED_PROJECT = Path("/Users/guanghu/Documents/genai/projects/sted")
sys.path.insert(0, str(STED_PROJECT))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Paths
LLM_RESULTS_DIR = STED_PROJECT / "llm_gen_results"
OUTPUT_DIR = Path(__file__).parent

# Model directories
MODEL_DIRS = {
    'Claude-3-Haiku': 'generations-claude-3-haiku',
    'Claude-3.5-Haiku': 'generations-claude3-5-haiku',
    'Claude-3.7-Sonnet': 'generations-claude3-7-sonnet',
    'DeepSeek-V3': 'generations-deepseek.v3-v1',
    'Gemini-2.5-Flash-Lite': 'generations-gemini-2.5-flash-lite',
    'GPT-4.1-Mini': 'generations-gpt-4.1-mini',
    'Llama-3.3-70B': 'generations-llama3-3-70b',
    'Nova-Pro-v1': 'generations-nova-pro-v1',
    'Qwen3-32B': 'generations-qwen3-32b-v1',
    'Qwen3-235B': 'generations-qwen3-235b-a22b-2507',
}

METRICS = ['ted', 'bertscore', 'deepdiff']


def extract_temperature_from_dir(dirname: str) -> float:
    """Extract temperature value from directory name."""
    match = re.search(r'temp_(\d+)_(\d+)', dirname)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None


def calculate_pairwise_similarities(responses: List[Dict], evaluator: SemanticJsonTreeConsistencyEvaluator,
                                     method: str) -> Dict[str, float]:
    """
    Calculate pairwise similarity statistics for a set of responses.

    Args:
        responses: List of JSON response dictionaries
        evaluator: The STED evaluator instance
        method: 'ted', 'bertscore', or 'deepdiff'

    Returns:
        Dictionary with mean, std, min, max, range statistics
    """
    if len(responses) < 2:
        return {'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0, 'range': 0.0, 'count': len(responses)}

    # Calculate all pairwise similarities
    similarities = []
    calc_method = evaluator.calculate_similarity_method[method]

    for r1, r2 in combinations(responses, 2):
        try:
            if method == 'bertscore':
                # BERTScore needs special handling
                sim = calc_method(r1, r2)
            else:
                sim = calc_method(r1, r2)
            similarities.append(sim)
        except Exception as e:
            # If comparison fails, skip this pair
            continue

    if not similarities:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'range': 0.0, 'count': 0}

    similarities = np.array(similarities)
    return {
        'mean': float(np.mean(similarities)),
        'std': float(np.std(similarities)),
        'std_cv': float(np.std(similarities) / np.mean(similarities)) if np.mean(similarities) > 0 else 0.0,
        'std_normalized': float(np.std(similarities) / (np.mean(similarities) * (1 - np.mean(similarities))))
                          if 0 < np.mean(similarities) < 1 else 0.0,
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'range': float(np.max(similarities) - np.min(similarities)),
        'count': len(similarities)
    }


def process_model_temperature(model_dir: Path, temp_dir: Path, evaluator: SemanticJsonTreeConsistencyEvaluator,
                               metric: str) -> List[Dict]:
    """
    Process all samples for a model at a specific temperature.

    Returns:
        List of result dictionaries, one per sample
    """
    all_results_path = temp_dir / 'all_results.json'

    if not all_results_path.exists():
        return []

    with open(all_results_path, 'r') as f:
        data = json.load(f)

    results = []

    for sample_idx, sample in enumerate(data.get('results', [])):
        responses = sample.get('responses', [])[:10]  # Limit to 10 responses

        # Filter out None/empty responses
        valid_responses = [r for r in responses if r is not None and r != {}]

        if len(valid_responses) < 2:
            continue

        stats = calculate_pairwise_similarities(valid_responses, evaluator, metric)
        stats['sample_id'] = sample_idx
        results.append(stats)

    return results


def compute_metrics_for_model(model_name: str, model_dirname: str,
                               evaluator: SemanticJsonTreeConsistencyEvaluator,
                               metrics: List[str] = METRICS,
                               force: bool = False) -> Dict[str, int]:
    """
    Compute all metrics for a single model.

    Args:
        model_name: Display name of the model
        model_dirname: Directory name containing model results
        evaluator: STED evaluator instance
        metrics: List of metrics to compute
        force: If True, recompute even if results exist

    Returns:
        Dictionary with count of processed temperature directories per metric
    """
    model_dir = LLM_RESULTS_DIR / model_dirname

    if not model_dir.exists():
        print(f"  Model directory not found: {model_dir}")
        return {}

    counts = {m: 0 for m in metrics}

    # Find all temperature directories
    temp_dirs = [d for d in model_dir.iterdir() if d.is_dir() and 'temp_' in d.name]

    for temp_dir in tqdm(temp_dirs, desc=f"  {model_name}", leave=False):
        temp = extract_temperature_from_dir(temp_dir.name)
        if temp is None:
            continue

        for metric in metrics:
            result_file = temp_dir / f'results_{metric}.json'

            # Skip if already computed (unless force)
            if result_file.exists() and not force:
                counts[metric] += 1
                continue

            # Compute metrics
            try:
                results = process_model_temperature(model_dir, temp_dir, evaluator, metric)

                if results:
                    with open(result_file, 'w') as f:
                        json.dump({'results': results}, f, indent=2)
                    counts[metric] += 1
            except Exception as e:
                print(f"    Error computing {metric} for {temp_dir.name}: {e}")

    return counts


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute consistency metrics for all LLM models')
    parser.add_argument('--models', nargs='*', help='Specific models to process (default: all)')
    parser.add_argument('--metrics', nargs='*', default=METRICS, help='Metrics to compute')
    parser.add_argument('--force', action='store_true', help='Recompute existing results')
    args = parser.parse_args()

    print("=" * 70)
    print("COMPUTING CONSISTENCY METRICS FOR ALL MODELS")
    print("=" * 70)

    # Initialize evaluator
    print("\n1. Initializing evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    print("   Evaluator initialized with all-MiniLM-L6-v2 model")

    # Determine which models to process
    if args.models:
        models_to_process = {k: v for k, v in MODEL_DIRS.items() if k in args.models}
    else:
        models_to_process = MODEL_DIRS

    print(f"\n2. Processing {len(models_to_process)} models with metrics: {args.metrics}")

    # Process each model
    all_results = {}

    for model_name, model_dirname in models_to_process.items():
        print(f"\n   Processing: {model_name}")
        counts = compute_metrics_for_model(
            model_name, model_dirname, evaluator,
            args.metrics, args.force
        )
        all_results[model_name] = counts
        print(f"   Processed: {counts}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} | " + " | ".join(f"{m:<10}" for m in args.metrics))
    print("-" * (25 + 3 + len(args.metrics) * 13))

    for model_name, counts in all_results.items():
        counts_str = " | ".join(f"{counts.get(m, 0):<10}" for m in args.metrics)
        print(f"{model_name:<25} | {counts_str}")

    print("\n" + "=" * 70)
    print("METRIC COMPUTATION COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
