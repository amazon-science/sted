#!/usr/bin/env python3
"""
Benchmark STED optimizations: compare original vs optimized versions.

Usage:
    python benchmark_sted_optimizations.py --results-dir llm_gen_results/toucan --num-samples 100
"""
import json
import os
import sys
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.json_tree_node import JsonNode


def load_sample_pairs(results_dir: str, num_samples: int) -> List[Tuple[Dict, Dict]]:
    """Load sample JSON pairs from results directory."""
    pairs = []

    # Find all result files (handle both flat and nested structures)
    result_files = []

    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == 'all_results.json':
                result_files.append(os.path.join(root, f))

    print(f"  Found {len(result_files)} result files")

    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            for sample in data.get('results', []):
                gt = sample.get('ground_truth')
                responses = sample.get('responses') or sample.get('generated_runs', [])

                if gt and responses:
                    for resp in responses[:2]:  # Take first 2 responses per sample
                        if resp:
                            pairs.append((gt, resp))
                            if len(pairs) >= num_samples:
                                return pairs
        except Exception as e:
            print(f"  Warning: Failed to load {result_file}: {e}")
            continue

    return pairs


def benchmark_original(evaluator: SemanticJsonTreeConsistencyEvaluator,
                       pairs: List[Tuple[Dict, Dict]],
                       variation_type: str = "combined") -> Tuple[List[float], float]:
    """Benchmark original _calculate_optimal_matching_cost method."""
    results = []

    start_time = time.time()
    for json1, json2 in pairs:
        # Convert to trees
        j1 = {"root": json1} if isinstance(json1, dict) else json1
        j2 = {"root": json2} if isinstance(json2, dict) else json2

        tree1 = JsonNode.from_dict(j1, sort_arrays=evaluator.sort_arrays,
                                   sort_keys=evaluator.sort_keys,
                                   order_sensitive_fields=evaluator.order_sensitive_fields)
        tree2 = JsonNode.from_dict(j2, sort_arrays=evaluator.sort_arrays,
                                   sort_keys=evaluator.sort_keys,
                                   order_sensitive_fields=evaluator.order_sensitive_fields)

        # Use original method
        cost = evaluator._calculate_optimal_matching_cost(tree1, tree2, variation_type)
        results.append(1.0 - cost)

    elapsed = time.time() - start_time
    return results, elapsed


def benchmark_optimized(evaluator: SemanticJsonTreeConsistencyEvaluator,
                        pairs: List[Tuple[Dict, Dict]],
                        variation_type: str = "combined",
                        use_greedy: bool = False) -> Tuple[List[float], float]:
    """Benchmark optimized _calculate_optimal_matching_cost_fast method."""
    results = []

    # Clear cache before benchmark
    evaluator.clear_subtree_cache()
    evaluator.use_greedy_matching = use_greedy

    start_time = time.time()
    for json1, json2 in pairs:
        # Convert to trees
        j1 = {"root": json1} if isinstance(json1, dict) else json1
        j2 = {"root": json2} if isinstance(json2, dict) else json2

        tree1 = JsonNode.from_dict(j1, sort_arrays=evaluator.sort_arrays,
                                   sort_keys=evaluator.sort_keys,
                                   order_sensitive_fields=evaluator.order_sensitive_fields)
        tree2 = JsonNode.from_dict(j2, sort_arrays=evaluator.sort_arrays,
                                   sort_keys=evaluator.sort_keys,
                                   order_sensitive_fields=evaluator.order_sensitive_fields)

        # Pre-compute leaf similarities (optimization 4)
        evaluator._precompute_leaf_similarities(tree1, tree2)

        # Use optimized method
        cost = evaluator._calculate_optimal_matching_cost_fast(tree1, tree2, variation_type)
        results.append(1.0 - cost)

    elapsed = time.time() - start_time
    return results, elapsed


def calculate_accuracy_metrics(original: List[float], optimized: List[float]) -> Dict[str, float]:
    """Calculate accuracy metrics comparing original and optimized results."""
    if len(original) != len(optimized):
        raise ValueError("Result lists must have same length")

    original = np.array(original)
    optimized = np.array(optimized)

    # Absolute differences
    abs_diff = np.abs(original - optimized)

    return {
        'mean_absolute_error': float(np.mean(abs_diff)),
        'max_absolute_error': float(np.max(abs_diff)),
        'correlation': float(np.corrcoef(original, optimized)[0, 1]) if len(original) > 1 else 1.0,
        'mean_original': float(np.mean(original)),
        'mean_optimized': float(np.mean(optimized)),
        'std_original': float(np.std(original)),
        'std_optimized': float(np.std(optimized)),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark STED optimizations')
    parser.add_argument('--results-dir', default='llm_gen_results/toucan',
                        help='Directory containing LLM generation results')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of sample pairs to benchmark')
    parser.add_argument('--model-id', default='all-MiniLM-L6-v2',
                        help='Embedding model ID')
    parser.add_argument('--variation-type', default='combined',
                        choices=['structural', 'content', 'combined'],
                        help='Variation type to benchmark')
    args = parser.parse_args()

    print("=" * 70)
    print("STED Algorithm Optimization Benchmark")
    print("=" * 70)

    # Load sample pairs
    print(f"\nLoading {args.num_samples} sample pairs from {args.results_dir}...")
    pairs = load_sample_pairs(args.results_dir, args.num_samples)
    print(f"Loaded {len(pairs)} sample pairs")

    if len(pairs) == 0:
        print("ERROR: No sample pairs found. Check the results directory.")
        return

    # Initialize evaluator
    print(f"\nInitializing evaluator with model: {args.model_id}")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id=args.model_id)

    # Pre-compute embeddings for all samples
    print("\nPre-computing embeddings...")
    all_json_objects = []
    for json1, json2 in pairs:
        all_json_objects.append(json1)
        all_json_objects.append(json2)
    evaluator.precompute_embeddings(all_json_objects, show_progress=True)

    print(f"\n{'=' * 70}")
    print(f"Benchmarking variation type: {args.variation_type}")
    print(f"{'=' * 70}")

    # Benchmark original
    print("\n[1/4] Running ORIGINAL algorithm...")
    original_results, original_time = benchmark_original(
        evaluator, pairs, args.variation_type
    )
    print(f"      Time: {original_time:.3f}s ({len(pairs)/original_time:.1f} pairs/sec)")

    # Benchmark optimized (without greedy)
    print("\n[2/4] Running OPTIMIZED algorithm (memoization + single-pass + pruning)...")
    optimized_results, optimized_time = benchmark_optimized(
        evaluator, pairs, args.variation_type, use_greedy=False
    )
    cache_stats = evaluator.get_cache_stats()
    print(f"      Time: {optimized_time:.3f}s ({len(pairs)/optimized_time:.1f} pairs/sec)")
    print(f"      Cache hits: {cache_stats['subtree_cache_hits']}, "
          f"misses: {cache_stats['subtree_cache_misses']}, "
          f"hit rate: {cache_stats['subtree_cache_hit_rate']:.1%}")

    # Benchmark optimized with greedy
    print("\n[3/4] Running OPTIMIZED + GREEDY algorithm...")
    greedy_results, greedy_time = benchmark_optimized(
        evaluator, pairs, args.variation_type, use_greedy=True
    )
    greedy_cache_stats = evaluator.get_cache_stats()
    print(f"      Time: {greedy_time:.3f}s ({len(pairs)/greedy_time:.1f} pairs/sec)")
    print(f"      Cache hits: {greedy_cache_stats['subtree_cache_hits']}, "
          f"misses: {greedy_cache_stats['subtree_cache_misses']}, "
          f"hit rate: {greedy_cache_stats['subtree_cache_hit_rate']:.1%}")

    # Run optimized again to measure cache benefit
    print("\n[4/4] Running OPTIMIZED (warm cache) to measure cache benefit...")
    evaluator.use_greedy_matching = False
    warm_start = time.time()
    for json1, json2 in pairs:
        j1 = {"root": json1} if isinstance(json1, dict) else json1
        j2 = {"root": json2} if isinstance(json2, dict) else json2
        tree1 = JsonNode.from_dict(j1, sort_arrays=evaluator.sort_arrays,
                                   sort_keys=evaluator.sort_keys,
                                   order_sensitive_fields=evaluator.order_sensitive_fields)
        tree2 = JsonNode.from_dict(j2, sort_arrays=evaluator.sort_arrays,
                                   sort_keys=evaluator.sort_keys,
                                   order_sensitive_fields=evaluator.order_sensitive_fields)
        evaluator._calculate_optimal_matching_cost_fast(tree1, tree2, args.variation_type)
    warm_time = time.time() - warm_start
    warm_cache_stats = evaluator.get_cache_stats()
    print(f"      Time: {warm_time:.3f}s ({len(pairs)/warm_time:.1f} pairs/sec)")
    print(f"      Cache hit rate: {warm_cache_stats['subtree_cache_hit_rate']:.1%}")

    # Calculate accuracy metrics
    print(f"\n{'=' * 70}")
    print("ACCURACY COMPARISON")
    print(f"{'=' * 70}")

    opt_accuracy = calculate_accuracy_metrics(original_results, optimized_results)
    greedy_accuracy = calculate_accuracy_metrics(original_results, greedy_results)

    print("\nOptimized vs Original:")
    print(f"  Mean Absolute Error: {opt_accuracy['mean_absolute_error']:.6f}")
    print(f"  Max Absolute Error:  {opt_accuracy['max_absolute_error']:.6f}")
    print(f"  Correlation:         {opt_accuracy['correlation']:.6f}")

    print("\nGreedy vs Original:")
    print(f"  Mean Absolute Error: {greedy_accuracy['mean_absolute_error']:.6f}")
    print(f"  Max Absolute Error:  {greedy_accuracy['max_absolute_error']:.6f}")
    print(f"  Correlation:         {greedy_accuracy['correlation']:.6f}")

    # Performance summary
    print(f"\n{'=' * 70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n{'Method':<35} {'Time (s)':<12} {'Speedup':<10} {'Accuracy':<12}")
    print("-" * 70)
    print(f"{'Original':<35} {original_time:<12.3f} {'1.00x':<10} {'baseline':<12}")
    print(f"{'Optimized (cold cache)':<35} {optimized_time:<12.3f} {original_time/optimized_time:<10.2f}x {opt_accuracy['correlation']:.4f}")
    print(f"{'Optimized (warm cache)':<35} {warm_time:<12.3f} {original_time/warm_time:<10.2f}x {opt_accuracy['correlation']:.4f}")
    print(f"{'Optimized + Greedy':<35} {greedy_time:<12.3f} {original_time/greedy_time:<10.2f}x {greedy_accuracy['correlation']:.4f}")

    # Show sample comparisons
    print(f"\n{'=' * 70}")
    print("SAMPLE COMPARISONS (first 10)")
    print(f"{'=' * 70}")
    print(f"\n{'#':<4} {'Original':<12} {'Optimized':<12} {'Greedy':<12} {'Diff(Opt)':<12} {'Diff(Grdy)':<12}")
    print("-" * 70)
    for i in range(min(10, len(pairs))):
        diff_opt = abs(original_results[i] - optimized_results[i])
        diff_greedy = abs(original_results[i] - greedy_results[i])
        print(f"{i+1:<4} {original_results[i]:<12.4f} {optimized_results[i]:<12.4f} "
              f"{greedy_results[i]:<12.4f} {diff_opt:<12.6f} {diff_greedy:<12.6f}")

    print(f"\n{'=' * 70}")
    print("Benchmark complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
