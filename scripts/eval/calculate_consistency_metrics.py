#!/usr/bin/env python3
"""
Calculate consistency metrics for LLM generation results.

Usage:
    python calculate_consistency_metrics.py --results-dir llm_gen_results --output-dir results
"""
import json
import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer
from tqdm import tqdm


def extract_temperature_from_path(path):
    """Extract temperature value from directory path."""
    match = re.search(r'temp_(\d+)_(\d+)', path)
    return float(f"{match.group(1)}.{match.group(2)}") if match else None


from sted.model_config import get_display_name


def process_single_sample(args_tuple):
    """
    Process a single sample for consistency calculation.
    Used for parallel processing with ThreadPoolExecutor.

    Args:
        args_tuple: (sample_idx, sample, analyzer, variation_type)

    Returns:
        Dictionary with all consistency metrics for this sample
    """
    sample_idx, sample, analyzer, variation_type = args_tuple

    gt = sample['ground_truth']
    responses = sample.get('responses') or sample.get('generated_runs', [])
    responses = responses[:10]

    valid_responses = [r for r in responses if r]
    total_runs = len(responses)
    valid_runs = len(valid_responses)
    validity_rate = valid_runs / total_runs if total_runs > 0 else 0.0

    if len(valid_responses) >= 1:
        report = analyzer.evaluate_structural_consistency(
            valid_responses, gt, method_name="sted", variation_type=variation_type,
            validity_rate=validity_rate
        )
        pairwise_similarities = report.get('raw_similarities', [])
    else:
        pairwise_similarities = []

    # Use unified metrics method that returns all metrics
    all_metrics = analyzer._calculate_consistency_metrics(pairwise_similarities, validity_rate)

    result = {
        'sample_idx': sample_idx,
        'validity_rate': validity_rate,
        'valid_runs': valid_runs,
        'total_runs': total_runs,
        # Interpretable metrics (ICML 2026)
        'mean_similarity': all_metrics['c_mean'],
        'c_mean': all_metrics['c_mean'],
        'd_std': all_metrics['d_std'],
        'd_std_normalized': all_metrics['d_std_normalized'],
        'r_v': all_metrics['r_v'],
        'c_adj': all_metrics['c_adj'],
        # Benchmarking metrics
        'stability_score': all_metrics['stability_score'],
        'ranking_score': all_metrics['ranking_score'],
        # Legacy metrics
        'consistency_coefficient': all_metrics['consistency_coefficient'],
        'normalized_cv': all_metrics['normalized_cv'],
        'empty_ratio': all_metrics['empty_ratio'],
        'penalized_consistency_coefficient': all_metrics['penalized_consistency_coefficient'],
        'penalized_stability_score': all_metrics['penalized_stability_score'],
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Calculate consistency metrics for LLM results')
    parser.add_argument('--results-dir', default='llm_gen_results', help='Directory containing LLM generation results')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for embedding precomputation')
    parser.add_argument('--embedding-cache', default=None, help='Path to embedding cache file (.npz). Will load if exists, save after computing.')
    parser.add_argument('--force-recompute', action='store_true', help='Force recompute embeddings even if cache exists')
    parser.add_argument('--model-id', default='all-MiniLM-L6-v2',
                        help='Embedding model ID. Use "amazon.titan-embed-text-v2:0" for Bedrock API')
    parser.add_argument('--dimension', type=int, default=512,
                        help='Embedding dimension for Bedrock models (256, 384, 512, or 1024). Default: 512')
    parser.add_argument('--region', default='us-west-2', help='AWS region for Bedrock API')
    parser.add_argument('--max-workers', type=int, default=10, help='Max parallel workers for Bedrock API calls')
    # Async API options
    parser.add_argument('--use-async', action='store_true',
                        help='Use async API calls with aioboto3 for Bedrock embeddings. '
                             'More efficient than ThreadPoolExecutor for I/O-bound operations. '
                             'Requires: pip install aioboto3 nest-asyncio')
    parser.add_argument('--max-concurrent', type=int, default=50,
                        help='Maximum concurrent API calls when using --use-async (default: 50)')
    # Bedrock Batch Inference options
    parser.add_argument('--use-batch-inference', action='store_true',
                        help='Use Bedrock Batch Inference (S3-based async) instead of parallel API calls. '
                             'Recommended for large datasets (>10,000 strings).')
    parser.add_argument('--s3-bucket', default=None,
                        help='S3 bucket for batch inference input/output. '
                             'Can also be set via BEDROCK_BATCH_S3_BUCKET env var.')
    parser.add_argument('--s3-prefix', default='bedrock-batch/embeddings',
                        help='S3 prefix for batch inference files')
    parser.add_argument('--role-arn', default=None,
                        help='IAM role ARN for Bedrock Batch Inference. '
                             'Can also be set via BEDROCK_BATCH_ROLE_ARN env var.')
    # Parallel sample processing
    parser.add_argument('--parallel-samples', type=int, default=1,
                        help='Number of parallel workers for sample processing. '
                             'Use 0 for auto (CPU count). Default: 1 (sequential)')
    # STED optimization options
    parser.add_argument('--use-greedy', action='store_true',
                        help='Use greedy matching approximation (O(B²) instead of O(B³) Hungarian). '
                             'Faster but potentially less accurate for large branching factors.')
    parser.add_argument('--early-pruning-threshold', type=float, default=0.8,
                        help='Early pruning threshold for type mismatches (0.0-1.0). Default: 0.8')
    # Note: --use-interpretable-metrics flag removed - unified method now returns all metrics
    # Auto-save cache option
    parser.add_argument('--auto-save-cache', action='store_true',
                        help='Automatically save embedding cache with model_id, dimension, and timestamp '
                             'in the filename (e.g., embeddings_titan-embed-text-v2_0_dim256_20260126_120000.npz)')
    parser.add_argument('--cache-dir', default=None,
                        help='Directory to save auto-generated cache files (default: results-dir)')
    parser.add_argument('--exact-match-fields', type=str, default=None,
                        help='Comma-separated list of field names to use exact matching for instead of semantic similarity. '
                             'For tool-calling datasets like Toucan, use --exact-match-fields=name to enforce exact '
                             'matching of tool names. Example: --exact-match-fields=name,type')
    parser.add_argument('--exact-match-all-keys', action='store_true',
                        help='Use exact string matching for ALL field names (keys) instead of semantic similarity. '
                             'Useful for tool calling where parameter names must match exactly.')
    args = parser.parse_args()

    # Resolve parallel workers
    if args.parallel_samples == 0:
        args.parallel_samples = multiprocessing.cpu_count()
    print(f"Using {args.parallel_samples} parallel workers for sample processing")
    print("Using unified metrics (ICML 2026 interpretable + benchmarking + legacy)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-generate embedding cache filename with model and dimension info
    if args.embedding_cache:
        # If user provided a cache path, check if it needs dimension suffix
        cache_path = args.embedding_cache
        if not any(f"dim{d}" in cache_path or f"_{d}" in cache_path for d in [256, 384, 512, 1024]):
            # Add dimension to cache filename to avoid mixing different dimension embeddings
            base, ext = os.path.splitext(cache_path)
            cache_path = f"{base}_dim{args.dimension}{ext}"
            print(f"Auto-adjusting cache filename to include dimension: {cache_path}")
        args.embedding_cache = cache_path

    # Parse exact match fields if provided
    exact_match_fields = None
    if args.exact_match_fields:
        exact_match_fields = set(f.strip() for f in args.exact_match_fields.split(','))
        print(f"Using exact match for fields: {exact_match_fields}")

    print(f"Using embedding model: {args.model_id} with dimension={args.dimension}")
    if args.exact_match_all_keys:
        print("Using exact match for ALL field names (keys)")
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id=args.model_id,
        region_name=args.region,
        embedding_dim=args.dimension,
        exact_match_fields=exact_match_fields,
        exact_match_all_keys=args.exact_match_all_keys
    )

    # Apply STED optimization settings
    if args.use_greedy:
        evaluator.set_greedy_matching(True)
        print("Using greedy matching approximation (O(B²) instead of O(B³))")
    evaluator.early_pruning_threshold = args.early_pruning_threshold

    analyzer = StructuralConsistencyAnalyzer(evaluator)

    # Try to load existing embedding cache
    embeddings_loaded = 0
    if args.embedding_cache and os.path.exists(args.embedding_cache) and not args.force_recompute:
        print(f"\n{'='*60}")
        print(f"Loading embedding cache from {args.embedding_cache}")
        print(f"{'='*60}")
        embeddings_loaded = evaluator.load_embedding_dict(args.embedding_cache)

    # Find all result directories containing all_results.json
    result_dirs = []
    for item in os.listdir(args.results_dir):
        item_path = os.path.join(args.results_dir, item)
        if not os.path.isdir(item_path):
            continue
        # Check if this directory contains all_results.json (flat structure)
        if os.path.exists(os.path.join(item_path, 'all_results.json')):
            result_dirs.append((item, item_path))
        else:
            # Nested structure: look for subdirectories
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path) and os.path.exists(os.path.join(subitem_path, 'all_results.json')):
                    result_dirs.append((subitem, subitem_path))

    # Pre-load all JSON data and collect all JSON objects for batch embedding precomputation
    print(f"\n{'='*60}")
    print("Loading all JSON data for batch embedding precomputation...")
    print(f"{'='*60}")

    all_data = {}  # dir_name -> (result_path, data)
    all_json_objects = []  # Collect all JSON objects for batch embedding

    for dir_name, result_path in tqdm(result_dirs, desc="Loading JSON files"):
        all_results_path = os.path.join(result_path, 'all_results.json')
        with open(all_results_path, 'r') as f:
            data = json.load(f)
        all_data[dir_name] = (result_path, data)

        # Collect all JSON objects (ground truths and responses)
        for sample in data['results']:
            gt = sample['ground_truth']
            if gt:
                all_json_objects.append(gt)
            responses = sample.get('responses') or sample.get('generated_runs', [])
            for resp in responses[:10]:
                if resp:
                    all_json_objects.append(resp)

    # Batch precompute embeddings for all unique strings (will skip already-loaded ones)
    print(f"\nCollected {len(all_json_objects)} JSON objects from {len(result_dirs)} directories")
    if embeddings_loaded > 0:
        print(f"Already loaded {embeddings_loaded} embeddings from cache, computing only new strings...")

    # Determine cache directory for auto-save
    auto_cache_dir = args.cache_dir if args.cache_dir else args.results_dir

    if args.use_batch_inference:
        print("Using Bedrock Batch Inference (S3-based async)...")
        num_embedded = evaluator.precompute_embeddings(
            all_json_objects, batch_size=args.batch_size,
            show_progress=True, max_workers=args.max_workers,
            use_batch_inference=True,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            role_arn=args.role_arn,
            auto_save_cache=args.auto_save_cache,
            cache_dir=auto_cache_dir
        )
    elif args.use_async:
        print(f"Using async Bedrock API calls (max {args.max_concurrent} concurrent)...")
        num_embedded = evaluator.precompute_embeddings(
            all_json_objects, batch_size=args.batch_size,
            show_progress=True, max_workers=args.max_workers,
            use_async=True,
            max_concurrent=args.max_concurrent,
            auto_save_cache=args.auto_save_cache,
            cache_dir=auto_cache_dir
        )
    else:
        num_embedded = evaluator.precompute_embeddings(
            all_json_objects, batch_size=args.batch_size,
            show_progress=True, max_workers=args.max_workers,
            auto_save_cache=args.auto_save_cache,
            cache_dir=auto_cache_dir
        )
    print(f"Pre-computed {num_embedded} new string embeddings\n")

    # Save embedding cache if path specified and embeddings were computed
    if args.embedding_cache:
        if num_embedded > 0 or (embeddings_loaded == 0 and not os.path.exists(args.embedding_cache)):
            print(f"Saving embedding cache to {args.embedding_cache}")
            evaluator.save_embedding_dict(args.embedding_cache)
        elif embeddings_loaded > 0:
            print(f"Using cached embeddings, no new embeddings to save.")

    variation_types = ["structural", "content", "combined"]

    for variation_type in variation_types:
        print(f"\n{'='*60}")
        print(f"Processing variation type: {variation_type}")
        print(f"{'='*60}")

        results = {}

        for dir_name, (result_path, data) in tqdm(all_data.items(), desc="Processing results"):
            
            metadata = data.get('metadata', {})
            # Support both 'display_name' (new format) and 'model_id'/'model' (for lookup)
            # Also handle alternative format where 'model' and 'temperature' are top-level keys
            model_name = metadata.get('display_name')
            if not model_name:
                model_id = metadata.get('model_id') or metadata.get('model', '')
                # Fallback to top-level 'model' key (alternative JSON format)
                if not model_id:
                    model_id = data.get('model', '')
                model_name = get_display_name(model_id) if model_id else "Unknown"

            # Try metadata.temperature first, then top-level temperature (alternative format)
            temperature = metadata.get('temperature')
            if temperature is None:
                temperature = data.get('temperature')
            
            if temperature is None:
                temperature = extract_temperature_from_path(dir_name)
            if temperature is None:
                continue
            
            if model_name not in results:
                results[model_name] = []

            samples = data['results']
            num_samples = len(samples)

            # Parallel sample processing
            if args.parallel_samples > 1 and num_samples > 10:
                # Use ThreadPoolExecutor for parallel processing (shares embedding cache)
                sample_args = [
                    (idx, sample, analyzer, variation_type)
                    for idx, sample in enumerate(samples)
                ]

                with ThreadPoolExecutor(max_workers=args.parallel_samples) as executor:
                    # Submit all tasks
                    futures = [executor.submit(process_single_sample, arg) for arg in sample_args]

                    # Collect results with progress bar
                    for future in tqdm(as_completed(futures), total=len(futures),
                                      desc=f"  {model_name[:20]}", leave=False):
                        result = future.result()
                        result['temperature'] = temperature
                        results[model_name].append(result)
            else:
                # Sequential processing (original behavior)
                for sample_idx, sample in enumerate(samples):
                    gt = sample['ground_truth']
                    responses = sample.get('responses') or sample.get('generated_runs', [])
                    responses = responses[:10]

                    valid_responses = [r for r in responses if r]
                    total_runs = len(responses)
                    valid_runs = len(valid_responses)
                    validity_rate = valid_runs / total_runs if total_runs > 0 else 0.0

                    if len(valid_responses) >= 1:
                        report = analyzer.evaluate_structural_consistency(
                            valid_responses, gt, method_name="sted", variation_type=variation_type,
                            validity_rate=validity_rate
                        )
                        pairwise_similarities = report.get('raw_similarities', [])
                    else:
                        pairwise_similarities = []

                    # Use unified metrics method that returns all metrics
                    all_metrics = analyzer._calculate_consistency_metrics(pairwise_similarities, validity_rate)

                    result = {
                        'temperature': temperature,
                        'sample_idx': sample_idx,
                        'validity_rate': validity_rate,
                        'valid_runs': valid_runs,
                        'total_runs': total_runs,
                        # Interpretable metrics (ICML 2026)
                        'mean_similarity': all_metrics['c_mean'],
                        'c_mean': all_metrics['c_mean'],
                        'd_std': all_metrics['d_std'],
                        'd_std_normalized': all_metrics['d_std_normalized'],
                        'r_v': all_metrics['r_v'],
                        'c_adj': all_metrics['c_adj'],
                        # Benchmarking metrics
                        'stability_score': all_metrics['stability_score'],
                        'ranking_score': all_metrics['ranking_score'],
                        # Legacy metrics
                        'consistency_coefficient': all_metrics['consistency_coefficient'],
                        'normalized_cv': all_metrics['normalized_cv'],
                        'empty_ratio': all_metrics['empty_ratio'],
                        'penalized_consistency_coefficient': all_metrics['penalized_consistency_coefficient'],
                        'penalized_stability_score': all_metrics['penalized_stability_score'],
                    }

                    results[model_name].append(result)

        # Save results
        output_file = os.path.join(args.output_dir, f'{variation_type}_consistency_metrics_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    print(f"\n{'='*60}")
    print("All consistency metrics calculated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
