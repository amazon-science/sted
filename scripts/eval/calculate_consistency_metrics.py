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

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer
from tqdm import tqdm


def extract_temperature_from_path(path):
    """Extract temperature value from directory path."""
    match = re.search(r'temp_(\d+)_(\d+)', path)
    return float(f"{match.group(1)}.{match.group(2)}") if match else None


from sted.model_config import get_display_name


def main():
    parser = argparse.ArgumentParser(description='Calculate consistency metrics for LLM results')
    parser.add_argument('--results-dir', default='llm_gen_results', help='Directory containing LLM generation results')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for embedding precomputation')
    parser.add_argument('--embedding-cache', default=None, help='Path to embedding cache file (.npz). Will load if exists, save after computing.')
    parser.add_argument('--force-recompute', action='store_true', help='Force recompute embeddings even if cache exists')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = SemanticJsonTreeConsistencyEvaluator()
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
    num_embedded = evaluator.precompute_embeddings(all_json_objects, batch_size=args.batch_size, show_progress=True)
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
            model_name = metadata.get('display_name')
            if not model_name:
                model_id = metadata.get('model_id') or metadata.get('model', '')
                model_name = get_display_name(model_id) if model_id else "Unknown"
            temperature = data.get('metadata', {}).get('temperature')
            
            if temperature is None:
                temperature = extract_temperature_from_path(dir_name)
            if temperature is None:
                continue
            
            if model_name not in results:
                results[model_name] = []
            
            for sample_idx, sample in enumerate(data['results']):
                gt = sample['ground_truth']
                # Support both 'responses' and 'generated_runs' keys
                responses = sample.get('responses') or sample.get('generated_runs', [])
                responses = responses[:10]

                # Filter out empty/invalid responses to avoid misleading consistency scores
                # Empty responses ([], {}, None, "") would give identical similarity scores
                # which incorrectly inflates consistency metrics
                valid_responses = [r for r in responses if r]
                total_runs = len(responses)
                valid_runs = len(valid_responses)
                validity_rate = valid_runs / total_runs if total_runs > 0 else 0.0

                # Only calculate consistency on valid responses
                if len(valid_responses) >= 2:
                    report = analyzer.evaluate_structural_consistency(
                        valid_responses, gt, method_name="sted", variation_type=variation_type
                    )
                    metrics = report.get('consistency_metrics', {})
                    mean_similarity = report['supporting_stats']['mean_similarity']
                elif len(valid_responses) == 1:
                    # Single valid response - calculate similarity to GT only
                    report = analyzer.evaluate_structural_consistency(
                        valid_responses, gt, method_name="sted", variation_type=variation_type
                    )
                    metrics = report.get('consistency_metrics', {})
                    mean_similarity = report['supporting_stats']['mean_similarity']
                else:
                    # No valid responses - assign minimum scores
                    metrics = {
                        'consistency_coefficient': 0.0,
                        'normalized_cv': 1.0,  # Maximum variability (undefined)
                        'stability_score': 0.0
                    }
                    mean_similarity = 0.0

                results[model_name].append({
                    'temperature': temperature,
                    'sample_idx': sample_idx,
                    'validity_rate': validity_rate,
                    'valid_runs': valid_runs,
                    'total_runs': total_runs,
                    'consistency_coefficient': metrics.get('consistency_coefficient', 0.0),
                    'normalized_cv': metrics.get('normalized_cv', 0.0),
                    'stability_score': metrics.get('stability_score', 0.0),
                    'mean_similarity': mean_similarity
                })
        
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
