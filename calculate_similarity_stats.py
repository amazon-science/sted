#!/usr/bin/env python
"""
Calculate mean similarity and standard deviation for JSON comparison methods.

This script compares ground truth JSON with generated JSON data using different methods:
- TED (Tree Edit Distance) 
- BERTScore
- DeepDiff
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import argparse
from pathlib import Path
import time

from semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
#from semantic_json_tree_consistency_v2 import CachedSemanticEvaluator


def calculate_similarity_statistics(
    ground_truth_list: List[Dict[str, Any]], 
    generated_list: List[Dict[str, Any]],
    methods: List[str] = ["ted", "bertscore", "deepdiff"],
    model_id: str = 'all-MiniLM-L6-v2'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate mean similarity and standard deviation for different comparison methods.
    
    Args:
        ground_truth_list: List of ground truth JSON objects
        generated_list: List of generated JSON objects  
        methods: List of methods to use for comparison
        model_id: Model ID for semantic similarity
        
    Returns:
        Dictionary with statistics for each method
    """
    
    if len(ground_truth_list) != len(generated_list):
        raise ValueError("Ground truth and generated lists must have the same length")
    
    # Initialize evaluator
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id=model_id)
    
    results = {}
    
    for method in methods:
        print(f"Calculating similarities using {method}...")
        similarities = []
        
        for i, (gt, gen) in enumerate(tqdm(zip(ground_truth_list, generated_list), 
                                          total=len(ground_truth_list),
                                          desc=f"Processing {method}")):
            try:
                if method == "ted":
                    # Tree Edit Distance (lower is better, so convert to similarity)
                    distance = evaluator.calculate_tree_edit_distance(gt, gen)                    
                    # Convert distance to similarity (0-1 scale)
                    similarity = 1.0 / (1.0 + distance)
                    
                elif method == "bertscore":
                    similarity = evaluator.calculate_bertscore(gt, gen)
                    
                elif method == "deepdiff":
                    similarity = evaluator.calculate_similarity_with_deepdiff(gt, gen)
                    
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                similarities.append(similarity)
                
            except Exception as e:
                print(f"Error processing pair {i} with {method}: {e}")
                similarities.append(0.0)  # Default to 0 similarity on error
        
        # Calculate statistics
        similarities = np.array(similarities)
        results[method] = {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities)),
            'count': len(similarities)
        }
    
    return results


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of JSON objects
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different data formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Check if it's a results file with specific structure
        if 'results' in data:
            return [item.get('ground_truth', {}) for item in data['results']]
        else:
            return [data]
    else:
        raise ValueError("Unsupported data format")


def load_generation_results(file_path: str) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Load generation results and extract ground truth and generated data.
    
    Args:
        file_path: Path to generation results JSON file
        
    Returns:
        Tuple of (ground_truth_list, generated_responses_list)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    ground_truth_list = []
    generated_responses_list = []
    
    for result in data.get('results', []):
        ground_truth = result.get('ground_truth', {})
        responses = result.get('responses', [])
        
        # Filter out empty responses
        valid_responses = [r for r in responses if r]
        
        if ground_truth and valid_responses:
            ground_truth_list.append(ground_truth)
            generated_responses_list.append(valid_responses)
    
    return ground_truth_list, generated_responses_list


def compare_with_multiple_generations(
    ground_truth_list: List[Dict[str, Any]], 
    generated_responses_list: List[List[Dict[str, Any]]],
    methods: List[str] = ["ted", "bertscore", "deepdiff"],
    model_id: str = 'all-MiniLM-L6-v2'
) -> Dict[str, Dict[str, Any]]:
    """
    Compare ground truth with multiple generations per sample.
    
    Args:
        ground_truth_list: List of ground truth JSON objects
        generated_responses_list: List of lists of generated responses
        methods: List of methods to use for comparison
        model_id: Model ID for semantic similarity
        
    Returns:
        Dictionary with detailed statistics for each method
    """
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id=model_id)
    results = {}
    
    for method in methods:
       
        print(f"Calculating similarities using {method}...")
        
        all_similarities = []
        sample_stats = []
        
        for i, (gt, responses) in enumerate(tqdm(zip(ground_truth_list, generated_responses_list),
                                                total=len(ground_truth_list),
                                                desc=f"Processing {method}")):
            sample_similarities = []            
            all_pairs = evaluator.collect_all_string_pairs(responses, gt)
            
            evaluator.batch_compute_similarities(all_pairs)
            print(f"preparation is done..")
            start = time.time()
            for response in responses:
                if method == "ted":
                    similarity = evaluator.calculate_tree_edit_distance(gt, response)
                elif method == "bertscore":
                    similarity = evaluator.calculate_bertscore(gt, response)
                elif method == "deepdiff":
                    similarity = evaluator.calculate_similarity_with_deepdiff(gt, response)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                sample_similarities.append(similarity)
            
            print(f"{method} - sample_similarities: {sample_similarities}")
            print(f"Processing sample {len(response)} response with {method} took average {(time.time() - start)/len(response)} seconds")
            #sample_similarities = evaluator.evaluate_structural_consistency(responses, gt, method)
            if sample_similarities:
                sample_stats.append({
                    'sample_id': i,
                    'mean': np.mean(sample_similarities),
                    'std': np.std(sample_similarities),
                    'min': np.min(sample_similarities),
                    'max': np.max(sample_similarities),
                    'count': len(sample_similarities)
                })
                all_similarities.extend(sample_similarities)
        
        print(f"sample_stats: {sample_stats}")
        # save result into file
        with open(f"results_{method}.json", "w") as f:
            json.dump({"results": sample_stats}, f)
        
        # Overall statistics
        all_similarities = np.array(all_similarities)
        sample_means = [s['mean'] for s in sample_stats]
        sample_stds = [s['std'] for s in sample_stats]
        
        results[method] = {
            'overall_stats': {
                'mean': float(np.mean(all_similarities)),
                'std': float(np.std(all_similarities)),
                'min': float(np.min(all_similarities)),
                'max': float(np.max(all_similarities)),
                'median': float(np.median(all_similarities)),
                'count': len(all_similarities)
            },
            'sample_level_stats': {
                'mean_of_means': float(np.mean(sample_means)),
                'std_of_means': float(np.std(sample_means)),
                'mean_of_stds': float(np.mean(sample_stds)),
                'std_of_stds': float(np.std(sample_stds))
            },
            'per_sample_stats': sample_stats
        }
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """
    Print a formatted comparison table of the results.
    
    Args:
        results: Results dictionary from comparison functions
    """
    print("\n" + "="*80)
    print("SIMILARITY COMPARISON RESULTS")
    print("="*80)
    
    # Create comparison table
    methods = list(results.keys())
    
    print(f"\n{'Method':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8} {'Count':<8}")
    print("-" * 80)
    
    for method in methods:
        stats = results[method]
        
        # Handle different result structures
        if 'overall_stats' in stats:
            # Multiple generations format
            s = stats['overall_stats']
        else:
            # Single generation format
            s = stats
            
        print(f"{method:<12} {s['mean']:<8.4f} {s['std']:<8.4f} {s['min']:<8.4f} "
              f"{s['max']:<8.4f} {s['median']:<8.4f} {s['count']:<8}")
    
    # If we have sample-level stats, show those too
    if any('sample_level_stats' in results[method] for method in methods):
        print(f"\n{'Method':<12} {'Mean of':<10} {'Std of':<10} {'Mean of':<10} {'Std of':<10}")
        print(f"{'':12} {'Means':<10} {'Means':<10} {'Stds':<10} {'Stds':<10}")
        print("-" * 80)
        
        for method in methods:
            if 'sample_level_stats' in results[method]:
                s = results[method]['sample_level_stats']
                print(f"{method:<12} {s['mean_of_means']:<10.4f} {s['std_of_means']:<10.4f} "
                      f"{s['mean_of_stds']:<10.4f} {s['std_of_stds']:<10.4f}")


def save_results(results: Dict[str, Dict[str, Any]], output_file: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Calculate similarity statistics for JSON comparison methods")
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth JSON file")
    parser.add_argument("--generated", type=str, help="Path to generated JSON file")
    parser.add_argument("--generation-results", type=str, help="Path to generation results file (alternative to separate files)")
    parser.add_argument("--methods", nargs='+', default=["ted", "bertscore", "deepdiff"], 
                       help="Methods to use for comparison")
    parser.add_argument("--model-id", type=str, default='all-MiniLM-L6-v2',
                       help="Model ID for semantic similarity")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.generation_results and (not args.ground_truth or not args.generated):
        parser.error("Either --generation-results or both --ground-truth and --generated must be provided")
    
    try:
        if args.generation_results:
            # Load from generation results file
            ground_truth_list, generated_responses_list = load_generation_results(args.generation_results)
            
            print(f"Loaded {len(ground_truth_list)} samples with multiple generations each")
            
            # Calculate statistics with multiple generations
            results = compare_with_multiple_generations(
                ground_truth_list, 
                generated_responses_list,
                methods=args.methods,
                model_id=args.model_id
            )
            
        else:
            # Load from separate files
            ground_truth_list = load_json_data(args.ground_truth)
            generated_list = load_json_data(args.generated)
            
            print(f"Loaded {len(ground_truth_list)} ground truth and {len(generated_list)} generated samples")
            
            # Calculate statistics
            results = calculate_similarity_statistics(
                ground_truth_list, 
                generated_list,
                methods=args.methods,
                model_id=args.model_id
            )
        
        # Print results
        print_comparison_table(results)
        
        # Save results if output file specified
        if args.output:
            save_results(results, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())