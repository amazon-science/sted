#!/usr/bin/env python3
import json
import numpy as np
import argparse
from itertools import combinations
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from tqdm import tqdm

def is_empty_output(output):
    """Check if output is empty or invalid"""
    if output is None:
        return True
    if isinstance(output, dict) and len(output) == 0:
        return True
    if isinstance(output, list) and len(output) == 0:
        return True
    return False

def calculate_variation_consistency(variations, evaluator, method='sted', variation_type='combined'):
    """
    Calculate consistency metrics for a set of variations using pairwise distances.
    
    Returns:
        dict with empty_ratio, consistency_score, and penalized_consistency
    """
    # Count empty outputs
    empty_count = sum(1 for v in variations if is_empty_output(v))
    total_count = len(variations)
    empty_ratio = empty_count / total_count if total_count > 0 else 0.0
    
    # Filter out empty variations for distance calculation
    valid_variations = [v for v in variations if not is_empty_output(v)]
    
    if len(valid_variations) < 2:
        # Not enough valid variations to calculate consistency
        return {
            'empty_ratio': empty_ratio,
            'consistency_score': float('inf'),  # Undefined consistency
            'penalized_consistency': float('inf'),
            'pairwise_distances': [],
            'valid_count': len(valid_variations)
        }
    
    # Calculate all pairwise distances
    pairwise_distances = []
    for v1, v2 in combinations(valid_variations, 2):
        try:
            if method == 'sted':
                similarity = evaluator.calculate_tree_edit_distance_opt(v1, v2, variation_type=variation_type)
            else:
                similarity = evaluator.calculate_similarity_method[method](v1, v2)
            distance = 1.0 - similarity
            pairwise_distances.append(distance)
        except Exception as e:
            print(f"Error calculating distance: {e}")
            continue
    
    if not pairwise_distances:
        return {
            'empty_ratio': empty_ratio,
            'consistency_score': float('inf'),
            'penalized_consistency': float('inf'),
            'pairwise_distances': [],
            'valid_count': len(valid_variations)
        }
    
    # Calculate dispersion (standard deviation)
    consistency_score = np.std(pairwise_distances)
    
    # Apply penalty based on empty ratio
    penalized_consistency = consistency_score * (1 + empty_ratio)
    
    return {
        'empty_ratio': empty_ratio,
        'consistency_score': consistency_score,
        'penalized_consistency': penalized_consistency,
        'pairwise_distances': pairwise_distances,
        'mean_distance': np.mean(pairwise_distances),
        'valid_count': len(valid_variations)
    }

def analyze_variation_consistency_progression(file_path, method='sted', variation_type='combined'):
    """Analyze variation consistency across different variation ratios"""
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='amazon.titan-embed-text-v2:0')
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Results by variation ratio
    results_by_ratio = {}
    
    for sample in tqdm(data, desc="Processing samples"):
        base_sample = sample.get('base_sample', {})
        variants = sample.get('variants', [])
        
        if not variants:
            continue
        
        # Group by variation ratio
        for variant in variants:
            variation_ratio = round(variant.get('variation_ratio'), 1)
            variation = variant.get('variation', {})
            
            if variation_ratio not in results_by_ratio:
                results_by_ratio[variation_ratio] = []
            
            # Collect all variations including base sample for this ratio
            all_variations = [base_sample, variation]
            results_by_ratio[variation_ratio].append(all_variations)
    
    # Calculate consistency metrics for each ratio
    consistency_results = {}
    
    for ratio in sorted(results_by_ratio.keys()):
        ratio_metrics = []
        
        for variations in results_by_ratio[ratio]:
            metrics = calculate_variation_consistency(variations, evaluator, method, variation_type)
            ratio_metrics.append(metrics)
        
        # Aggregate metrics
        consistency_results[ratio] = {
            'avg_empty_ratio': np.mean([m['empty_ratio'] for m in ratio_metrics]),
            'avg_consistency_score': np.mean([m['consistency_score'] for m in ratio_metrics if m['consistency_score'] != float('inf')]),
            'avg_penalized_consistency': np.mean([m['penalized_consistency'] for m in ratio_metrics if m['penalized_consistency'] != float('inf')]),
            'avg_mean_distance': np.mean([m['mean_distance'] for m in ratio_metrics if 'mean_distance' in m]),
            'samples_count': len(ratio_metrics)
        }
    
    return consistency_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze variation consistency using pairwise distances')
    parser.add_argument('file', help='Dataset file to analyze')
    parser.add_argument('--method', default='sted', choices=['ted', 'sted', 'bertscore', 'deepdiff', 'gnn'], 
                        help='Similarity calculation method')
    parser.add_argument('--variation-type', default='combined', choices=['structural', 'content', 'combined'],
                        help='Type of variation to analyze')
    parser.add_argument('--output-dir', default='results', help='Directory to save output files')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f'variation_consistency_{args.method}_{args.variation_type}.json'
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Analyzing variation consistency for: {args.file}")
    print(f"Method: {args.method}, Variation Type: {args.variation_type}")
    
    results = analyze_variation_consistency_progression(args.file, args.method, args.variation_type)
    
    # Save results
    output_data = {
        'file_path': args.file,
        'method': args.method,
        'variation_type': args.variation_type,
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("Variation Consistency Analysis Summary")
    print("="*80)
    print(f"\n{'Ratio':<8} {'Empty%':<10} {'Consistency':<15} {'Penalized':<15} {'Mean Dist':<12}")
    print("-"*70)
    
    for ratio in sorted(results.keys()):
        r = results[ratio]
        print(f"{ratio:<8.1f} {r['avg_empty_ratio']*100:<10.2f} {r['avg_consistency_score']:<15.4f} "
              f"{r['avg_penalized_consistency']:<15.4f} {r['avg_mean_distance']:<12.4f}")
