#!/usr/bin/env python3
"""
Analyze how similarity changes as variation ratio increases in synthetic datasets.

Usage:
    python analyze_semantic_expression_variation_progression.py \
        synthetic_dataset/expression_variation_dataset_*.json \
        synthetic_dataset/semantic_variation_dataset_*.json \
        --output-dir results/variation_progression
"""
import json
import argparse
import os

import numpy as np
from tqdm import tqdm

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


def get_variation_type_from_filename(filename):
    """Extract variation type from filename."""
    basename = os.path.basename(filename).lower()
    if 'expression' in basename:
        return 'expression'
    elif 'semantic' in basename:
        return 'semantic'
    elif 'schema' in basename:
        return 'schema'
    return 'unknown'


def analyze_variation_progression(file_paths, output_dir='results'):
    """Analyze how similarity changes as variation ratio increases."""
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='amazon.titan-embed-text-v2:0')
    methods = ['sted', 'bertscore', 'deepdiff']
    variation_types_map = {'structural': 'structural', 'content': 'content', 'combined': 'combined'}
    
    all_results = {}
    
    for file_path in file_paths:
        variation_type = get_variation_type_from_filename(file_path)
        print(f"\nProcessing {variation_type} variation: {file_path}")
        
        # Results: {method: {variation_type: {ratio: [similarities]}}}
        results = {
            method: {vt: {round(i/10, 1): [] for i in range(1, 11)} for vt in variation_types_map}
            for method in methods
        }
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for sample in tqdm(data, desc=f"Processing samples"):
            base_sample = sample.get('base_sample', {})
            variants = sample.get('variants', [])
            
            if not base_sample or not variants:
                continue
            
            for variant in variants:
                variation_ratio = round(variant.get('variation_ratio', 0), 1)
                variation = variant.get('variation', {})
                
                if not variation or variation_ratio == 0:
                    continue
                
                for method in methods:
                    for vt_key, vt_name in variation_types_map.items():
                        try:
                            if method == 'sted':
                                similarity = evaluator.calculate_tree_edit_distance_opt(
                                    base_sample, variation, variation_type=vt_key
                                )
                            elif method == 'bertscore':
                                similarity = evaluator.calculate_bertscore(
                                    base_sample, variation
                                )
                            elif method == 'deepdiff':
                                similarity = evaluator.calculate_similarity_with_deepdiff(
                                    base_sample, variation
                                )
                            results[method][vt_name][variation_ratio].append(similarity)
                        except Exception as e:
                            print(f"Error: {method}/{vt_name} at ratio {variation_ratio}: {e}")
        
        all_results[file_path] = {'type': variation_type, 'results': results}
    
    # Save results for each file
    for file_path, file_data in all_results.items():
        variation_type = file_data['type']
        results = file_data['results']
        variation_ratios = [round(i/10, 1) for i in range(1, 11)]
        
        # Calculate averages
        avg_results = {
            method: {
                vt: [np.mean(results[method][vt][r]) if results[method][vt][r] else 0 
                     for r in variation_ratios]
                for vt in variation_types_map.values()
            }
            for method in methods
        }
        
        # Save to JSON
        output_data = {
            'variation_type': variation_type,
            'file_path': file_path,
            'variation_ratios': variation_ratios,
            'average_similarities': avg_results,
        }
        
        output_filename = os.path.join(output_dir, f'{variation_type}_variation_progression_results.json')
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_filename}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"{variation_type.title()} Variation Progression Summary")
        print(f"{'='*60}")
        
        for vt in variation_types_map.values():
            print(f"\n{vt.title()} similarity type:")
            print(f"{'Ratio':<6} {'STED':<8} {'BERT':<8} {'DEEP':<8}")
            print("-" * 35)
            for i, ratio in enumerate(variation_ratios):
                print(f"{ratio:<6} {avg_results['sted'][vt][i]:<8.3f} "
                      f"{avg_results['bertscore'][vt][i]:<8.3f} "
                      f"{avg_results['deepdiff'][vt][i]:<8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze variation progression from dataset files')
    parser.add_argument('files', nargs='+', help='Dataset files to analyze')
    parser.add_argument('--output-dir', default='results', help='Directory to save output files')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    analyze_variation_progression(args.files, args.output_dir)
