#!/usr/bin/env python3
import json
import os
import re
import sys
import argparse
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from tqdm import tqdm

def extract_temperature_from_path(path):
    match = re.search(r'temp_(\d+)_(\d+)', path)
    return float(f"{match.group(1)}.{match.group(2)}") if match else None

from sted.model_config import get_display_name

def main(results_dir="llm_gen_results", output_dir="."):
    evaluator = SemanticJsonTreeConsistencyEvaluator()
    results = {}
    
    variation_types = ["structural", "content", "combined"]
    
    for variation_type in variation_types:
        print(f"\n{'='*80}")
        print(f"Processing variation type: {variation_type}")
        print('='*80)
        
        for model_dir in tqdm(os.listdir(results_dir), desc=f"Processing models"):
            model_path = os.path.join(results_dir, model_dir)
            if not os.path.isdir(model_path): continue
            
            for result_dir in sorted(os.listdir(model_path)):
                result_path = os.path.join(model_path, result_dir)
                if not os.path.isdir(result_path): continue
                
                all_results_path = os.path.join(result_path, 'all_results.json')
                if not os.path.exists(all_results_path): continue
                
                with open(all_results_path, 'r') as f:
                    data = json.load(f)
                
                model_id = data.get('metadata', {}).get('model_id', '')
                model_name = get_display_name(model_id) if model_id else "Unknown"
                temperature = data.get('metadata', {}).get('temperature')
                if temperature is None:
                    temperature = extract_temperature_from_path(result_dir)
                if temperature is None: continue
                
                if model_name not in results:
                    results[model_name] = []
                
                # Process each sample
                for sample_idx, sample in enumerate(data['results']):
                    responses = sample['responses'][:10]
                    
                    # Calculate variation consistency metrics with power transformation
                    metrics = evaluator.calculate_variation_consistency(
                        responses, 
                        method='sted', 
                        variation_type=variation_type,
                        apply_power_transform=True,
                        steepness_factor=20
                    )
                    
                    results[model_name].append({
                        'temperature': temperature,
                        'sample_idx': sample_idx,
                        'variation_type': variation_type,
                        'empty_ratio': metrics['empty_ratio'],
                        'consistency_score': metrics['consistency_score'],
                        'penalized_consistency': metrics['penalized_consistency'],
                        'mean_distance': metrics.get('mean_distance', 0.0),
                        'std_distance': metrics.get('std_distance', 0.0),
                        'valid_count': metrics['valid_count']
                    })
                    
                    print(f"{model_name} T={temperature} S={sample_idx}: "
                          f"Empty={metrics['empty_ratio']:.2%}, "
                          f"Consistency={metrics['consistency_score']:.4f}, "
                          f"Std={metrics.get('std_distance', 0):.4f}")
        
        # Save results for this variation type
        output_file = os.path.join(output_dir, f'{variation_type}_variation_consistency_metrics.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    print("\n" + "="*80)
    print("All variation types processed successfully!")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate variation consistency metrics for LLM generation results')
    parser.add_argument('--results-dir', default='llm_gen_results', help='Directory containing LLM generation results')
    parser.add_argument('--output-dir', default='.', help='Directory to save output files')
    
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)
