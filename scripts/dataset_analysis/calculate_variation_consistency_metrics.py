#!/usr/bin/env python3
import json
import os
import re
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer
from tqdm import tqdm

def extract_temperature_from_path(path):
    match = re.search(r'temp_(\d+)_(\d+)', path)
    return float(f"{match.group(1)}.{match.group(2)}") if match else None

from sted.model_config import get_display_name

def main(results_dir="llm_gen_results", output_dir="."):
    evaluator = SemanticJsonTreeConsistencyEvaluator()
    analyzer = StructuralConsistencyAnalyzer(evaluator)
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

                    # Calculate variation consistency metrics using analyzer
                    report = analyzer.evaluate_structural_consistency(
                        json_outputs=responses,
                        method_name='sted',
                        variation_type=variation_type
                    )

                    # Extract metrics from report
                    consistency_metrics = report.get('consistency_metrics', {})
                    supporting_stats = report.get('supporting_stats', {})

                    empty_ratio = consistency_metrics.get('empty_ratio', 0.0)
                    consistency_score = consistency_metrics.get('stability_score', 0.0)
                    penalized_consistency = consistency_metrics.get('penalized_stability_score', 0.0)
                    mean_distance = 1.0 - supporting_stats.get('mean_similarity', 0.0)
                    std_distance = supporting_stats.get('std_deviation', 0.0)
                    valid_count = report.get('num_outputs_analyzed', 0)

                    results[model_name].append({
                        'temperature': temperature,
                        'sample_idx': sample_idx,
                        'variation_type': variation_type,
                        'empty_ratio': empty_ratio,
                        'consistency_score': consistency_score,
                        'penalized_consistency': penalized_consistency,
                        'mean_distance': mean_distance,
                        'std_distance': std_distance,
                        'valid_count': valid_count
                    })

                    print(f"{model_name} T={temperature} S={sample_idx}: "
                          f"Empty={empty_ratio:.2%}, "
                          f"Consistency={consistency_score:.4f}, "
                          f"Std={std_distance:.4f}")
        
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
