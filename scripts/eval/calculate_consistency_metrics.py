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


def extract_model_name(path):
    """Extract human-readable model name from directory path."""
    model_mappings = {
        'claude-3-7-sonnet': 'Claude-3.7-Sonnet',
        'claude3-7-sonnet': 'Claude-3.7-Sonnet',
        'claude-3-5-haiku': 'Claude-3.5-Haiku',
        'claude3-5-haiku': 'Claude-3.5-Haiku',
        'claude-3-haiku': 'Claude-3-Haiku',
        'claude3-5-sonnet': 'Claude-3.5-Sonnet-v2',
        'llama3-3-70b': 'Llama-3.3-70B',
        'nova-pro-v1': 'Nova-Pro',
        'deepseek.v3-v1': 'DeepSeek-V3.1',
        'gemini-2.5-flash-lite': 'Gemini-2.5-Flash-Lite',
        'gpt-4.1-mini': 'GPT-4.1-Mini',
        'qwen3-32b-v1': 'Qwen3-32B',
        'qwen3-235b-a22b-2507': 'Qwen3-235B-A22B',
    }
    for key, name in model_mappings.items():
        if key in path:
            return name
    return 'Unknown'


def main():
    parser = argparse.ArgumentParser(description='Calculate consistency metrics for LLM results')
    parser.add_argument('--results-dir', default='llm_gen_results', help='Directory containing LLM generation results')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = SemanticJsonTreeConsistencyEvaluator()
    analyzer = StructuralConsistencyAnalyzer(evaluator)
    
    variation_types = ["structural", "content", "combined"]
    
    for variation_type in variation_types:
        print(f"\n{'='*60}")
        print(f"Processing variation type: {variation_type}")
        print(f"{'='*60}")
        
        results = {}
        
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
        
        for dir_name, result_path in tqdm(result_dirs, desc="Processing results"):
            model_name = extract_model_name(dir_name)
            temperature = extract_temperature_from_path(dir_name)
            
            if temperature is None:
                continue
            
            if model_name not in results:
                results[model_name] = []
            
            all_results_path = os.path.join(result_path, 'all_results.json')
            with open(all_results_path, 'r') as f:
                data = json.load(f)
                
                for sample_idx, sample in enumerate(data['results']):
                    gt = sample['ground_truth']
                    responses = sample['responses'][:10]
                    
                    report = analyzer.evaluate_structural_consistency(
                        responses, gt, method_name="sted", variation_type=variation_type
                    )
                    metrics = report.get('consistency_metrics', {})
                    
                    results[model_name].append({
                        'temperature': temperature,
                        'sample_idx': sample_idx,
                        'consistency_coefficient': metrics.get('consistency_coefficient', 0.0),
                        'normalized_cv': metrics.get('normalized_cv', 0.0),
                        'stability_score': metrics.get('stability_score', 0.0),
                        'mean_similarity': report['supporting_stats']['mean_similarity']
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
