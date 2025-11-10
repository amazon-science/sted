#!/usr/bin/env python3
import json
import os
import re
import argparse
import matplotlib.pyplot as plt
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer
from tqdm import tqdm

def extract_temperature_from_path(path):
    match = re.search(r'temp_(\d+)_(\d+)', path)
    return float(f"{match.group(1)}.{match.group(2)}") if match else None

def extract_model_name(path):
    if 'claude3-5-haiku' in path: return 'Claude-3.5-Haiku'
    elif 'claude-3-haiku' in path: return 'Claude-3-Haiku'
    elif 'llama3-3-70b' in path: return 'Llama-3.3-70B'
    elif 'claude3-7-sonnet' in path: return 'Claude-3.7-Sonnet'
    elif 'claude3-5-sonnet' in path: return 'Claude-3.5-Sonnet-v2'
    elif 'nova-pro-v1' in path: return 'Nova-Pro'
    elif 'deepseek.v3-v1' in path: return 'DeepSeek-V3.1'
    elif 'gemini-2.5-flash-lite' in path: return 'Gemini 2.5 Flash Lite'
    elif 'gpt-4.1-mini' in path: return 'GPT-4.1 Mini'
    elif 'qwen3-32b-v1' in path: return 'Qwen3-32B'
    elif 'qwen3-235b-a22b-2507' in path: return 'Qwen3-235B-A22B-Instruct-2507'
    return 'Unknown'

def main():
    parser = argparse.ArgumentParser(description='Calculate consistency metrics for LLM results')
    parser.add_argument('--results-dir', default='llm_gen_results', help='Directory containing LLM generation results')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = SemanticJsonTreeConsistencyEvaluator()
    analyzer = StructuralConsistencyAnalyzer(evaluator)
    results = {}
    
    variation_types = ["structural", "content", "combined"]
    
    for variation_type in variation_types:
        for model_dir in tqdm(os.listdir(args.results_dir), desc=f"{variation_type} Processing models"):
            model_path = os.path.join(args.results_dir, model_dir)
            if not os.path.isdir(model_path): continue
            
            model_name = extract_model_name(model_dir)
            results[model_name] = []
            
            for result_dir in tqdm(sorted(os.listdir(model_path)), desc=f"{variation_type} Processing temperatures"):
                result_path = os.path.join(model_path, result_dir)
                if not os.path.isdir(result_path): continue
                
                temperature = extract_temperature_from_path(result_dir)
                if temperature is None: continue
                
                all_results_path = os.path.join(result_path, 'all_results.json')
                if not os.path.exists(all_results_path): continue
                
                with open(all_results_path, 'r') as f:
                    data = json.load(f)
                
                # Process all samples and all temperatures
                for sample_idx, sample in enumerate(data['results']):
                    gt = sample['ground_truth']
                    responses = sample['responses'][:10]
                    
                    # Use structural consistency analyzer
                    report = analyzer.evaluate_structural_consistency(responses, gt, method_name="sted", variation_type=variation_type)
                    metrics = report.get('consistency_metrics', {})
                    
                    results[model_name].append({
                        'temperature': temperature,
                        'sample_idx': sample_idx,
                        'consistency_coefficient': metrics.get('consistency_coefficient', 0.0),
                        'normalized_cv': metrics.get('normalized_cv', 0.0),
                        'stability_score': metrics.get('stability_score', 0.0),
                        'mean_similarity': report['supporting_stats']['mean_similarity']
                    })
                    
                    print(f"{model_name} T={temperature} S={sample_idx}: CC={metrics.get('consistency_coefficient', 0):.3f}, "
                        f"CV={metrics.get('normalized_cv', 0):.3f}, SS={metrics.get('stability_score', 0):.3f}")
        
        output_file = os.path.join(args.output_dir, f'{variation_type}_consistency_metrics_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Create visualization using mean metrics at each temperature
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['consistency_coefficient', 'normalized_cv', 'stability_score']
    titles = ['Consistency Coefficient', 'Normalized CV', 'Stability Score']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        for model_name, model_results in results.items():
            # Group by temperature and calculate means
            temp_groups = {}
            for result in model_results:
                temp = result['temperature']
                if temp not in temp_groups:
                    temp_groups[temp] = []
                temp_groups[temp].append(result[metric])
            
            temperatures = sorted(temp_groups.keys())
            mean_values = [sum(temp_groups[temp]) / len(temp_groups[temp]) for temp in temperatures]
            
            ax.plot(temperatures, mean_values, marker='o', label=model_name, linewidth=2, markersize=6)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Temperature (Mean)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if metric == 'normalized_cv':
            ax.set_ylim(0, max(0.5, max([max([sum(temp_groups[temp]) / len(temp_groups[temp]) 
                                             for temp in sorted(temp_groups.keys())]) 
                                        for model_results in results.values() 
                                        for temp_groups in [{}] 
                                        if temp_groups.update({result['temperature']: temp_groups.get(result['temperature'], []) + [result[metric]] 
                                                              for result in model_results}) or True]) * 1.1))
        else:
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_png = os.path.join(args.output_dir, f'{variation_type}_consistency_metrics_comparison.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to {args.output_dir}/{variation_type}_consistency_metrics_results.json")
    print(f"Visualization saved to {output_png}")

if __name__ == "__main__":
    main()
