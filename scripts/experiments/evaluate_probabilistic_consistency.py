#!/usr/bin/env python3
"""
Evaluate Probabilistic Consistency on Real LLM Data

Compares probabilistic consistency with power transformation.
"""

import sys
import os
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.probabilistic_consistency import ProbabilisticConsistency

def extract_temperature_from_path(path):
    match = re.search(r'temp_(\d+)_(\d+)', path)
    return float(f"{match.group(1)}.{match.group(2)}") if match else None

def extract_model_name(path):
    if 'claude3-5-haiku' in path: return 'Claude-3.5-Haiku'
    elif 'claude-3-haiku' in path: return 'Claude-3-Haiku'
    elif 'llama3-3-70b' in path: return 'Llama-3.3-70B'
    elif 'claude3-7-sonnet' in path: return 'Claude-3.7-Sonnet'
    elif 'nova-pro-v1' in path: return 'Nova-Pro'
    return 'Unknown'

def main():
    evaluator = SemanticJsonTreeConsistencyEvaluator()
    
    # Distance function
    def distance_fn(v1, v2):
        sim = evaluator.calculate_tree_edit_distance_opt(v1, v2, variation_type='combined')
        return 1.0 - sim
    
    # Create probabilistic consistency metric
    prob_metric = ProbabilisticConsistency(distance_fn, sigma_0=None, adaptive=True)
    
    parser = argparse.ArgumentParser(description='Evaluate probabilistic consistency')
    parser.add_argument('--results-dir', default='llm_gen_results', help='LLM results directory')
    parser.add_argument('--output-file', default='results/probabilistic_consistency_results.json', help='Output file')
    args = parser.parse_args()
    
    results = {}
    
    print("\n" + "="*80)
    print("PROBABILISTIC CONSISTENCY EVALUATION ON REAL LLM DATA")
    print("="*80)
    
    for model_dir in tqdm(os.listdir(args.results_dir), desc="Processing models"):
        model_path = os.path.join(args.results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        model_name = extract_model_name(model_dir)
        results[model_name] = []
        
        for result_dir in sorted(os.listdir(model_path)):
            result_path = os.path.join(model_path, result_dir)
            if not os.path.isdir(result_path):
                continue
            
            temperature = extract_temperature_from_path(result_dir)
            if temperature is None:
                continue
            
            all_results_path = os.path.join(result_path, 'all_results.json')
            if not os.path.exists(all_results_path):
                continue
            
            with open(all_results_path, 'r') as f:
                data = json.load(f)
            
            # Process each sample
            for sample_idx, sample in enumerate(data['results']):
                responses = sample['responses'][:10]
                
                # Probabilistic consistency
                prob_result = prob_metric.compute_consistency(responses, return_details=True)
                
                # Power transformation (for comparison)
                distances = prob_metric.compute_distances(responses)
                if len(distances) > 0:
                    std_dist = np.std(distances)
                    n = len(responses)
                    max_std = np.sqrt((n//2) * (n - n//2) / n**2)
                    sigma_norm = std_dist / max_std if max_std > 0 else 0
                    power_score = (1.0 / (1.0 + 2 * sigma_norm)) ** 20
                else:
                    power_score = 1.0
                
                results[model_name].append({
                    'temperature': temperature,
                    'sample_idx': sample_idx,
                    'prob_consistency': prob_result['consistency'],
                    'power_consistency': power_score,
                    'sigma_0': prob_result['sigma_0'],
                    'mean_distance': prob_result['mean_distance'],
                    'std_distance': prob_result['std_distance']
                })
                
        break
    
    # Save results
    output_file = 'probabilistic_consistency_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Analyze and visualize
    analyze_results(results)

def analyze_results(results):
    """Analyze and visualize results"""
    print("\n" + "="*80)
    print("ANALYSIS: Probabilistic vs Power Transformation")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Consistency vs Temperature (Probabilistic)
    ax1 = axes[0, 0]
    for model_name, model_results in results.items():
        temp_groups = {}
        for r in model_results:
            temp = r['temperature']
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(r['prob_consistency'])
        
        temps = sorted(temp_groups.keys())
        means = [np.mean(temp_groups[t]) for t in temps]
        ax1.plot(temps, means, marker='o', label=model_name, linewidth=2)
    
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Probabilistic Consistency')
    ax1.set_title('Probabilistic Consistency vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Consistency vs Temperature (Power)
    ax2 = axes[0, 1]
    for model_name, model_results in results.items():
        temp_groups = {}
        for r in model_results:
            temp = r['temperature']
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(r['power_consistency'])
        
        temps = sorted(temp_groups.keys())
        means = [np.mean(temp_groups[t]) for t in temps]
        ax2.plot(temps, means, marker='o', label=model_name, linewidth=2)
    
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Power Consistency')
    ax2.set_title('Power Consistency vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correlation between methods
    ax3 = axes[1, 0]
    all_prob = []
    all_power = []
    for model_results in results.values():
        for r in model_results:
            all_prob.append(r['prob_consistency'])
            all_power.append(r['power_consistency'])
    
    ax3.scatter(all_prob, all_power, alpha=0.3)
    ax3.plot([0, 1], [0, 1], 'r--', label='y=x')
    ax3.set_xlabel('Probabilistic Consistency')
    ax3.set_ylabel('Power Consistency')
    ax3.set_title('Correlation Between Methods')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Compute correlation
    from scipy.stats import spearmanr, pearsonr
    spearman_r, _ = spearmanr(all_prob, all_power)
    pearson_r, _ = pearsonr(all_prob, all_power)
    ax3.text(0.05, 0.95, f'Spearman: {spearman_r:.3f}\nPearson: {pearson_r:.3f}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Distribution comparison
    ax4 = axes[1, 1]
    ax4.hist(all_prob, bins=30, alpha=0.5, label='Probabilistic', density=True)
    ax4.hist(all_power, bins=30, alpha=0.5, label='Power', density=True)
    ax4.set_xlabel('Consistency Score')
    ax4.set_ylabel('Density')
    ax4.set_title('Score Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probabilistic_vs_power_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: probabilistic_vs_power_comparison.png")
    
    # Print statistics
    print(f"\nCorrelation Analysis:")
    print(f"  Spearman ρ: {spearman_r:.4f}")
    print(f"  Pearson r: {pearson_r:.4f}")
    
    print(f"\nScore Statistics:")
    print(f"  Probabilistic - Mean: {np.mean(all_prob):.4f}, Std: {np.std(all_prob):.4f}")
    print(f"  Power - Mean: {np.mean(all_power):.4f}, Std: {np.std(all_power):.4f}")
    
    print(f"\nDiscrimination (Range):")
    print(f"  Probabilistic: {np.max(all_prob) - np.min(all_prob):.4f}")
    print(f"  Power: {np.max(all_power) - np.min(all_power):.4f}")

if __name__ == "__main__":
    main()
