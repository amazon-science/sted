#!/usr/bin/env python3
import sys
import os
import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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

evaluator = SemanticJsonTreeConsistencyEvaluator()

def distance_fn(v1, v2):
    sim = evaluator.calculate_tree_edit_distance_opt(v1, v2, variation_type='combined')
    return 1.0 - sim

# Fixed sigma_0 = 0.05 (5% tolerance)
prob_metric = ProbabilisticConsistency(distance_fn, sigma_0=0.05, adaptive=False)

results = {}
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="llm_gen_results", help="LLM results directory")
args = parser.parse_args()
results_dir = args.results_dir

print("\n" + "="*80)
print("PROBABILISTIC CONSISTENCY (σ₀=0.05) vs POWER TRANSFORMATION")
print("="*80)

for model_dir in tqdm(os.listdir(results_dir), desc="Models"):
    model_path = os.path.join(results_dir, model_dir)
    if not os.path.isdir(model_path): continue
    
    model_name = extract_model_name(model_dir)
    results[model_name] = []
    
    for result_dir in sorted(os.listdir(model_path)):
        result_path = os.path.join(model_path, result_dir)
        if not os.path.isdir(result_path): continue
        
        temperature = extract_temperature_from_path(result_dir)
        if temperature is None: continue
        
        all_results_path = os.path.join(result_path, 'all_results.json')
        if not os.path.exists(all_results_path): continue
        
        with open(all_results_path, 'r') as f:
            data = json.load(f)
        
        for sample_idx, sample in enumerate(data['results']):
            responses = sample['responses'][:10]
            
            prob_result = prob_metric.compute_consistency(responses, return_details=True)
            
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
                'mean_distance': prob_result['mean_distance']
            })

with open('prob_fixed_sigma_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved")

# Analysis
all_prob = []
all_power = []
for model_results in results.values():
    for r in model_results:
        all_prob.append(r['prob_consistency'])
        all_power.append(r['power_consistency'])

print(f"\nProbabilistic (σ₀=0.05):")
print(f"  Mean: {np.mean(all_prob):.4f}, Range: {np.max(all_prob)-np.min(all_prob):.4f}")
print(f"\nPower (β=20):")
print(f"  Mean: {np.mean(all_power):.4f}, Range: {np.max(all_power)-np.min(all_power):.4f}")
