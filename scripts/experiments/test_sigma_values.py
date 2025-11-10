#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.probabilistic_consistency import ProbabilisticConsistency

# Load results
with open('probabilistic_consistency_results.json', 'r') as f:
    data = json.load(f)

# Get sample outputs
evaluator = SemanticJsonTreeConsistencyEvaluator()

def distance_fn(v1, v2):
    sim = evaluator.calculate_tree_edit_distance_opt(v1, v2, variation_type='combined')
    return 1.0 - sim

# Test different sigma_0 values
sigma_values = [0.01, 0.05, 0.1, 0.15, 0.2, 'adaptive']

print("\n" + "="*80)
print("TESTING DIFFERENT σ₀ VALUES")
print("="*80)

# Collect distances from existing results
all_distances = []
for model_results in data.values():
    for r in model_results:
        if r['mean_distance'] > 0:
            all_distances.append(r['mean_distance'])

print(f"\nDistance statistics:")
print(f"  Mean: {np.mean(all_distances):.4f}")
print(f"  Median: {np.median(all_distances):.4f}")
print(f"  Std: {np.std(all_distances):.4f}")

# Simulate consistency scores for different sigma_0
results_by_sigma = {}

for sigma in sigma_values:
    scores = []
    for dist in all_distances:
        if sigma == 'adaptive':
            s = np.median(all_distances)
        else:
            s = sigma
        score = np.exp(-dist**2 / (2 * s**2))
        scores.append(score)
    
    results_by_sigma[str(sigma)] = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'range': np.max(scores) - np.min(scores)
    }
    
    print(f"\nσ₀ = {sigma}:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print(f"  Spread: {np.max(scores) - np.min(scores):.4f}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, sigma in enumerate(sigma_values):
    scores = []
    for dist in all_distances:
        if sigma == 'adaptive':
            s = np.median(all_distances)
        else:
            s = sigma
        score = np.exp(-dist**2 / (2 * s**2))
        scores.append(score)
    
    axes[idx].hist(scores, bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'σ₀ = {sigma}')
    axes[idx].set_xlabel('Consistency Score')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axvline(np.mean(scores), color='r', linestyle='--', label=f'Mean={np.mean(scores):.2f}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('sigma_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: sigma_comparison.png")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nBased on discrimination (range):")
for sigma in sigma_values:
    r = results_by_sigma[str(sigma)]
    print(f"  σ₀={sigma:10s}: Range={r['range']:.4f}")

best_sigma = max(sigma_values[:-1], key=lambda s: results_by_sigma[str(s)]['range'])
print(f"\n✓ Best σ₀ for discrimination: {best_sigma}")
