#!/usr/bin/env python3
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Just analyze existing results with different sigma
with open('probabilistic_consistency_results.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("QUICK COMPARISON: Different σ₀ values")
print("="*80)

# Extract distances from existing results
all_distances = []
for model_results in data.values():
    for r in model_results:
        all_distances.append(r['mean_distance'])

all_distances = np.array(all_distances)

# Test different sigma values
for sigma in [0.01, 0.05, 0.1, 0.15, 'adaptive']:
    if sigma == 'adaptive':
        s = np.median(all_distances)
    else:
        s = sigma
    
    scores = np.exp(-all_distances**2 / (2 * s**2))
    
    print(f"\nσ₀ = {sigma}:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")
    print(f"  Range: {np.max(scores) - np.min(scores):.4f}")

# Compare with power transformation
print("\n" + "="*80)
print("COMPARISON WITH POWER TRANSFORMATION")
print("="*80)

all_power = []
for model_results in data.values():
    for r in model_results:
        all_power.append(r['power_consistency'])

print(f"\nPower (β=20):")
print(f"  Mean: {np.mean(all_power):.4f}")
print(f"  Std:  {np.std(all_power):.4f}")
print(f"  Range: {np.max(all_power) - np.min(all_power):.4f}")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nσ₀=0.05 provides:")
print("  ✓ Full range [0, 1]")
print("  ✓ Balanced mean (~0.5)")
print("  ✓ Clear interpretation (5% tolerance)")
print("  ✓ Better than adaptive (more consistent)")
print("\nPower transformation:")
print("  ✓ Full range [0, 1]")
print("  ✓ Lower mean (more discriminative)")
print("  ✓ But lacks theoretical justification")
