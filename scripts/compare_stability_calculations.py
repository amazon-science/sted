#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def current_calculation(similarity_values):
    """Current calculation using original formula with normalized std"""
    n = len(similarity_values)
    if n <= 1:
        return 1.0
    
    std_sim = float(np.std(similarity_values))
    zeros = n // 2
    ones = n - zeros
    max_possible_std = float(np.std([0] * zeros + [1] * ones))
    
    normalized_std = std_sim / max_possible_std if max_possible_std > 0 else 0.0
    
    # Current approach: original formula with normalized std
    stability_score = 1.0 / (1.0 + normalized_std*2)
    stability_score = stability_score ** 20
    return stability_score

def proposed_calculation(similarity_values):
    """Proposed calculation using (1.0 - normalized_std) ** 20"""
    n = len(similarity_values)
    if n <= 1:
        return 1.0
    
    std_sim = float(np.std(similarity_values))
    zeros = n // 2
    ones = n - zeros
    max_possible_std = float(np.std([0] * zeros + [1] * ones))
    
    normalized_std = std_sim / max_possible_std if max_possible_std > 0 else 0.0
    
    # Proposed approach: direct exponential with factor 20
    stability_score = (1.0 - normalized_std) ** 20
    return stability_score

# Test cases with different std levels
test_cases = [
    [1.0, 1.0, 1.0, 1.0],  # Perfect consistency (std=0)
    [0.95, 0.95, 0.95, 0.95],  # Perfect consistency, different values
    [0.9, 0.95, 0.92, 0.93],  # Very high consistency
    [0.8, 0.85, 0.82, 0.88],  # High consistency
    [0.7, 0.8, 0.75, 0.85],  # Medium consistency
    [0.5, 0.6, 0.4, 0.7],  # Low consistency
    [0.2, 0.8, 0.3, 0.9],  # Very low consistency
    [0.0, 1.0, 0.0, 1.0],  # Maximum inconsistency
]

print("Stability Score Comparison:")
print("Values\t\t\tStd\tCurrent\tProposed (**20)")
print("-" * 65)

results_current = []
results_proposed = []
std_values = []

for values in test_cases:
    std_val = np.std(values)
    current_score = current_calculation(values)
    proposed_score = proposed_calculation(values)
    
    results_current.append(current_score)
    results_proposed.append(proposed_score)
    std_values.append(std_val)
    
    print(f"{values}\t{std_val:.3f}\t{current_score:.3f}\t{proposed_score:.3f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Comparison by std values
ax1.scatter(std_values, results_current, c='blue', s=100, label='Current Method', alpha=0.7)
ax1.scatter(std_values, results_proposed, c='red', s=100, label='Proposed (**20)', alpha=0.7)
ax1.set_xlabel('Standard Deviation')
ax1.set_ylabel('Stability Score')
ax1.set_title('Stability Score vs Standard Deviation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Direct comparison
x_pos = np.arange(len(test_cases))
width = 0.35

ax2.bar(x_pos - width/2, results_current, width, label='Current Method', alpha=0.7)
ax2.bar(x_pos + width/2, results_proposed, width, label='Proposed (**20)', alpha=0.7)
ax2.set_xlabel('Test Case')
ax2.set_ylabel('Stability Score')
ax2.set_title('Direct Comparison of Methods')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'Case {i+1}' for i in range(len(test_cases))])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stability_calculation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Generate continuous comparison
print("\nContinuous Comparison (normalized std from 0 to 1):")
normalized_std_range = np.linspace(0, 1, 21)
current_continuous = []
proposed_continuous = []

for norm_std in normalized_std_range:
    # Current method
    current_val = 1.0 / (1.0 + norm_std*2)
    current_val = current_val ** 20
    current_continuous.append(current_val)
    
    # Proposed method
    proposed_val = (1.0 - norm_std) ** 20
    proposed_continuous.append(proposed_val)

print("Norm_Std\tCurrent\tProposed")
for i, norm_std in enumerate(normalized_std_range):
    print(f"{norm_std:.2f}\t\t{current_continuous[i]:.3f}\t{proposed_continuous[i]:.3f}")

# Plot continuous comparison
plt.figure(figsize=(10, 6))
plt.plot(normalized_std_range, current_continuous, 'b-', linewidth=2, label='Current Method')
plt.plot(normalized_std_range, proposed_continuous, 'r-', linewidth=2, label='Proposed (**20)')
plt.xlabel('Normalized Standard Deviation')
plt.ylabel('Stability Score')
plt.title('Continuous Comparison: Current vs Proposed (**20)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('continuous_stability_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
