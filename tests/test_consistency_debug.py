#!/usr/bin/env python3
import numpy as np
from itertools import combinations

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

evaluator = SemanticJsonTreeConsistencyEvaluator()

variations = [
    {'name': 'John', 'age': 30},
    {'name': 'John', 'age': 31},
    {'name': 'John', 'age': 32},
    {'name': 'Jane', 'age': 30},
]

print("Testing pairwise distances:")
print("="*60)
pairwise_distances = []
for i, (v1, v2) in enumerate(combinations(variations, 2)):
    similarity = evaluator.calculate_tree_edit_distance_opt(v1, v2, variation_type='combined')
    distance = 1.0 - similarity
    pairwise_distances.append(distance)
    print(f"Pair {i+1}: similarity={similarity:.4f}, distance={distance:.4f}")

print(f"\nStd deviation: {np.std(pairwise_distances):.4f}")

print("\n" + "="*60)
print("WITHOUT Power Transformation:")
metrics = evaluator.calculate_variation_consistency(variations, method='sted', variation_type='combined', apply_power_transform=False)
print(f"Consistency score: {metrics['consistency_score']:.4f}")
print(f"Penalized: {metrics['penalized_consistency']:.4f}")

print("\n" + "="*60)
print("WITH Power Transformation:")
metrics = evaluator.calculate_variation_consistency(variations, method='sted', variation_type='combined', apply_power_transform=True)
print(f"Consistency score: {metrics['consistency_score']:.4f}")
print(f"Penalized: {metrics['penalized_consistency']:.4f}")
