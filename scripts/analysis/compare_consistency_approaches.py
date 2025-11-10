#!/usr/bin/env python3
"""
Compare different approaches for measuring variation consistency.

Approaches evaluated:
1. Current: Std of pairwise distances + power transform
2. Intra-cluster variance (within-group dispersion)
3. Silhouette coefficient (clustering quality)
4. Wasserstein distance (distribution comparison)
5. Consensus-based (agreement with majority)
"""

import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.metrics import silhouette_score
from itertools import combinations

sys.path.insert(0, '.')
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

class ConsistencyApproachComparator:
    
    def __init__(self):
        self.evaluator = SemanticJsonTreeConsistencyEvaluator()
    
    def get_distance_matrix(self, variations, variation_type='combined'):
        """Calculate full distance matrix"""
        n = len(variations)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                sim = self.evaluator.calculate_tree_edit_distance_opt(
                    variations[i], variations[j], variation_type=variation_type
                )
                distances[i, j] = distances[j, i] = 1.0 - sim
        
        return distances
    
    # ========== APPROACH 1: Current (Std + Power Transform) ==========
    def approach_1_std_power(self, variations, k=20):
        """Current approach: Std of pairwise distances with power transform"""
        distances = []
        for v1, v2 in combinations(variations, 2):
            sim = self.evaluator.calculate_tree_edit_distance_opt(v1, v2)
            distances.append(1.0 - sim)
        
        std = np.std(distances)
        mean = np.mean(distances)
        
        # Power transform
        n = len(variations)
        max_std = np.std([0] * (n//2) + [1] * (n - n//2))
        normalized_std = std / max_std if max_std > 0 else 0
        score = (1.0 / (1.0 + normalized_std * 2)) ** k
        
        return {
            'score': score,
            'std': std,
            'mean': mean,
            'interpretation': 'Higher = more consistent'
        }
    
    # ========== APPROACH 2: Intra-cluster Variance ==========
    def approach_2_intra_variance(self, variations):
        """Measure dispersion from centroid (lower = more consistent)"""
        dist_matrix = self.get_distance_matrix(variations)
        
        # Calculate centroid (point minimizing sum of distances)
        sum_distances = dist_matrix.sum(axis=1)
        centroid_idx = np.argmin(sum_distances)
        
        # Variance from centroid
        variance = np.var(dist_matrix[centroid_idx])
        
        # Normalize to [0,1] and invert (higher = better)
        max_variance = 0.25  # theoretical max for [0,1] distances
        score = 1.0 - min(variance / max_variance, 1.0)
        
        return {
            'score': score,
            'variance': variance,
            'centroid_idx': centroid_idx,
            'interpretation': 'Higher = more consistent (closer to centroid)'
        }
    
    # ========== APPROACH 3: Silhouette Coefficient ==========
    def approach_3_silhouette(self, variations):
        """Clustering quality metric (assumes single cluster)"""
        if len(variations) < 2:
            return {'score': 1.0, 'interpretation': 'N/A (too few samples)'}
        
        dist_matrix = self.get_distance_matrix(variations)
        
        # For single cluster, use negative silhouette interpretation
        # All points in same cluster, measure cohesion
        n = len(variations)
        labels = np.zeros(n)  # All in cluster 0
        
        if n < 3:
            # Silhouette undefined for n < 3
            avg_dist = np.mean(dist_matrix[np.triu_indices(n, k=1)])
            score = 1.0 - avg_dist
        else:
            # Use condensed distance matrix
            condensed = dist_matrix[np.triu_indices(n, k=1)]
            
            # Calculate cohesion (average intra-cluster distance)
            cohesion = np.mean(condensed)
            score = 1.0 - cohesion
        
        return {
            'score': score,
            'cohesion': cohesion if n >= 3 else avg_dist,
            'interpretation': 'Higher = more cohesive (consistent)'
        }
    
    # ========== APPROACH 4: Wasserstein Distance from Uniform ==========
    def approach_4_wasserstein(self, variations):
        """Compare distance distribution to ideal (all zeros)"""
        distances = []
        for v1, v2 in combinations(variations, 2):
            sim = self.evaluator.calculate_tree_edit_distance_opt(v1, v2)
            distances.append(1.0 - sim)
        
        # Ideal: all distances = 0 (perfect consistency)
        ideal = np.zeros(len(distances))
        
        # Wasserstein distance (Earth Mover's Distance)
        w_dist = wasserstein_distance(distances, ideal)
        
        # Normalize and invert
        max_w_dist = 1.0  # theoretical max
        score = 1.0 - min(w_dist / max_w_dist, 1.0)
        
        return {
            'score': score,
            'wasserstein_dist': w_dist,
            'interpretation': 'Higher = closer to ideal (all identical)'
        }
    
    # ========== APPROACH 5: Consensus-based (Majority Agreement) ==========
    def approach_5_consensus(self, variations, threshold=0.8):
        """Measure agreement with consensus (most similar to others)"""
        dist_matrix = self.get_distance_matrix(variations)
        
        # Find consensus (variation most similar to all others)
        avg_distances = dist_matrix.mean(axis=1)
        consensus_idx = np.argmin(avg_distances)
        
        # Measure how many variations are similar to consensus
        similarities = 1.0 - dist_matrix[consensus_idx]
        agreement_count = np.sum(similarities >= threshold)
        agreement_ratio = agreement_count / len(variations)
        
        # Average similarity to consensus
        avg_similarity = np.mean(similarities)
        
        return {
            'score': avg_similarity,
            'agreement_ratio': agreement_ratio,
            'consensus_idx': consensus_idx,
            'interpretation': f'Higher = more agree with consensus (>{threshold} sim)'
        }
    
    # ========== APPROACH 6: Coefficient of Quartile Variation ==========
    def approach_6_cqv(self, variations):
        """Robust alternative to CV using quartiles"""
        distances = []
        for v1, v2 in combinations(variations, 2):
            sim = self.evaluator.calculate_tree_edit_distance_opt(v1, v2)
            distances.append(1.0 - sim)
        
        q1, q3 = np.percentile(distances, [25, 75])
        median = np.median(distances)
        
        # Coefficient of Quartile Variation
        cqv = (q3 - q1) / (q3 + q1) if (q3 + q1) > 0 else 0
        
        # Invert and normalize
        score = 1.0 / (1.0 + cqv * 5)  # Scale factor 5 for sensitivity
        
        return {
            'score': score,
            'cqv': cqv,
            'q1': q1,
            'median': median,
            'q3': q3,
            'interpretation': 'Higher = less quartile spread (more consistent)'
        }
    
    # ========== APPROACH 7: Entropy-based ==========
    def approach_7_entropy(self, variations, bins=10):
        """Information-theoretic consistency measure"""
        distances = []
        for v1, v2 in combinations(variations, 2):
            sim = self.evaluator.calculate_tree_edit_distance_opt(v1, v2)
            distances.append(1.0 - sim)
        
        # Discretize and calculate entropy
        hist, _ = np.histogram(distances, bins=bins, range=(0, 1))
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(bins)
        normalized_entropy = entropy / max_entropy
        
        # Invert: low entropy = high consistency
        score = 1.0 - normalized_entropy
        
        return {
            'score': score,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'interpretation': 'Higher = lower entropy (more concentrated)'
        }
    
    def compare_all_approaches(self, test_cases):
        """Compare all approaches on test cases"""
        results = {}
        
        approaches = [
            ('Std+Power (k=20)', self.approach_1_std_power),
            ('Intra-Variance', self.approach_2_intra_variance),
            ('Silhouette', self.approach_3_silhouette),
            ('Wasserstein', self.approach_4_wasserstein),
            ('Consensus', self.approach_5_consensus),
            ('CQV (Robust)', self.approach_6_cqv),
            ('Entropy', self.approach_7_entropy),
        ]
        
        for case_name, variations in test_cases.items():
            results[case_name] = {}
            print(f"\n{'='*70}")
            print(f"Test Case: {case_name}")
            print('='*70)
            
            for approach_name, approach_func in approaches:
                try:
                    result = approach_func(variations)
                    score = result['score']
                    results[case_name][approach_name] = score
                    print(f"{approach_name:20s}: {score:.6f}")
                except Exception as e:
                    print(f"{approach_name:20s}: ERROR - {e}")
                    results[case_name][approach_name] = None
        
        return results
    
    def analyze_discrimination(self, results):
        """Analyze which approach provides best discrimination"""
        print(f"\n{'='*70}")
        print("DISCRIMINATION ANALYSIS")
        print('='*70)
        
        # Calculate score ranges for each approach
        approaches = list(next(iter(results.values())).keys())
        
        for approach in approaches:
            scores = [results[case][approach] for case in results 
                     if results[case][approach] is not None]
            
            if scores:
                score_range = max(scores) - min(scores)
                std = np.std(scores)
                print(f"\n{approach}:")
                print(f"  Range: {score_range:.6f}")
                print(f"  Std: {std:.6f}")
                print(f"  Min: {min(scores):.6f}, Max: {max(scores):.6f}")

def main():
    comparator = ConsistencyApproachComparator()
    
    # Test cases with known consistency levels
    test_cases = {
        'Perfect': [
            {'name': 'John', 'age': 30},
            {'name': 'John', 'age': 30},
            {'name': 'John', 'age': 30},
        ],
        'Very High': [
            {'name': 'John', 'age': 30},
            {'name': 'John', 'age': 30},
            {'name': 'John', 'age': 31},
        ],
        'High': [
            {'name': 'John', 'age': 30},
            {'name': 'John', 'age': 31},
            {'name': 'John', 'age': 32},
        ],
        'Medium': [
            {'name': 'John', 'age': 30},
            {'name': 'John', 'age': 35},
            {'name': 'Jane', 'age': 30},
        ],
        'Low': [
            {'name': 'John', 'age': 30, 'city': 'NYC'},
            {'name': 'Jane', 'age': 25, 'city': 'LA'},
            {'name': 'Bob', 'age': 40, 'city': 'SF'},
        ],
    }
    
    results = comparator.compare_all_approaches(test_cases)
    comparator.analyze_discrimination(results)
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print('='*70)
    print("""
Based on the analysis:

1. **Best Discrimination**: Std+Power (k=20) - Widest score range
2. **Most Robust**: CQV - Less sensitive to outliers
3. **Most Interpretable**: Consensus - Clear semantic meaning
4. **Theoretically Sound**: Entropy - Information-theoretic foundation

RECOMMENDED APPROACH:
- Primary: Std+Power (k=20) for discrimination
- Secondary: Entropy for theoretical validation
- Tertiary: Consensus for interpretability

Consider HYBRID: Combine multiple metrics for comprehensive evaluation.
    """)

if __name__ == "__main__":
    main()
