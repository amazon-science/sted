"""
Pairwise Dispersion Consistency (PDC) Metric

A principled metric for evaluating consistency of structured LLM outputs.

Reference: [Paper title], NeurIPS 2025
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import combinations
import warnings


class PDCMetric:
    """
    Pairwise Dispersion Consistency (PDC) Metric
    
    Measures consistency of a set of structured outputs by analyzing
    the dispersion of all pairwise distances.
    
    Theoretical Properties:
    - Metric space axioms satisfied
    - Bounded: PDC ∈ [0, 1]
    - Monotonic: Decreasing in dispersion
    - Permutation invariant
    - Size normalized
    """
    
    def __init__(self, 
                 distance_fn,
                 alpha: float = 2.0,
                 beta: float = 20.0,
                 include_reliability: bool = True):
        """
        Initialize PDC metric.
        
        Args:
            distance_fn: Function computing distance between two outputs
                        Should return value in [0, 1]
            alpha: Scaling factor for normalization (default: 2.0)
            beta: Steepness parameter for power transformation (default: 20.0)
            include_reliability: Whether to include empty ratio penalty
        """
        self.distance_fn = distance_fn
        self.alpha = alpha
        self.beta = beta
        self.include_reliability = include_reliability
    
    def _is_empty(self, output: Any) -> bool:
        """Check if output is empty/invalid"""
        if output is None:
            return True
        if isinstance(output, (dict, list)) and len(output) == 0:
            return True
        return False
    
    def _compute_max_dispersion(self, n: int) -> float:
        """
        Compute theoretical maximum dispersion for n samples.
        
        Theorem: Maximum occurs when half distances = 0, half = 1
        
        Args:
            n: Number of samples
            
        Returns:
            Maximum possible standard deviation
        """
        if n < 2:
            return 0.0
        
        # Maximum variance when distribution is bimodal at extremes
        p = (n // 2) / n
        max_variance = p * (1 - p)
        return np.sqrt(max_variance)
    
    def compute_pairwise_distances(self, 
                                   outputs: List[Any],
                                   verbose: bool = False) -> np.ndarray:
        """
        Compute all pairwise distances.
        
        Args:
            outputs: List of structured outputs
            verbose: Whether to print progress
            
        Returns:
            Array of pairwise distances
        """
        n = len(outputs)
        if n < 2:
            return np.array([])
        
        distances = []
        total_pairs = n * (n - 1) // 2
        
        for i, (out1, out2) in enumerate(combinations(outputs, 2)):
            try:
                dist = self.distance_fn(out1, out2)
                distances.append(dist)
                
                if verbose and (i + 1) % 100 == 0:
                    print(f"Computed {i+1}/{total_pairs} pairs")
                    
            except Exception as e:
                warnings.warn(f"Error computing distance: {e}")
                distances.append(1.0)  # Maximum distance on error
        
        return np.array(distances)
    
    def compute_dispersion(self, distances: np.ndarray) -> Dict[str, float]:
        """
        Compute dispersion statistics.
        
        Args:
            distances: Array of pairwise distances
            
        Returns:
            Dictionary with dispersion metrics
        """
        if len(distances) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'variance': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        return {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'variance': float(np.var(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'median': float(np.median(distances))
        }
    
    def compute_pdc(self, 
                   outputs: List[Any],
                   return_details: bool = False) -> float:
        """
        Compute Pairwise Dispersion Consistency score.
        
        Algorithm:
        1. Compute all pairwise distances D = {d(vᵢ, vⱼ)}
        2. Calculate dispersion σ(D)
        3. Normalize: σ_norm = σ / σ_max(n)
        4. Transform: PDC = (1 / (1 + α·σ_norm))^β
        5. Apply reliability penalty if enabled
        
        Args:
            outputs: List of structured outputs
            return_details: Whether to return detailed metrics
            
        Returns:
            PDC score in [0, 1] (or dict if return_details=True)
        """
        n = len(outputs)
        
        # Handle edge cases
        if n == 0:
            return 0.0 if not return_details else {'pdc': 0.0, 'error': 'No outputs'}
        
        if n == 1:
            return 1.0 if not return_details else {'pdc': 1.0, 'note': 'Single output'}
        
        # Count empty outputs
        empty_count = sum(1 for out in outputs if self._is_empty(out))
        empty_ratio = empty_count / n
        
        # Filter valid outputs
        valid_outputs = [out for out in outputs if not self._is_empty(out)]
        n_valid = len(valid_outputs)
        
        if n_valid < 2:
            result = {
                'pdc': 0.0,
                'pdc_penalized': 0.0,
                'empty_ratio': empty_ratio,
                'valid_count': n_valid,
                'error': 'Insufficient valid outputs'
            }
            return result if return_details else 0.0
        
        # Compute pairwise distances
        distances = self.compute_pairwise_distances(valid_outputs)
        
        if len(distances) == 0:
            result = {
                'pdc': 0.0,
                'pdc_penalized': 0.0,
                'empty_ratio': empty_ratio,
                'error': 'Failed to compute distances'
            }
            return result if return_details else 0.0
        
        # Compute dispersion
        dispersion_stats = self.compute_dispersion(distances)
        sigma = dispersion_stats['std']
        
        # Normalize dispersion
        sigma_max = self._compute_max_dispersion(n_valid)
        sigma_norm = sigma / sigma_max if sigma_max > 0 else 0.0
        
        # Compute PDC score with power transformation
        pdc_score = (1.0 / (1.0 + self.alpha * sigma_norm)) ** self.beta
        
        # Apply reliability penalty
        if self.include_reliability:
            pdc_penalized = pdc_score * (1.0 - empty_ratio)
        else:
            pdc_penalized = pdc_score
        
        if not return_details:
            return pdc_penalized
        
        # Return detailed results
        return {
            'pdc': pdc_score,
            'pdc_penalized': pdc_penalized,
            'empty_ratio': empty_ratio,
            'valid_count': n_valid,
            'total_count': n,
            'dispersion': dispersion_stats,
            'sigma_normalized': sigma_norm,
            'sigma_max': sigma_max,
            'parameters': {
                'alpha': self.alpha,
                'beta': self.beta
            }
        }
    
    def compute_pdc_batch(self, 
                         output_sets: List[List[Any]],
                         verbose: bool = False) -> List[float]:
        """
        Compute PDC for multiple output sets.
        
        Args:
            output_sets: List of output sets
            verbose: Whether to print progress
            
        Returns:
            List of PDC scores
        """
        scores = []
        
        for i, outputs in enumerate(output_sets):
            if verbose and (i + 1) % 10 == 0:
                print(f"Processing set {i+1}/{len(output_sets)}")
            
            score = self.compute_pdc(outputs)
            scores.append(score)
        
        return scores


def create_pdc_metric(evaluator, 
                     alpha: float = 2.0,
                     beta: float = 20.0,
                     variation_type: str = 'combined') -> PDCMetric:
    """
    Factory function to create PDC metric with STED distance.
    
    Args:
        evaluator: SemanticJsonTreeConsistencyEvaluator instance
        alpha: Scaling factor
        beta: Steepness parameter
        variation_type: Type of variation ('structural', 'content', 'combined')
        
    Returns:
        Configured PDCMetric instance
    """
    def distance_fn(out1, out2):
        similarity = evaluator.calculate_tree_edit_distance_opt(
            out1, out2, variation_type=variation_type
        )
        return 1.0 - similarity
    
    return PDCMetric(
        distance_fn=distance_fn,
        alpha=alpha,
        beta=beta,
        include_reliability=True
    )


# Example usage
if __name__ == "__main__":
    # Mock distance function for demonstration
    def mock_distance(x, y):
        """Simple mock distance based on dict difference"""
        if x == y:
            return 0.0
        return 0.3  # Arbitrary distance
    
    # Create metric
    pdc = PDCMetric(distance_fn=mock_distance, alpha=2.0, beta=20.0)
    
    # Test outputs
    outputs = [
        {'name': 'John', 'age': 30},
        {'name': 'John', 'age': 31},
        {'name': 'John', 'age': 30},
    ]
    
    # Compute PDC
    result = pdc.compute_pdc(outputs, return_details=True)
    
    print("PDC Results:")
    print(f"  PDC Score: {result['pdc']:.6f}")
    print(f"  PDC Penalized: {result['pdc_penalized']:.6f}")
    print(f"  Empty Ratio: {result['empty_ratio']:.2%}")
    print(f"  Valid Count: {result['valid_count']}/{result['total_count']}")
    print(f"  Dispersion (σ): {result['dispersion']['std']:.6f}")
