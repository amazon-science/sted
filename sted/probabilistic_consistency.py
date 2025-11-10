"""
Probabilistic Consistency Metric

Based on kernel density estimation with adaptive bandwidth (Silverman's rule).
Consistency = Average probability that pairs are similar.

Reference: Silverman (1986) "Density Estimation for Statistics and Data Analysis"
"""

import numpy as np
from typing import List, Dict, Any
from itertools import combinations


class ProbabilisticConsistency:
    """
    Probabilistic consistency metric using Gaussian kernel.
    
    C(D) = E[exp(-d²/2σ₀²)]
    
    where σ₀ = std(D) (adaptive bandwidth following Silverman's rule)
    
    Interpretation: Average probability that random pair is similar
                   relative to typical dispersion.
    """
    
    def __init__(self, distance_fn, sigma_0=None, adaptive=True):
        """
        Args:
            distance_fn: Function computing distance between outputs
            sigma_0: Fixed tolerance threshold (None = adaptive)
            adaptive: Whether to adapt sigma_0 from data (recommended)
        """
        self.distance_fn = distance_fn
        self.sigma_0 = sigma_0
        self.adaptive = adaptive
    
    def compute_distances(self, outputs):
        """Compute all pairwise distances"""
        distances = []
        for v1, v2 in combinations(outputs, 2):
            d = self.distance_fn(v1, v2)
            distances.append(d)
        return np.array(distances)
    
    def compute_sigma(self, distances):
        """
        Compute adaptive sigma_0 using Silverman's bandwidth rule.
        
        Uses standard deviation of distances (measures dispersion).
        This is the standard approach in kernel density estimation.
        
        Reference: Silverman (1986), Section 3.4
        """
        if len(distances) == 0:
            return 0.02  # Default for typical LLM outputs
        
        # Use std of distances as bandwidth
        # This directly measures dispersion
        sigma = np.std(distances)
        
        # Avoid zero sigma (perfect consistency case)
        return max(sigma, 1e-6)
    
    def compute_consistency(self, outputs, return_details=False):
        """
        Compute probabilistic consistency score.
        
        Returns:
            Consistency score in [0, 1]
        """
        if len(outputs) < 2:
            return 1.0 if not return_details else {'consistency': 1.0}
        
        # Compute distances
        distances = self.compute_distances(outputs)
        
        if len(distances) == 0:
            return 1.0 if not return_details else {'consistency': 1.0}
        
        # Determine sigma_0
        if self.adaptive:
            sigma_0 = self.compute_sigma(distances)
        else:
            sigma_0 = self.sigma_0 if self.sigma_0 is not None else 0.02
        
        # Ensure sigma_0 is not zero
        sigma_0 = max(sigma_0, 1e-6)
        
        # Probabilistic consistency (Gaussian kernel)
        similarities = np.exp(-distances**2 / (2 * sigma_0**2))
        consistency = float(np.mean(similarities))
        
        if not return_details:
            return consistency
        
        return {
            'consistency': consistency,
            'sigma_0': sigma_0,
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'median_distance': float(np.median(distances)),
            'n_pairs': len(distances)
        }
