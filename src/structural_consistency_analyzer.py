"""
Structural Consistency Analyzer for evaluating consistency across multiple JSON outputs.

This module provides high-level analysis capabilities for batch evaluation of JSON structural consistency,
separate from the core pairwise comparison logic.
"""

import datetime
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from bert_score import score as bert_score

from .utils import collect_all_values
from .semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

class StructuralConsistencyAnalyzer:
    """High-level analyzer for evaluating consistency across multiple JSON outputs."""
    
    def __init__(self, evaluator):
        """
        Initialize the analyzer with a semantic evaluator.
        
        Args:
            evaluator: Instance of SemanticJsonTreeConsistencyEvaluator
        """
        self.evaluator = evaluator
    
    def collect_all_string_pairs(self, json_outputs: Union[Dict, List[Dict]], gt: Union[Dict, List[Dict], None] = None) -> List[Tuple[str, str]]:
        # get all values from json_outputs
        output_values = collect_all_values(json_outputs)
        
        pairs = []
        seen = set()
            
        ref_values = collect_all_values(gt) if gt else output_values.copy()
            
        # create all pairs from output_values and gt_values
        for item1 in ref_values:
            for item2 in output_values:
                # Sort the pair to ensure consistent ordering
                pair_tuple = tuple(sorted([str(item1), str(item2)]))
                if pair_tuple not in seen:
                    seen.add(pair_tuple)
                    pairs.append((item1, item2))
        return pairs
    
    def batch_compute_similarities(self, pairs: List[Tuple[str, str]], method='cosine') -> Dict[Tuple[str, str], float]:
        """
        Batch compute similarities for all unique pairs using the underlying evaluator.
        
        Args:
            pairs: List of string pairs to compute similarities for
            method: Similarity computation method ('cosine' or 'bertscore')
            
        Returns:
            Dictionary mapping pairs to similarity scores
        """
        uncached_pairs = [(s1, s2) for s1, s2 in pairs if self.evaluator.cache.get(s1, s2) is None]
    
        if not uncached_pairs:
            # Return cached values for requested pairs
            return {(s1, s2): self.evaluator.cache.get(s1, s2) for s1, s2 in pairs 
                    if self.evaluator.cache.get(s1, s2) is not None}
        
        batch_size = min(self.evaluator.batch_size_bertscore, len(uncached_pairs))
        
        # Process pairs by batch
        for i in range(0, len(uncached_pairs), batch_size):
            batch = uncached_pairs[i:i+batch_size]
            refs, cands = zip(*batch)
            
            if method == "bertscore":
                P, R, F1 = bert_score(list(cands), list(refs), lang="en", verbose=False)
                scores = [float(f.item()) for f in F1]
            else:
                scores = [self.evaluator._calculate_semantic_similarity(s1, s2) for s1, s2 in zip(list(cands), list(refs))]
            
            self.evaluator.cache.batch_set(batch, scores)
        
        return self.evaluator.cache.cache
    
    def evaluate_structural_consistency(self, json_outputs: List[Dict[str, Any]], 
                                      gt: Dict[str, Any] = None, 
                                      method_name: str = "ted") -> Dict[str, Any]:
        """
        Evaluate structural consistency across multiple JSON outputs with enhanced metrics.
        
        Args:
            json_outputs: List of JSON objects to evaluate
            gt: Ground truth JSON object (optional)
            method_name: Similarity method to use ('ted', 'bertscore', 'deepdiff')
            
        Returns:
            Dictionary with comprehensive consistency metrics
        """
        n = len(json_outputs)
        if gt is None and n < 2:
            return {
                "error": "Need at least 2 outputs to evaluate consistency",
                "valid_count": n
            }
        
        # Prepare similarity computation by collecting all string pairs
        if gt:
            all_pairs = self.collect_all_string_pairs(json_outputs, gt)
            self.batch_compute_similarities(all_pairs)
            
            # Calculate similarities between ground truth and each output
            similarity_values = [
                self.evaluator.calculate_similarity_method[method_name](gt, json_output) 
                for json_output in json_outputs
            ]
        else:
            all_pairs = self.collect_all_string_pairs(json_outputs)
            self.batch_compute_similarities(all_pairs)
            
            # Calculate pairwise similarities between all outputs
            similarity_values = []
            for i in range(n-1):
                for j in range(i+1, n):
                    sim = self.evaluator.calculate_similarity_method[method_name](json_outputs[i], json_outputs[j])
                    similarity_values.append(sim)

        # Calculate essential consistency metrics (this includes mean/std calculation internally)
        consistency_metrics = self._calculate_advanced_metrics(similarity_values, gt is not None)
        
        # Prepare focused report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_outputs_analyzed": n,
            "method_used": method_name,
            "has_ground_truth": gt is not None,
            
            # Primary consistency metrics (what actually matters for LLM evaluation)
            "consistency_metrics": consistency_metrics,
            
            # Supporting data for debugging/analysis (but not primary metrics)
            "supporting_stats": {
                "mean_similarity": sum(similarity_values) / len(similarity_values) if similarity_values else 1.0,
                "std_deviation": float(np.std(similarity_values)) if len(similarity_values) > 1 else 0.0,
                "min_similarity": min(similarity_values) if similarity_values else 1.0,
                "max_similarity": max(similarity_values) if similarity_values else 1.0,
                "median_similarity": float(np.median(similarity_values)) if similarity_values else 1.0,
                "count": len(similarity_values)
            },
            
            # Raw similarity values for further analysis
            "raw_similarities": similarity_values
        }
        
        return report
    
    def _calculate_advanced_metrics(self, similarity_values: List[float], has_gt: bool) -> Dict[str, float]:
        """
        Calculate essential consistency metrics for LLM output evaluation.
        
        Args:
            similarity_values: List of similarity scores
            has_gt: Whether ground truth was used
            
        Returns:
            Dictionary of essential metrics
        """
        if not similarity_values:
            return {}
        
        similarity_array = np.array(similarity_values)
        mean_sim = float(np.mean(similarity_array))
        std_sim = float(np.std(similarity_array))
        
        # Calculate coefficient of variation for use in metrics
        cv = std_sim / mean_sim if mean_sim > 1e-10 else 0.0
        
        # === PRIMARY METRIC: Consistency Coefficient ===
        # Combines accuracy (mean) with stability (penalizes high variance)
        # This is the most comprehensive single metric
        if mean_sim > 1e-10:
            variance_penalty = min(cv ** 1.5, 1.0)
            consistency_coefficient = mean_sim * (1 - variance_penalty)
        else:
            consistency_coefficient = 0.0
        
        # === SUPPORTING METRICS (choose based on analysis needs) ===
        
        # Option A: Normalized CV - for relative variability analysis
        normalized_cv = cv
        
        # Option B: Stability Score - for absolute variability analysis  
        stability_score = 1.0 / (1.0 + std_sim)
        
        return {
            # Primary comprehensive metric
            "consistency_coefficient": consistency_coefficient,
            
            # Choose one based on your analysis focus:
            "normalized_cv": normalized_cv,        # Relative variability (recommended for comparing across different similarity scales)
            "stability_score": stability_score     # Absolute variability (recommended for fixed similarity scales)
        }