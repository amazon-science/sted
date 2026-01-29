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
        uncached_pairs = [(s1, s2) for s1, s2 in pairs if self.evaluator._cache.get(s1, s2) is None]

        if not uncached_pairs:
            # Return cached values for requested pairs
            return {(s1, s2): self.evaluator._cache.get(s1, s2) for s1, s2 in pairs
                    if self.evaluator._cache.get(s1, s2) is not None}

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

            self.evaluator._cache.batch_set(batch, scores)

        return self.evaluator._cache.cache

    def evaluate_structural_consistency(self, json_outputs: List[Dict[str, Any]],
                                      gt: Dict[str, Any] = None,
                                      method_name: str = "ted", variation_type="combined",
                                      validity_rate: float = None) -> Dict[str, Any]:
        """
        Evaluate structural consistency across multiple JSON outputs with enhanced metrics.

        Args:
            json_outputs: List of JSON objects to evaluate
            gt: Ground truth JSON object (optional)
            method_name: Similarity method to use ('ted', 'bertscore', 'deepdiff')
            variation_type: Type of variation to consider ('structural', 'content', 'combined')
            validity_rate: Pre-calculated validity rate from caller (if None, will calculate from inputs)

        Returns:
            Dictionary with comprehensive consistency metrics
        """
        # Filter out empty/invalid responses to avoid misleading consistency scores
        # Empty responses ([], {}, None, "") would give identical similarity scores
        # which incorrectly inflates consistency metrics (especially without GT)
        total_outputs = len(json_outputs)
        valid_outputs = [output for output in json_outputs if output]
        valid_count = len(valid_outputs)
        # Use passed validity_rate if provided (from caller who knows original total),
        # otherwise calculate from what we receive (fallback for backward compatibility)
        if validity_rate is None:
            validity_rate = valid_count / total_outputs if total_outputs > 0 else 0.0

        n = valid_count

        # Handle edge cases with insufficient valid outputs
        if n == 0:
            # No valid outputs - return minimum scores
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "num_outputs_analyzed": 0,
                "num_outputs_total": total_outputs,
                "validity_rate": 0.0,
                "method_used": method_name,
                "has_ground_truth": gt is not None,
                # New interpretable metrics (ICML 2026 formulation)
                "c_mean": 0.0,
                "d_std": 0.0,
                "r_v": 0.0,
                "c_adj": 0.0,
                # Legacy consistency_metrics for backward compatibility
                "consistency_metrics": {
                    "c_mean": 0.0,
                    "d_std": 0.0,
                    "d_std_normalized": 0.0,
                    "r_v": 0.0,
                    "c_adj": 0.0,
                    "consistency_coefficient": 0.0,
                    "stability_score": 0.0,
                },
                "supporting_stats": {
                    "mean_similarity": 0.0,
                    "std_deviation": 0.0,
                    "min_similarity": 0.0,
                    "max_similarity": 0.0,
                    "median_similarity": 0.0,
                    "count": 0
                },
                "raw_similarities": []
            }

        if gt is None and n < 2:
            return {
                "error": "Need at least 2 valid outputs to evaluate consistency without ground truth",
                "valid_count": n,
                "num_outputs_total": total_outputs,
                "validity_rate": validity_rate
            }

        # Use filtered valid_outputs for consistency calculation
        json_outputs = valid_outputs

        # Prepare similarity computation by collecting all string pairs
        if gt:

            if isinstance(gt, list) and len(gt) == 1:
                gt = gt[0]

            all_pairs = self.collect_all_string_pairs(json_outputs, gt)
            self.batch_compute_similarities(all_pairs)

            # Calculate similarities between ground truth and each output
            similarity_values = [
                self.evaluator.calculate_similarity_method[method_name](gt, json_output, variation_type)
                for json_output in json_outputs
            ]
        else:
            all_pairs = self.collect_all_string_pairs(json_outputs)
            self.batch_compute_similarities(all_pairs)

            # Calculate pairwise similarities between all outputs
            similarity_values = []
            for i in range(n-1):
                for j in range(i+1, n):
                    sim = self.evaluator.calculate_similarity_method[method_name](json_outputs[i], json_outputs[j], variation_type)
                    similarity_values.append(sim)

        # Calculate all consistency metrics using unified method
        # Returns interpretable (ICML 2026) + benchmarking + legacy metrics
        consistency_metrics = self._calculate_consistency_metrics(similarity_values, validity_rate=validity_rate)

        # Prepare focused report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_outputs_analyzed": n,
            "num_outputs_total": total_outputs,
            "validity_rate": validity_rate,
            "method_used": method_name,
            "has_ground_truth": gt is not None,

            # Primary consistency metrics (what actually matters for LLM evaluation)
            "consistency_metrics": consistency_metrics,

            # Supporting data for debugging/analysis (but not primary metrics)
            "supporting_stats": {
                "mean_similarity": sum(similarity_values) / len(similarity_values) if similarity_values else 0.0,
                "std_deviation": float(np.std(similarity_values)) if len(similarity_values) > 1 else 0.0,
                "min_similarity": min(similarity_values) if similarity_values else 0.0,
                "max_similarity": max(similarity_values) if similarity_values else 0.0,
                "median_similarity": float(np.median(similarity_values)) if similarity_values else 0.0,
                "count": len(similarity_values)
            },

            # Raw similarity values for further analysis
            "raw_similarities": similarity_values
        }

        return report

    def _calculate_consistency_metrics(self, similarity_values: List[float],
                                         validity_rate: float = 1.0,
                                         steepness_factor: int = 20) -> Dict[str, float]:
        """
        Calculate unified consistency metrics combining ICML 2026 interpretable statistics
        with benchmarking-oriented stability scores.

        This unified method computes all metrics in a single pass:

        **Interpretable Metrics (ICML 2026 - for reporting):**
        - C_mean: Mean pairwise consistency (average STED similarity)
        - D_std: Dispersion (standard deviation of pairwise similarities)
        - r_v: Validity rate (fraction of parseable outputs)
        - C_adj: Validity-adjusted consistency = r_v * C_mean

        **Benchmarking Metrics (for model selection/optimization):**
        - S_α: Stability score with amplified discrimination
        - R: Combined ranking scalar = r_v * C_mean * S_α

        Args:
            similarity_values: List of pairwise similarity scores
            validity_rate: Ratio of valid (non-empty) outputs (default: 1.0)
            steepness_factor: Exponent α for stability score (default: 20)

        Returns:
            Dictionary with all consistency metrics
        """
        if not similarity_values:
            return {
                # Interpretable metrics (ICML 2026)
                "c_mean": 0.0,
                "d_std": 0.0,
                "d_std_normalized": 0.0,
                "r_v": validity_rate,
                "c_adj": 0.0,
                # Benchmarking metrics
                "stability_score": 0.0 if validity_rate == 0 else 1.0,
                "ranking_score": 0.0,
                # Legacy metrics
                "consistency_coefficient": 0.0,
                "normalized_cv": 0.0,
                "empty_ratio": 1.0 - validity_rate,
                "penalized_consistency_coefficient": 0.0,
                "penalized_stability_score": 0.0,
                # Alias
                "mean_similarity": 0.0,
            }

        similarity_array = np.array(similarity_values)
        n = len(similarity_values)

        # === Core Statistics ===
        # C_mean: Mean Pairwise Consistency
        # "If we sample two runs at random, how similar are they on average?"
        c_mean = float(np.mean(similarity_array))

        # D_std: Dispersion (standard deviation of pairwise similarities)
        # Captures stability - low D_std means consistent outputs
        d_std = float(np.std(similarity_array)) if n > 1 else 0.0

        # D_std_normalized: Normalize by max achievable std for n values in [0,1]
        if n > 1:
            # Max std achieved when half values are 0 and half are 1
            k_zeros = n // 2
            k_ones = n - k_zeros
            d_max = float(np.std([0.0] * k_zeros + [1.0] * k_ones))
            d_std_normalized = d_std / d_max if d_max > 0 else 0.0
        else:
            d_std_normalized = 0.0

        # r_v: Validity rate (passed from caller)
        r_v = validity_rate

        # C_adj: Validity-Adjusted Consistency
        # Single deployment-oriented score: r_v * C_mean
        c_adj = r_v * c_mean

        # === Benchmarking Metrics ===
        # S_α: Stability Score with power transformation for amplified discrimination
        # Formula: S_α = (1 / (1 + 2 * D̂_std))^α
        if n <= 1:
            stability_score = 1.0
        else:
            stability_score = (1.0 / (1.0 + d_std_normalized * 2)) ** steepness_factor

        # R: Combined Ranking Scalar for model selection
        # R = r_v * C_mean * S_α (from ICML paper)
        ranking_score = r_v * c_mean * stability_score

        # === Legacy Metrics (for backward compatibility) ===
        # Coefficient of Variation
        cv = d_std / c_mean if c_mean > 1e-10 else 0.0

        # Consistency Coefficient with variance penalty
        if c_mean > 1e-10:
            variance_penalty = min(cv ** 1.5, 1.0)
            consistency_coefficient = (c_mean * (1 - variance_penalty)) ** 5
        else:
            consistency_coefficient = 0.0

        # Empty ratio and penalized scores
        empty_ratio = 1.0 - r_v

        return {
            # Interpretable metrics (ICML 2026 - primary reporting)
            "c_mean": c_mean,
            "d_std": d_std,
            "d_std_normalized": d_std_normalized,
            "r_v": r_v,
            "c_adj": c_adj,
            # Benchmarking metrics (model selection/optimization)
            "stability_score": stability_score,
            "ranking_score": ranking_score,
            # Legacy metrics (backward compatibility)
            "consistency_coefficient": consistency_coefficient,
            "normalized_cv": min(cv * 20.0, 1.0),
            "empty_ratio": empty_ratio,
            "penalized_consistency_coefficient": consistency_coefficient * r_v,
            "penalized_stability_score": stability_score * r_v,
            # Aliases for convenience
            "mean_similarity": c_mean,
        }

    def evaluate_field_level_consistency(self, json_outputs: List[Dict[str, Any]],
                                       gt: Dict[str, Any] = None,
                                       exact_match_fields: set = None) -> Dict[str, Any]:
        """
        Evaluate consistency at field level across multiple JSON outputs.
        
        Args:
            json_outputs: List of JSON objects to evaluate
            gt: Ground truth JSON object (optional)
            exact_match_fields: Set of field names requiring exact match
            
        Returns:
            Dictionary with field-level consistency metrics
        """
        n = len(json_outputs)
        if gt is None and n < 2:
            return {"error": "Need at least 2 outputs to evaluate consistency"}

        field_consistency = {}

        if gt:
            # Compare each output against ground truth
            for i, output in enumerate(json_outputs):
                field_similarities = self.evaluator.calculate_field_level_similarity(
                    gt, output, exact_match_fields
                )

                for field_path, result in field_similarities.items():
                    if field_path not in field_consistency:
                        field_consistency[field_path] = []
                    field_consistency[field_path].append(result['similarity'])
        else:
            # Pairwise comparison between all outputs
            for i in range(n-1):
                for j in range(i+1, n):
                    field_similarities = self.evaluator.calculate_field_level_similarity(
                        json_outputs[i], json_outputs[j], exact_match_fields
                    )

                    for field_path, result in field_similarities.items():
                        if field_path not in field_consistency:
                            field_consistency[field_path] = []
                        field_consistency[field_path].append(result['similarity'])

        # Calculate consistency metrics for each field
        field_metrics = {}
        for field_path, similarities in field_consistency.items():
            if similarities:
                field_metrics[field_path] = {
                    "mean_similarity": float(np.mean(similarities)),
                    "std_deviation": float(np.std(similarities)),
                    "min_similarity": float(np.min(similarities)),
                    "max_similarity": float(np.max(similarities)),
                    "consistency_coefficient": self._calculate_field_consistency_coefficient(similarities)
                }

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_outputs_analyzed": n,
            "has_ground_truth": gt is not None,
            "field_level_metrics": field_metrics,
            "overall_field_consistency": float(np.mean([m["consistency_coefficient"] for m in field_metrics.values()])) if field_metrics else 0.0
        }

    def _calculate_field_consistency_coefficient(self, similarities: List[float]) -> float:
        """Calculate consistency coefficient for a single field."""
        if not similarities:
            return 0.0

        # Simple and intuitive: just use mean similarity
        # This aligns with human expectation that 2/3 matches = ~0.67 consistency
        return float(np.mean(similarities))
