#!/usr/bin/env python
"""
Generate human validation dataset for STED.

This script creates a stratified sample of JSON pairs for human annotation:
1. Synthetic pairs (controlled variations with known ground truth)
2. Real LLM output pairs (actual model generations)

Output: JSON file ready for human annotation interface.
"""

import json
import os
import argparse
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer


def compute_structural_metrics(json_obj: Any) -> Dict[str, Any]:
    """
    Compute structural metrics for a JSON object.

    Returns:
        depth: Maximum nesting depth
        node_count: Total number of nodes (keys + values)
        array_count: Number of arrays
        object_count: Number of nested objects
        leaf_count: Number of leaf values (strings, numbers, bools, nulls)
        max_array_length: Maximum length of any array
        complexity_category: 'simple', 'medium', or 'complex'
    """
    metrics = {
        "depth": 0,
        "node_count": 0,
        "array_count": 0,
        "object_count": 0,
        "leaf_count": 0,
        "max_array_length": 0,
        "string_count": 0,
        "number_count": 0,
    }

    def traverse(obj, current_depth=0):
        metrics["depth"] = max(metrics["depth"], current_depth)
        metrics["node_count"] += 1

        if isinstance(obj, dict):
            metrics["object_count"] += 1
            for key, value in obj.items():
                metrics["node_count"] += 1  # Count key as node
                traverse(value, current_depth + 1)
        elif isinstance(obj, list):
            metrics["array_count"] += 1
            metrics["max_array_length"] = max(metrics["max_array_length"], len(obj))
            for item in obj:
                traverse(item, current_depth + 1)
        else:
            # Leaf node
            metrics["leaf_count"] += 1
            if isinstance(obj, str):
                metrics["string_count"] += 1
            elif isinstance(obj, (int, float)):
                metrics["number_count"] += 1

    if json_obj is not None:
        traverse(json_obj)

    # Compute complexity category
    total_complexity = metrics["depth"] + metrics["node_count"] / 10 + metrics["array_count"]
    if total_complexity < 5:
        metrics["complexity_category"] = "simple"
    elif total_complexity < 15:
        metrics["complexity_category"] = "medium"
    else:
        metrics["complexity_category"] = "complex"

    return metrics


def compute_pair_structural_metrics(json_a: Any, json_b: Any) -> Dict[str, Any]:
    """Compute structural metrics for a pair of JSON objects."""
    metrics_a = compute_structural_metrics(json_a)
    metrics_b = compute_structural_metrics(json_b)

    # Compute combined/averaged metrics
    return {
        "json_a": metrics_a,
        "json_b": metrics_b,
        "avg_depth": (metrics_a["depth"] + metrics_b["depth"]) / 2,
        "avg_node_count": (metrics_a["node_count"] + metrics_b["node_count"]) / 2,
        "max_depth": max(metrics_a["depth"], metrics_b["depth"]),
        "max_node_count": max(metrics_a["node_count"], metrics_b["node_count"]),
        "complexity_category": (
            "complex" if metrics_a["complexity_category"] == "complex" or metrics_b["complexity_category"] == "complex"
            else "medium" if metrics_a["complexity_category"] == "medium" or metrics_b["complexity_category"] == "medium"
            else "simple"
        ),
    }


class HumanValidationDatasetGenerator:
    """Generate stratified sample pairs for human validation study."""

    def __init__(
        self,
        model_id: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 512,
    ):
        """Initialize STED evaluator for computing similarity scores."""
        print("Initializing STED evaluator...")
        self.evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id=model_id,
            embedding_dim=embedding_dim,
        )
        self.analyzer = StructuralConsistencyAnalyzer(self.evaluator)
        print("STED evaluator initialized.")

    def compute_all_scores(self, json_a: Dict, json_b: Dict) -> Dict[str, float]:
        """Compute STED and baseline scores for a pair."""
        scores = {}

        # STED scores
        try:
            scores["sted_combined"] = self.evaluator.calculate_tree_edit_distance_fast(
                json_a, json_b, variation_type="combined"
            )
            scores["sted_structural"] = self.evaluator.calculate_tree_edit_distance_fast(
                json_a, json_b, variation_type="structural"
            )
            scores["sted_semantic"] = self.evaluator.calculate_tree_edit_distance_fast(
                json_a, json_b, variation_type="content"
            )
        except Exception as e:
            print(f"STED error: {e}")
            scores["sted_combined"] = None
            scores["sted_structural"] = None
            scores["sted_semantic"] = None

        # DeepDiff baseline
        try:
            scores["deepdiff"] = self.evaluator.calculate_similarity_with_deepdiff(json_a, json_b)
        except Exception as e:
            print(f"DeepDiff error: {e}")
            scores["deepdiff"] = None

        # TED baseline (using ZSS)
        try:
            scores["ted"] = self.evaluator.calculate_tree_edit_distance(
                json_a, json_b, original_zss=True
            )
        except Exception as e:
            print(f"TED error: {e}")
            scores["ted"] = None

        # BERTScore baseline
        try:
            scores["bertscore"] = self.evaluator.calculate_bertscore(json_a, json_b)
        except Exception as e:
            print(f"BERTScore error: {e}")
            scores["bertscore"] = None

        return scores

    def load_synthetic_pairs(self, synthetic_dir: str) -> List[Dict]:
        """Load pairs from synthetic dataset files."""
        pairs = []

        if not os.path.exists(synthetic_dir):
            print(f"Synthetic directory not found: {synthetic_dir}")
            return pairs

        # Find all synthetic dataset files
        for filename in os.listdir(synthetic_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(synthetic_dir, filename)
            print(f"Loading synthetic data from {filename}...")

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Determine variation type from filename
                if "tool_call" in filename.lower():
                    variation_category = "tool_call"
                elif "schema" in filename.lower():
                    variation_category = "schema"
                elif "semantic" in filename.lower() or "sematic" in filename.lower():
                    variation_category = "semantic"
                elif "expression" in filename.lower():
                    variation_category = "expression"
                else:
                    variation_category = "unknown"

                # Extract pairs from synthetic data
                for sample in data:
                    base_sample = sample.get("base_sample") or sample.get("ground_truth")
                    variants = sample.get("variants", [])

                    # Handle flat/nested structure samples (no variants list)
                    if not variants and "variation" in sample:
                        pairs.append({
                            "source": "synthetic",
                            "variation_category": variation_category,
                            "variation_type": sample.get("variation_type", "unknown"),
                            "variation_ratio": 1.0,
                            "sample_id": sample.get("sample_id", "unknown"),
                            "json_a": base_sample,
                            "json_b": sample["variation"],
                        })
                    else:
                        # Sample from different variation ratios
                        for variant in variants:
                            pairs.append({
                                "source": "synthetic",
                                "variation_category": variation_category,
                                "variation_type": sample.get("variation_type", "unknown"),
                                "variation_ratio": variant.get("variation_ratio", 0.0),
                                "sample_id": sample.get("sample_id", "unknown"),
                                "json_a": base_sample,
                                "json_b": variant.get("variation"),
                            })

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        print(f"Loaded {len(pairs)} synthetic pairs")
        return pairs

    def load_toucan_ground_truth_pairs(
        self,
        toucan_data_path: str,
        results_dir: str,
        max_samples_per_model: int = 50,
    ) -> List[Dict]:
        """
        Load Toucan tool call pairs comparing LLM outputs to ground truth.

        This creates pairs where json_a is ground truth and json_b is LLM output,
        which is clearer for human annotators than comparing two LLM outputs.
        """
        pairs = []

        # Load ground truth
        try:
            with open(toucan_data_path, "r") as f:
                toucan_data = json.load(f)
            gt_by_id = {item["id"]: item["tool_calls"] for item in toucan_data if "tool_calls" in item}
        except Exception as e:
            print(f"Error loading Toucan ground truth: {e}")
            return pairs

        # Find LLM generation results
        dataset_dir = os.path.join(results_dir, "toucan")
        if not os.path.exists(dataset_dir):
            print(f"Toucan results directory not found: {dataset_dir}")
            return pairs

        model_dirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("generations-")
        ]

        for model_dir in model_dirs:
            model_name = model_dir.replace("generations-", "")
            # Remove date suffix if present
            model_name = "-".join(model_name.split("-")[:-1]) if model_name.split("-")[-1].isdigit() else model_name
            model_path = os.path.join(dataset_dir, model_dir)

            # Find lowest temperature run for most consistent outputs
            # Handle both "temp_*" and "run_*" naming conventions
            results_file = None
            for d in os.listdir(model_path):
                if not os.path.isdir(os.path.join(model_path, d)):
                    continue
                if d.startswith("temp_0.0") or d.startswith("temp_0.1"):
                    candidate = os.path.join(model_path, d, "all_results.json")
                    if os.path.exists(candidate):
                        results_file = candidate
                        break
                elif d.startswith("run_") and "_temp_0_0" in d:
                    candidate = os.path.join(model_path, d, "all_results.json")
                    if os.path.exists(candidate):
                        results_file = candidate
                        break
                elif d.startswith("run_") and "_temp_0_1" in d:
                    candidate = os.path.join(model_path, d, "all_results.json")
                    if os.path.exists(candidate):
                        results_file = candidate
                        break

            if results_file is None:
                # Try any run directory
                for d in os.listdir(model_path):
                    if os.path.isdir(os.path.join(model_path, d)):
                        candidate = os.path.join(model_path, d, "all_results.json")
                        if os.path.exists(candidate):
                            results_file = candidate
                            break

            if results_file is None:
                continue

            try:
                with open(results_file, "r") as f:
                    data = json.load(f)

                # Handle both formats: {"results": [...]} and direct list
                if isinstance(data, dict):
                    results = data.get("results", [])
                else:
                    results = data

                model_pairs = []
                for sample_data in results:
                    sample_id = sample_data.get("sample_id") or sample_data.get("id")
                    if sample_id not in gt_by_id:
                        continue

                    gt_tool_calls = gt_by_id[sample_id]
                    # Handle both "responses" and "generated_runs" keys
                    responses = sample_data.get("generated_runs") or sample_data.get("responses", [])

                    # Get first valid LLM response
                    for resp in responses:
                        if resp is None:
                            continue
                        if isinstance(resp, list) and len(resp) > 0:
                            # Tool calls format
                            model_pairs.append({
                                "source": "toucan_gt_comparison",
                                "model": model_name,
                                "sample_id": sample_id,
                                "json_a": {"tool_calls": gt_tool_calls},
                                "json_b": {"tool_calls": resp},
                            })
                            break

                # Sample from model pairs
                random.shuffle(model_pairs)
                pairs.extend(model_pairs[:max_samples_per_model])

            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue

        print(f"Loaded {len(pairs)} Toucan ground-truth comparison pairs")
        return pairs

    def load_llm_output_pairs(
        self,
        results_dir: str,
        dataset: str = "sharegpt",
        max_samples_per_model: int = 50,
        balanced_sampling: bool = True,
        inconsistency_ratio: float = 0.5,
    ) -> List[Dict]:
        """
        Load pairs from real LLM generation results.

        Args:
            results_dir: Directory containing LLM results
            dataset: Dataset name ('sharegpt' or 'toucan')
            max_samples_per_model: Max pairs to sample per model
            balanced_sampling: If True, balance consistent vs inconsistent pairs
            inconsistency_ratio: Target ratio of inconsistent pairs (default 0.5)
        """
        all_pairs = []

        dataset_dir = os.path.join(results_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            return all_pairs

        # Find all model generation directories
        model_dirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("generations-")
        ]

        for model_dir in model_dirs:
            model_name = model_dir.replace("generations-", "")
            # Remove date suffix if present (e.g., "claude-sonnet-4.5-20251224" -> "claude-sonnet-4.5")
            model_name = "-".join(model_name.split("-")[:-1]) if model_name.split("-")[-1].isdigit() else model_name
            model_path = os.path.join(dataset_dir, model_dir)

            # Find run directories - handle both "temp_*" and "run_*" naming conventions
            run_dirs = []
            for d in os.listdir(model_path):
                if not os.path.isdir(os.path.join(model_path, d)):
                    continue
                if d.startswith("temp_"):
                    # Old format: temp_0.1, temp_0.2, etc.
                    try:
                        temp = float(d.replace("temp_", ""))
                        run_dirs.append((d, temp))
                    except ValueError:
                        continue
                elif d.startswith("run_"):
                    # New format: run_Model_temp_0_10_date
                    # Extract temperature from directory name
                    parts = d.split("_temp_")
                    if len(parts) >= 2:
                        temp_part = parts[1].split("_")[0:2]  # Get "0_10" -> ["0", "10"]
                        try:
                            temp = float(f"{temp_part[0]}.{temp_part[1]}")
                            run_dirs.append((d, temp))
                        except (ValueError, IndexError):
                            continue

            if not run_dirs:
                continue

            # Sort by temperature (higher temps first for more variation)
            run_dirs.sort(key=lambda x: x[1], reverse=True)

            print(f"Loading LLM outputs from {model_name} ({len(run_dirs)} temperature runs)...")
            model_pairs = []

            for run_dir, temperature in run_dirs:
                run_path = os.path.join(model_path, run_dir)
                results_file = os.path.join(run_path, "all_results.json")

                if not os.path.exists(results_file):
                    continue

                try:
                    with open(results_file, "r") as f:
                        data = json.load(f)

                    # Handle both formats: {"results": [...]} and direct list
                    if isinstance(data, dict):
                        results = data.get("results", [])
                    else:
                        results = data

                    # Extract pairs from runs within each sample
                    for sample_idx, sample_data in enumerate(results):
                        # Handle both "responses" and "generated_runs" keys
                        responses = sample_data.get("generated_runs") or sample_data.get("responses", [])
                        valid_responses = []

                        # Filter valid JSON responses
                        for resp in responses:
                            if resp is None:
                                continue
                            # Handle different response formats
                            if isinstance(resp, dict):
                                if "modelReport" in resp:
                                    valid_responses.append(resp["modelReport"])
                                else:
                                    valid_responses.append(resp)
                            elif isinstance(resp, list):
                                # Tool calls format - wrap in dict for consistency
                                valid_responses.append({"tool_calls": resp})

                        # Create pairs from different runs (not just consecutive)
                        if len(valid_responses) >= 2:
                            # Pair first with last (likely more different)
                            model_pairs.append({
                                "source": "llm_output",
                                "model": model_name,
                                "temperature": temperature,
                                "sample_idx": sample_idx,
                                "run_pair": f"0-{len(valid_responses)-1}",
                                "dataset": dataset,
                                "json_a": valid_responses[0],
                                "json_b": valid_responses[-1],
                            })
                            # Also pair consecutive for comparison
                            if len(valid_responses) >= 3:
                                model_pairs.append({
                                    "source": "llm_output",
                                    "model": model_name,
                                    "temperature": temperature,
                                    "sample_idx": sample_idx,
                                    "run_pair": "0-1",
                                    "dataset": dataset,
                                    "json_a": valid_responses[0],
                                    "json_b": valid_responses[1],
                                })

                except Exception as e:
                    print(f"Error loading {results_file}: {e}")
                    continue

            # Balanced sampling: compute STED and select diverse pairs
            if balanced_sampling and model_pairs:
                model_pairs = self._balanced_sample_pairs(
                    model_pairs,
                    max_samples=max_samples_per_model,
                    inconsistency_ratio=inconsistency_ratio,
                )

            all_pairs.extend(model_pairs[:max_samples_per_model])

        print(f"Loaded {len(all_pairs)} LLM output pairs")
        return all_pairs

    def _balanced_sample_pairs(
        self,
        pairs: List[Dict],
        max_samples: int,
        inconsistency_ratio: float = 0.5,
        consistency_threshold: float = 0.8,
    ) -> List[Dict]:
        """
        Sample pairs to balance consistent vs inconsistent examples.

        Args:
            pairs: List of pair dicts
            max_samples: Maximum number of pairs to return
            inconsistency_ratio: Target ratio of inconsistent pairs
            consistency_threshold: STED score below this = inconsistent
        """
        # Compute STED scores for all pairs
        print(f"  Computing STED for {len(pairs)} pairs for balanced sampling...")
        for pair in pairs:
            try:
                pair["_sted_score"] = self.evaluator.calculate_tree_edit_distance_fast(
                    pair["json_a"], pair["json_b"], variation_type="combined"
                )
            except:
                pair["_sted_score"] = None

        # Split into consistent vs inconsistent
        consistent = [p for p in pairs if p.get("_sted_score") is not None and p["_sted_score"] >= consistency_threshold]
        inconsistent = [p for p in pairs if p.get("_sted_score") is not None and p["_sted_score"] < consistency_threshold]

        print(f"  Found {len(consistent)} consistent, {len(inconsistent)} inconsistent pairs")

        # Calculate target counts
        n_inconsistent = min(int(max_samples * inconsistency_ratio), len(inconsistent))
        n_consistent = min(max_samples - n_inconsistent, len(consistent))

        # If not enough inconsistent, take more consistent
        if n_inconsistent < int(max_samples * inconsistency_ratio):
            n_consistent = min(max_samples - n_inconsistent, len(consistent))

        # Sample
        random.shuffle(consistent)
        random.shuffle(inconsistent)

        selected = inconsistent[:n_inconsistent] + consistent[:n_consistent]
        random.shuffle(selected)

        print(f"  Selected {n_inconsistent} inconsistent + {n_consistent} consistent = {len(selected)} pairs")

        # Clean up temp field
        for p in selected:
            if "_sted_score" in p:
                p["sted_score"] = p.pop("_sted_score")

        return selected

    def stratify_by_sted_score(
        self,
        pairs: List[Dict],
        n_per_stratum: int = 30,
        strata: List[Tuple[float, float]] = None,
        min_complexity: int = 3,  # Not used for filtering, only for categorization
        prioritize_disagreement: bool = True,
        ensure_complexity_coverage: bool = True,
    ) -> List[Dict]:
        """
        Stratify pairs by STED score AND complexity to ensure full coverage.

        Args:
            pairs: List of pair dicts
            n_per_stratum: Pairs per similarity stratum
            strata: List of (low, high) tuples for stratification
            min_complexity: Threshold for simple vs complex categorization
            prioritize_disagreement: If True, prioritize pairs where methods disagree
            ensure_complexity_coverage: If True, ensure mix of simple/medium/complex in each stratum
        """

        if strata is None:
            strata = [
                (0.0, 0.2),   # Low similarity
                (0.2, 0.4),   # Medium-low
                (0.4, 0.6),   # Medium
                (0.6, 0.8),   # Medium-high
                (0.8, 1.0),   # High similarity
            ]

        print("Computing STED scores and structural metrics for stratification...")

        # Compute STED scores and structural metrics for all pairs
        for pair in tqdm(pairs, desc="Computing scores"):
            if pair.get("json_a") is None or pair.get("json_b") is None:
                pair["sted_score"] = None
                pair["_complexity"] = 0
                pair["_complexity_category"] = "unknown"
                continue

            try:
                pair["sted_score"] = self.evaluator.calculate_tree_edit_distance_fast(
                    pair["json_a"], pair["json_b"], variation_type="combined"
                )
                # Compute complexity
                metrics = compute_pair_structural_metrics(pair["json_a"], pair["json_b"])
                pair["_complexity"] = metrics["avg_node_count"]
                pair["_structural"] = metrics
                pair["_complexity_category"] = metrics["complexity_category"]
            except Exception as e:
                print(f"Error computing STED: {e}")
                pair["sted_score"] = None
                pair["_complexity"] = 0
                pair["_complexity_category"] = "unknown"

        # Filter out pairs with no score
        scored_pairs = [p for p in pairs if p.get("sted_score") is not None]

        # Report complexity distribution
        complexity_dist = defaultdict(int)
        for p in scored_pairs:
            complexity_dist[p.get("_complexity_category", "unknown")] += 1
        print(f"Complexity distribution: simple={complexity_dist['simple']}, medium={complexity_dist['medium']}, complex={complexity_dist['complex']}")

        # Group by stratum AND complexity
        stratified = defaultdict(lambda: {"simple": [], "medium": [], "complex": []})
        for pair in scored_pairs:
            score = pair["sted_score"]
            complexity_cat = pair.get("_complexity_category", "medium")
            if complexity_cat not in ["simple", "medium", "complex"]:
                complexity_cat = "medium"

            for low, high in strata:
                if low <= score < high or (high == 1.0 and score == 1.0):
                    stratified[(low, high)][complexity_cat].append(pair)
                    break

        # Sample from each stratum with complexity coverage
        selected = []
        for (low, high), complexity_groups in stratified.items():
            n_simple = len(complexity_groups["simple"])
            n_medium = len(complexity_groups["medium"])
            n_complex = len(complexity_groups["complex"])
            total = n_simple + n_medium + n_complex

            print(f"Stratum [{low:.1f}, {high:.1f}): {total} pairs (simple={n_simple}, medium={n_medium}, complex={n_complex})")

            if ensure_complexity_coverage and total > 0:
                # Allocate samples proportionally but ensure minimum coverage
                n_sample = min(n_per_stratum, total)

                # Target: at least 20% simple, 30% medium, 30% complex (if available)
                target_simple = max(2, int(n_sample * 0.2)) if n_simple > 0 else 0
                target_medium = max(3, int(n_sample * 0.3)) if n_medium > 0 else 0
                target_complex = max(3, int(n_sample * 0.3)) if n_complex > 0 else 0

                # Adjust to available
                target_simple = min(target_simple, n_simple)
                target_medium = min(target_medium, n_medium)
                target_complex = min(target_complex, n_complex)

                # Fill remaining quota
                remaining = n_sample - target_simple - target_medium - target_complex

                # Process each complexity group
                stratum_selected = []
                for cat, target in [("simple", target_simple), ("medium", target_medium), ("complex", target_complex)]:
                    group_pairs = complexity_groups[cat]
                    if not group_pairs:
                        continue

                    if prioritize_disagreement:
                        group_pairs = self._compute_method_disagreement(group_pairs)
                        group_pairs.sort(key=lambda x: x.get("disagreement_score", 0), reverse=True)
                    else:
                        random.shuffle(group_pairs)

                    stratum_selected.extend(group_pairs[:target])

                # Add remaining from any category (prioritize complex for interesting cases)
                if remaining > 0:
                    all_remaining = []
                    for cat in ["complex", "medium", "simple"]:
                        target = {"simple": target_simple, "medium": target_medium, "complex": target_complex}[cat]
                        all_remaining.extend(complexity_groups[cat][target:])

                    if prioritize_disagreement and all_remaining:
                        all_remaining = self._compute_method_disagreement(all_remaining)
                        all_remaining.sort(key=lambda x: x.get("disagreement_score", 0), reverse=True)
                    else:
                        random.shuffle(all_remaining)

                    stratum_selected.extend(all_remaining[:remaining])

                selected.extend(stratum_selected)
            else:
                # Simple stratification without complexity coverage
                all_pairs_in_stratum = []
                for cat in ["simple", "medium", "complex"]:
                    all_pairs_in_stratum.extend(complexity_groups[cat])

                if prioritize_disagreement:
                    all_pairs_in_stratum = self._compute_method_disagreement(all_pairs_in_stratum)
                    all_pairs_in_stratum.sort(key=lambda x: x.get("disagreement_score", 0), reverse=True)
                else:
                    random.shuffle(all_pairs_in_stratum)

                n_sample = min(n_per_stratum, len(all_pairs_in_stratum))
                selected.extend(all_pairs_in_stratum[:n_sample])

        print(f"Selected {len(selected)} pairs after stratification")

        # Print final complexity distribution
        final_dist = defaultdict(int)
        for p in selected:
            final_dist[p.get("_complexity_category", "unknown")] += 1
        print(f"Final complexity distribution: simple={final_dist['simple']}, medium={final_dist['medium']}, complex={final_dist['complex']}")

        # Print disagreement statistics
        if prioritize_disagreement:
            disagreements = [p.get("disagreement_score", 0) for p in selected if p.get("disagreement_score") is not None]
            if disagreements:
                print(f"Disagreement score stats: mean={np.mean(disagreements):.3f}, max={np.max(disagreements):.3f}")

        return selected

    def _compute_method_disagreement(self, pairs: List[Dict]) -> List[Dict]:
        """Compute disagreement score between STED and baselines."""

        for pair in tqdm(pairs, desc="Computing baseline scores", leave=False):
            if "all_scores" in pair:
                continue

            try:
                pair["all_scores"] = self.compute_all_scores(pair["json_a"], pair["json_b"])

                # Compute disagreement as max difference from STED
                sted = pair["all_scores"].get("sted_combined")
                if sted is not None:
                    disagreements = []
                    for method in ["deepdiff", "ted", "bertscore"]:
                        baseline = pair["all_scores"].get(method)
                        if baseline is not None:
                            disagreements.append(abs(sted - baseline))
                    pair["disagreement_score"] = max(disagreements) if disagreements else 0
                else:
                    pair["disagreement_score"] = 0

            except Exception as e:
                print(f"Error computing scores: {e}")
                pair["all_scores"] = {}
                pair["disagreement_score"] = 0

        return pairs

    def format_for_annotation(self, pairs: List[Dict]) -> List[Dict]:
        """Format pairs for human annotation interface."""

        annotation_items = []

        for idx, pair in enumerate(pairs):
            # Compute structural metrics for the pair
            structural_metrics = compute_pair_structural_metrics(
                pair.get("json_a"), pair.get("json_b")
            )

            item = {
                "id": f"pair_{idx:04d}",
                "json_a": pair["json_a"],
                "json_b": pair["json_b"],
                "metadata": {
                    "source": pair.get("source"),
                    "sted_score": pair.get("sted_score"),
                    "all_scores": pair.get("all_scores", {}),
                    "structural": structural_metrics,
                },
            }

            # Add source-specific metadata
            if pair.get("source") == "synthetic":
                item["metadata"]["variation_category"] = pair.get("variation_category")
                item["metadata"]["variation_type"] = pair.get("variation_type")
                item["metadata"]["variation_ratio"] = pair.get("variation_ratio")
            elif pair.get("source") == "llm_output":
                item["metadata"]["model"] = pair.get("model")
                item["metadata"]["temperature"] = pair.get("temperature")
                item["metadata"]["dataset"] = pair.get("dataset")
            elif pair.get("source") == "toucan_gt_comparison":
                item["metadata"]["model"] = pair.get("model")
                item["metadata"]["sample_id"] = pair.get("sample_id")

            # Placeholder for human annotation
            item["annotation"] = {
                "rating": None,  # 1-5 scale
                "annotator_id": None,
                "timestamp": None,
                "notes": None,
            }

            annotation_items.append(item)

        return annotation_items

    def generate_dataset(
        self,
        synthetic_dir: str = None,
        llm_results_dir: str = None,
        toucan_data_path: str = None,
        n_per_stratum: int = 30,
        output_path: str = "human_validation_dataset.json",
        include_synthetic: bool = True,
        include_llm: bool = True,
        include_toucan_gt: bool = False,
        data_source: str = "both",  # "sharegpt", "toucan", or "both"
        min_complexity: int = 3,
        prioritize_disagreement: bool = True,
    ) -> List[Dict]:
        """Generate complete human validation dataset.

        Args:
            synthetic_dir: Directory with synthetic variation datasets
            llm_results_dir: Directory with LLM generation results
            toucan_data_path: Path to Toucan ground truth JSON
            n_per_stratum: Pairs per similarity stratum
            output_path: Output file path
            include_synthetic: Include synthetic variation pairs
            include_llm: Include LLM output pairs (comparing two LLM outputs)
            include_toucan_gt: Include Toucan ground truth comparisons
            data_source: "sharegpt", "toucan", or "both"
            min_complexity: Minimum node count to filter trivially simple pairs
            prioritize_disagreement: Prioritize pairs where STED disagrees with baselines
        """

        all_pairs = []

        # Load synthetic pairs
        if include_synthetic and synthetic_dir:
            synthetic_pairs = self.load_synthetic_pairs(synthetic_dir)
            all_pairs.extend(synthetic_pairs)

        # Load Toucan ground truth comparison pairs (clearer for human annotation)
        if include_toucan_gt and toucan_data_path and llm_results_dir:
            toucan_gt_pairs = self.load_toucan_ground_truth_pairs(
                toucan_data_path, llm_results_dir, max_samples_per_model=30
            )
            all_pairs.extend(toucan_gt_pairs)

        # Load LLM output pairs (comparing two LLM outputs)
        if include_llm and llm_results_dir:
            datasets_to_load = []
            if data_source in ["sharegpt", "both"]:
                datasets_to_load.append("sharegpt")
            if data_source in ["toucan", "both"]:
                datasets_to_load.append("toucan")

            for dataset in datasets_to_load:
                llm_pairs = self.load_llm_output_pairs(
                    llm_results_dir, dataset=dataset, max_samples_per_model=30
                )
                all_pairs.extend(llm_pairs)

        if not all_pairs:
            print("No pairs loaded!")
            return []

        print(f"\nTotal pairs before stratification: {len(all_pairs)}")

        # Separate pairs by data source for independent stratification
        toucan_pairs = [p for p in all_pairs if p.get("dataset") == "toucan" or p.get("source") == "toucan_gt_comparison"]
        sharegpt_pairs = [p for p in all_pairs if p.get("dataset") == "sharegpt"]
        synthetic_pairs = [p for p in all_pairs if p.get("source") == "synthetic"]
        other_pairs = [p for p in all_pairs if p not in toucan_pairs and p not in sharegpt_pairs and p not in synthetic_pairs]

        print(f"\nBy data source: Toucan={len(toucan_pairs)}, ShareGPT={len(sharegpt_pairs)}, Synthetic={len(synthetic_pairs)}, Other={len(other_pairs)}")

        selected_pairs = []

        # Stratify each data source separately
        # Allocate n_per_stratum proportionally based on available data
        total_sources = sum([1 for x in [toucan_pairs, sharegpt_pairs, synthetic_pairs] if x])

        if toucan_pairs:
            n_toucan = n_per_stratum // total_sources if total_sources > 1 else n_per_stratum
            print(f"\n--- Stratifying Toucan pairs (target {n_toucan} per stratum) ---")
            toucan_selected = self.stratify_by_sted_score(
                toucan_pairs,
                n_per_stratum=n_toucan,
                min_complexity=min_complexity,
                prioritize_disagreement=prioritize_disagreement,
            )
            selected_pairs.extend(toucan_selected)

        if sharegpt_pairs:
            n_sharegpt = n_per_stratum // total_sources if total_sources > 1 else n_per_stratum
            print(f"\n--- Stratifying ShareGPT pairs (target {n_sharegpt} per stratum) ---")
            sharegpt_selected = self.stratify_by_sted_score(
                sharegpt_pairs,
                n_per_stratum=n_sharegpt,
                min_complexity=min_complexity,
                prioritize_disagreement=prioritize_disagreement,
            )
            selected_pairs.extend(sharegpt_selected)

        if synthetic_pairs:
            n_synthetic = n_per_stratum // total_sources if total_sources > 1 else n_per_stratum
            print(f"\n--- Stratifying Synthetic pairs (target {n_synthetic} per stratum) ---")
            synthetic_selected = self.stratify_by_sted_score(
                synthetic_pairs,
                n_per_stratum=n_synthetic,
                min_complexity=min_complexity,
                prioritize_disagreement=prioritize_disagreement,
            )
            selected_pairs.extend(synthetic_selected)

        if other_pairs:
            print(f"\n--- Stratifying Other pairs ---")
            other_selected = self.stratify_by_sted_score(
                other_pairs,
                n_per_stratum=5,  # Small allocation for miscellaneous
                min_complexity=min_complexity,
                prioritize_disagreement=prioritize_disagreement,
            )
            selected_pairs.extend(other_selected)

        print(f"\n=== Total selected: {len(selected_pairs)} pairs ===")

        # Format for annotation
        annotation_dataset = self.format_for_annotation(selected_pairs)

        # Add dataset metadata
        output_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "n_pairs": len(annotation_dataset),
                "n_per_stratum": n_per_stratum,
                "sources": {
                    "synthetic": include_synthetic,
                    "llm_outputs": include_llm,
                },
                "strata": ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"],
            },
            "annotation_guidelines": {
                "scale": {
                    "1": "Completely different (incompatible)",
                    "2": "Mostly different (major structural/semantic gaps)",
                    "3": "Somewhat similar (some overlap, notable differences)",
                    "4": "Mostly similar (minor differences, largely compatible)",
                    "5": "Identical or equivalent (fully interchangeable)",
                },
                "instructions": "Rate how similar these JSON outputs are. Consider functional equivalence for downstream applications.",
            },
            "pairs": annotation_dataset,
        }

        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved {len(annotation_dataset)} pairs to {output_path}")

        # Print summary statistics
        self._print_summary(annotation_dataset)

        return annotation_dataset

    def _print_summary(self, pairs: List[Dict]):
        """Print summary statistics of the dataset."""

        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)

        # By source
        sources = defaultdict(int)
        for p in pairs:
            sources[p["metadata"].get("source", "unknown")] += 1
        print("\nBy source:")
        for source, count in sources.items():
            print(f"  {source}: {count}")

        # By STED score stratum
        strata_counts = defaultdict(int)
        for p in pairs:
            score = p["metadata"].get("sted_score")
            if score is not None:
                if score < 0.2:
                    strata_counts["[0.0-0.2)"] += 1
                elif score < 0.4:
                    strata_counts["[0.2-0.4)"] += 1
                elif score < 0.6:
                    strata_counts["[0.4-0.6)"] += 1
                elif score < 0.8:
                    strata_counts["[0.6-0.8)"] += 1
                else:
                    strata_counts["[0.8-1.0]"] += 1

        print("\nBy STED score stratum:")
        for stratum in ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]:
            print(f"  {stratum}: {strata_counts.get(stratum, 0)}")

        # Score statistics
        sted_scores = [p["metadata"]["sted_score"] for p in pairs if p["metadata"].get("sted_score") is not None]
        if sted_scores:
            print(f"\nSTED score statistics:")
            print(f"  Mean: {np.mean(sted_scores):.3f}")
            print(f"  Std:  {np.std(sted_scores):.3f}")
            print(f"  Min:  {np.min(sted_scores):.3f}")
            print(f"  Max:  {np.max(sted_scores):.3f}")

        # Structural metrics coverage
        print("\n" + "-"*50)
        print("STRUCTURAL COVERAGE")
        print("-"*50)

        # By complexity category
        complexity_counts = defaultdict(int)
        depths = []
        node_counts = []
        array_counts = []

        for p in pairs:
            structural = p["metadata"].get("structural", {})
            if structural:
                complexity_counts[structural.get("complexity_category", "unknown")] += 1
                if "max_depth" in structural:
                    depths.append(structural["max_depth"])
                if "avg_node_count" in structural:
                    node_counts.append(structural["avg_node_count"])
                # Count arrays from both json_a and json_b
                json_a_metrics = structural.get("json_a", {})
                json_b_metrics = structural.get("json_b", {})
                array_counts.append(json_a_metrics.get("array_count", 0) + json_b_metrics.get("array_count", 0))

        print("\nBy complexity category:")
        for cat in ["simple", "medium", "complex"]:
            count = complexity_counts.get(cat, 0)
            pct = count / len(pairs) * 100 if pairs else 0
            print(f"  {cat}: {count} ({pct:.1f}%)")

        if depths:
            print(f"\nDepth statistics:")
            print(f"  Mean: {np.mean(depths):.1f}")
            print(f"  Min:  {np.min(depths)}")
            print(f"  Max:  {np.max(depths)}")

        if node_counts:
            print(f"\nNode count statistics:")
            print(f"  Mean: {np.mean(node_counts):.1f}")
            print(f"  Min:  {np.min(node_counts):.0f}")
            print(f"  Max:  {np.max(node_counts):.0f}")

        # Check for arrays
        pairs_with_arrays = sum(1 for c in array_counts if c > 0)
        print(f"\nArray coverage:")
        print(f"  Pairs with arrays: {pairs_with_arrays} ({pairs_with_arrays/len(pairs)*100:.1f}%)")

        # Coverage verification
        print("\n" + "-"*50)
        print("COVERAGE VERIFICATION")
        print("-"*50)

        warnings = []

        # Check complexity coverage
        if complexity_counts.get("simple", 0) < 10:
            warnings.append("WARNING: Low coverage of simple structures (<10 pairs)")
        if complexity_counts.get("complex", 0) < 10:
            warnings.append("WARNING: Low coverage of complex structures (<10 pairs)")

        # Check depth coverage
        if depths and max(depths) < 3:
            warnings.append("WARNING: No deeply nested structures (max depth < 3)")

        # Check array coverage
        if pairs_with_arrays < len(pairs) * 0.2:
            warnings.append("WARNING: Low array coverage (<20% of pairs)")

        # Check stratum coverage
        for stratum in ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]:
            if strata_counts.get(stratum, 0) < 10:
                warnings.append(f"WARNING: Low coverage in stratum {stratum} (<10 pairs)")

        if warnings:
            for w in warnings:
                print(f"  {w}")
        else:
            print("  All coverage checks passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate human validation dataset for STED"
    )
    parser.add_argument(
        "--synthetic-dir",
        default="synthetic_dataset",
        help="Directory containing synthetic variation datasets",
    )
    parser.add_argument(
        "--llm-results-dir",
        default="llm_gen_results",
        help="Directory containing LLM generation results",
    )
    parser.add_argument(
        "--toucan-data-path",
        default="toucan_data/toucan_tool_calls_1006.json",
        help="Path to Toucan ground truth JSON file",
    )
    parser.add_argument(
        "--output",
        default="human_validation_dataset.json",
        help="Output file path",
    )
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=30,
        help="Number of pairs per similarity stratum (default: 30)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Exclude synthetic pairs",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Exclude LLM output pairs",
    )
    parser.add_argument(
        "--include-toucan-gt",
        action="store_true",
        help="Include Toucan ground truth comparisons (LLM vs ground truth)",
    )
    parser.add_argument(
        "--data-source",
        choices=["sharegpt", "toucan", "both"],
        default="both",
        help="Data source for LLM pairs: 'sharegpt', 'toucan', or 'both' (default: both)",
    )
    parser.add_argument(
        "--toucan-only",
        action="store_true",
        help="Shortcut: Use only Toucan data (sets --data-source toucan --include-toucan-gt)",
    )
    parser.add_argument(
        "--model-id",
        default="all-MiniLM-L6-v2",
        help="Embedding model ID",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-balanced-sampling",
        action="store_true",
        help="Disable balanced sampling for LLM pairs (default: enabled)",
    )
    parser.add_argument(
        "--inconsistency-ratio",
        type=float,
        default=0.5,
        help="Target ratio of inconsistent pairs in LLM samples (default: 0.5)",
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.8,
        help="STED score threshold for consistent vs inconsistent (default: 0.8)",
    )
    parser.add_argument(
        "--min-complexity",
        type=int,
        default=3,
        help="Minimum node count for pairs (filters trivially simple pairs, default: 3)",
    )
    parser.add_argument(
        "--no-prioritize-disagreement",
        action="store_true",
        help="Disable prioritizing pairs where methods disagree (default: enabled)",
    )

    args = parser.parse_args()

    # Handle --toucan-only shortcut
    if args.toucan_only:
        args.data_source = "toucan"
        args.include_toucan_gt = True

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generate dataset
    generator = HumanValidationDatasetGenerator(model_id=args.model_id)
    generator.generate_dataset(
        synthetic_dir=args.synthetic_dir,
        llm_results_dir=args.llm_results_dir,
        toucan_data_path=args.toucan_data_path,
        n_per_stratum=args.n_per_stratum,
        output_path=args.output,
        include_synthetic=not args.no_synthetic,
        include_llm=not args.no_llm,
        include_toucan_gt=args.include_toucan_gt,
        data_source=args.data_source,
        min_complexity=args.min_complexity,
        prioritize_disagreement=not args.no_prioritize_disagreement,
    )


if __name__ == "__main__":
    main()
