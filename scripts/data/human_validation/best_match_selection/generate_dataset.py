#!/usr/bin/env python
"""
Generate ranking-based human validation dataset for STED.

This approach is more objective than rating scales:
1. For each sample, show Ground Truth (GT) and multiple LLM responses
2. Each similarity method (STED, BERTScore, DeepDiff, TED) picks its "most similar" response
3. Human chooses which response is actually most similar to GT
4. Calculate win rate / MRR for each method

This is essentially a "best-match retrieval" task with high inter-annotator agreement.
"""

import json
import os
import argparse
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Final models to include (matches visualize_consistency_scores.py)
FINAL_MODELS = [
    'Qwen3-235B-A22B',
    'Claude-3.5-Sonnet',
    'Claude-Haiku-4.5',
    'Claude-3.7-Sonnet',
    'Claude-3.5-Haiku',
    'Claude-Opus-4.5',
    'Claude-Opus-4',
    'Claude-Sonnet-4',
    'Claude-Sonnet-4.5',
    'Qwen3-32B',
    'Llama-3.3-70B',
    'Nova-2-Lite',
    'Mimo-V2-Flash',
    'Grok-4.1-Fast',
    'Minimax-M2',
    'GPT-4.1-Mini',
    'Gemini-2.5-Flash-Lite',
    'GPT-OSS-120B',
]


def is_final_model(model_name: str) -> bool:
    """Check if model name matches any FINAL_MODELS (case-insensitive substring match)."""
    model_name_lower = model_name.lower()
    for final_model in FINAL_MODELS:
        final_model_lower = final_model.lower()
        if final_model_lower in model_name_lower or model_name_lower in final_model_lower:
            return True
    return False


# Global evaluator for worker processes
_worker_evaluator = None


def _init_worker(model_id: str):
    """Initialize evaluator in worker process."""
    global _worker_evaluator
    _worker_evaluator = SemanticJsonTreeConsistencyEvaluator(model_id=model_id)


def _compute_sample_scores(args: Tuple) -> Optional[Dict]:
    """Compute scores for a single sample (for parallel processing)."""
    sample, methods = args
    global _worker_evaluator

    gt = sample["ground_truth"]
    responses = sample["responses"]

    if len(responses) < 2:
        return None

    # Compute all scores for each response
    response_scores = []
    for resp_data in responses:
        # Toucan responses are lists (need wrapping), ShareGPT responses are dicts (use directly)
        raw_resp = resp_data["response"]
        if isinstance(raw_resp, list):
            resp = {"tool_calls": raw_resp}  # Toucan format
        else:
            resp = raw_resp  # ShareGPT format - already a dict
        scores = {}

        # STED score
        try:
            scores["sted"] = _worker_evaluator.calculate_tree_edit_distance_opt(
                gt, resp, variation_type="combined"
            )
        except Exception:
            scores["sted"] = None

        # DeepDiff baseline
        try:
            scores["deepdiff"] = _worker_evaluator.calculate_similarity_with_deepdiff(gt, resp)
        except Exception:
            scores["deepdiff"] = None

        # TED baseline (using ZSS)
        try:
            scores["ted"] = _worker_evaluator.calculate_tree_edit_distance(
                gt, resp, original_zss=True
            )
        except Exception:
            scores["ted"] = None

        # BERTScore baseline
        try:
            scores["bertscore"] = _worker_evaluator.calculate_bertscore(gt, resp)
        except Exception:
            scores["bertscore"] = None

        # Skip if any method failed to compute
        if None in scores.values():
            continue

        response_scores.append({
            "model": resp_data["model"],
            "response": resp,
            "scores": scores,
        })

    if len(response_scores) < 2:
        return None

    # Find top pick for each method
    method_picks = {}
    for method in methods:
        valid_responses = [r for r in response_scores if r["scores"].get(method) is not None]
        if valid_responses:
            top_pick = max(valid_responses, key=lambda x: x["scores"][method])
            method_picks[method] = {
                "model": top_pick["model"],
                "response": top_pick["response"],
                "score": top_pick["scores"][method],
            }

    # Check if methods disagree
    unique_picks = set()
    for method, pick in method_picks.items():
        pick_key = json.dumps(pick["response"], sort_keys=True)
        unique_picks.add(pick_key)

    return {
        "sample_id": sample["sample_id"],
        "ground_truth": gt,
        "method_picks": method_picks,
        "unique_picks": unique_picks,
        "n_unique_picks": len(unique_picks),
    }


def compute_structural_metrics(json_obj: Any) -> Dict[str, Any]:
    """Compute structural metrics for a JSON object."""
    metrics = {
        "depth": 0,
        "node_count": 0,
        "array_count": 0,
        "object_count": 0,
    }

    def traverse(obj, current_depth=0):
        metrics["depth"] = max(metrics["depth"], current_depth)
        metrics["node_count"] += 1

        if isinstance(obj, dict):
            metrics["object_count"] += 1
            for key, value in obj.items():
                metrics["node_count"] += 1
                traverse(value, current_depth + 1)
        elif isinstance(obj, list):
            metrics["array_count"] += 1
            for item in obj:
                traverse(item, current_depth + 1)

    if json_obj is not None:
        traverse(json_obj)

    return metrics


class RankingValidationDatasetGenerator:
    """Generate ranking-based validation dataset."""

    def __init__(
        self,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        """Initialize STED evaluator."""
        self.model_id = model_id
        print("Initializing STED evaluator...")
        self.evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id=model_id,
        )
        print("STED evaluator initialized.")
        print(f"Has calculate_tree_edit_distance_opt: {hasattr(self.evaluator, 'calculate_tree_edit_distance_opt')}")
        print(f"Evaluator methods: {[m for m in dir(self.evaluator) if 'calculate' in m]}")

    def compute_all_scores(self, json_a: Dict, json_b: Dict) -> Dict[str, float]:
        """Compute all similarity scores for a pair."""
        scores = {}

        # STED score
        try:
            scores["sted"] = self.evaluator.calculate_tree_edit_distance_opt(
                json_a, json_b, variation_type="combined"
            )
            if scores["sted"] is None:
                print(f"STED returned None (not an exception)")
        except Exception as e:
            print(f"STED error: {e}")
            import traceback
            traceback.print_exc()
            scores["sted"] = None

        # DeepDiff baseline
        try:
            scores["deepdiff"] = self.evaluator.calculate_similarity_with_deepdiff(json_a, json_b)
        except Exception as e:
            scores["deepdiff"] = None

        # TED baseline (using ZSS)
        try:
            scores["ted"] = self.evaluator.calculate_tree_edit_distance(
                json_a, json_b, original_zss=True
            )
        except Exception as e:
            scores["ted"] = None

        # BERTScore baseline
        try:
            scores["bertscore"] = self.evaluator.calculate_bertscore(json_a, json_b)
        except Exception as e:
            scores["bertscore"] = None

        return scores

    def load_toucan_samples(
        self,
        toucan_data_path: str,
        results_dir: str,
        min_responses: int = 4,
        max_samples: int = 200,
        final_models_only: bool = False,
        exclude_models: List[str] = None,
    ) -> List[Dict]:
        """
        Load samples with ground truth and multiple LLM responses.

        Returns list of samples, each containing:
        - sample_id
        - ground_truth: the expected tool calls
        - responses: list of (model_name, response) tuples
        """
        samples = []
        exclude_models = exclude_models or []

        # Load ground truth
        print(f"Loading Toucan ground truth from {toucan_data_path}...")
        with open(toucan_data_path, "r") as f:
            toucan_data = json.load(f)
        gt_by_id = {item["id"]: item["tool_calls"] for item in toucan_data if "tool_calls" in item}
        print(f"Loaded {len(gt_by_id)} ground truth samples")

        # Find all model results
        dataset_dir = os.path.join(results_dir, "toucan")
        if not os.path.exists(dataset_dir):
            print(f"Toucan results directory not found: {dataset_dir}")
            return samples

        model_dirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("generations-")
        ]

        # Filter to final models only if requested
        if final_models_only:
            original_count = len(model_dirs)
            model_dirs = [d for d in model_dirs if is_final_model(d.replace("generations-", ""))]
            print(f"Final models filter: {original_count} -> {len(model_dirs)} model directories")

        # Filter out excluded models
        if exclude_models:
            original_count = len(model_dirs)
            model_dirs = [
                d for d in model_dirs
                if not any(excl in d for excl in exclude_models)
            ]
            print(f"Exclude models filter: {original_count} -> {len(model_dirs)} model directories")

        # Collect responses per sample_id
        responses_by_sample = defaultdict(list)

        for model_dir in tqdm(model_dirs, desc="Loading model results"):
            model_name = model_dir.replace("generations-", "")
            # Clean up model name (remove date suffix)
            parts = model_name.split("-")
            if parts[-1].isdigit() and len(parts[-1]) == 8:
                model_name = "-".join(parts[:-1])

            model_path = os.path.join(dataset_dir, model_dir)

            # Find results file (prefer T=0.0 for most consistent outputs)
            results_file = None
            for d in os.listdir(model_path):
                if not os.path.isdir(os.path.join(model_path, d)):
                    continue
                # Try different naming conventions
                if d.startswith("temp_0.0") or d.startswith("temp_0.1"):
                    candidate = os.path.join(model_path, d, "all_results.json")
                    if os.path.exists(candidate):
                        results_file = candidate
                        break
                elif "_temp_0_0" in d or "_temp_0_1" in d:
                    candidate = os.path.join(model_path, d, "all_results.json")
                    if os.path.exists(candidate):
                        results_file = candidate
                        break

            if results_file is None:
                # Try any run directory
                for d in os.listdir(model_path):
                    candidate = os.path.join(model_path, d, "all_results.json")
                    if os.path.exists(candidate):
                        results_file = candidate
                        break

            if results_file is None:
                continue

            try:
                with open(results_file, "r") as f:
                    data = json.load(f)

                results = data.get("results", data) if isinstance(data, dict) else data

                for sample_data in results:
                    sample_id = sample_data.get("sample_id") or sample_data.get("id")
                    if sample_id not in gt_by_id:
                        continue

                    responses = sample_data.get("generated_runs") or sample_data.get("responses", [])

                    # Get first valid response
                    for resp in responses:
                        if resp is not None and isinstance(resp, list) and len(resp) > 0:
                            responses_by_sample[sample_id].append({
                                "model": model_name,
                                "response": resp,
                            })
                            break

            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue

        # Filter samples with enough responses
        valid_samples = [
            (sid, resps) for sid, resps in responses_by_sample.items()
            if len(resps) >= min_responses
        ]

        print(f"Found {len(valid_samples)} samples with >= {min_responses} model responses")

        # Create sample entries
        for sample_id, responses in valid_samples[:max_samples]:
            samples.append({
                "sample_id": sample_id,
                "ground_truth": {"tool_calls": gt_by_id[sample_id]},
                "responses": responses,
            })

        return samples

    def load_sharegpt_samples(
        self,
        results_dir: str,
        min_responses: int = 4,
        max_samples: int = 200,
        final_models_only: bool = False,
        exclude_models: List[str] = None,
    ) -> List[Dict]:
        """
        Load ShareGPT samples with ground truth and multiple LLM responses.

        ShareGPT data has ground_truth embedded in each results file.
        """
        samples = []
        exclude_models = exclude_models or []

        # Find ShareGPT results directory
        dataset_dir = os.path.join(results_dir, "sharegpt")
        if not os.path.exists(dataset_dir):
            print(f"ShareGPT results directory not found: {dataset_dir}")
            return samples

        model_dirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("generations-")
        ]

        # Filter to final models only if requested
        if final_models_only:
            original_count = len(model_dirs)
            model_dirs = [d for d in model_dirs if is_final_model(d.replace("generations-", ""))]
            print(f"Final models filter: {original_count} -> {len(model_dirs)} model directories")

        # Filter out excluded models
        if exclude_models:
            original_count = len(model_dirs)
            model_dirs = [
                d for d in model_dirs
                if not any(excl in d for excl in exclude_models)
            ]
            print(f"Exclude models filter: {original_count} -> {len(model_dirs)} model directories")

        # Collect responses per sample_id
        responses_by_sample = defaultdict(list)
        gt_by_sample = {}

        for model_dir in tqdm(model_dirs, desc="Loading ShareGPT model results"):
            model_name = model_dir.replace("generations-", "")
            # Clean up model name (remove date suffix)
            parts = model_name.split("-")
            if parts[-1].isdigit() and len(parts[-1]) == 8:
                model_name = "-".join(parts[:-1])

            model_path = os.path.join(dataset_dir, model_dir)

            # Find results file (prefer T=0.0 for most consistent outputs)
            results_file = None
            for d in os.listdir(model_path):
                subdir = os.path.join(model_path, d)
                if not os.path.isdir(subdir):
                    continue
                # Look for temp_0_00 directories
                if "_temp_0_00" in d or "_temp_0_0_" in d:
                    candidate = os.path.join(subdir, "all_results.json")
                    if os.path.exists(candidate):
                        results_file = candidate
                        break

            if results_file is None:
                # Try any run directory
                for d in os.listdir(model_path):
                    subdir = os.path.join(model_path, d)
                    if os.path.isdir(subdir):
                        candidate = os.path.join(subdir, "all_results.json")
                        if os.path.exists(candidate):
                            results_file = candidate
                            break

            if results_file is None:
                continue

            try:
                with open(results_file, "r") as f:
                    data = json.load(f)

                results = data.get("results", [])

                for sample_data in results:
                    sample_id = sample_data.get("sample_id", "")
                    if not sample_id:
                        continue

                    # Get ground truth (only need to store once)
                    gt = sample_data.get("ground_truth")
                    if gt and sample_id not in gt_by_sample:
                        gt_by_sample[sample_id] = gt

                    # Get generated responses (ShareGPT uses "responses" with dicts, not "generated_runs")
                    responses = sample_data.get("responses", [])
                    for resp in responses:
                        # ShareGPT responses are dicts (e.g., {"modelReport": {...}})
                        if resp is not None and isinstance(resp, dict) and len(resp) > 0:
                            responses_by_sample[sample_id].append({
                                "model": model_name,
                                "response": resp,
                            })
                            break  # Only take first valid response per model

            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue

        # Filter samples with enough responses and valid ground truth
        valid_samples = [
            (sid, resps) for sid, resps in responses_by_sample.items()
            if len(resps) >= min_responses and sid in gt_by_sample
        ]

        print(f"Found {len(valid_samples)} ShareGPT samples with >= {min_responses} model responses")

        # Create sample entries
        for sample_id, responses in valid_samples[:max_samples]:
            samples.append({
                "sample_id": sample_id,
                "ground_truth": gt_by_sample[sample_id],
                "responses": responses,
            })

        return samples

    def generate_ranking_items(
        self,
        samples: List[Dict],
        methods: List[str] = ["sted", "deepdiff", "ted", "bertscore"],
        require_disagreement: bool = True,
        max_items: int = 10000,  # High default to include all disagreement samples
        n_workers: int = 1,
    ) -> List[Dict]:
        """
        Generate ranking validation items.

        For each sample:
        1. Compute similarity scores between GT and each response using all methods
        2. Each method picks its top response
        3. If methods disagree, include as validation item

        Returns list of items for human annotation.
        """
        items = []

        print(f"Processing {len(samples)} samples with {n_workers} workers...")

        if n_workers > 1:
            # Parallel processing
            results = self._generate_ranking_items_parallel(samples, methods, n_workers)
        else:
            # Sequential processing (original behavior)
            results = self._generate_ranking_items_sequential(samples, methods)

        # Debug: count result types
        n_none = sum(1 for r in results if r is None)
        n_valid = sum(1 for r in results if r is not None)
        n_agree = sum(1 for r in results if r is not None and r.get("n_unique_picks", 0) <= 1)
        n_disagree = sum(1 for r in results if r is not None and r.get("n_unique_picks", 0) > 1)
        print(f"Debug: {n_none} None, {n_valid} valid, {n_agree} methods agree, {n_disagree} methods disagree")

        # Filter and format results
        for result in results:
            if result is None:
                continue

            if require_disagreement and result["n_unique_picks"] <= 1:
                continue

            method_picks = result["method_picks"]
            unique_picks = result["unique_picks"]

            # Build candidate set (unique responses picked by any method)
            candidates = {}
            for method, pick in method_picks.items():
                pick_key = json.dumps(pick["response"], sort_keys=True)
                if pick_key not in candidates:
                    candidates[pick_key] = {
                        "response": pick["response"],
                        "model": pick["model"],
                        "picked_by": [method],
                        "scores": {method: pick["score"]},
                    }
                else:
                    candidates[pick_key]["picked_by"].append(method)
                    candidates[pick_key]["scores"][method] = pick["score"]

            # Convert to list and add labels
            candidate_list = list(candidates.values())
            for i, cand in enumerate(candidate_list):
                cand["label"] = chr(ord("A") + i)

            # Compute GT structural metrics
            gt_metrics = compute_structural_metrics(result["ground_truth"])

            items.append({
                "sample_id": result["sample_id"],
                "ground_truth": result["ground_truth"],
                "candidates": candidate_list,
                "method_picks": {m: candidates[json.dumps(p["response"], sort_keys=True)]["label"]
                                for m, p in method_picks.items()},
                "n_unique_picks": len(unique_picks),
                "gt_metrics": gt_metrics,
            })

            if len(items) >= max_items:
                break

        print(f"Generated {len(items)} ranking items (methods disagreed)")
        return items

    def _generate_ranking_items_sequential(
        self,
        samples: List[Dict],
        methods: List[str],
    ) -> List[Optional[Dict]]:
        """Sequential score computation."""
        results = []
        debug_first = True
        for sample in tqdm(samples, desc="Computing scores"):
            gt = sample["ground_truth"]
            responses = sample["responses"]

            if len(responses) < 2:
                results.append(None)
                continue

            response_scores = []
            for resp_data in responses:
                # Toucan responses are lists (need wrapping), ShareGPT responses are dicts (use directly)
                raw_resp = resp_data["response"]
                if isinstance(raw_resp, list):
                    resp = {"tool_calls": raw_resp}  # Toucan format
                else:
                    resp = raw_resp  # ShareGPT format - already a dict
                scores = self.compute_all_scores(gt, resp)

                # Debug: print first sample's scores
                if debug_first:
                    print(f"\nDebug first sample scores: {scores}")
                    debug_first = False

                # Allow partial scores - only skip if ALL scores are None
                valid_scores = {k: v for k, v in scores.items() if v is not None}
                if len(valid_scores) < 2:  # Need at least 2 methods with valid scores
                    continue

                response_scores.append({
                    "model": resp_data["model"],
                    "response": resp,
                    "scores": scores,
                })

            if len(response_scores) < 2:
                results.append(None)
                continue

            method_picks = {}
            for method in methods:
                valid_responses = [r for r in response_scores if r["scores"].get(method) is not None]
                if valid_responses:
                    top_pick = max(valid_responses, key=lambda x: x["scores"][method])
                    method_picks[method] = {
                        "model": top_pick["model"],
                        "response": top_pick["response"],
                        "score": top_pick["scores"][method],
                    }

            unique_picks = set()
            for method, pick in method_picks.items():
                pick_key = json.dumps(pick["response"], sort_keys=True)
                unique_picks.add(pick_key)

            results.append({
                "sample_id": sample["sample_id"],
                "ground_truth": gt,
                "method_picks": method_picks,
                "unique_picks": unique_picks,
                "n_unique_picks": len(unique_picks),
            })

        return results

    def _generate_ranking_items_parallel(
        self,
        samples: List[Dict],
        methods: List[str],
        n_workers: int,
    ) -> List[Optional[Dict]]:
        """Parallel score computation using ProcessPoolExecutor."""
        results = [None] * len(samples)

        # Prepare arguments for parallel processing
        args_list = [(sample, methods) for sample in samples]

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(self.model_id,),
        ) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_compute_sample_scores, args): idx
                for idx, args in enumerate(args_list)
            }

            # Collect results with progress bar
            with tqdm(total=len(samples), desc="Computing scores (parallel)") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Error processing sample {idx}: {e}")
                        results[idx] = None
                    pbar.update(1)

        return results

    def stratify_by_complexity(
        self,
        items: List[Dict],
        n_per_stratum: int = 50,
    ) -> List[Dict]:
        """
        Stratify items by GT complexity to ensure coverage.

        Strata:
        - Simple: depth <= 2, nodes <= 10
        - Medium: depth 3-4, nodes 10-30
        - Complex: depth >= 5 or nodes >= 30
        """
        simple = []
        medium = []
        complex_items = []

        for item in items:
            metrics = item["gt_metrics"]
            depth = metrics["depth"]
            nodes = metrics["node_count"]

            if depth <= 2 and nodes <= 10:
                simple.append(item)
            elif depth >= 5 or nodes >= 30:
                complex_items.append(item)
            else:
                medium.append(item)

        print(f"Complexity distribution: simple={len(simple)}, medium={len(medium)}, complex={len(complex_items)}")

        # Sample from each stratum
        random.shuffle(simple)
        random.shuffle(medium)
        random.shuffle(complex_items)

        selected = []
        n_each = n_per_stratum // 3

        selected.extend(simple[:n_each])
        selected.extend(medium[:n_each])
        selected.extend(complex_items[:n_each])

        # Fill remaining with any
        remaining = n_per_stratum - len(selected)
        all_remaining = simple[n_each:] + medium[n_each:] + complex_items[n_each:]
        random.shuffle(all_remaining)
        selected.extend(all_remaining[:remaining])

        print(f"Selected {len(selected)} items after stratification")
        return selected

    def format_for_annotation(self, items: List[Dict], dataset_name: str = "unknown") -> Dict:
        """Format items for human annotation."""
        annotation_items = []

        for idx, item in enumerate(items):
            # Randomize candidate order to avoid position bias
            candidates = item["candidates"].copy()
            random.shuffle(candidates)

            # Re-assign labels after shuffle
            for i, cand in enumerate(candidates):
                cand["label"] = chr(ord("A") + i)

            # Update method_picks to reflect new labels
            response_to_label = {
                json.dumps(c["response"], sort_keys=True): c["label"]
                for c in candidates
            }

            method_picks = {}
            for method in ["sted", "deepdiff", "ted", "bertscore"]:
                for cand in item["candidates"]:
                    if method in cand.get("picked_by", []):
                        method_picks[method] = response_to_label[json.dumps(cand["response"], sort_keys=True)]
                        break

            # Get structural metrics for ground truth
            gt_metrics = item["gt_metrics"]

            # Get unique models in candidates
            candidate_models = [c["model"] for c in candidates]

            annotation_items.append({
                "id": f"rank_{idx:04d}",
                "sample_id": item["sample_id"],
                "ground_truth": item["ground_truth"],
                "candidates": [
                    {
                        "label": c["label"],
                        "response": c["response"],
                        "model": c["model"],
                    }
                    for c in candidates
                ],
                "metadata": {
                    "dataset": dataset_name,
                    "depth": gt_metrics.get("depth", 0),
                    "node_count": gt_metrics.get("node_count", 0),
                    "array_count": gt_metrics.get("array_count", 0),
                    "object_count": gt_metrics.get("object_count", 0),
                    "method_picks": method_picks,  # Hidden from annotators
                    "n_candidates": len(candidates),
                    "candidate_models": candidate_models,
                    "gt_metrics": gt_metrics,  # Full metrics object for reference
                },
                "annotation": {
                    "choice": None,  # A, B, C, D, etc.
                    "confidence": None,  # 1-5
                    "annotator_id": None,
                    "timestamp": None,
                },
            })

        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "dataset": dataset_name,
                "n_items": len(annotation_items),
                "task_type": "ranking",
                "methods_compared": ["sted", "deepdiff", "ted", "bertscore"],
                "temperature": "0.0",
                "embedding_model": "all-MiniLM-L6-v2",
            },
            "annotation_guidelines": {
                "task": "Given the Ground Truth, choose which candidate response is MOST similar.",
                "criteria": [
                    "Same tool names and function calls",
                    "Same parameter names and values",
                    "Same structure (order may differ)",
                    "Could be used interchangeably in downstream system",
                ],
                "confidence_scale": {
                    "1": "Very uncertain - all look equally similar/different",
                    "2": "Somewhat uncertain",
                    "3": "Moderately confident",
                    "4": "Confident",
                    "5": "Very confident - clearly best match",
                },
            },
            "items": annotation_items,
        }

    def generate_dataset(
        self,
        toucan_data_path: str,
        results_dir: str,
        output_path: str = "ranking_validation_dataset.json",
        n_items: int = 10000,  # High default to include all disagreement samples
        require_disagreement: bool = True,
        max_samples: int = 2000,  # High default to process all Toucan samples
        n_workers: int = 1,
        final_models_only: bool = False,
        exclude_models: List[str] = None,
        skip_stratification: bool = False,
    ) -> Dict:
        """Generate complete ranking validation dataset (Toucan only)."""

        # Load Toucan samples only
        samples = self.load_toucan_samples(
            toucan_data_path,
            results_dir,
            min_responses=4,
            max_samples=max_samples,
            final_models_only=final_models_only,
            exclude_models=exclude_models,
        )

        if not samples:
            print("No valid samples found!")
            return {}

        # Generate ranking items (all disagreement samples)
        items = self.generate_ranking_items(
            samples,
            require_disagreement=require_disagreement,
            max_items=n_items,  # Include all disagreement samples
            n_workers=n_workers,
        )

        # Optionally stratify by complexity (disabled by default for full dataset)
        if not skip_stratification and n_items < len(items):
            items = self.stratify_by_complexity(items, n_per_stratum=n_items)

        # Format for annotation
        result = self.format_for_annotation(items, dataset_name="toucan")

        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nSaved {len(result['items'])} ranking items to {output_path}")

        # Print summary
        self._print_summary(result)

        return result

    def _print_summary(self, dataset: Dict):
        """Print summary statistics."""
        items = dataset["items"]

        print("\n" + "=" * 60)
        print("RANKING VALIDATION DATASET SUMMARY")
        print("=" * 60)

        print(f"\nTotal items: {len(items)}")

        # Method pick distribution
        print("\nMethod pick distribution:")
        pick_counts = defaultdict(lambda: defaultdict(int))
        for item in items:
            method_picks = item["metadata"]["method_picks"]
            for method, label in method_picks.items():
                pick_counts[method][label] += 1

        for method in ["sted", "deepdiff", "ted", "bertscore"]:
            counts = pick_counts[method]
            print(f"  {method}: {dict(counts)}")

        # Complexity distribution
        print("\nComplexity distribution:")
        simple = medium = complex_count = 0
        for item in items:
            metrics = item["metadata"]["gt_metrics"]
            if metrics["depth"] <= 2 and metrics["node_count"] <= 10:
                simple += 1
            elif metrics["depth"] >= 5 or metrics["node_count"] >= 30:
                complex_count += 1
            else:
                medium += 1

        print(f"  Simple: {simple}")
        print(f"  Medium: {medium}")
        print(f"  Complex: {complex_count}")

        # Candidate count distribution
        n_candidates = [item["metadata"]["n_candidates"] for item in items]
        if n_candidates:
            print(f"\nCandidates per item: mean={np.mean(n_candidates):.1f}, min={min(n_candidates)}, max={max(n_candidates)}")
        else:
            print("\nNo items generated - all methods may have agreed on the same response.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ranking-based human validation dataset (Toucan tool calling only)"
    )
    parser.add_argument(
        "--toucan-data-path",
        default="toucan_data/toucan_tool_calls_1006.json",
        help="Path to Toucan ground truth JSON",
    )
    parser.add_argument(
        "--results-dir",
        default="llm_gen_results",
        help="Directory containing LLM results",
    )
    parser.add_argument(
        "--output",
        default="scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json",
        help="Output file path",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=10000,
        help="Maximum number of ranking items to generate (default: 10000, i.e., all disagreement samples)",
    )
    parser.add_argument(
        "--include-agreements",
        action="store_true",
        help="Include items where all methods agree (default: only disagreements)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum samples to process (default: 2000, covers all 1006 Toucan samples)",
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
        help="Random seed",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use CPU count for max parallelism)",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=[],
        help="List of model names to exclude (substring match). E.g., --exclude-models nemotron-3-nano",
    )
    parser.add_argument(
        "--final-models-only",
        action="store_true",
        help="Only include results from FINAL_MODELS (same as visualize_consistency_scores.py)",
    )
    parser.add_argument(
        "--skip-stratification",
        action="store_true",
        default=True,
        help="Skip complexity stratification to include all disagreement samples (default: True)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Determine number of workers
    n_workers = args.n_workers
    if n_workers <= 0:
        n_workers = mp.cpu_count()
    print(f"Using {n_workers} workers (CPU count: {mp.cpu_count()})")

    generator = RankingValidationDatasetGenerator(model_id=args.model_id)
    generator.generate_dataset(
        toucan_data_path=args.toucan_data_path,
        results_dir=args.results_dir,
        output_path=args.output,
        n_items=args.n_items,
        require_disagreement=not args.include_agreements,
        max_samples=args.max_samples,
        n_workers=n_workers,
        final_models_only=args.final_models_only,
        exclude_models=args.exclude_models,
        skip_stratification=args.skip_stratification,
    )


if __name__ == "__main__":
    main()
