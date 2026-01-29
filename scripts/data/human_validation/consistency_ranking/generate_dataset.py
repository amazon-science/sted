#!/usr/bin/env python
"""
Generate consistency ranking dataset for validating STED consistency scores.

This script creates pairs/groups of output sets for human ranking:
- Each "set" contains N outputs from the same prompt
- Annotators rank which set is more consistent
- Validates that STED consistency score matches human consistency judgment

Output: JSON file with set pairs ready for ranking annotation.
"""

import json
import os
import argparse
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from itertools import combinations

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


class ConsistencyRankingDatasetGenerator:
    """Generate set pairs for human consistency ranking validation."""

    def __init__(
        self,
        model_id: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 512,
    ):
        """Initialize STED evaluator for computing consistency scores."""
        print("Initializing STED evaluator...")
        self.evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id=model_id,
            embedding_dim=embedding_dim,
        )
        print("STED evaluator initialized.")

    def compute_set_consistency(self, outputs: List[Dict]) -> Dict[str, float]:
        """
        Compute consistency score for a set of outputs.

        Returns dict with:
        - sted_consistency: mean pairwise STED similarity
        - sted_min: minimum pairwise STED similarity
        - sted_std: standard deviation of pairwise similarities
        """
        if len(outputs) < 2:
            return {"sted_consistency": 1.0, "sted_min": 1.0, "sted_std": 0.0}

        pairwise_scores = []
        for i, j in combinations(range(len(outputs)), 2):
            try:
                score = self.evaluator.calculate_tree_edit_distance_fast(
                    outputs[i], outputs[j], variation_type="combined"
                )
                pairwise_scores.append(score)
            except Exception as e:
                print(f"Error computing STED: {e}")
                continue

        if not pairwise_scores:
            return {"sted_consistency": None, "sted_min": None, "sted_std": None}

        return {
            "sted_consistency": np.mean(pairwise_scores),
            "sted_min": np.min(pairwise_scores),
            "sted_std": np.std(pairwise_scores),
            "n_pairs": len(pairwise_scores),
        }

    def load_output_sets_from_llm_results(
        self,
        results_dir: str,
        dataset: str = "toucan",
        min_outputs_per_set: int = 3,
        max_outputs_per_set: int = 5,
    ) -> List[Dict]:
        """
        Load sets of outputs from LLM generation results.

        Each set contains multiple outputs for the same prompt.
        """
        all_sets = []

        dataset_dir = os.path.join(results_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            return all_sets

        # Find all model generation directories
        model_dirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("generations-")
        ]

        for model_dir in model_dirs:
            model_name = model_dir.replace("generations-", "")
            model_path = os.path.join(dataset_dir, model_dir)

            # Find temperature directories
            temp_dirs = [
                d for d in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, d)) and d.startswith("temp_")
            ]

            for temp_dir in temp_dirs:
                temp_path = os.path.join(model_path, temp_dir)
                results_file = os.path.join(temp_path, "all_results.json")

                if not os.path.exists(results_file):
                    continue

                try:
                    with open(results_file, "r") as f:
                        results = json.load(f)

                    temperature = float(temp_dir.replace("temp_", ""))

                    for sample_idx, sample_data in enumerate(results):
                        responses = sample_data.get("responses", [])

                        # Filter valid responses
                        valid_outputs = []
                        for resp in responses:
                            if resp is None:
                                continue
                            if isinstance(resp, dict):
                                if "modelReport" in resp:
                                    valid_outputs.append(resp["modelReport"])
                                else:
                                    valid_outputs.append(resp)
                            elif isinstance(resp, list):
                                valid_outputs.append({"tool_calls": resp})

                        # Only include if we have enough outputs
                        if len(valid_outputs) >= min_outputs_per_set:
                            # Limit to max outputs
                            valid_outputs = valid_outputs[:max_outputs_per_set]

                            all_sets.append({
                                "source": "llm_output",
                                "model": model_name,
                                "temperature": temperature,
                                "sample_idx": sample_idx,
                                "sample_id": sample_data.get("id", f"{model_name}_{sample_idx}"),
                                "dataset": dataset,
                                "outputs": valid_outputs,
                                "n_outputs": len(valid_outputs),
                            })

                except Exception as e:
                    print(f"Error loading {results_file}: {e}")
                    continue

        print(f"Loaded {len(all_sets)} output sets from {dataset}")
        return all_sets

    def create_ranking_pairs(
        self,
        output_sets: List[Dict],
        n_pairs: int = 100,
        min_consistency_diff: float = 0.1,
    ) -> List[Dict]:
        """
        Create pairs of sets for ranking comparison.

        Pairs are selected to have diverse consistency differences.
        """
        print(f"Computing consistency scores for {len(output_sets)} sets...")

        # Compute consistency for all sets
        for output_set in tqdm(output_sets, desc="Computing consistency"):
            scores = self.compute_set_consistency(output_set["outputs"])
            output_set["consistency_scores"] = scores

        # Filter out sets with invalid scores
        valid_sets = [s for s in output_sets if s["consistency_scores"].get("sted_consistency") is not None]
        print(f"Valid sets with consistency scores: {len(valid_sets)}")

        if len(valid_sets) < 2:
            print("Not enough valid sets for ranking pairs")
            return []

        # Sort by consistency score
        valid_sets.sort(key=lambda x: x["consistency_scores"]["sted_consistency"])

        # Create pairs with diverse consistency differences
        ranking_pairs = []

        # Strategy: pair sets from different consistency levels
        n_bins = 5
        bin_size = len(valid_sets) // n_bins
        bins = [valid_sets[i*bin_size:(i+1)*bin_size] for i in range(n_bins)]

        # Handle remainder
        if len(valid_sets) % n_bins > 0:
            bins[-1].extend(valid_sets[n_bins*bin_size:])

        pair_id = 0
        attempts = 0
        max_attempts = n_pairs * 10

        while len(ranking_pairs) < n_pairs and attempts < max_attempts:
            attempts += 1

            # Select two different bins
            bin_indices = random.sample(range(n_bins), 2)
            bin_a, bin_b = bins[bin_indices[0]], bins[bin_indices[1]]

            if not bin_a or not bin_b:
                continue

            set_a = random.choice(bin_a)
            set_b = random.choice(bin_b)

            # Ensure sufficient consistency difference
            cons_a = set_a["consistency_scores"]["sted_consistency"]
            cons_b = set_b["consistency_scores"]["sted_consistency"]

            if abs(cons_a - cons_b) < min_consistency_diff:
                continue

            # Avoid duplicate pairs
            pair_key = tuple(sorted([set_a["sample_id"], set_b["sample_id"]]))
            if any(p.get("_key") == pair_key for p in ranking_pairs):
                continue

            # Randomly order (so annotators don't learn pattern)
            if random.random() > 0.5:
                set_a, set_b = set_b, set_a
                cons_a, cons_b = cons_b, cons_a

            ranking_pairs.append({
                "pair_id": f"rank_{pair_id:04d}",
                "_key": pair_key,
                "set_a": {
                    "outputs": set_a["outputs"],
                    "n_outputs": set_a["n_outputs"],
                    "model": set_a.get("model"),
                    "temperature": set_a.get("temperature"),
                },
                "set_b": {
                    "outputs": set_b["outputs"],
                    "n_outputs": set_b["n_outputs"],
                    "model": set_b.get("model"),
                    "temperature": set_b.get("temperature"),
                },
                "metadata": {
                    "sted_consistency_a": cons_a,
                    "sted_consistency_b": cons_b,
                    "consistency_diff": abs(cons_a - cons_b),
                    "expected_answer": "A" if cons_a > cons_b else "B",
                },
            })
            pair_id += 1

        # Remove internal keys
        for pair in ranking_pairs:
            pair.pop("_key", None)

        print(f"Created {len(ranking_pairs)} ranking pairs")
        return ranking_pairs

    def stratify_by_difficulty(
        self,
        ranking_pairs: List[Dict],
        n_per_difficulty: int = 25,
    ) -> List[Dict]:
        """
        Stratify ranking pairs by difficulty (consistency difference).

        Difficulty levels:
        - Easy: large difference (>0.3)
        - Medium: moderate difference (0.15-0.3)
        - Hard: small difference (0.05-0.15)
        - Very Hard: tiny difference (<0.05)
        """
        difficulty_bins = {
            "easy": (0.3, 1.0),
            "medium": (0.15, 0.3),
            "hard": (0.05, 0.15),
            "very_hard": (0.0, 0.05),
        }

        stratified = []

        for difficulty, (low, high) in difficulty_bins.items():
            matching = [
                p for p in ranking_pairs
                if low <= p["metadata"]["consistency_diff"] < high
            ]

            # Add difficulty label
            for p in matching:
                p["metadata"]["difficulty"] = difficulty

            random.shuffle(matching)
            selected = matching[:n_per_difficulty]
            stratified.extend(selected)

            print(f"  {difficulty}: {len(matching)} available, selected {len(selected)}")

        random.shuffle(stratified)
        return stratified

    def format_for_annotation(self, ranking_pairs: List[Dict]) -> Dict:
        """Format ranking pairs for annotation interface."""

        annotation_items = []

        for pair in ranking_pairs:
            item = {
                "id": pair["pair_id"],
                "set_a": pair["set_a"],
                "set_b": pair["set_b"],
                "metadata": {
                    "difficulty": pair["metadata"].get("difficulty"),
                    # Hide ground truth from annotators
                },
                "annotation": {
                    "choice": None,  # "A", "B", or "equal"
                    "confidence": None,  # 1-5
                    "annotator_id": None,
                    "timestamp": None,
                    "notes": None,
                },
                # Store ground truth separately for analysis
                "_ground_truth": {
                    "sted_consistency_a": pair["metadata"]["sted_consistency_a"],
                    "sted_consistency_b": pair["metadata"]["sted_consistency_b"],
                    "expected_answer": pair["metadata"]["expected_answer"],
                    "consistency_diff": pair["metadata"]["consistency_diff"],
                },
            }
            annotation_items.append(item)

        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "n_pairs": len(annotation_items),
                "task_type": "consistency_ranking",
                "description": "Rank which set of outputs is more consistent",
            },
            "annotation_guidelines": {
                "task": "For each pair of output sets, indicate which set shows more consistent outputs.",
                "options": {
                    "A": "Set A is more consistent",
                    "B": "Set B is more consistent",
                    "equal": "Both sets are equally consistent",
                },
                "confidence_scale": {
                    "1": "Very uncertain",
                    "2": "Somewhat uncertain",
                    "3": "Neutral",
                    "4": "Somewhat confident",
                    "5": "Very confident",
                },
                "definition": "Consistency means the outputs convey similar information and structure. A consistent set would work interchangeably in downstream applications.",
            },
            "pairs": annotation_items,
        }

    def generate_dataset(
        self,
        llm_results_dir: str,
        output_path: str = "consistency_ranking_dataset.json",
        dataset: str = "toucan",
        n_pairs: int = 100,
        n_per_difficulty: int = 25,
        min_outputs_per_set: int = 3,
        max_outputs_per_set: int = 5,
    ) -> List[Dict]:
        """Generate complete consistency ranking dataset."""

        # Load output sets
        output_sets = self.load_output_sets_from_llm_results(
            llm_results_dir,
            dataset=dataset,
            min_outputs_per_set=min_outputs_per_set,
            max_outputs_per_set=max_outputs_per_set,
        )

        if not output_sets:
            print("No output sets loaded!")
            return []

        # Create ranking pairs
        ranking_pairs = self.create_ranking_pairs(
            output_sets,
            n_pairs=n_pairs * 2,  # Create more to allow stratification
            min_consistency_diff=0.02,
        )

        if not ranking_pairs:
            print("No ranking pairs created!")
            return []

        # Stratify by difficulty
        stratified_pairs = self.stratify_by_difficulty(
            ranking_pairs,
            n_per_difficulty=n_per_difficulty,
        )

        # Format for annotation
        output_data = self.format_for_annotation(stratified_pairs)

        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved {len(stratified_pairs)} ranking pairs to {output_path}")
        self._print_summary(stratified_pairs)

        return stratified_pairs

    def _print_summary(self, pairs: List[Dict]):
        """Print summary statistics."""
        print("\n" + "=" * 50)
        print("CONSISTENCY RANKING DATASET SUMMARY")
        print("=" * 50)

        # By difficulty
        difficulties = defaultdict(int)
        for p in pairs:
            diff = p["metadata"].get("difficulty", "unknown")
            difficulties[diff] += 1

        print("\nBy difficulty:")
        for diff in ["easy", "medium", "hard", "very_hard"]:
            print(f"  {diff}: {difficulties.get(diff, 0)}")

        # Consistency difference stats
        diffs = [p["metadata"]["consistency_diff"] for p in pairs]
        print(f"\nConsistency difference statistics:")
        print(f"  Mean: {np.mean(diffs):.3f}")
        print(f"  Std:  {np.std(diffs):.3f}")
        print(f"  Min:  {np.min(diffs):.3f}")
        print(f"  Max:  {np.max(diffs):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate consistency ranking dataset for STED validation"
    )
    parser.add_argument(
        "--llm-results-dir",
        default="llm_gen_results",
        help="Directory containing LLM generation results",
    )
    parser.add_argument(
        "--output",
        default="consistency_ranking_dataset.json",
        help="Output file path",
    )
    parser.add_argument(
        "--dataset",
        choices=["sharegpt", "toucan"],
        default="toucan",
        help="Dataset to use for output sets",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        help="Target number of ranking pairs",
    )
    parser.add_argument(
        "--n-per-difficulty",
        type=int,
        default=25,
        help="Pairs per difficulty level (easy/medium/hard/very_hard)",
    )
    parser.add_argument(
        "--min-outputs",
        type=int,
        default=3,
        help="Minimum outputs per set",
    )
    parser.add_argument(
        "--max-outputs",
        type=int,
        default=5,
        help="Maximum outputs per set",
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

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    generator = ConsistencyRankingDatasetGenerator(model_id=args.model_id)
    generator.generate_dataset(
        llm_results_dir=args.llm_results_dir,
        output_path=args.output,
        dataset=args.dataset,
        n_pairs=args.n_pairs,
        n_per_difficulty=args.n_per_difficulty,
        min_outputs_per_set=args.min_outputs,
        max_outputs_per_set=args.max_outputs,
    )


if __name__ == "__main__":
    main()
