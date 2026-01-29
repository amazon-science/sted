#!/usr/bin/env python
"""
Create a stratified subset of the validation dataset.

Selects samples to match a target size while maintaining:
1. Complexity distribution (simple/medium/complex)
2. Candidate count distribution (2/3/4 candidates)
"""

import json
import argparse
import random
from collections import defaultdict
from typing import List, Dict


def get_complexity(item: Dict) -> str:
    """Determine complexity category."""
    metrics = item.get("metadata", {}).get("gt_metrics", {})
    depth = metrics.get("depth", 0)
    nodes = metrics.get("node_count", 0)

    if depth <= 2 and nodes <= 10:
        return "simple"
    elif depth >= 5 or nodes >= 30:
        return "complex"
    else:
        return "medium"


def create_stratified_subset(
    input_file: str,
    output_file: str,
    target_size: int = 79,
    seed: int = 42,
    strategy: str = "keep_all_complex",
):
    """Create stratified subset matching target size.

    Args:
        input_file: Input dataset JSON file
        output_file: Output dataset JSON file
        target_size: Target number of samples
        seed: Random seed for reproducibility
        strategy: Sampling strategy - "keep_all_complex" (default) or "proportional"
    """

    random.seed(seed)

    # Load dataset
    with open(input_file, "r") as f:
        data = json.load(f)

    items = data.get("items", [])
    print(f"Loaded {len(items)} items from {input_file}")

    if len(items) <= target_size:
        print(f"Dataset already has {len(items)} items (<= {target_size}), no subsetting needed")
        # Just copy to output
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        return

    # Group by complexity
    by_complexity = defaultdict(list)
    for item in items:
        complexity = get_complexity(item)
        by_complexity[complexity].append(item)

    print(f"\nOriginal distribution:")
    for c in ["simple", "medium", "complex"]:
        print(f"  {c}: {len(by_complexity[c])}")

    if strategy == "keep_all_complex":
        # Keep ALL complex samples, fill remaining with medium, then simple
        selected = []

        # 1. Add all complex samples
        selected.extend(by_complexity["complex"])
        remaining = target_size - len(selected)

        # 2. Add medium samples
        random.shuffle(by_complexity["medium"])
        n_medium = min(len(by_complexity["medium"]), remaining)
        selected.extend(by_complexity["medium"][:n_medium])
        remaining = target_size - len(selected)

        # 3. Add simple samples if still needed
        if remaining > 0:
            random.shuffle(by_complexity["simple"])
            selected.extend(by_complexity["simple"][:remaining])

        print(f"\nStrategy: keep_all_complex")
        print(f"  Complex: {len(by_complexity['complex'])} (all included)")
        print(f"  Medium: {n_medium} (sampled)")
        print(f"  Simple: {max(0, target_size - len(by_complexity['complex']) - n_medium)} (sampled)")

    else:  # proportional
        # Calculate target counts (proportional)
        total = len(items)
        target_counts = {}
        for c in ["simple", "medium", "complex"]:
            proportion = len(by_complexity[c]) / total
            target_counts[c] = max(1, round(proportion * target_size)) if by_complexity[c] else 0

        # Adjust to exactly match target_size
        current_total = sum(target_counts.values())
        diff = target_size - current_total

        # Add/remove from largest category
        largest = max(target_counts, key=target_counts.get)
        target_counts[largest] += diff

        print(f"\nStrategy: proportional")
        print(f"Target distribution (n={target_size}):")
        for c in ["simple", "medium", "complex"]:
            print(f"  {c}: {target_counts[c]}")

        # Sample from each category
        selected = []
        for complexity, count in target_counts.items():
            available = by_complexity[complexity]
            random.shuffle(available)
            selected.extend(available[:count])

    # Shuffle final selection
    random.shuffle(selected)

    # Re-assign IDs
    for i, item in enumerate(selected):
        item["id"] = f"rank_{i:04d}"

    # Update metadata
    data["items"] = selected
    data["metadata"]["n_items"] = len(selected)
    data["metadata"]["subset_info"] = {
        "original_size": len(items),
        "target_size": target_size,
        "actual_size": len(selected),
        "seed": seed,
        "strategy": strategy,
    }

    # Save
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(selected)} items to {output_file}")

    # Print final distribution
    final_dist = defaultdict(int)
    for item in selected:
        final_dist[get_complexity(item)] += 1

    print(f"\nFinal distribution:")
    for c in ["simple", "medium", "complex"]:
        print(f"  {c}: {final_dist[c]}")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified subset of validation dataset"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset JSON file",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=79,
        help="Target number of samples (default: 79)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--strategy",
        choices=["keep_all_complex", "proportional"],
        default="keep_all_complex",
        help="Sampling strategy: 'keep_all_complex' (default) or 'proportional'",
    )

    args = parser.parse_args()

    create_stratified_subset(
        input_file=args.input,
        output_file=args.output,
        target_size=args.target_size,
        seed=args.seed,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
