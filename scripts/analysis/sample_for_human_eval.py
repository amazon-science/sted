#!/usr/bin/env python3
"""
Sample output pairs for human evaluation.

Stratifies by STED score:
- 30% high STED (>0.9) - easy cases
- 40% medium STED (0.5-0.9) - discriminative cases
- 30% low STED (<0.5) - clear differences

Output: 150 pairs (100 Toucan, 50 ShareGPT) for annotation
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
import csv

# Set random seed for reproducibility
random.seed(42)

# Paths
BASE_DIR = Path("/Users/guanghu/Documents/genai/projects/sted-internal")
TOUCAN_CONSISTENCY = BASE_DIR / "llm_gen_results/toucan/consistency_results/combined_consistency_metrics_results.json"
TOUCAN_DATA = BASE_DIR / "toucan_data/toucan_tool_calls_1006.json"
TOUCAN_GENERATIONS_DIR = BASE_DIR / "llm_gen_results/toucan"
SHAREGPT_DIR = BASE_DIR / "llm_gen_results/sharegpt"
OUTPUT_DIR = BASE_DIR / "human_evaluation"


def load_toucan_data():
    """Load original Toucan queries."""
    with open(TOUCAN_DATA) as f:
        data = json.load(f)
    return {i: item for i, item in enumerate(data)}


def load_toucan_consistency():
    """Load consistency metrics for all models."""
    with open(TOUCAN_CONSISTENCY) as f:
        return json.load(f)


def load_toucan_generations(model_name: str, sample_idx: int, temp: float):
    """Load actual generation outputs for a sample."""
    # Map model names to directory names
    model_dir_map = {
        "Claude-3.5-Sonnet": "generations-claude-3.5-sonnet-20251223",
        "Claude-3.5-Haiku": "generations-claude-3.5-haiku-20251224",
        "Claude-3.7-Sonnet": "generations-claude37-sonnet-20251229",
        "Claude-Opus-4": "generations-claude-opus-4-20251222",
        "Claude-Opus-4.5": "generations-claude-opus-4.5-20251224",
        "Claude-Sonnet-4": "generations-claude-sonnet-4-20251223",
        "Claude-Sonnet-4.5": "generations-claude-sonnet-4.5-20251224",
        "Claude-Haiku-4.5": "generations-claude-haiku-4.5-20251223",
        "Qwen3-235B-A22B": "generations-qwen3-235b-a22b-20251229",
        "Qwen3-32B": "generations-qwen3-32b-20251224",
        "Llama-3.3-70B": "generations-llama-3.3-70b-20251223",
        "Nova-2-Lite": "generations-nova2-lite-20251222_075929",
        "Mimo-V2-Flash:free": "generations-mimo-v2-flash-20251229",
    }

    if model_name not in model_dir_map:
        return None

    model_dir = TOUCAN_GENERATIONS_DIR / model_dir_map[model_name]
    if not model_dir.exists():
        return None

    # Find temperature directory - format: temp_0_50 for 0.5, temp_0_10 for 0.1
    temp_int = int(temp)
    temp_dec = int(round((temp - temp_int) * 100))  # 0.5 -> 50, 0.1 -> 10
    temp_str = f"temp_{temp_int}_{temp_dec:02d}"

    for subdir in model_dir.iterdir():
        if temp_str in subdir.name and subdir.is_dir():
            # Try all_results.json first, then generation_results.json
            for results_name in ["all_results.json", "generation_results.json"]:
                results_file = subdir / results_name
                if results_file.exists():
                    try:
                        with open(results_file) as f:
                            data = json.load(f)
                            if "results" in data and sample_idx < len(data["results"]):
                                result = data["results"][sample_idx]
                                # Normalize the runs field name
                                if "generated_runs" in result:
                                    result["runs"] = result["generated_runs"]
                                return result
                    except Exception as e:
                        print(f"Error loading {results_file}: {e}")
                        continue
    return None


def stratified_sample_toucan(consistency_data: dict, n_samples: int = 100):
    """
    Stratified sampling from Toucan based on STED score.

    Returns samples with diverse STED scores and different models.
    """
    # Collect all samples with their STED scores
    all_samples = []

    # Focus on good models for meaningful comparisons
    target_models = [
        "Claude-3.5-Sonnet", "Claude-3.5-Haiku", "Claude-3.7-Sonnet",
        "Claude-Opus-4", "Qwen3-235B-A22B", "Llama-3.3-70B", "Nova-2-Lite"
    ]

    for model, samples in consistency_data.items():
        if model not in target_models:
            continue

        for sample in samples:
            if sample.get("validity_rate", 0) > 0:  # Only valid samples
                sted_score = sample.get("mean_similarity", 0)
                all_samples.append({
                    "model": model,
                    "sample_idx": sample["sample_idx"],
                    "temperature": sample["temperature"],
                    "sted_score": sted_score,
                    "validity_rate": sample["validity_rate"],
                    "consistency_score": sample.get("penalized_consistency_coefficient", 0)
                })

    # Stratify by STED score
    high_sted = [s for s in all_samples if s["sted_score"] > 0.9]
    medium_sted = [s for s in all_samples if 0.5 <= s["sted_score"] <= 0.9]
    low_sted = [s for s in all_samples if s["sted_score"] < 0.5]

    print(f"Toucan samples available:")
    print(f"  High STED (>0.9): {len(high_sted)}")
    print(f"  Medium STED (0.5-0.9): {len(medium_sted)}")
    print(f"  Low STED (<0.5): {len(low_sted)}")

    # Sample with stratification: 30% high, 40% medium, 30% low
    n_high = int(n_samples * 0.3)
    n_medium = int(n_samples * 0.4)
    n_low = n_samples - n_high - n_medium

    selected = []

    if len(high_sted) >= n_high:
        selected.extend(random.sample(high_sted, n_high))
    else:
        selected.extend(high_sted)

    if len(medium_sted) >= n_medium:
        selected.extend(random.sample(medium_sted, n_medium))
    else:
        selected.extend(medium_sted)

    if len(low_sted) >= n_low:
        selected.extend(random.sample(low_sted, n_low))
    else:
        selected.extend(low_sted)

    print(f"\nSelected {len(selected)} Toucan samples")
    return selected


def load_sharegpt_consistency():
    """Load ShareGPT consistency results if available."""
    # Look for consistency results in ShareGPT directory
    sharegpt_consistency_dir = SHAREGPT_DIR / "consistency_results"
    if sharegpt_consistency_dir.exists():
        for f in sharegpt_consistency_dir.glob("*.json"):
            with open(f) as fp:
                return json.load(fp)

    # Fallback: generate from generation files
    print("No pre-computed ShareGPT consistency found, sampling from generations...")
    return None


def sample_sharegpt_from_generations(n_samples: int = 50):
    """Sample ShareGPT outputs directly from generation files."""
    samples = []

    # Look for generation directories
    for model_dir in SHAREGPT_DIR.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("generations-"):
            continue

        model_name = model_dir.name.replace("generations-", "").split("-2025")[0].split("-2026")[0]

        # Find temperature directories
        for temp_dir in sorted(model_dir.iterdir()):
            if not temp_dir.is_dir() or "temp_" not in temp_dir.name:
                continue

            results_file = temp_dir / "generation_results.json"
            if not results_file.exists():
                continue

            try:
                with open(results_file) as f:
                    data = json.load(f)

                if "results" not in data:
                    continue

                temp = data.get("metadata", {}).get("temperature", 0.5)

                for idx, result in enumerate(data["results"]):
                    if result.get("valid_runs", 0) >= 2:  # Need at least 2 valid for comparison
                        samples.append({
                            "model": model_name,
                            "sample_idx": idx,
                            "temperature": temp,
                            "valid_runs": result.get("valid_runs", 0),
                            "runs": result.get("runs", []),
                            "avg_similarity": result.get("avg_similarity", 0)
                        })
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue

    print(f"Found {len(samples)} ShareGPT samples with valid runs")

    if len(samples) == 0:
        return []

    # Stratify by similarity score
    high_sim = [s for s in samples if s.get("avg_similarity", 0) > 0.9]
    medium_sim = [s for s in samples if 0.5 <= s.get("avg_similarity", 0) <= 0.9]
    low_sim = [s for s in samples if s.get("avg_similarity", 0) < 0.5]

    print(f"ShareGPT samples by similarity:")
    print(f"  High (>0.9): {len(high_sim)}")
    print(f"  Medium (0.5-0.9): {len(medium_sim)}")
    print(f"  Low (<0.5): {len(low_sim)}")

    # Sample with stratification
    n_high = int(n_samples * 0.3)
    n_medium = int(n_samples * 0.4)
    n_low = n_samples - n_high - n_medium

    selected = []
    if high_sim:
        selected.extend(random.sample(high_sim, min(n_high, len(high_sim))))
    if medium_sim:
        selected.extend(random.sample(medium_sim, min(n_medium, len(medium_sim))))
    if low_sim:
        selected.extend(random.sample(low_sim, min(n_low, len(low_sim))))

    return selected[:n_samples]


def create_annotation_csv(toucan_samples: list, sharegpt_samples: list, toucan_data: dict):
    """Create CSV file for human annotation."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows = []
    pair_id = 0

    # Process Toucan samples
    loaded_count = 0
    for sample in toucan_samples:
        sample_idx = sample["sample_idx"]
        original_data = toucan_data.get(sample_idx, {})

        # Load actual generations
        generations = load_toucan_generations(
            sample["model"],
            sample_idx,
            sample["temperature"]
        )

        if generations and "runs" in generations and len(generations["runs"]) >= 2:
            # Take first two runs as the pair to compare
            run1 = generations["runs"][0]
            run2 = generations["runs"][1]

            # Get query from generations if available, otherwise from original data
            query = generations.get("query", original_data.get("question", ""))

            rows.append({
                "pair_id": pair_id,
                "dataset": "toucan",
                "model": sample["model"],
                "sample_idx": sample_idx,
                "temperature": sample["temperature"],
                "sted_score": round(sample["sted_score"], 4),
                "query": query[:500],  # Truncate long queries
                "output_1": json.dumps(run1, indent=2)[:2000],  # Truncate if too long
                "output_2": json.dumps(run2, indent=2)[:2000],
                "human_rating": "",  # To be filled by annotator (1-5 scale)
                "notes": ""
            })
            pair_id += 1
            loaded_count += 1
        else:
            if generations is None:
                print(f"  Could not load generations for {sample['model']} sample {sample_idx} temp {sample['temperature']}")
            elif "runs" not in generations:
                print(f"  No 'runs' field for {sample['model']} sample {sample_idx}")
            elif len(generations.get("runs", [])) < 2:
                print(f"  Not enough runs for {sample['model']} sample {sample_idx}: {len(generations.get('runs', []))}")

    print(f"Successfully loaded {loaded_count} Toucan generations")

    # Process ShareGPT samples
    for sample in sharegpt_samples:
        if "runs" in sample and len(sample["runs"]) >= 2:
            run1 = sample["runs"][0]
            run2 = sample["runs"][1]

            rows.append({
                "pair_id": pair_id,
                "dataset": "sharegpt",
                "model": sample["model"],
                "sample_idx": sample["sample_idx"],
                "temperature": sample["temperature"],
                "sted_score": round(sample.get("avg_similarity", 0), 4),
                "query": "",  # Would need to load from original data
                "output_1": json.dumps(run1, indent=2)[:2000] if isinstance(run1, (dict, list)) else str(run1)[:2000],
                "output_2": json.dumps(run2, indent=2)[:2000] if isinstance(run2, (dict, list)) else str(run2)[:2000],
                "human_rating": "",
                "notes": ""
            })
            pair_id += 1

    # Write CSV
    output_file = OUTPUT_DIR / "human_eval_samples.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "dataset", "model", "sample_idx", "temperature",
            "sted_score", "query", "output_1", "output_2", "human_rating", "notes"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCreated annotation file: {output_file}")
    print(f"Total pairs: {len(rows)}")

    # Also create a simpler JSON format for easier loading
    json_file = OUTPUT_DIR / "human_eval_samples.json"
    with open(json_file, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Created JSON file: {json_file}")

    # Print summary statistics
    print("\n=== Sample Summary ===")
    sted_scores = [r["sted_score"] for r in rows]
    print(f"STED score distribution:")
    print(f"  High (>0.9): {sum(1 for s in sted_scores if s > 0.9)}")
    print(f"  Medium (0.5-0.9): {sum(1 for s in sted_scores if 0.5 <= s <= 0.9)}")
    print(f"  Low (<0.5): {sum(1 for s in sted_scores if s < 0.5)}")

    models = defaultdict(int)
    for r in rows:
        models[r["model"]] += 1
    print(f"\nModel distribution:")
    for model, count in sorted(models.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count}")

    return rows


def main():
    print("Loading Toucan data...")
    toucan_data = load_toucan_data()

    print("Loading Toucan consistency metrics...")
    toucan_consistency = load_toucan_consistency()

    print("\n=== Sampling Toucan ===")
    toucan_samples = stratified_sample_toucan(toucan_consistency, n_samples=100)

    print("\n=== Sampling ShareGPT ===")
    sharegpt_samples = sample_sharegpt_from_generations(n_samples=50)

    print("\n=== Creating Annotation Files ===")
    create_annotation_csv(toucan_samples, sharegpt_samples, toucan_data)

    print("\n=== Done ===")
    print("Next steps:")
    print("1. Open human_evaluation/human_eval_samples.csv")
    print("2. Have 3 annotators rate each pair on 1-5 scale:")
    print("   1 = Completely different")
    print("   2 = Mostly different")
    print("   3 = Somewhat similar")
    print("   4 = Mostly similar")
    print("   5 = Identical/nearly identical")
    print("3. Compute inter-annotator agreement (Fleiss' kappa)")
    print("4. Correlate mean human ratings with STED scores")


if __name__ == "__main__":
    main()
