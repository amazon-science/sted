#!/usr/bin/env python3
"""
Bidirectional Intervention Study: Testing causal effects of "should"/"must" on consistency and accuracy.

This script tests whether adding/removing directive language causally affects:
1. Consistency (stability score across 10 runs)
2. Accuracy (STED similarity to ground truth)

Experimental Design:
- 100 samples total: 50 WITH directive (→ remove) + 50 WITHOUT directive (→ add)
- 3 temperatures: T=0.0, T=0.5, T=1.0
- 3 models: Claude-Sonnet-4, Qwen3-235B, Nova-2-Lite
- 10 runs per condition
- max_tokens = 800 (aligned with previous experiments)

Usage:
    python run_directive_intervention.py --dataset toucan --num-samples 100
"""

import argparse
import json
import os
import re
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.eval.generate_tool_calls import (
    get_bedrock_client,
    get_openai_client,
    generate_tool_calls_bedrock,
    generate_tool_calls_openai,
    load_toucan_dataset,
    get_provider,
    get_display_name,
)
from sted.model_config import MODEL_REGISTRY


# ============================================================================
# Prompt Rewriting Functions
# ============================================================================

DIRECTIVE_PATTERNS = [
    (r'\bYou should\b', 'should'),
    (r'\byou should\b', 'should'),
    (r'\bYou must\b', 'must'),
    (r'\byou must\b', 'must'),
    (r'\bshould\b', 'should'),
    (r'\bmust\b', 'must'),
]


def has_directive_language(text: str) -> bool:
    """Check if text contains directive language (should/must)."""
    text_lower = text.lower()
    return bool(re.search(r'\bshould\b', text_lower)) or bool(re.search(r'\bmust\b', text_lower))


def get_directive_type(text: str) -> Optional[str]:
    """Get the primary directive type in text."""
    text_lower = text.lower()
    if re.search(r'\bshould\b', text_lower):
        return 'should'
    if re.search(r'\bmust\b', text_lower):
        return 'must'
    return None


def add_directive_to_query(query: str, directive: str = "should") -> str:
    """
    Add directive language to a neutral query.

    Strategy: Intelligently prepend "You should" or "You must" based on query structure.
    """
    query = query.strip()

    # Don't modify if already has directive
    if has_directive_language(query):
        return query

    # Check if query starts with common patterns
    lower_query = query.lower()

    # If starts with "I want", "I need", "I would like" - rephrase
    if lower_query.startswith(('i want ', 'i need ', 'i would like ')):
        # Extract the action part
        for prefix in ['i want to ', 'i need to ', 'i would like to ', 'i want ', 'i need ', 'i would like ']:
            if lower_query.startswith(prefix):
                rest = query[len(prefix):]
                return f"You {directive} {rest}"

    # If starts with a question word, add at beginning with "First, you should..."
    if lower_query.startswith(('how ', 'what ', 'when ', 'where ', 'why ', 'which ', 'can ', 'could ')):
        return f"You {directive} help me with this: {query}"

    # If starts with imperative verb, prepend "You should"
    imperative_starters = ['get', 'find', 'search', 'create', 'make', 'check', 'look', 'help',
                          'tell', 'show', 'give', 'list', 'provide', 'calculate', 'convert']
    first_word = lower_query.split()[0] if query else ''
    if first_word in imperative_starters:
        return f"You {directive} {query[0].lower()}{query[1:]}"

    # Default: prepend "You should"
    if query[0].isupper():
        return f"You {directive} {query[0].lower()}{query[1:]}"
    return f"You {directive} {query}"


def remove_directive_from_query(query: str) -> str:
    """
    Remove directive language from a query while preserving meaning.

    Strategy: Remove "You should/must" patterns and rephrase naturally.
    """
    query = query.strip()

    # Don't modify if no directive
    if not has_directive_language(query):
        return query

    modified = query

    # Remove "You should/must" at the start
    patterns_to_remove = [
        (r'^You should\s+', ''),
        (r'^You must\s+', ''),
        (r'^you should\s+', ''),
        (r'^you must\s+', ''),
    ]

    for pattern, replacement in patterns_to_remove:
        modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)

    # Replace mid-sentence "should"/"must" with neutral alternatives
    modified = re.sub(r'\bshould\b', 'can', modified, flags=re.IGNORECASE)
    modified = re.sub(r'\bmust\b', 'can', modified, flags=re.IGNORECASE)

    # Capitalize first letter if needed
    if modified and modified[0].islower():
        modified = modified[0].upper() + modified[1:]

    return modified


def rewrite_query(query: str, intervention_type: str) -> Tuple[str, str]:
    """
    Rewrite query based on intervention type.

    Args:
        query: Original query
        intervention_type: "add_directive" or "remove_directive"

    Returns:
        Tuple of (rewritten_query, directive_word_used)
    """
    if intervention_type == "add_directive":
        directive = random.choice(['should', 'must'])
        rewritten = add_directive_to_query(query, directive)
        return rewritten, directive
    else:  # remove_directive
        directive = get_directive_type(query) or 'should'
        rewritten = remove_directive_from_query(query)
        return rewritten, directive


# ============================================================================
# Consistency and Accuracy Calculation
# ============================================================================

def calculate_consistency(runs: List[List[Dict]], alpha: float = 20) -> float:
    """
    Calculate consistency (stability score) across multiple runs.

    Uses simplified pairwise similarity based on tool call matching.
    """
    # Filter valid runs
    valid_runs = [r for r in runs if r is not None and len(r) > 0]

    if len(valid_runs) < 2:
        return 1.0 if len(valid_runs) == 1 else 0.0

    # Calculate pairwise similarities
    similarities = []
    n = len(valid_runs)

    for i in range(n):
        for j in range(i + 1, n):
            sim = calculate_run_similarity(valid_runs[i], valid_runs[j])
            similarities.append(sim)

    if not similarities:
        return 0.0

    # Calculate stability score: S_alpha = exp(-alpha * sigma_d^2)
    distances = [1 - s for s in similarities]
    sigma_d = np.std(distances) if len(distances) > 1 else 0.0
    stability_score = np.exp(-alpha * sigma_d ** 2)

    # Also compute c_mean for reference
    c_mean = np.mean(similarities)

    return c_mean  # Return c_mean as the consistency metric


def calculate_run_similarity(run1: List[Dict], run2: List[Dict]) -> float:
    """Calculate similarity between two runs based on tool calls."""
    if not run1 and not run2:
        return 1.0
    if not run1 or not run2:
        return 0.0

    # Exact match
    if json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True):
        return 1.0

    # Tool name Jaccard similarity
    names1 = set(tc.get('name', '') for tc in run1 if isinstance(tc, dict))
    names2 = set(tc.get('name', '') for tc in run2 if isinstance(tc, dict))

    if not names1 and not names2:
        return 1.0
    if not names1 or not names2:
        return 0.0

    jaccard = len(names1 & names2) / len(names1 | names2)

    # Argument similarity (simplified)
    args_sim = 0.0
    common_names = names1 & names2
    if common_names:
        for name in common_names:
            args1 = [tc.get('arguments', {}) for tc in run1 if tc.get('name') == name]
            args2 = [tc.get('arguments', {}) for tc in run2 if tc.get('name') == name]
            if args1 and args2:
                if json.dumps(args1[0], sort_keys=True) == json.dumps(args2[0], sort_keys=True):
                    args_sim += 1.0
        args_sim /= len(common_names)

    # Weighted combination
    return 0.6 * jaccard + 0.4 * args_sim


def calculate_accuracy(runs: List[List[Dict]], ground_truth: List[Dict]) -> float:
    """Calculate accuracy as max similarity to ground truth across runs."""
    if not ground_truth:
        return 0.0

    valid_runs = [r for r in runs if r is not None and len(r) > 0]
    if not valid_runs:
        return 0.0

    # Calculate similarity of each run to ground truth
    accuracies = []
    gt_names = set(tc.get('name', '') for tc in ground_truth if isinstance(tc, dict))

    for run in valid_runs:
        run_names = set(tc.get('name', '') for tc in run if isinstance(tc, dict))

        if not gt_names:
            accuracies.append(0.0)
            continue

        # Tool name recall (what fraction of GT tools were called)
        recall = len(gt_names & run_names) / len(gt_names) if gt_names else 0.0

        # Precision (what fraction of called tools are in GT)
        precision = len(gt_names & run_names) / len(run_names) if run_names else 0.0

        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracies.append(f1)

    return max(accuracies) if accuracies else 0.0


# ============================================================================
# Main Experiment
# ============================================================================

@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""
    sample_id: str
    original_query: str
    rewritten_query: str
    intervention_type: str
    directive_word: str
    temperature: float
    model: str

    # Original prompt metrics
    original_consistency: float
    original_accuracy: float
    original_num_valid_runs: int

    # Rewritten prompt metrics
    rewritten_consistency: float
    rewritten_accuracy: float
    rewritten_num_valid_runs: int

    # Deltas
    delta_consistency: float
    delta_accuracy: float


def select_samples_for_intervention(
    dataset: List[Dict],
    num_with_directive: int = 50,
    num_without_directive: int = 50,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Select samples for intervention study."""
    random.seed(seed)

    with_directive = []
    without_directive = []

    for sample in dataset:
        query = sample.get('query', '')
        if has_directive_language(query):
            with_directive.append(sample)
        else:
            without_directive.append(sample)

    print(f"Found {len(with_directive)} samples WITH directive language")
    print(f"Found {len(without_directive)} samples WITHOUT directive language")

    # Randomly select
    selected_with = random.sample(with_directive, min(num_with_directive, len(with_directive)))
    selected_without = random.sample(without_directive, min(num_without_directive, len(without_directive)))

    return selected_with, selected_without


def run_intervention_experiment(
    client,
    provider: str,
    model_id: str,
    sample: Dict,
    intervention_type: str,
    temperature: float,
    num_runs: int = 10,
    max_tokens: int = 800,
    max_workers: int = 5,
) -> Optional[InterventionResult]:
    """Run intervention experiment for a single sample."""

    original_query = sample.get('query', '')
    tools = sample.get('tools', [])
    ground_truth = sample.get('answers', [])
    sample_id = sample.get('id', 'unknown')

    # Rewrite query
    rewritten_query, directive_word = rewrite_query(original_query, intervention_type)

    # Skip if rewriting didn't change anything meaningful
    if original_query.strip().lower() == rewritten_query.strip().lower():
        return None

    # Select generation function
    if provider == "bedrock":
        generate_fn = generate_tool_calls_bedrock
    else:
        generate_fn = generate_tool_calls_openai

    # Generate with original query
    try:
        original_runs = generate_fn(
            client=client,
            model_id=model_id,
            query=original_query,
            tools=tools,
            num_runs=num_runs,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers,
        )
    except Exception as e:
        print(f"  Original query failed: {e}")
        original_runs = [[] for _ in range(num_runs)]

    # Generate with rewritten query
    try:
        rewritten_runs = generate_fn(
            client=client,
            model_id=model_id,
            query=rewritten_query,
            tools=tools,
            num_runs=num_runs,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers,
        )
    except Exception as e:
        print(f"  Rewritten query failed: {e}")
        rewritten_runs = [[] for _ in range(num_runs)]

    # Calculate metrics
    original_consistency = calculate_consistency(original_runs)
    rewritten_consistency = calculate_consistency(rewritten_runs)

    original_accuracy = calculate_accuracy(original_runs, ground_truth)
    rewritten_accuracy = calculate_accuracy(rewritten_runs, ground_truth)

    original_valid = sum(1 for r in original_runs if r)
    rewritten_valid = sum(1 for r in rewritten_runs if r)

    return InterventionResult(
        sample_id=sample_id,
        original_query=original_query,
        rewritten_query=rewritten_query,
        intervention_type=intervention_type,
        directive_word=directive_word,
        temperature=temperature,
        model=get_display_name(model_id),
        original_consistency=original_consistency,
        original_accuracy=original_accuracy,
        original_num_valid_runs=original_valid,
        rewritten_consistency=rewritten_consistency,
        rewritten_accuracy=rewritten_accuracy,
        rewritten_num_valid_runs=rewritten_valid,
        delta_consistency=rewritten_consistency - original_consistency,
        delta_accuracy=rewritten_accuracy - original_accuracy,
    )


def statistical_analysis(results: List[InterventionResult]) -> Dict[str, Any]:
    """Perform statistical analysis on intervention results."""
    if not results:
        return {}

    # Separate by intervention type
    add_results = [r for r in results if r.intervention_type == "add_directive"]
    remove_results = [r for r in results if r.intervention_type == "remove_directive"]

    analysis = {"overall": {}, "by_intervention": {}, "by_temperature": {}, "by_model": {}}

    # Overall analysis
    all_delta_cons = [r.delta_consistency for r in results]
    all_delta_acc = [r.delta_accuracy for r in results]

    if all_delta_cons:
        t_cons, p_cons = stats.ttest_1samp(all_delta_cons, 0)
        t_acc, p_acc = stats.ttest_1samp(all_delta_acc, 0)

        analysis["overall"] = {
            "n_samples": len(results),
            "consistency": {
                "mean_delta": float(np.mean(all_delta_cons)),
                "std_delta": float(np.std(all_delta_cons)),
                "t_statistic": float(t_cons),
                "p_value": float(p_cons),
                "cohens_d": float(np.mean(all_delta_cons) / np.std(all_delta_cons)) if np.std(all_delta_cons) > 0 else 0,
            },
            "accuracy": {
                "mean_delta": float(np.mean(all_delta_acc)),
                "std_delta": float(np.std(all_delta_acc)),
                "t_statistic": float(t_acc),
                "p_value": float(p_acc),
                "cohens_d": float(np.mean(all_delta_acc) / np.std(all_delta_acc)) if np.std(all_delta_acc) > 0 else 0,
            }
        }

    # By intervention type
    for int_type, int_results in [("add_directive", add_results), ("remove_directive", remove_results)]:
        if not int_results:
            continue

        delta_cons = [r.delta_consistency for r in int_results]
        delta_acc = [r.delta_accuracy for r in int_results]

        t_cons, p_cons = stats.ttest_1samp(delta_cons, 0) if len(delta_cons) > 1 else (0, 1)
        t_acc, p_acc = stats.ttest_1samp(delta_acc, 0) if len(delta_acc) > 1 else (0, 1)

        analysis["by_intervention"][int_type] = {
            "n_samples": len(int_results),
            "consistency": {
                "original_mean": float(np.mean([r.original_consistency for r in int_results])),
                "rewritten_mean": float(np.mean([r.rewritten_consistency for r in int_results])),
                "mean_delta": float(np.mean(delta_cons)),
                "std_delta": float(np.std(delta_cons)),
                "t_statistic": float(t_cons),
                "p_value": float(p_cons),
                "improved": sum(1 for d in delta_cons if d > 0),
                "worsened": sum(1 for d in delta_cons if d < 0),
            },
            "accuracy": {
                "original_mean": float(np.mean([r.original_accuracy for r in int_results])),
                "rewritten_mean": float(np.mean([r.rewritten_accuracy for r in int_results])),
                "mean_delta": float(np.mean(delta_acc)),
                "std_delta": float(np.std(delta_acc)),
                "t_statistic": float(t_acc),
                "p_value": float(p_acc),
                "improved": sum(1 for d in delta_acc if d > 0),
                "worsened": sum(1 for d in delta_acc if d < 0),
            }
        }

    # By temperature
    for temp in sorted(set(r.temperature for r in results)):
        temp_results = [r for r in results if r.temperature == temp]
        if temp_results:
            analysis["by_temperature"][str(temp)] = {
                "n_samples": len(temp_results),
                "mean_delta_consistency": float(np.mean([r.delta_consistency for r in temp_results])),
                "mean_delta_accuracy": float(np.mean([r.delta_accuracy for r in temp_results])),
            }

    # By model
    for model in sorted(set(r.model for r in results)):
        model_results = [r for r in results if r.model == model]
        if model_results:
            analysis["by_model"][model] = {
                "n_samples": len(model_results),
                "mean_delta_consistency": float(np.mean([r.delta_consistency for r in model_results])),
                "mean_delta_accuracy": float(np.mean([r.delta_accuracy for r in model_results])),
            }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Bidirectional Intervention Study")
    parser.add_argument("--dataset-path", type=str, default="toucan_data/toucan_tool_calls_1006.json",
                        help="Path to Toucan dataset")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Total samples (split evenly: add + remove)")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of runs per condition")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Max tokens (aligned with previous experiments)")
    parser.add_argument("--output-dir", type=str, default="results/directive_intervention",
                        help="Output directory")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Max parallel workers per inference batch")

    args = parser.parse_args()

    # Configuration: 4 representative models from different families
    MODELS = [
        "us.anthropic.claude-sonnet-4-20250514-v1:0",  # Anthropic
        "us.amazon.nova-2-lite-v1:0",                   # Amazon
        "us.meta.llama3-3-70b-instruct-v1:0",          # Meta
        "qwen.qwen3-235b-a22b-2507-v1:0",              # Qwen (Alibaba)
    ]
    TEMPERATURES = [0.0, 0.5, 1.0]  # Low, medium, high variance

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Bidirectional Intervention Study")
    print("=" * 70)
    print(f"Models: {len(MODELS)}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Samples: {args.num_samples} ({args.num_samples // 2} add + {args.num_samples // 2} remove)")
    print(f"Runs per condition: {args.num_runs}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load dataset
    print("\nLoading Toucan dataset...")
    dataset = load_toucan_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Select samples
    samples_with_directive, samples_without_directive = select_samples_for_intervention(
        dataset,
        num_with_directive=args.num_samples // 2,
        num_without_directive=args.num_samples // 2
    )

    print(f"\nSelected {len(samples_with_directive)} samples for REMOVE intervention")
    print(f"Selected {len(samples_without_directive)} samples for ADD intervention")

    # Initialize clients
    print("\nInitializing clients...")
    bedrock_client = get_bedrock_client()

    all_results = []

    # Run experiments
    for model_id in MODELS:
        provider = get_provider(model_id)
        display_name = get_display_name(model_id)

        if provider == "bedrock":
            client = bedrock_client
        else:
            try:
                client = get_openai_client()
            except Exception as e:
                print(f"Skipping {display_name}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Model: {display_name}")
        print(f"{'='*60}")

        for temperature in TEMPERATURES:
            print(f"\n--- Temperature: {temperature} ---")

            # REMOVE directive experiments
            print(f"\nRemove Directive ({len(samples_with_directive)} samples):")
            for i, sample in enumerate(tqdm(samples_with_directive, desc="Remove")):
                result = run_intervention_experiment(
                    client=client,
                    provider=provider,
                    model_id=model_id,
                    sample=sample,
                    intervention_type="remove_directive",
                    temperature=temperature,
                    num_runs=args.num_runs,
                    max_tokens=args.max_tokens,
                    max_workers=args.max_workers,
                )
                if result:
                    all_results.append(result)

            # ADD directive experiments
            print(f"\nAdd Directive ({len(samples_without_directive)} samples):")
            for i, sample in enumerate(tqdm(samples_without_directive, desc="Add")):
                result = run_intervention_experiment(
                    client=client,
                    provider=provider,
                    model_id=model_id,
                    sample=sample,
                    intervention_type="add_directive",
                    temperature=temperature,
                    num_runs=args.num_runs,
                    max_tokens=args.max_tokens,
                    max_workers=args.max_workers,
                )
                if result:
                    all_results.append(result)

            # Save intermediate results
            intermediate_file = output_dir / "intermediate_results.json"
            with open(intermediate_file, 'w') as f:
                json.dump([asdict(r) for r in all_results], f, indent=2)

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    analysis = statistical_analysis(all_results)

    # Print overall results
    if "overall" in analysis and analysis["overall"]:
        overall = analysis["overall"]
        print(f"\nOVERALL (n={overall['n_samples']})")
        print("-" * 40)

        cons = overall.get("consistency", {})
        print(f"CONSISTENCY:")
        print(f"  Mean Δ: {cons.get('mean_delta', 0):+.4f} (std: {cons.get('std_delta', 0):.4f})")
        print(f"  t-stat: {cons.get('t_statistic', 0):.3f}, p-value: {cons.get('p_value', 1):.4f}")
        print(f"  Cohen's d: {cons.get('cohens_d', 0):.3f}")

        acc = overall.get("accuracy", {})
        print(f"ACCURACY:")
        print(f"  Mean Δ: {acc.get('mean_delta', 0):+.4f} (std: {acc.get('std_delta', 0):.4f})")
        print(f"  t-stat: {acc.get('t_statistic', 0):.3f}, p-value: {acc.get('p_value', 1):.4f}")
        print(f"  Cohen's d: {acc.get('cohens_d', 0):.3f}")

    # Print by intervention type
    for int_type in ["add_directive", "remove_directive"]:
        if int_type in analysis.get("by_intervention", {}):
            stats_data = analysis["by_intervention"][int_type]
            print(f"\n{int_type.upper()} (n={stats_data['n_samples']})")
            print("-" * 40)

            cons = stats_data["consistency"]
            print(f"CONSISTENCY: {cons['original_mean']:.4f} → {cons['rewritten_mean']:.4f}")
            print(f"  Δ: {cons['mean_delta']:+.4f}, p={cons['p_value']:.4f}")
            print(f"  Improved: {cons['improved']}, Worsened: {cons['worsened']}")

            acc = stats_data["accuracy"]
            print(f"ACCURACY: {acc['original_mean']:.4f} → {acc['rewritten_mean']:.4f}")
            print(f"  Δ: {acc['mean_delta']:+.4f}, p={acc['p_value']:.4f}")
            print(f"  Improved: {acc['improved']}, Worsened: {acc['worsened']}")

    # Save final results
    final_output = {
        "metadata": {
            "models": [get_display_name(m) for m in MODELS],
            "temperatures": TEMPERATURES,
            "num_runs": args.num_runs,
            "max_tokens": args.max_tokens,
            "total_samples": len(all_results),
            "timestamp": timestamp,
        },
        "results": [asdict(r) for r in all_results],
        "analysis": analysis,
    }

    results_file = output_dir / "final_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
