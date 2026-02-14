#!/usr/bin/env python3
"""
Feature Importance Validation: Causal intervention for each feature.

Tests whether modifying specific prompt features causally affects consistency and accuracy.

Features to test (from paper's Feature Importance Validation):
1. word_count (0.673) - Shorten prompts
2. has_should (0.071) - Add/remove "should"
3. has_if (0.070) - Add/remove conditionals
4. has_can_you (0.065) - Add/remove polite phrasing
5. has_must (0.040) - Add/remove "must"
6. has_numbered_list (0.016) - Add/remove numbered lists

Usage:
    python run_feature_intervention.py --feature word_count --num-samples 50
    python run_feature_intervention.py --feature all --num-samples 50
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
    generate_tool_calls_bedrock,
    load_toucan_dataset,
    get_provider,
    get_display_name,
)
from sted.model_config import MODEL_REGISTRY


# ============================================================================
# Feature Detection Functions
# ============================================================================

def has_should(text: str) -> bool:
    return bool(re.search(r'\bshould\b', text, re.IGNORECASE))

def has_must(text: str) -> bool:
    return bool(re.search(r'\bmust\b', text, re.IGNORECASE))

def has_if(text: str) -> bool:
    return bool(re.search(r'\bif\b', text, re.IGNORECASE))

def has_can_you(text: str) -> bool:
    return bool(re.search(r'\bcan you\b', text, re.IGNORECASE))

def has_numbered_list(text: str) -> bool:
    return bool(re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))

def get_word_count(text: str) -> int:
    return len(text.split())


# ============================================================================
# Feature Rewriting Functions
# ============================================================================

def add_should(query: str) -> str:
    """Add 'should' to a query."""
    if has_should(query):
        return query

    lower = query.lower()
    # If starts with imperative, prepend "You should"
    imperative_starters = ['get', 'find', 'search', 'create', 'make', 'check', 'look', 'help',
                          'tell', 'show', 'give', 'list', 'provide', 'calculate', 'convert',
                          'use', 'set', 'update', 'delete', 'add', 'remove']
    first_word = lower.split()[0] if query else ''

    if first_word in imperative_starters:
        return f"You should {query[0].lower()}{query[1:]}"

    # If starts with "I want/need", rephrase
    for prefix in ['i want to ', 'i need to ', 'i would like to ']:
        if lower.startswith(prefix):
            rest = query[len(prefix):]
            return f"You should {rest}"

    # Default: prepend
    if query[0].isupper():
        return f"You should {query[0].lower()}{query[1:]}"
    return f"You should {query}"


def remove_should(query: str) -> str:
    """Remove 'should' from a query."""
    if not has_should(query):
        return query

    modified = query
    # Remove "You should" at start
    modified = re.sub(r'^You should\s+', '', modified, flags=re.IGNORECASE)
    # Replace remaining "should" with neutral alternatives
    modified = re.sub(r'\bshould\b', 'can', modified, flags=re.IGNORECASE)

    # Capitalize first letter
    if modified and modified[0].islower():
        modified = modified[0].upper() + modified[1:]
    return modified


def add_must(query: str) -> str:
    """Add 'must' to a query."""
    if has_must(query):
        return query

    lower = query.lower()
    imperative_starters = ['get', 'find', 'search', 'create', 'make', 'check', 'look', 'help',
                          'tell', 'show', 'give', 'list', 'provide', 'calculate', 'convert']
    first_word = lower.split()[0] if query else ''

    if first_word in imperative_starters:
        return f"You must {query[0].lower()}{query[1:]}"

    if query[0].isupper():
        return f"You must {query[0].lower()}{query[1:]}"
    return f"You must {query}"


def remove_must(query: str) -> str:
    """Remove 'must' from a query."""
    if not has_must(query):
        return query

    modified = query
    modified = re.sub(r'^You must\s+', '', modified, flags=re.IGNORECASE)
    modified = re.sub(r'\bmust\b', 'can', modified, flags=re.IGNORECASE)

    if modified and modified[0].islower():
        modified = modified[0].upper() + modified[1:]
    return modified


def add_if(query: str) -> str:
    """Add conditional 'if' clause to a query."""
    if has_if(query):
        return query

    # Add a conditional clause at the end
    conditionals = [
        "if possible",
        "if available",
        "if it exists",
        "if you can",
    ]
    return f"{query.rstrip('.')} {random.choice(conditionals)}."


def remove_if(query: str) -> str:
    """Remove conditional 'if' clauses from a query."""
    if not has_if(query):
        return query

    modified = query
    # Remove common conditional phrases
    patterns = [
        r',?\s*if possible\.?',
        r',?\s*if available\.?',
        r',?\s*if it exists\.?',
        r',?\s*if you can\.?',
        r',?\s*if needed\.?',
        r',?\s*if necessary\.?',
    ]
    for pattern in patterns:
        modified = re.sub(pattern, '', modified, flags=re.IGNORECASE)

    # Remove standalone "if" clauses more carefully
    # This is tricky - we don't want to break the sentence
    modified = re.sub(r'\s+if\s+\w+\s+is\s+\w+', '', modified, flags=re.IGNORECASE)

    return modified.strip()


def add_can_you(query: str) -> str:
    """Add polite 'can you' phrasing."""
    if has_can_you(query):
        return query

    lower = query.lower()

    # If already a question or polite form, modify appropriately
    if lower.startswith(('please ', 'could you ', 'would you ')):
        return re.sub(r'^(please|could you|would you)\s+', 'Can you ', query, flags=re.IGNORECASE)

    # If imperative, convert to polite request
    imperative_starters = ['get', 'find', 'search', 'create', 'make', 'check', 'look', 'help',
                          'tell', 'show', 'give', 'list', 'provide', 'calculate', 'convert']
    first_word = lower.split()[0] if query else ''

    if first_word in imperative_starters:
        return f"Can you {query[0].lower()}{query[1:]}"

    # Default
    if query[0].isupper():
        return f"Can you {query[0].lower()}{query[1:]}"
    return f"Can you {query}"


def remove_can_you(query: str) -> str:
    """Remove polite 'can you' phrasing."""
    if not has_can_you(query):
        return query

    modified = re.sub(r'^can you\s+', '', query, flags=re.IGNORECASE)

    if modified and modified[0].islower():
        modified = modified[0].upper() + modified[1:]
    return modified


def add_numbered_list(query: str) -> str:
    """Convert query to use numbered list format."""
    if has_numbered_list(query):
        return query

    # If query has multiple parts (commas, "and"), convert to numbered list
    # Split on common conjunctions
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', query)

    if len(parts) >= 2:
        numbered = []
        for i, part in enumerate(parts, 1):
            part = part.strip()
            if part:
                numbered.append(f"{i}. {part}")
        return "\n".join(numbered)

    # Single item - make it a single-item list
    return f"1. {query}"


def remove_numbered_list(query: str) -> str:
    """Remove numbered list formatting."""
    if not has_numbered_list(query):
        return query

    # Extract list items and join with commas
    items = re.findall(r'^\s*\d+[\.\)]\s*(.+)$', query, re.MULTILINE)

    if items:
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"

    return query


def shorten_query(query: str) -> str:
    """Shorten a query by removing redundant words and phrases."""
    modified = query

    # Remove filler phrases
    fillers = [
        r'\bplease\b',
        r'\bkindly\b',
        r'\bI would like you to\b',
        r'\bI want you to\b',
        r'\bI need you to\b',
        r'\bcould you please\b',
        r'\bwould you please\b',
        r'\bif possible\b',
        r'\bif you can\b',
        r'\bfor me\b',
        r'\bthat I need\b',
        r'\bthat I want\b',
    ]

    for filler in fillers:
        modified = re.sub(filler, '', modified, flags=re.IGNORECASE)

    # Clean up extra spaces
    modified = re.sub(r'\s+', ' ', modified).strip()

    # Remove trailing punctuation repetition
    modified = re.sub(r'[.!?]+$', '.', modified)

    return modified


def lengthen_query(query: str) -> str:
    """Lengthen a query by adding polite phrases and context."""
    additions = [
        "Please ",
        "I would like you to ",
        "Could you kindly ",
    ]

    suffixes = [
        " for me",
        " if possible",
        ", please",
    ]

    modified = random.choice(additions) + query[0].lower() + query[1:]
    modified = modified.rstrip('.') + random.choice(suffixes) + "."

    return modified


# ============================================================================
# Feature Intervention Mapping
# ============================================================================

FEATURE_CONFIG = {
    "word_count": {
        "detect_has": lambda q: get_word_count(q) > 15,  # "long" query
        "detect_lacks": lambda q: get_word_count(q) <= 15,  # "short" query
        "add": lengthen_query,
        "remove": shorten_query,
        "description": "Query length (word count)",
    },
    "has_should": {
        "detect_has": has_should,
        "detect_lacks": lambda q: not has_should(q),
        "add": add_should,
        "remove": remove_should,
        "description": "Directive 'should'",
    },
    "has_must": {
        "detect_has": has_must,
        "detect_lacks": lambda q: not has_must(q),
        "add": add_must,
        "remove": remove_must,
        "description": "Directive 'must'",
    },
    "has_if": {
        "detect_has": has_if,
        "detect_lacks": lambda q: not has_if(q),
        "add": add_if,
        "remove": remove_if,
        "description": "Conditional 'if'",
    },
    "has_can_you": {
        "detect_has": has_can_you,
        "detect_lacks": lambda q: not has_can_you(q),
        "add": add_can_you,
        "remove": remove_can_you,
        "description": "Polite 'can you'",
    },
    "has_numbered_list": {
        "detect_has": has_numbered_list,
        "detect_lacks": lambda q: not has_numbered_list(q),
        "add": add_numbered_list,
        "remove": remove_numbered_list,
        "description": "Numbered list format",
    },
}


# ============================================================================
# Consistency and Accuracy Calculation
# ============================================================================

def calculate_consistency(runs: List[List[Dict]]) -> float:
    """Calculate mean pairwise similarity across runs."""
    valid_runs = [r for r in runs if r is not None and len(r) > 0]

    if len(valid_runs) < 2:
        return 1.0 if len(valid_runs) == 1 else 0.0

    similarities = []
    n = len(valid_runs)

    for i in range(n):
        for j in range(i + 1, n):
            sim = calculate_run_similarity(valid_runs[i], valid_runs[j])
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def calculate_run_similarity(run1: List[Dict], run2: List[Dict]) -> float:
    """Calculate similarity between two runs."""
    if not run1 and not run2:
        return 1.0
    if not run1 or not run2:
        return 0.0

    if json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True):
        return 1.0

    names1 = set(tc.get('name', '') for tc in run1 if isinstance(tc, dict))
    names2 = set(tc.get('name', '') for tc in run2 if isinstance(tc, dict))

    if not names1 and not names2:
        return 1.0
    if not names1 or not names2:
        return 0.0

    jaccard = len(names1 & names2) / len(names1 | names2)

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

    return 0.6 * jaccard + 0.4 * args_sim


def calculate_accuracy(runs: List[List[Dict]], ground_truth: List[Dict]) -> float:
    """Calculate accuracy as max F1 to ground truth across runs."""
    if not ground_truth:
        return 0.0

    valid_runs = [r for r in runs if r is not None and len(r) > 0]
    if not valid_runs:
        return 0.0

    gt_names = set(tc.get('name', '') for tc in ground_truth if isinstance(tc, dict))

    accuracies = []
    for run in valid_runs:
        run_names = set(tc.get('name', '') for tc in run if isinstance(tc, dict))

        if not gt_names:
            accuracies.append(0.0)
            continue

        recall = len(gt_names & run_names) / len(gt_names) if gt_names else 0.0
        precision = len(gt_names & run_names) / len(run_names) if run_names else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracies.append(f1)

    return max(accuracies) if accuracies else 0.0


# ============================================================================
# Main Experiment
# ============================================================================

@dataclass
class FeatureInterventionResult:
    """Result of a single feature intervention."""
    sample_id: str
    feature: str
    intervention_type: str  # "add" or "remove"
    original_query: str
    rewritten_query: str
    temperature: float
    model: str

    original_consistency: float
    original_accuracy: float
    rewritten_consistency: float
    rewritten_accuracy: float

    delta_consistency: float
    delta_accuracy: float


def run_feature_experiment(
    client,
    model_id: str,
    sample: Dict,
    feature: str,
    intervention_type: str,
    temperature: float,
    num_runs: int = 10,
    max_tokens: int = 8000,
    max_workers: int = 5,
) -> Optional[FeatureInterventionResult]:
    """Run intervention experiment for a single feature."""

    config = FEATURE_CONFIG[feature]
    original_query = sample.get('query', '')
    tools = sample.get('tools', [])
    ground_truth = sample.get('answers', [])
    sample_id = sample.get('id', 'unknown')

    # Rewrite query
    if intervention_type == "add":
        rewritten_query = config["add"](original_query)
    else:
        rewritten_query = config["remove"](original_query)

    # Skip if no meaningful change
    if original_query.strip().lower() == rewritten_query.strip().lower():
        return None

    # Generate with original query
    try:
        original_runs = generate_tool_calls_bedrock(
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
        print(f"  Original failed: {e}")
        original_runs = [[] for _ in range(num_runs)]

    # Generate with rewritten query
    try:
        rewritten_runs = generate_tool_calls_bedrock(
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
        print(f"  Rewritten failed: {e}")
        rewritten_runs = [[] for _ in range(num_runs)]

    # Calculate metrics
    original_consistency = calculate_consistency(original_runs)
    rewritten_consistency = calculate_consistency(rewritten_runs)
    original_accuracy = calculate_accuracy(original_runs, ground_truth)
    rewritten_accuracy = calculate_accuracy(rewritten_runs, ground_truth)

    return FeatureInterventionResult(
        sample_id=sample_id,
        feature=feature,
        intervention_type=intervention_type,
        original_query=original_query,
        rewritten_query=rewritten_query,
        temperature=temperature,
        model=get_display_name(model_id),
        original_consistency=original_consistency,
        original_accuracy=original_accuracy,
        rewritten_consistency=rewritten_consistency,
        rewritten_accuracy=rewritten_accuracy,
        delta_consistency=rewritten_consistency - original_consistency,
        delta_accuracy=rewritten_accuracy - original_accuracy,
    )


def select_samples_for_feature(
    dataset: List[Dict],
    feature: str,
    num_add: int = 25,
    num_remove: int = 25,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Select samples for feature intervention."""
    random.seed(seed)

    config = FEATURE_CONFIG[feature]

    has_feature = [s for s in dataset if config["detect_has"](s.get('query', ''))]
    lacks_feature = [s for s in dataset if config["detect_lacks"](s.get('query', ''))]

    print(f"  Found {len(has_feature)} samples WITH {feature}")
    print(f"  Found {len(lacks_feature)} samples WITHOUT {feature}")

    selected_add = random.sample(lacks_feature, min(num_add, len(lacks_feature)))
    selected_remove = random.sample(has_feature, min(num_remove, len(has_feature)))

    return selected_add, selected_remove


def statistical_analysis(results: List[FeatureInterventionResult]) -> Dict:
    """Perform statistical analysis on results."""
    if not results:
        return {}

    analysis = {"overall": {}, "by_intervention": {}, "by_feature": {}}

    # Overall
    delta_cons = [r.delta_consistency for r in results]
    delta_acc = [r.delta_accuracy for r in results]

    if len(delta_cons) > 1:
        t_cons, p_cons = stats.ttest_1samp(delta_cons, 0)
        t_acc, p_acc = stats.ttest_1samp(delta_acc, 0)

        analysis["overall"] = {
            "n_samples": len(results),
            "consistency": {
                "mean_delta": float(np.mean(delta_cons)),
                "std_delta": float(np.std(delta_cons)),
                "t_statistic": float(t_cons),
                "p_value": float(p_cons),
                "cohens_d": float(np.mean(delta_cons) / np.std(delta_cons)) if np.std(delta_cons) > 0 else 0,
            },
            "accuracy": {
                "mean_delta": float(np.mean(delta_acc)),
                "std_delta": float(np.std(delta_acc)),
                "t_statistic": float(t_acc),
                "p_value": float(p_acc),
                "cohens_d": float(np.mean(delta_acc) / np.std(delta_acc)) if np.std(delta_acc) > 0 else 0,
            }
        }

    # By intervention type
    for int_type in ["add", "remove"]:
        int_results = [r for r in results if r.intervention_type == int_type]
        if len(int_results) > 1:
            d_cons = [r.delta_consistency for r in int_results]
            d_acc = [r.delta_accuracy for r in int_results]
            t_c, p_c = stats.ttest_1samp(d_cons, 0)
            t_a, p_a = stats.ttest_1samp(d_acc, 0)

            analysis["by_intervention"][int_type] = {
                "n_samples": len(int_results),
                "consistency": {
                    "original_mean": float(np.mean([r.original_consistency for r in int_results])),
                    "rewritten_mean": float(np.mean([r.rewritten_consistency for r in int_results])),
                    "mean_delta": float(np.mean(d_cons)),
                    "p_value": float(p_c),
                },
                "accuracy": {
                    "original_mean": float(np.mean([r.original_accuracy for r in int_results])),
                    "rewritten_mean": float(np.mean([r.rewritten_accuracy for r in int_results])),
                    "mean_delta": float(np.mean(d_acc)),
                    "p_value": float(p_a),
                }
            }

    # By feature
    for feature in set(r.feature for r in results):
        feat_results = [r for r in results if r.feature == feature]
        if len(feat_results) > 1:
            d_cons = [r.delta_consistency for r in feat_results]
            d_acc = [r.delta_accuracy for r in feat_results]
            t_c, p_c = stats.ttest_1samp(d_cons, 0)
            t_a, p_a = stats.ttest_1samp(d_acc, 0)

            analysis["by_feature"][feature] = {
                "n_samples": len(feat_results),
                "mean_delta_consistency": float(np.mean(d_cons)),
                "mean_delta_accuracy": float(np.mean(d_acc)),
                "p_consistency": float(p_c),
                "p_accuracy": float(p_a),
            }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Feature Importance Causal Validation")
    parser.add_argument("--dataset-path", type=str, default="toucan_data/toucan_tool_calls_1006.json")
    parser.add_argument("--feature", type=str, default="all",
                        choices=["all"] + list(FEATURE_CONFIG.keys()),
                        help="Feature to test (or 'all')")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Samples per feature (split: add + remove)")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--temperatures", type=str, default="0.0,0.5,1.0",
                        help="Comma-separated temperatures")
    parser.add_argument("--model", type=str, default="us.anthropic.claude-sonnet-4-20250514-v1:0")
    parser.add_argument("--output-dir", type=str, default="results/feature_intervention")
    parser.add_argument("--max-workers", type=int, default=5)

    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]
    features = list(FEATURE_CONFIG.keys()) if args.feature == "all" else [args.feature]

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Feature Importance Causal Validation")
    print("=" * 70)
    print(f"Features: {features}")
    print(f"Model: {get_display_name(args.model)}")
    print(f"Temperatures: {temperatures}")
    print(f"Samples per feature: {args.num_samples}")
    print(f"Runs per condition: {args.num_runs}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_toucan_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Initialize client
    client = get_bedrock_client()

    all_results = []

    for feature in features:
        print(f"\n{'='*60}")
        print(f"Feature: {feature} - {FEATURE_CONFIG[feature]['description']}")
        print(f"{'='*60}")

        # Select samples
        samples_add, samples_remove = select_samples_for_feature(
            dataset, feature,
            num_add=args.num_samples // 2,
            num_remove=args.num_samples // 2
        )

        for temperature in temperatures:
            print(f"\n--- Temperature: {temperature} ---")

            # ADD feature
            if samples_add:
                print(f"\nAdding {feature} ({len(samples_add)} samples):")
                for sample in tqdm(samples_add, desc="Add"):
                    result = run_feature_experiment(
                        client=client,
                        model_id=args.model,
                        sample=sample,
                        feature=feature,
                        intervention_type="add",
                        temperature=temperature,
                        num_runs=args.num_runs,
                        max_tokens=args.max_tokens,
                        max_workers=args.max_workers,
                    )
                    if result:
                        all_results.append(result)

            # REMOVE feature
            if samples_remove:
                print(f"\nRemoving {feature} ({len(samples_remove)} samples):")
                for sample in tqdm(samples_remove, desc="Remove"):
                    result = run_feature_experiment(
                        client=client,
                        model_id=args.model,
                        sample=sample,
                        feature=feature,
                        intervention_type="remove",
                        temperature=temperature,
                        num_runs=args.num_runs,
                        max_tokens=args.max_tokens,
                        max_workers=args.max_workers,
                    )
                    if result:
                        all_results.append(result)

            # Save intermediate
            intermediate_file = output_dir / "intermediate_results.json"
            with open(intermediate_file, 'w') as f:
                json.dump([asdict(r) for r in all_results], f, indent=2)

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    analysis = statistical_analysis(all_results)

    # Print by feature
    print("\nRESULTS BY FEATURE:")
    print("-" * 70)
    print(f"{'Feature':<20} {'Δ Cons':>10} {'p-value':>10} {'Δ Acc':>10} {'p-value':>10} {'n':>5}")
    print("-" * 70)

    for feature, stats_data in analysis.get("by_feature", {}).items():
        print(f"{feature:<20} {stats_data['mean_delta_consistency']:>+10.4f} {stats_data['p_consistency']:>10.4f} "
              f"{stats_data['mean_delta_accuracy']:>+10.4f} {stats_data['p_accuracy']:>10.4f} {stats_data['n_samples']:>5}")

    # Save final results
    final_output = {
        "metadata": {
            "features": features,
            "model": get_display_name(args.model),
            "temperatures": temperatures,
            "num_runs": args.num_runs,
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
