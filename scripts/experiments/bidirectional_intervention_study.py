#!/usr/bin/env python3
"""
Bidirectional Intervention Study: Testing causal effects of "should"/"must" on consistency and accuracy.

This script tests:
1. Adding "should"/"must" to neutral prompts → measure consistency/accuracy change
2. Removing "should"/"must" from directive prompts → measure consistency/accuracy change

Design:
- Select 50 prompts WITH directive language ("should", "must")
- Select 50 prompts WITHOUT directive language (neutral)
- Create matched pairs by adding/removing directive phrases
- Run each version 10 times
- Calculate stability score (consistency) and accuracy against ground truth
"""

import json
import re
import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import random
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.bedrock_utils import inference_with_converse_api
from sted.model_config import FINAL_MODELS
import boto3
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directive patterns
DIRECTIVE_PATTERNS = [
    r'\bshould\b',
    r'\bmust\b',
    r'\bneed to\b',
    r'\bhave to\b',
]

# Templates for adding directive language
ADD_TEMPLATES = [
    ("You should ", ""),  # Prepend "You should" to the query
    ("You must ", ""),
    (" You should do this.", ""),  # Append
]

# Templates for removing directive language
REMOVE_PATTERNS = [
    (r'\bYou should\b', ''),
    (r'\bYou must\b', ''),
    (r'\byou should\b', ''),
    (r'\byou must\b', ''),
    (r'\bshould\b', 'can'),  # Replace "should" with "can"
    (r'\bmust\b', 'can'),    # Replace "must" with "can"
]


@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""
    sample_id: str
    original_query: str
    modified_query: str
    intervention_type: str  # "add_directive" or "remove_directive"
    directive_word: str  # "should" or "must"

    # Original prompt results
    original_runs: List[Any]
    original_consistency: float
    original_accuracy: float

    # Modified prompt results
    modified_runs: List[Any]
    modified_consistency: float
    modified_accuracy: float

    # Deltas
    delta_consistency: float
    delta_accuracy: float


def has_directive_language(text: str) -> bool:
    """Check if text contains directive language (should/must)."""
    text_lower = text.lower()
    for pattern in DIRECTIVE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def get_directive_words(text: str) -> List[str]:
    """Get list of directive words found in text."""
    text_lower = text.lower()
    found = []
    if re.search(r'\bshould\b', text_lower):
        found.append('should')
    if re.search(r'\bmust\b', text_lower):
        found.append('must')
    return found


def add_directive_to_query(query: str, directive: str = "should") -> str:
    """Add directive language to a neutral query."""
    # Strategy: Prepend "You should" or "You must" to the query
    if directive == "should":
        # If query starts with a verb or action word, prepend "You should"
        if query[0].isupper():
            return f"You should {query[0].lower()}{query[1:]}"
        return f"You should {query}"
    elif directive == "must":
        if query[0].isupper():
            return f"You must {query[0].lower()}{query[1:]}"
        return f"You must {query}"
    return query


def remove_directive_from_query(query: str) -> str:
    """Remove directive language from a query."""
    modified = query
    # Remove "You should" / "You must" patterns
    modified = re.sub(r'\bYou should\s+', '', modified, flags=re.IGNORECASE)
    modified = re.sub(r'\bYou must\s+', '', modified, flags=re.IGNORECASE)
    # Replace remaining "should"/"must" with neutral alternatives
    modified = re.sub(r'\bshould\b', 'can', modified, flags=re.IGNORECASE)
    modified = re.sub(r'\bmust\b', 'can', modified, flags=re.IGNORECASE)
    # Capitalize first letter if needed
    if modified and modified[0].islower():
        modified = modified[0].upper() + modified[1:]
    return modified


def calculate_accuracy(generated: List[Dict], ground_truth: List[Dict], evaluator: SemanticJsonTreeConsistencyEvaluator) -> float:
    """Calculate accuracy of generated tool calls against ground truth using STED similarity."""
    if not generated or not ground_truth:
        return 0.0

    try:
        # Use STED to compute similarity (accepts dict/list directly)
        similarity = evaluator.calculate_similarity_with_deepdiff_opt(
            generated, ground_truth, variation_type="combined"
        )
        return similarity
    except Exception as e:
        logger.warning(f"Error calculating accuracy: {e}")
        return 0.0


def calculate_consistency(runs: List[Any], evaluator: SemanticJsonTreeConsistencyEvaluator) -> float:
    """Calculate consistency (stability score) across multiple runs."""
    if len(runs) < 2:
        return 1.0

    # Filter out None/failed runs
    valid_runs = [r for r in runs if r is not None]
    if len(valid_runs) < 2:
        return 0.0

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(valid_runs)):
        for j in range(i + 1, len(valid_runs)):
            try:
                sim = evaluator.calculate_similarity_with_deepdiff_opt(
                    valid_runs[i], valid_runs[j], variation_type="combined"
                )
                similarities.append(sim)
            except Exception as e:
                logger.warning(f"Error computing similarity: {e}")
                similarities.append(0.0)

    if not similarities:
        return 0.0

    # Calculate stability score: S_alpha = exp(-alpha * sigma_d^2)
    # where sigma_d is std of distances (1 - similarity)
    distances = [1 - s for s in similarities]
    sigma_d = np.std(distances)
    alpha = 20  # Same as paper
    stability_score = np.exp(-alpha * sigma_d ** 2)

    return stability_score


def convert_openai_tools_to_bedrock(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI format tools to Bedrock Converse API format."""
    bedrock_tools = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            bedrock_tool = {
                "toolSpec": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "inputSchema": {
                        "json": func.get("parameters", {"type": "object", "properties": {}})
                    }
                }
            }
            bedrock_tools.append(bedrock_tool)
    return bedrock_tools


async def run_single_query(
    bedrock_client,
    model_id: str,
    query: str,
    tools: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Optional[List[Dict]]:
    """Run a single LLM query and return tool calls."""
    messages = [{"role": "user", "content": [{"text": query}]}]

    # Convert tools from OpenAI format to Bedrock format
    bedrock_tools = convert_openai_tools_to_bedrock(tools)

    try:
        response = inference_with_converse_api(
            bedrock_client=bedrock_client,
            model_id=model_id,
            messages=messages,
            tools=bedrock_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            return_content=True
        )

        if response is None:
            return None

        # Extract tool calls from response
        tool_calls = []
        if 'output' in response and 'message' in response['output']:
            content = response['output']['message'].get('content', [])
            for item in content:
                if 'toolUse' in item:
                    tool_use = item['toolUse']
                    tool_calls.append({
                        'name': tool_use.get('name', ''),
                        'arguments': tool_use.get('input', {})
                    })

        return tool_calls if tool_calls else None

    except Exception as e:
        logger.warning(f"Error running query: {e}")
        return None


async def run_intervention_experiment(
    bedrock_client,
    model_id: str,
    sample: Dict,
    intervention_type: str,
    evaluator: SemanticJsonTreeConsistencyEvaluator,
    num_runs: int = 10,
    temperature: float = 0.7
) -> Optional[InterventionResult]:
    """Run intervention experiment for a single sample."""

    original_query = sample['question']
    ground_truth = sample.get('tool_calls', [])
    tools = sample.get('tools', [])

    # Create modified query based on intervention type
    if intervention_type == "add_directive":
        # Add "should" to neutral prompts
        directive = random.choice(['should', 'must'])
        modified_query = add_directive_to_query(original_query, directive)
    else:  # remove_directive
        directive = get_directive_words(original_query)[0] if get_directive_words(original_query) else 'should'
        modified_query = remove_directive_from_query(original_query)

    # Skip if modification didn't change anything meaningful
    if original_query.strip().lower() == modified_query.strip().lower():
        logger.info(f"Skipping sample {sample['id']}: no meaningful modification")
        return None

    logger.info(f"Running intervention: {intervention_type}")
    logger.info(f"Original: {original_query[:100]}...")
    logger.info(f"Modified: {modified_query[:100]}...")

    # Run original query multiple times
    original_runs = []
    for i in range(num_runs):
        result = await run_single_query(bedrock_client, model_id, original_query, tools, temperature)
        original_runs.append(result)
        await asyncio.sleep(0.5)  # Rate limiting

    # Run modified query multiple times
    modified_runs = []
    for i in range(num_runs):
        result = await run_single_query(bedrock_client, model_id, modified_query, tools, temperature)
        modified_runs.append(result)
        await asyncio.sleep(0.5)  # Rate limiting

    # Calculate metrics
    original_consistency = calculate_consistency(original_runs, evaluator)
    modified_consistency = calculate_consistency(modified_runs, evaluator)

    # Calculate accuracy (mean across runs)
    original_accuracies = [calculate_accuracy(r, ground_truth, evaluator) for r in original_runs if r]
    modified_accuracies = [calculate_accuracy(r, ground_truth, evaluator) for r in modified_runs if r]

    original_accuracy = np.mean(original_accuracies) if original_accuracies else 0.0
    modified_accuracy = np.mean(modified_accuracies) if modified_accuracies else 0.0

    return InterventionResult(
        sample_id=sample['id'],
        original_query=original_query,
        modified_query=modified_query,
        intervention_type=intervention_type,
        directive_word=directive,
        original_runs=original_runs,
        original_consistency=original_consistency,
        original_accuracy=original_accuracy,
        modified_runs=modified_runs,
        modified_consistency=modified_consistency,
        modified_accuracy=modified_accuracy,
        delta_consistency=modified_consistency - original_consistency,
        delta_accuracy=modified_accuracy - original_accuracy
    )


def select_samples_for_intervention(
    data: List[Dict],
    num_with_directive: int = 25,
    num_without_directive: int = 25
) -> Tuple[List[Dict], List[Dict]]:
    """Select samples for intervention study."""

    with_directive = []
    without_directive = []

    for sample in data:
        query = sample.get('question', '')
        if has_directive_language(query):
            with_directive.append(sample)
        else:
            without_directive.append(sample)

    logger.info(f"Found {len(with_directive)} samples with directive language")
    logger.info(f"Found {len(without_directive)} samples without directive language")

    # Randomly select samples
    random.seed(42)  # Reproducibility
    selected_with = random.sample(with_directive, min(num_with_directive, len(with_directive)))
    selected_without = random.sample(without_directive, min(num_without_directive, len(without_directive)))

    return selected_with, selected_without


def statistical_analysis(results: List[InterventionResult]) -> Dict[str, Any]:
    """Perform statistical analysis on intervention results."""

    # Separate by intervention type
    add_results = [r for r in results if r.intervention_type == "add_directive"]
    remove_results = [r for r in results if r.intervention_type == "remove_directive"]

    analysis = {}

    for intervention_type, intervention_results in [("add_directive", add_results), ("remove_directive", remove_results)]:
        if not intervention_results:
            continue

        delta_consistency = [r.delta_consistency for r in intervention_results]
        delta_accuracy = [r.delta_accuracy for r in intervention_results]

        # Paired t-test for consistency
        t_stat_cons, p_val_cons = stats.ttest_1samp(delta_consistency, 0)

        # Paired t-test for accuracy
        t_stat_acc, p_val_acc = stats.ttest_1samp(delta_accuracy, 0)

        # Effect sizes (Cohen's d)
        cohens_d_cons = np.mean(delta_consistency) / np.std(delta_consistency) if np.std(delta_consistency) > 0 else 0
        cohens_d_acc = np.mean(delta_accuracy) / np.std(delta_accuracy) if np.std(delta_accuracy) > 0 else 0

        # Count improvements vs degradations
        improved_cons = sum(1 for d in delta_consistency if d > 0)
        worsened_cons = sum(1 for d in delta_consistency if d < 0)
        improved_acc = sum(1 for d in delta_accuracy if d > 0)
        worsened_acc = sum(1 for d in delta_accuracy if d < 0)

        analysis[intervention_type] = {
            "n_samples": len(intervention_results),
            "consistency": {
                "mean_delta": np.mean(delta_consistency),
                "std_delta": np.std(delta_consistency),
                "t_statistic": t_stat_cons,
                "p_value": p_val_cons,
                "cohens_d": cohens_d_cons,
                "improved": improved_cons,
                "worsened": worsened_cons,
                "no_change": len(intervention_results) - improved_cons - worsened_cons
            },
            "accuracy": {
                "mean_delta": np.mean(delta_accuracy),
                "std_delta": np.std(delta_accuracy),
                "t_statistic": t_stat_acc,
                "p_value": p_val_acc,
                "cohens_d": cohens_d_acc,
                "improved": improved_acc,
                "worsened": worsened_acc,
                "no_change": len(intervention_results) - improved_acc - worsened_acc
            },
            "original_consistency_mean": np.mean([r.original_consistency for r in intervention_results]),
            "modified_consistency_mean": np.mean([r.modified_consistency for r in intervention_results]),
            "original_accuracy_mean": np.mean([r.original_accuracy for r in intervention_results]),
            "modified_accuracy_mean": np.mean([r.modified_accuracy for r in intervention_results])
        }

    return analysis


async def main():
    parser = argparse.ArgumentParser(description="Bidirectional Intervention Study")
    parser.add_argument("--dataset", type=str, choices=["toucan", "sharegpt", "both"], default="toucan",
                        help="Dataset to use")
    parser.add_argument("--model", type=str, default="us.anthropic.claude-sonnet-4-20250514-v1:0",
                        help="Model ID to use")
    parser.add_argument("--num-samples", type=int, default=25,
                        help="Number of samples per intervention type")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of runs per prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--output-dir", type=str, default="results/intervention_study",
                        help="Output directory for results")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Bedrock client
    boto_config = Config(
        retries={'max_attempts': 10, 'mode': 'adaptive'},
        read_timeout=300,
        connect_timeout=30
    )
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1", config=boto_config)

    # Initialize STED evaluator
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id="amazon.titan-embed-text-v2:0",
        embedding_dim=512,
        region_name="us-east-1"
    )

    datasets_to_process = []
    if args.dataset in ["toucan", "both"]:
        datasets_to_process.append(("toucan", "toucan_data/toucan_tool_calls_1006.json"))
    if args.dataset in ["sharegpt", "both"]:
        datasets_to_process.append(("sharegpt", "sharegpt_data/sharegpt_samples.json"))

    all_results = {}

    for dataset_name, dataset_path in datasets_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        # Load dataset
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} samples from {dataset_name}")

        # Select samples for intervention
        samples_with_directive, samples_without_directive = select_samples_for_intervention(
            data,
            num_with_directive=args.num_samples,
            num_without_directive=args.num_samples
        )

        results = []

        # Run "remove directive" experiments on samples WITH directive language
        logger.info(f"\n--- Remove Directive Experiments ({len(samples_with_directive)} samples) ---")
        for i, sample in enumerate(samples_with_directive):
            logger.info(f"Processing sample {i+1}/{len(samples_with_directive)}: {sample['id']}")
            result = await run_intervention_experiment(
                bedrock_client=bedrock_client,
                model_id=args.model,
                sample=sample,
                intervention_type="remove_directive",
                evaluator=evaluator,
                num_runs=args.num_runs,
                temperature=args.temperature
            )
            if result:
                results.append(result)
                logger.info(f"  Δ Consistency: {result.delta_consistency:+.4f}, Δ Accuracy: {result.delta_accuracy:+.4f}")

        # Run "add directive" experiments on samples WITHOUT directive language
        logger.info(f"\n--- Add Directive Experiments ({len(samples_without_directive)} samples) ---")
        for i, sample in enumerate(samples_without_directive):
            logger.info(f"Processing sample {i+1}/{len(samples_without_directive)}: {sample['id']}")
            result = await run_intervention_experiment(
                bedrock_client=bedrock_client,
                model_id=args.model,
                sample=sample,
                intervention_type="add_directive",
                evaluator=evaluator,
                num_runs=args.num_runs,
                temperature=args.temperature
            )
            if result:
                results.append(result)
                logger.info(f"  Δ Consistency: {result.delta_consistency:+.4f}, Δ Accuracy: {result.delta_accuracy:+.4f}")

        # Statistical analysis
        analysis = statistical_analysis(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"{dataset_name}_intervention_results_{timestamp}.json"

        output_data = {
            "metadata": {
                "dataset": dataset_name,
                "model": args.model,
                "num_runs": args.num_runs,
                "temperature": args.temperature,
                "timestamp": timestamp
            },
            "results": [asdict(r) for r in results],
            "analysis": analysis
        }

        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"\nResults saved to {results_file}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {dataset_name}")
        print(f"{'='*60}")

        for intervention_type, stats in analysis.items():
            print(f"\n{intervention_type.upper()} (n={stats['n_samples']})")
            print("-" * 40)
            print(f"CONSISTENCY:")
            print(f"  Original Mean: {stats['original_consistency_mean']:.4f}")
            print(f"  Modified Mean: {stats['modified_consistency_mean']:.4f}")
            print(f"  Mean Δ: {stats['consistency']['mean_delta']:+.4f} (std: {stats['consistency']['std_delta']:.4f})")
            print(f"  t-stat: {stats['consistency']['t_statistic']:.3f}, p-value: {stats['consistency']['p_value']:.4f}")
            print(f"  Cohen's d: {stats['consistency']['cohens_d']:.3f}")
            print(f"  Improved: {stats['consistency']['improved']}, Worsened: {stats['consistency']['worsened']}")

            print(f"\nACCURACY:")
            print(f"  Original Mean: {stats['original_accuracy_mean']:.4f}")
            print(f"  Modified Mean: {stats['modified_accuracy_mean']:.4f}")
            print(f"  Mean Δ: {stats['accuracy']['mean_delta']:+.4f} (std: {stats['accuracy']['std_delta']:.4f})")
            print(f"  t-stat: {stats['accuracy']['t_statistic']:.3f}, p-value: {stats['accuracy']['p_value']:.4f}")
            print(f"  Cohen's d: {stats['accuracy']['cohens_d']:.3f}")
            print(f"  Improved: {stats['accuracy']['improved']}, Worsened: {stats['accuracy']['worsened']}")

        all_results[dataset_name] = {
            "results": results,
            "analysis": analysis
        }

    # Save combined summary
    summary_file = output_dir / f"intervention_study_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        summary = {
            "metadata": {
                "model": args.model,
                "num_runs": args.num_runs,
                "temperature": args.temperature,
                "datasets": list(all_results.keys())
            },
            "analysis_by_dataset": {k: v["analysis"] for k, v in all_results.items()}
        }
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nCombined summary saved to {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
