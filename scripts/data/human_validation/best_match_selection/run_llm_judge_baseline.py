#!/usr/bin/env python3
"""
Run LLM Judge as a baseline for Best-Match Selection Task.

Uses the existing LLMJudge class to score each candidate against ground truth
and pick the most similar one, providing a baseline comparison against
STED, BERTScore, DeepDiff, and TED methods.

NEW: Runs multiple times per sample to measure LLM-as-judge consistency.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from sted.llm_judge import LLMJudge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def score_candidates(
    judge: LLMJudge,
    ground_truth: dict,
    candidates: list,
    temperature: float = 0.0
) -> dict:
    """
    Score each candidate against ground truth using LLM judge.

    Args:
        judge: LLMJudge instance
        ground_truth: Ground truth JSON
        candidates: List of candidate responses with 'label' and 'response'
        temperature: Temperature for LLM calls

    Returns:
        Dict with scores for each candidate and the best pick
    """
    scores = {}

    for candidate in candidates:
        label = candidate['label']
        response = candidate['response']

        # Calculate similarity between ground truth and candidate
        score = judge.calculate_similarity(ground_truth, response)
        scores[label] = score

    # Find best match (highest score)
    if scores:
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
    else:
        best_label = None
        best_score = 0.0

    return {
        'scores': scores,
        'pick': best_label,
        'best_score': best_score
    }


def single_run(args):
    """Execute a single LLM judge run (for parallel execution)."""
    judge, ground_truth, candidates, run_idx, item_id = args
    try:
        result = score_candidates(judge, ground_truth, candidates)
        return {
            'run': run_idx,
            'pick': result['pick'],
            'scores': result['scores'],
            'best_score': result['best_score'],
            'error': None
        }
    except Exception as e:
        logger.error(f"Error processing {item_id} run {run_idx}: {e}")
        return {
            'run': run_idx,
            'pick': None,
            'scores': {},
            'best_score': 0.0,
            'error': str(e)
        }


def process_sample_multi_run(args_tuple):
    """Process a single sample multiple times to measure consistency (with parallel runs)."""
    judge, sample, n_runs, temperature, parallel_runs = args_tuple

    item_id = sample.get('id', 'unknown')
    ground_truth = sample.get('ground_truth', {})
    candidates = sample.get('candidates', [])

    all_runs = []
    all_picks = []
    all_scores = []
    errors = 0

    if parallel_runs and n_runs > 1:
        # Run in parallel using ThreadPoolExecutor
        run_args = [(judge, ground_truth, candidates, run_idx, item_id) for run_idx in range(n_runs)]
        with ThreadPoolExecutor(max_workers=min(n_runs, 5)) as executor:
            results = list(executor.map(single_run, run_args))

        for result in results:
            all_runs.append(result)
            if result['error']:
                errors += 1
            else:
                all_picks.append(result['pick'])
                all_scores.append(result['scores'])
    else:
        # Sequential execution
        for run_idx in range(n_runs):
            try:
                result = score_candidates(judge, ground_truth, candidates, temperature)
                all_runs.append({
                    'run': run_idx,
                    'pick': result['pick'],
                    'scores': result['scores'],
                    'best_score': result['best_score']
                })
                all_picks.append(result['pick'])
                all_scores.append(result['scores'])
            except Exception as e:
                logger.error(f"Error processing {item_id} run {run_idx}: {e}")
                all_runs.append({
                    'run': run_idx,
                    'pick': None,
                    'scores': {},
                    'best_score': 0.0,
                    'error': str(e)
                })
                errors += 1

    # Compute consistency metrics
    valid_picks = [p for p in all_picks if p is not None]

    if valid_picks:
        # Count how many times each pick was selected
        pick_counts = Counter(valid_picks)
        most_common_pick, most_common_count = pick_counts.most_common(1)[0]

        # Consistency = fraction of runs that agree with majority
        consistency = most_common_count / len(valid_picks)

        # All-agree = did all runs agree?
        all_agree = len(set(valid_picks)) == 1

        # Score variance per candidate
        score_variance = {}
        for label in candidates[0]['response'] if candidates else []:
            label_scores = [s.get(label, 0) for s in all_scores if label in s]
            if label_scores:
                score_variance[label] = np.var(label_scores)
    else:
        most_common_pick = None
        consistency = 0.0
        all_agree = False
        score_variance = {}

    return {
        'item_id': item_id,
        'n_runs': n_runs,
        'n_errors': errors,
        'picks': all_picks,
        'pick_counts': dict(Counter(valid_picks)) if valid_picks else {},
        'majority_pick': most_common_pick,
        'consistency': consistency,
        'all_agree': all_agree,
        'runs': all_runs,
        'error': errors == n_runs  # Only error if ALL runs failed
    }


def run_llm_judge_consistency(
    dataset_path: str,
    output_path: str,
    model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    region: str = "us-west-2",
    n_runs: int = 5,
    temperature: float = 0.7,
    n_workers: int = 3,
    limit: int = None,
    resume: bool = True,
    parallel_runs: bool = True
):
    """
    Run LLM judge multiple times per sample to measure consistency.

    Args:
        dataset_path: Path to validation dataset JSON
        output_path: Path to save results
        model_id: Bedrock model ID for the judge
        region: AWS region
        n_runs: Number of runs per sample to measure consistency
        temperature: Temperature for LLM calls (higher = more variability)
        n_workers: Number of parallel workers for processing samples
        limit: Limit number of items to process
        resume: Resume from existing results
        parallel_runs: Run the n_runs in parallel within each sample
    """

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    items = dataset.get('items', [])
    logger.info(f"Loaded {len(items)} items")

    if limit:
        items = items[:limit]
        logger.info(f"Limited to {limit} items")

    # Load existing results if resuming
    existing_results = {}
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
                for result in existing_data.get('results', []):
                    if not result.get('error'):
                        existing_results[result['item_id']] = result
            logger.info(f"Loaded {len(existing_results)} existing results")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")

    # Filter items that need processing
    items_to_process = [item for item in items if item.get('id') not in existing_results]
    logger.info(f"Items to process: {len(items_to_process)} (skipping {len(existing_results)} existing)")

    if not items_to_process:
        logger.info("All items already processed!")
        results = list(existing_results.values())
        compute_consistency_statistics(results, dataset)
        return results

    # Create LLM Judge with specified temperature
    logger.info(f"Creating LLM Judge with model: {model_id}, temperature: {temperature}")
    judge = LLMJudge(
        provider="bedrock",
        model_id=model_id,
        region_name=region,
        temperature=temperature
    )

    results = list(existing_results.values())
    errors = 0
    completed = 0

    # Process items with parallel workers
    logger.info(f"Processing {len(items_to_process)} items with {n_runs} runs each (workers={n_workers}, parallel_runs={parallel_runs})")

    def process_item(item):
        """Process a single item."""
        try:
            return process_sample_multi_run((judge, item, n_runs, temperature, parallel_runs))
        except Exception as e:
            logger.error(f"Error processing {item.get('id')}: {e}")
            return {
                'item_id': item.get('id'),
                'n_runs': n_runs,
                'n_errors': n_runs,
                'picks': [],
                'majority_pick': None,
                'consistency': 0.0,
                'all_agree': False,
                'runs': [],
                'error': True,
                'error_message': str(e)
            }

    if n_workers > 1:
        # Parallel processing of samples
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_item, item): item.get('id') for item in items_to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM Judge ({n_runs} runs × {n_workers} workers)"):
                result = future.result()
                results.append(result)
                completed += 1
                if result.get('error'):
                    errors += 1
                # Save intermediate results every 10 items
                if completed % 10 == 0:
                    save_consistency_results(results, output_path, model_id, n_runs, temperature, dataset)
    else:
        # Sequential processing
        for item in tqdm(items_to_process, desc=f"LLM Judge ({n_runs} runs each)"):
            result = process_item(item)
            results.append(result)
            completed += 1
            if result.get('error'):
                errors += 1
            # Save intermediate results every 10 items
            if completed % 10 == 0:
                save_consistency_results(results, output_path, model_id, n_runs, temperature, dataset)

    # Save final results
    save_consistency_results(results, output_path, model_id, n_runs, temperature, dataset)

    # Compute and print statistics
    compute_consistency_statistics(results, dataset)

    return results


def save_consistency_results(results: list, output_path: str, model_id: str, n_runs: int, temperature: float, dataset: dict):
    """Save results to JSON file."""

    # Compute overall consistency stats
    valid_results = [r for r in results if not r.get('error')]

    if valid_results:
        avg_consistency = np.mean([r['consistency'] for r in valid_results])
        all_agree_rate = np.mean([1 if r['all_agree'] else 0 for r in valid_results])
    else:
        avg_consistency = 0.0
        all_agree_rate = 0.0

    output_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'model': model_id,
            'n_runs_per_sample': n_runs,
            'temperature': temperature,
            'n_items': len(results),
            'n_errors': sum(1 for r in results if r.get('error')),
            'method': 'llm_judge_consistency',
            'source_dataset': dataset.get('metadata', {})
        },
        'summary': {
            'avg_consistency': avg_consistency,
            'all_agree_rate': all_agree_rate,
            'n_valid': len(valid_results)
        },
        'results': results
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(results)} results to {output_path}")


def compute_consistency_statistics(results: list, dataset: dict):
    """Compute and print consistency statistics."""

    valid_results = [r for r in results if not r.get('error')]

    if not valid_results:
        print("\nNo valid results to analyze.")
        return

    # Consistency metrics
    consistencies = [r['consistency'] for r in valid_results]
    all_agree_flags = [r['all_agree'] for r in valid_results]

    # Build method picks map from dataset for comparison
    method_picks = {}
    for item in dataset.get('items', []):
        item_id = item.get('id')
        picks = item.get('metadata', {}).get('method_picks', {})
        method_picks[item_id] = picks

    # Count agreements with each method (using majority pick)
    methods = ['sted', 'deepdiff', 'ted', 'bertscore']
    agreement_counts = {method: 0 for method in methods}
    total_comparable = 0

    for result in valid_results:
        item_id = result['item_id']
        majority_pick = result.get('majority_pick')

        if item_id not in method_picks or majority_pick is None:
            continue

        total_comparable += 1

        for method in methods:
            if method_picks[item_id].get(method) == majority_pick:
                agreement_counts[method] += 1

    # Print results
    print("\n" + "=" * 70)
    print("LLM-AS-JUDGE CONSISTENCY ANALYSIS")
    print("=" * 70)
    print(f"\nSamples analyzed: {len(valid_results)}")
    print(f"Runs per sample: {valid_results[0].get('n_runs', 'N/A') if valid_results else 'N/A'}")

    print(f"\n{'='*40}")
    print("CONSISTENCY METRICS (Is LLM-as-judge self-consistent?)")
    print(f"{'='*40}")
    print(f"  Average consistency:     {np.mean(consistencies):.1%}")
    print(f"  Min consistency:         {np.min(consistencies):.1%}")
    print(f"  Max consistency:         {np.max(consistencies):.1%}")
    print(f"  Std consistency:         {np.std(consistencies):.1%}")
    print(f"  All runs agree rate:     {np.mean(all_agree_flags):.1%}")

    # Breakdown by consistency level
    perfect = sum(1 for c in consistencies if c == 1.0)
    high = sum(1 for c in consistencies if 0.8 <= c < 1.0)
    medium = sum(1 for c in consistencies if 0.6 <= c < 0.8)
    low = sum(1 for c in consistencies if c < 0.6)

    print(f"\n  Consistency breakdown:")
    print(f"    Perfect (100%):        {perfect}/{len(valid_results)} ({perfect/len(valid_results)*100:.1f}%)")
    print(f"    High (80-99%):         {high}/{len(valid_results)} ({high/len(valid_results)*100:.1f}%)")
    print(f"    Medium (60-79%):       {medium}/{len(valid_results)} ({medium/len(valid_results)*100:.1f}%)")
    print(f"    Low (<60%):            {low}/{len(valid_results)} ({low/len(valid_results)*100:.1f}%)")

    print(f"\n{'='*40}")
    print("AGREEMENT WITH OTHER METHODS")
    print(f"{'='*40}")
    if total_comparable > 0:
        for method in methods:
            rate = agreement_counts[method] / total_comparable * 100
            print(f"  {method.upper():12s}: {agreement_counts[method]:3d}/{total_comparable} ({rate:.1f}%)")
    else:
        print("  No comparable items found.")

    print("=" * 70)

    # Return for programmatic use
    return {
        'avg_consistency': np.mean(consistencies),
        'all_agree_rate': np.mean(all_agree_flags),
        'agreement_with_methods': agreement_counts,
        'total_comparable': total_comparable
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Judge Consistency Analysis")
    parser.add_argument(
        '--dataset',
        type=str,
        default='scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json',
        help='Path to validation dataset JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_results.json',
        help='Path to save results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='us.anthropic.claude-3-5-haiku-20241022-v1:0',
        help='Bedrock model ID for the judge'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='us-west-2',
        help='AWS region'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=5,
        help='Number of runs per sample to measure consistency'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for LLM calls (higher = more variability expected)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='Number of parallel workers for processing samples'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of items to process'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from existing results'
    )
    parser.add_argument(
        '--no-parallel-runs',
        action='store_true',
        help='Disable parallel execution of runs within each sample'
    )

    args = parser.parse_args()

    run_llm_judge_consistency(
        dataset_path=args.dataset,
        output_path=args.output,
        model_id=args.model,
        region=args.region,
        n_runs=args.n_runs,
        temperature=args.temperature,
        n_workers=args.workers,
        limit=args.limit,
        resume=not args.no_resume,
        parallel_runs=not args.no_parallel_runs
    )


if __name__ == '__main__':
    main()
