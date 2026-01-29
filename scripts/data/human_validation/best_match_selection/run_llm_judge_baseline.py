#!/usr/bin/env python3
"""
Run LLM Judge as a baseline for Best-Match Selection Task.

Uses the existing LLMJudge class to score each candidate against ground truth
and pick the most similar one, providing a baseline comparison against
STED, BERTScore, DeepDiff, and TED methods.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from sted.llm_judge import LLMJudge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def score_candidates(
    judge: LLMJudge,
    ground_truth: dict,
    candidates: list
) -> dict:
    """
    Score each candidate against ground truth using LLM judge.

    Args:
        judge: LLMJudge instance
        ground_truth: Ground truth JSON
        candidates: List of candidate responses with 'label' and 'response'

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


def process_sample(args_tuple):
    """Process a single sample (for parallel execution)."""
    judge, sample = args_tuple

    item_id = sample.get('id', 'unknown')
    ground_truth = sample.get('ground_truth', {})
    candidates = sample.get('candidates', [])

    try:
        result = score_candidates(judge, ground_truth, candidates)
        return {
            'item_id': item_id,
            'pick': result['pick'],
            'scores': result['scores'],
            'best_score': result['best_score'],
            'error': False
        }
    except Exception as e:
        logger.error(f"Error processing {item_id}: {e}")
        return {
            'item_id': item_id,
            'pick': None,
            'scores': {},
            'best_score': 0.0,
            'error': True,
            'error_message': str(e)
        }


def run_llm_judge_baseline(
    dataset_path: str,
    output_path: str,
    model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    region: str = "us-west-2",
    n_workers: int = 3,
    limit: int = None,
    resume: bool = True
):
    """
    Run LLM judge baseline on the validation dataset.

    Args:
        dataset_path: Path to validation dataset JSON
        output_path: Path to save results
        model_id: Bedrock model ID for the judge
        region: AWS region
        n_workers: Number of parallel workers
        limit: Limit number of items to process
        resume: Resume from existing results
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
        return list(existing_results.values())

    # Create LLM Judge
    logger.info(f"Creating LLM Judge with model: {model_id}")
    judge = LLMJudge(
        provider="bedrock",
        model_id=model_id,
        region_name=region,
        temperature=0.0
    )

    results = list(existing_results.values())
    errors = 0
    completed = 0

    # Process items in parallel using ThreadPoolExecutor
    logger.info(f"Processing with {n_workers} parallel workers")
    with tqdm(total=len(items_to_process), desc="LLM Judge Baseline") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_sample, (judge, item)): item.get('id') for item in items_to_process}

            for future in as_completed(futures):
                item_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if result.get('error'):
                        errors += 1

                except Exception as e:
                    logger.error(f"Error processing {item_id}: {e}")
                    results.append({
                        'item_id': item_id,
                        'pick': None,
                        'scores': {},
                        'best_score': 0.0,
                        'error': True,
                        'error_message': str(e)
                    })
                    errors += 1
                    completed += 1

                pbar.update(1)
                pbar.set_postfix({'errors': errors, 'done': completed})

                # Save intermediate results every 20 items
                if completed % 20 == 0:
                    save_results(results, output_path, model_id, dataset)

    # Save final results
    save_results(results, output_path, model_id, dataset)

    # Compute and print statistics
    compute_statistics(results, dataset)

    return results


def save_results(results: list, output_path: str, model_id: str, dataset: dict):
    """Save results to JSON file."""

    # Build method picks mapping
    method_picks = {'llm': {}}
    for result in results:
        if not result.get('error'):
            method_picks['llm'][result['item_id']] = result['pick']

    output_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'model': model_id,
            'n_items': len(results),
            'n_errors': sum(1 for r in results if r.get('error')),
            'method': 'llm_judge',
            'source_dataset': dataset.get('metadata', {})
        },
        'method_picks': method_picks,
        'results': results
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(results)} results to {output_path}")

    # Also save as CSV in human annotation format
    csv_path = output_path.replace('.json', '.csv')
    save_results_csv(results, csv_path, model_id)


def save_results_csv(results: list, csv_path: str, model_id: str):
    """Save results in CSV format matching human annotation format."""
    import csv

    # Sort results by item_id for consistent ordering
    sorted_results = sorted(results, key=lambda x: x.get('item_id', ''))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header matching human annotation format
        writer.writerow(['item_id', 'choice', 'confidence', 'skipped', 'skip_reason', 'notes', 'timestamp'])

        for result in sorted_results:
            item_id = result.get('item_id', '')
            choice = result.get('pick', '')
            error = result.get('error', False)
            timestamp = datetime.now().isoformat()

            # Mark as skipped if there was an error
            skipped = 'true' if error else 'false'
            skip_reason = result.get('error_message', '') if error else ''
            notes = f"LLM Judge ({model_id}), score={result.get('best_score', 0):.3f}"

            writer.writerow([item_id, choice, '', skipped, skip_reason, notes, timestamp])

    logger.info(f"Saved CSV results to {csv_path}")


def compute_statistics(results: list, dataset: dict):
    """Compute and print statistics comparing LLM judge to other methods."""

    # Build method picks map from dataset
    method_picks = {}
    for item in dataset.get('items', []):
        item_id = item.get('id')
        picks = item.get('metadata', {}).get('method_picks', {})
        method_picks[item_id] = picks

    # Count agreements with each method
    methods = ['sted', 'deepdiff', 'ted', 'bertscore']
    agreement_counts = {method: 0 for method in methods}
    total_valid = 0

    llm_picks = {}
    for result in results:
        if result.get('error'):
            continue

        item_id = result['item_id']
        choice = result.get('pick')
        llm_picks[item_id] = choice

        if item_id not in method_picks:
            continue

        total_valid += 1

        for method in methods:
            if method_picks[item_id].get(method) == choice:
                agreement_counts[method] += 1

    # Print results
    print("\n" + "=" * 60)
    print("LLM Judge Baseline Results Summary")
    print("=" * 60)
    print(f"Total items judged: {len(results)}")
    print(f"Valid judgments: {total_valid}")
    print(f"Errors: {len(results) - total_valid}")
    print("")
    print("Agreement with other methods:")
    for method in methods:
        if total_valid > 0:
            rate = agreement_counts[method] / total_valid * 100
            print(f"  {method.upper():12s}: {agreement_counts[method]:3d}/{total_valid} ({rate:.1f}%)")
    print("=" * 60)

    return agreement_counts, total_valid


def main():
    parser = argparse.ArgumentParser(description="LLM Judge Baseline for Best-Match Selection")
    parser.add_argument(
        '--dataset',
        type=str,
        default='scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json',
        help='Path to validation dataset JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scripts/data/human_validation/best_match_selection/data/llm_judge_baseline_results.json',
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
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (keep low to avoid rate limits)'
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

    args = parser.parse_args()

    run_llm_judge_baseline(
        dataset_path=args.dataset,
        output_path=args.output,
        model_id=args.model,
        region=args.region,
        n_workers=args.workers,
        limit=args.limit,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
