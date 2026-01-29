#!/usr/bin/env python3
"""
LLM Judge for Best-Match Selection Task.

Uses an LLM to select which candidate response is most similar to the ground truth,
providing a baseline comparison against STED, BERTScore, DeepDiff, and TED methods.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

import boto3
from botocore.config import Config
from sted.bedrock_utils import inference_with_converse_api, boto_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model configurations for LLM judge
LLM_JUDGE_MODELS = {
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-sonnet-3.5": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    "nova-lite": "us.amazon.nova-lite-v1:0",
}


SYSTEM_PROMPT = """You are an expert evaluator for structured JSON data similarity. Your task is to identify which candidate response is MOST similar to the ground truth.

When comparing JSON structures, consider:
1. **Structural Similarity**: Same keys, nesting levels, and overall structure
2. **Semantic Similarity**: Same meaning even if naming conventions differ (e.g., "user_name" vs "userName")
3. **Value Similarity**: Same or equivalent values
4. **Functional Equivalence**: Whether the responses could be used interchangeably in a downstream system

Focus on semantic and functional similarity rather than exact string matching."""


def create_judge_prompt(ground_truth: dict, candidates: list) -> str:
    """Create the prompt for LLM to judge which candidate is most similar."""

    # Format ground truth
    gt_str = json.dumps(ground_truth, indent=2, ensure_ascii=False)

    # Format candidates
    candidate_strs = []
    for c in candidates:
        label = c['label']
        response = json.dumps(c['response'], indent=2, ensure_ascii=False)
        candidate_strs.append(f"**Candidate {label}:**\n```json\n{response}\n```")

    candidates_text = "\n\n".join(candidate_strs)

    prompt = f"""Given the Ground Truth JSON below, determine which candidate response is MOST similar.

**Ground Truth:**
```json
{gt_str}
```

{candidates_text}

Compare each candidate to the ground truth and select the one that is most similar in terms of structure, semantics, and values.

Respond with ONLY the letter of your choice (A, B, C, or D) and a brief explanation in the following format:
Choice: [LETTER]
Reason: [Brief explanation of why this candidate is most similar]"""

    return prompt


def judge_sample(
    bedrock_client,
    model_id: str,
    sample: dict,
    temperature: float = 0.0,
    max_tokens: int = 500
) -> dict:
    """Use LLM to judge which candidate is most similar to ground truth."""

    item_id = sample.get('id', 'unknown')
    ground_truth = sample.get('ground_truth', {})
    candidates = sample.get('candidates', [])

    if not candidates:
        return {
            'item_id': item_id,
            'choice': None,
            'reason': 'No candidates provided',
            'error': True
        }

    prompt = create_judge_prompt(ground_truth, candidates)

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    try:
        response = inference_with_converse_api(
            bedrock_client=bedrock_client,
            model_id=model_id,
            messages=messages,
            system_prompts=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if not response:
            return {
                'item_id': item_id,
                'choice': None,
                'reason': 'Empty response from LLM',
                'error': True
            }

        # Parse response text
        response_text = ""
        for content in response:
            if 'text' in content:
                response_text += content['text']

        # Extract choice and reason
        choice = None
        reason = response_text

        # Try to parse structured response
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.lower().startswith('choice:'):
                choice_part = line.split(':', 1)[1].strip()
                # Extract just the letter
                for c in choice_part:
                    if c.upper() in ['A', 'B', 'C', 'D']:
                        choice = c.upper()
                        break
            elif line.lower().startswith('reason:'):
                reason = line.split(':', 1)[1].strip()

        # If no structured format, try to find letter at beginning
        if choice is None:
            first_char = response_text.strip()[0].upper() if response_text.strip() else ''
            if first_char in ['A', 'B', 'C', 'D']:
                choice = first_char

        return {
            'item_id': item_id,
            'choice': choice,
            'reason': reason,
            'raw_response': response_text,
            'error': choice is None
        }

    except Exception as e:
        logger.error(f"Error judging sample {item_id}: {e}")
        return {
            'item_id': item_id,
            'choice': None,
            'reason': str(e),
            'error': True
        }


def run_llm_judge(
    dataset_path: str,
    output_path: str,
    model_name: str = "claude-haiku",
    region: str = "us-west-2",
    n_workers: int = 5,
    temperature: float = 0.0,
    limit: int = None,
    resume: bool = True
):
    """Run LLM judge on the validation dataset."""

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
        return existing_results

    # Get model ID
    model_id = LLM_JUDGE_MODELS.get(model_name)
    if not model_id:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(LLM_JUDGE_MODELS.keys())}")

    logger.info(f"Using model: {model_name} ({model_id})")

    # Create Bedrock client
    bedrock_client = boto3.client("bedrock-runtime", region_name=region, config=boto_config)

    results = list(existing_results.values())
    errors = 0

    # Process items with progress bar
    with tqdm(total=len(items_to_process), desc="LLM Judge") as pbar:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for item in items_to_process:
                future = executor.submit(
                    judge_sample,
                    bedrock_client,
                    model_id,
                    item,
                    temperature
                )
                futures[future] = item.get('id')

            for future in as_completed(futures):
                item_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.get('error'):
                        errors += 1
                        pbar.set_postfix({'errors': errors})

                except Exception as e:
                    logger.error(f"Error processing {item_id}: {e}")
                    results.append({
                        'item_id': item_id,
                        'choice': None,
                        'reason': str(e),
                        'error': True
                    })
                    errors += 1

                pbar.update(1)

                # Save intermediate results every 50 items
                if len(results) % 50 == 0:
                    save_results(results, output_path, model_name, dataset)

    # Save final results
    save_results(results, output_path, model_name, dataset)

    # Compute statistics
    compute_statistics(results, dataset)

    return results


def save_results(results: list, output_path: str, model_name: str, dataset: dict):
    """Save results to JSON file."""

    output_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'model': model_name,
            'n_items': len(results),
            'n_errors': sum(1 for r in results if r.get('error')),
            'source_dataset': dataset.get('metadata', {})
        },
        'results': results
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(results)} results to {output_path}")


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
        choice = result.get('choice')
        llm_picks[item_id] = choice

        if item_id not in method_picks:
            continue

        total_valid += 1

        for method in methods:
            if method_picks[item_id].get(method) == choice:
                agreement_counts[method] += 1

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("LLM Judge Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total items judged: {len(results)}")
    logger.info(f"Valid judgments: {total_valid}")
    logger.info(f"Errors: {len(results) - total_valid}")
    logger.info("")
    logger.info("Agreement with other methods:")
    for method in methods:
        if total_valid > 0:
            rate = agreement_counts[method] / total_valid * 100
            logger.info(f"  {method.upper():12s}: {agreement_counts[method]:3d}/{total_valid} ({rate:.1f}%)")
    logger.info("=" * 60)

    return agreement_counts, total_valid


def main():
    parser = argparse.ArgumentParser(description="LLM Judge for Best-Match Selection")
    parser.add_argument(
        '--dataset',
        type=str,
        default='scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json',
        help='Path to validation dataset JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scripts/data/human_validation/best_match_selection/data/llm_judge_results.json',
        help='Path to save results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-haiku',
        choices=list(LLM_JUDGE_MODELS.keys()),
        help='LLM model to use for judging'
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
        default=5,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for LLM sampling'
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

    run_llm_judge(
        dataset_path=args.dataset,
        output_path=args.output,
        model_name=args.model,
        region=args.region,
        n_workers=args.workers,
        temperature=args.temperature,
        limit=args.limit,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
