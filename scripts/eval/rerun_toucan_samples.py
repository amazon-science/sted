#!/usr/bin/env python3
"""
Re-run specific Toucan samples from existing experiment results.

Re-runs only the high-token samples (ground truth > threshold) with higher max_tokens
to handle samples that may have been truncated in the original run.

Usage:
    # Re-run high-token samples for a specific experiment
    python rerun_toucan_samples.py \
        --experiment-dir llm_gen_results/toucan/generations-gemini-2.5-flash-lite-20260118 \
        --model-id google/gemini-2.5-flash-lite \
        --max-tokens 8000

    # Re-run specific sample IDs
    python rerun_toucan_samples.py \
        --experiment-dir llm_gen_results/toucan/generations-gpt-oss-120b-20260118 \
        --model-id openai.gpt-oss-120b-1:0 \
        --sample-ids f454d555-46c7-53d3-8a77-df4d667248c5,f96b2358-f347-5032-909e-4631a2968467 \
        --max-tokens 8000

    # Dry run to see what would be done
    python rerun_toucan_samples.py \
        --experiment-dir llm_gen_results/toucan/generations-mimo-v2-flash-20251229 \
        --model-id xiaomi/mimo-v2-flash:free \
        --high-token-only \
        --dry-run
"""

import argparse
import concurrent.futures
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(override=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted.model_config import get_provider, get_display_name

# High-token sample IDs (ground truth > 500 tokens)
HIGH_TOKEN_SAMPLE_IDS = [
    "f454d555-46c7-53d3-8a77-df4d667248c5",  # 3133 tokens
    "f96b2358-f347-5032-909e-4631a2968467",  # 2556 tokens
    "c0920df5-d6c9-5cb3-9672-7baa7a6eee91",  # 2287 tokens
    "36d73413-81aa-5044-a953-391c1eb81818",  # 2082 tokens
    "3f7ee455-365c-5842-8eeb-0bf2f79f8406",  # 1193 tokens
    "e8ae6446-dda7-5d9d-a3f7-cf2e031b80c3",  # 1162 tokens
    "43948be7-bb3d-53db-a288-1de3f8f35dfa",  # 1116 tokens
    "11c01a7c-1b22-59f7-b1ea-0462ef5ed1bd",  # 734 tokens
    "e718da06-3cfb-5059-bf37-5474102087e7",  # 667 tokens
    "11e76ab8-e318-52ad-a79c-16881b9f9f10",  # 633 tokens
    "150675e9-325e-5516-98bd-a60feb2ef05b",  # 607 tokens
    "bc60ada3-ea6c-5b07-85e6-f2cf9bd48b91",  # 565 tokens
    "57956a88-6984-5b00-aebd-6e093987fc67",  # 563 tokens
    "6c3b0f0c-8f0e-59a7-90f6-64cd9e3912fb",  # 545 tokens
]


def get_bedrock_client():
    """Get AWS Bedrock client with retry configuration."""
    import boto3
    from botocore.config import Config

    boto_config = Config(
        retries={'max_attempts': 20, 'mode': 'adaptive'},
        max_pool_connections=100,
        connect_timeout=30,
        read_timeout=120
    )
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
        config=boto_config
    )


def get_openai_client():
    """Get OpenAI-compatible client."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def rerun_single_sample_bedrock(
    client,
    model_id: str,
    query: str,
    tools: List[Dict],
    num_runs: int,
    temperature: float,
    max_tokens: int,
    max_workers: int,
) -> List[List[Dict]]:
    """Re-run a single sample using Bedrock."""
    from scripts.eval.generate_tool_calls import (
        generate_tool_calls_bedrock,
        xlam_tool_to_bedrock_tool,
    )

    return generate_tool_calls_bedrock(
        client=client,
        model_id=model_id,
        query=query,
        tools=tools,
        num_runs=num_runs,
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=max_workers,
    )


def rerun_single_sample_openai(
    client,
    model_id: str,
    query: str,
    tools: List[Dict],
    num_runs: int,
    temperature: float,
    max_tokens: int,
    max_workers: int,
) -> List[List[Dict]]:
    """Re-run a single sample using OpenAI-compatible API."""
    from scripts.eval.generate_tool_calls import generate_tool_calls_openai

    return generate_tool_calls_openai(
        client=client,
        model_id=model_id,
        query=query,
        tools=tools,
        num_runs=num_runs,
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=max_workers,
    )


def process_sample(
    client,
    model_id: str,
    provider: str,
    result: Dict,
    temperature: float,
    max_tokens: int,
    num_runs: int,
    max_workers: int,
) -> Dict:
    """Process a single sample and return updated result."""
    sample_id = result.get("sample_id", "unknown")
    query = result.get("query", "")
    tools = result.get("tools", [])

    try:
        if provider == "bedrock":
            new_runs = rerun_single_sample_bedrock(
                client, model_id, query, tools,
                num_runs, temperature, max_tokens, max_workers
            )
        else:
            new_runs = rerun_single_sample_openai(
                client, model_id, query, tools,
                num_runs, temperature, max_tokens, max_workers
            )

        # Update result
        result["generated_runs"] = new_runs
        result["num_valid_runs"] = sum(1 for r in new_runs if r)
        result["rerun_metadata"] = {
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat(),
        }

        return result, None
    except Exception as e:
        return result, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Re-run specific Toucan samples with higher max_tokens"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment directory (e.g., llm_gen_results/toucan/generations-xxx)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID for inference"
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        help="Comma-separated sample IDs to re-run"
    )
    parser.add_argument(
        "--high-token-only",
        action="store_true",
        help="Re-run only the 14 high-token samples (GT > 500 tokens)"
    )
    parser.add_argument(
        "--gt-threshold",
        type=int,
        default=500,
        help="Ground truth token threshold for auto-detection (default: 500)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Max tokens for generation (default: 8000)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs per sample (default: 10)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Max parallel workers per sample (default: 10)"
    )
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=1,
        help="Number of samples to process in parallel (default: 1)"
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs='+',
        help="Specific temperatures to re-run (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Error: Directory not found: {exp_dir}")
        return 1

    # Determine sample IDs to re-run
    if args.high_token_only:
        sample_ids = HIGH_TOKEN_SAMPLE_IDS
        print(f"Using predefined high-token samples: {len(sample_ids)} samples")
    elif args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",")]
        print(f"Using specified sample IDs: {len(sample_ids)} samples")
    else:
        print("Error: Specify --sample-ids or --high-token-only")
        return 1

    # Get provider
    provider = get_provider(args.model_id)
    display_name = get_display_name(args.model_id)

    print(f"\n{'='*70}")
    print(f"Toucan Sample Re-run with max_tokens={args.max_tokens}")
    print(f"{'='*70}")
    print(f"Experiment: {exp_dir.name}")
    print(f"Model: {display_name} ({args.model_id})")
    print(f"Provider: {provider}")
    print(f"Samples to re-run: {len(sample_ids)}")
    print(f"Sample IDs: {', '.join(sample_ids[:3])}{'...' if len(sample_ids) > 3 else ''}")
    print(f"Runs per sample: {args.num_runs}")
    print(f"Max workers: {args.max_workers}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("[DRY RUN MODE - No changes will be made]\n")

    # Find temperature directories
    temp_dirs = sorted(exp_dir.glob("run_*_temp_*"))
    if not temp_dirs:
        print(f"Error: No temperature directories found in {exp_dir}")
        return 1

    print(f"Found {len(temp_dirs)} temperature directories")

    # Filter by specified temperatures
    if args.temperatures:
        print(f"Filtering to temperatures: {args.temperatures}")

    # Initialize client
    if not args.dry_run:
        print(f"\nInitializing {provider} client...")
        if provider == "bedrock":
            client = get_bedrock_client()
        else:
            client = get_openai_client()
    else:
        client = None

    # Process each temperature directory
    total_updated = 0
    total_errors = 0

    for temp_dir in temp_dirs:
        # Extract temperature from directory name
        temp_parts = temp_dir.name.split('temp_')[-1].split('_')[:2]
        try:
            temperature = float(f"{temp_parts[0]}.{temp_parts[1]}")
        except (ValueError, IndexError):
            temperature = 0.0

        # Skip if not in specified temperatures
        if args.temperatures and temperature not in args.temperatures:
            continue

        print(f"\nTemperature {temperature}:")

        # Load all_results.json
        all_results_file = temp_dir / "all_results.json"
        if not all_results_file.exists():
            print(f"  Warning: all_results.json not found, skipping")
            continue

        with open(all_results_file) as f:
            all_results = json.load(f)

        # Find samples to update
        results_to_update = []
        for result in all_results.get("results", []):
            sample_id = result.get("sample_id", "")
            if sample_id in sample_ids:
                results_to_update.append(result)

        if not results_to_update:
            print(f"  No matching samples found")
            continue

        print(f"  Found {len(results_to_update)} samples to re-run")

        if args.dry_run:
            for result in results_to_update:
                print(f"    {result.get('sample_id', 'unknown')}: [DRY RUN] Would re-run")
            continue

        # Process samples
        updated_count = 0
        error_count = 0

        for result in results_to_update:
            sample_id = result.get("sample_id", "unknown")
            print(f"    {sample_id}: Running...", end=" ", flush=True)

            updated_result, error = process_sample(
                client=client,
                model_id=args.model_id,
                provider=provider,
                result=result,
                temperature=temperature,
                max_tokens=args.max_tokens,
                num_runs=args.num_runs,
                max_workers=args.max_workers,
            )

            if error:
                print(f"ERROR - {error}")
                error_count += 1
            else:
                valid = updated_result.get("num_valid_runs", 0)
                print(f"{valid}/{args.num_runs} valid")
                updated_count += 1

        # Update all_results.json
        if "metadata" not in all_results:
            all_results["metadata"] = {}
        all_results["metadata"]["rerun_timestamp"] = datetime.now().isoformat()
        all_results["metadata"]["rerun_max_tokens"] = args.max_tokens
        all_results["metadata"]["rerun_sample_ids"] = [
            r.get("sample_id") for r in results_to_update
        ]

        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"  Updated {updated_count} samples, {error_count} errors")
        total_updated += updated_count
        total_errors += error_count

    print(f"\n{'='*70}")
    print("Re-run Complete!")
    print(f"{'='*70}")
    print(f"Total samples updated: {total_updated}")
    print(f"Total errors: {total_errors}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
