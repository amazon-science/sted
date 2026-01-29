#!/usr/bin/env python
"""
Re-run specific samples from existing experiment results.

Reads system_prompt and modified_prompt from existing sample files,
runs inference with new max_tokens, and replaces the responses field.

Usage:
    python rerun_samples.py --experiment-dir llm_gen_results/sharegpt/generations-claude-3.5-haiku-20260102 \
        --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0 \
        --sample-ids sample_000,sample_003,sample_007 \
        --max-tokens 8000

    # Or use sample_token_counts.json to get high-token samples automatically:
    python rerun_samples.py --experiment-dir llm_gen_results/sharegpt/generations-claude-3.5-haiku-20260102 \
        --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0 \
        --high-token-only \
        --max-tokens 8000
"""

import argparse
import json
import os
import sys
import concurrent.futures
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.eval.generate_structured_outputs import run_inference


def process_single_sample(sample_file, model_id, temperature, max_tokens, run_num, max_workers):
    """Process a single sample file and return updated data."""
    sample_id = sample_file.stem

    try:
        with open(sample_file) as f:
            sample_data = json.load(f)

        system_prompt = sample_data.get('system_prompt', '')
        user_prompt = sample_data.get('modified_prompt', sample_data.get('original_prompt', ''))

        # Use run_inference from generate_structured_outputs.py
        new_responses = run_inference(
            model_id=model_id,
            user_prompt=user_prompt,
            system_prompts=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            run_num=run_num,
            max_workers=max_workers
        )

        # Count valid responses
        valid = sum(1 for r in new_responses if r and (isinstance(r, list) or (isinstance(r, dict) and '_extraction_error' not in r and '_api_error' not in r)))

        # Update sample data
        sample_data['responses'] = new_responses
        sample_data['metadata']['max_tokens'] = max_tokens
        sample_data['metadata']['rerun_timestamp'] = datetime.now().isoformat()

        # Save
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)

        return sample_id, valid, run_num, None
    except Exception as e:
        return sample_id, 0, run_num, str(e)


def main():
    parser = argparse.ArgumentParser(description="Re-run specific samples with new max_tokens")
    parser.add_argument("--experiment-dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID for inference")
    parser.add_argument("--sample-ids", type=str, help="Comma-separated sample IDs to re-run")
    parser.add_argument("--high-token-only", action="store_true", help="Re-run only high-token samples from sample_token_counts.json")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Max tokens for generation")
    parser.add_argument("--run-num", type=int, default=10, help="Number of runs per sample")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers per sample for inference")
    parser.add_argument("--sample-workers", type=int, default=1, help="Number of samples to process in parallel (default: 1 = sequential)")
    parser.add_argument("--temperatures", type=float, nargs='+', help="Specific temperatures to re-run (e.g., 0.7 0.8 0.9 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")

    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Error: Directory not found: {exp_dir}")
        return

    # Get sample IDs to re-run
    if args.high_token_only:
        token_file = exp_dir / "sample_token_counts.json"
        if not token_file.exists():
            print(f"Error: sample_token_counts.json not found in {exp_dir}")
            return
        with open(token_file) as f:
            token_data = json.load(f)
        sample_ids = token_data.get("high_token_samples", [])
        print(f"High-token samples: {len(sample_ids)}")
    elif args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",")]
    else:
        print("Error: Specify --sample-ids or --high-token-only")
        return

    print(f"\n{'='*60}")
    print(f"Re-running samples with max_tokens={args.max_tokens}")
    print(f"{'='*60}")
    print(f"Experiment: {exp_dir.name}")
    print(f"Model: {args.model_id}")
    print(f"Samples to re-run: {len(sample_ids)}")
    print(f"Sample IDs: {', '.join(sample_ids[:5])}{'...' if len(sample_ids) > 5 else ''}")
    print(f"Sample workers: {args.sample_workers}, Inference workers per sample: {args.max_workers}")
    print(f"Total max concurrent API calls: {args.sample_workers * args.max_workers}")
    print(f"{'='*60}\n")

    # Find all temperature directories (support multiple naming patterns)
    temp_dirs = sorted(exp_dir.glob("llm_gen_results_*_temp_*"))
    if not temp_dirs:
        # Try alternative pattern: run_*_temp_*
        temp_dirs = sorted(exp_dir.glob("run_*_temp_*"))
    if not temp_dirs:
        for subdir in exp_dir.iterdir():
            if subdir.is_dir():
                temp_dirs.extend(sorted(subdir.glob("llm_gen_results_*_temp_*")))
                if not temp_dirs:
                    temp_dirs.extend(sorted(subdir.glob("run_*_temp_*")))

    print(f"Found {len(temp_dirs)} temperature directories")

    # Filter by specified temperatures if provided
    if args.temperatures:
        print(f"Filtering to temperatures: {args.temperatures}\n")
    else:
        print()

    # Process each temperature directory
    for temp_dir in temp_dirs:
        # Extract temperature from directory name
        temp_parts = temp_dir.name.split('temp_')[-1].split('_')[:2]
        temperature = float(f"{temp_parts[0]}.{temp_parts[1]}") if len(temp_parts) >= 2 else 0.0

        # Skip if not in specified temperatures
        if args.temperatures and temperature not in args.temperatures:
            continue

        print(f"\nTemperature {temperature}:")

        # Collect sample files to process
        sample_files_to_process = []
        for sample_id in sample_ids:
            sample_file = temp_dir / f"{sample_id}.json"
            if not sample_file.exists():
                print(f"  {sample_id}: NOT FOUND")
                continue
            if args.dry_run:
                print(f"  {sample_id}: [DRY RUN] Would re-run")
                continue
            sample_files_to_process.append(sample_file)

        if not sample_files_to_process or args.dry_run:
            continue

        # Process samples in parallel
        if args.sample_workers > 1:
            print(f"  Processing {len(sample_files_to_process)} samples in parallel (workers={args.sample_workers})...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.sample_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_sample,
                        sample_file, args.model_id, temperature,
                        args.max_tokens, args.run_num, args.max_workers
                    ): sample_file.stem for sample_file in sample_files_to_process
                }

                for future in concurrent.futures.as_completed(futures):
                    sample_id, valid, run_num, error = future.result()
                    if error:
                        print(f"  {sample_id}: ERROR - {error}")
                    else:
                        print(f"  {sample_id}: {valid}/{run_num} valid")
        else:
            # Sequential processing
            for sample_file in sample_files_to_process:
                sample_id = sample_file.stem
                print(f"  {sample_id}: Running...", end=" ", flush=True)
                sample_id, valid, run_num, error = process_single_sample(
                    sample_file, args.model_id, temperature,
                    args.max_tokens, args.run_num, args.max_workers
                )
                if error:
                    print(f"ERROR - {error}")
                else:
                    print(f"{valid}/{run_num} valid")

        # Update all_results.json if exists
        all_results_file = temp_dir / "all_results.json"
        if all_results_file.exists() and not args.dry_run:
            with open(all_results_file) as f:
                all_results = json.load(f)

            for result in all_results.get('results', []):
                sample_id = result.get('sample_id', '')
                if sample_id in sample_ids:
                    sample_file = temp_dir / f"{sample_id}.json"
                    if sample_file.exists():
                        with open(sample_file) as f:
                            sample_data = json.load(f)
                        result['responses'] = sample_data.get('responses', [])
                        result['metadata'] = sample_data.get('metadata', {})

            with open(all_results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Re-run complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
