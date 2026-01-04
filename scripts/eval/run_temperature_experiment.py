#!/usr/bin/env python
"""
Temperature-Stability Correlation Experiment

This script runs a comprehensive experiment to analyze the relationship between
temperature settings and standard deviation of mean similarity in LLM generations.
It focuses on how temperature affects the variability of similarity scores.

Supports two modes:
1. Structured output generation (default) - for ShareGPT-style data
2. Tool calling generation - for Toucan-style tool calling data

Usage:
    # Structured outputs (ShareGPT)
    python run_temperature_experiment.py --data-dir sharegpt_data --output-dir ./temperature_experiment

    # Tool calling (Toucan)
    python run_temperature_experiment.py --mode tool-calling --dataset-path toucan_data/toucan_tool_calls.json --output-dir ./tool_calling_experiment
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

from tqdm import tqdm


def load_generation_results(file_path: str) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Load generation results and extract ground truth and generated data.

    Args:
        file_path: Path to generation results JSON file

    Returns:
        Tuple of (ground_truth_list, generated_responses_list)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    ground_truth_list = []
    generated_responses_list = []

    for result in data.get('results', []):
        ground_truth = result.get('ground_truth', {})
        responses = result.get('responses', [])

        # Filter out empty responses
        valid_responses = [r for r in responses if r]

        if ground_truth and valid_responses:
            ground_truth_list.append(ground_truth)
            generated_responses_list.append(valid_responses)

    return ground_truth_list, generated_responses_list


def run_tool_calling_generation(
    dataset_path: str,
    output_dir: str,
    temperature: float,
    run_num: int,
    model_id: str,
    sample_limit: int = 100,
    max_tokens: int = 1024,
    start_idx: int = 0,
    max_workers: int = 10
) -> str:
    """
    Run tool calling generation using generate_tool_calls.py.

    Args:
        dataset_path: Path to Toucan dataset JSON file
        output_dir: Directory to save generation results
        temperature: Temperature setting for generation
        run_num: Number of runs to perform
        model_id: Model ID to use for generation
        sample_limit: Maximum number of samples to process
        max_tokens: Maximum tokens for LLM generation
        start_idx: Starting index in dataset
        max_workers: Maximum parallel workers for inference

    Returns:
        Path to the generated results file
    """
    cmd = [
        "python3", "scripts/eval/generate_tool_calls.py",
        "--dataset-path", dataset_path,
        "--dataset-type", "toucan",
        "--output-dir", output_dir,
        "--temperature", str(temperature),
        "--num-runs", str(run_num),
        "--num-samples", str(sample_limit),
        "--model", model_id,
        "--max-tokens", str(max_tokens),
        "--start-idx", str(start_idx),
        "--max-workers", str(max_workers)
    ]

    print(f"Running tool calling generation with temperature {temperature}...")
    print(f"Command: {' '.join(cmd)}")

    # Set environment with PYTHONPATH
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env['PYTHONPATH'] = project_root

    result = subprocess.run(cmd, check=False, capture_output=True, cwd=project_root, env=env)

    # Decode with error handling for non-UTF8 responses
    stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
    stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''

    if result.returncode != 0:
        print(f"Subprocess failed with return code {result.returncode}")
        print(f"Subprocess stdout: {stdout}")
        print(f"Subprocess stderr: {stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, stdout, stderr)

    print(f"Subprocess stdout: {stdout[-2000:]}")  # Last 2000 chars
    if stderr:
        print(f"Subprocess stderr: {stderr[-1000:]}")

    # Find the most recent results directory for this temperature
    temp_str = f"temp_{temperature:.2f}".replace('.', '_')
    result_dirs = list(Path(output_dir).glob(f"run_*{temp_str}*"))

    if not result_dirs:
        raise FileNotFoundError(f"No results found for temperature {temperature}")

    # Sort by creation time (most recent first)
    result_dir = sorted(result_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    results_file = result_dir / "all_results.json"

    return str(results_file)


def run_generation(data_dir: str, output_dir: str, temperature: float, run_num: int, include_schema: bool, model_id: str, sample_limit: int = 40, max_tokens: int = 3000) -> str:
    """
    Run LLM generation with specified parameters.
    
    Args:
        data_dir: Directory containing the data files
        output_dir: Directory to save generation results
        temperature: Temperature setting for generation
        run_num: Number of runs to perform
        include_schema: Whether to include schema in the prompt
        model_id: Model ID to use for generation
        sample_limit: Maximum number of samples to process
        max_tokens: Maximum tokens for LLM generation
        
    Returns:
        Path to the generated results file
    """
    cmd = [
        "python3", "scripts/eval/generate_structured_outputs.py",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--temperature", str(temperature),
        "--run-num", str(run_num),
        "--sample-limit", str(sample_limit),
        "--model-id", model_id,
        "--max-tokens", str(max_tokens)
    ]
    
    if include_schema:
        cmd.append("--include-schema")
    
    print(f"Running generation with temperature {temperature}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Set environment with PYTHONPATH
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env['PYTHONPATH'] = project_root
    
    result = subprocess.run(cmd, check=False, capture_output=True, cwd=project_root, env=env)

    # Decode with error handling for non-UTF8 responses
    stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
    stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''

    if result.returncode != 0:
        print(f"Subprocess failed with return code {result.returncode}")
        print(f"Subprocess stdout: {stdout}")
        print(f"Subprocess stderr: {stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, stdout, stderr)

    print(f"Subprocess stdout: {stdout}")
    print(f"Subprocess stderr: {stderr}")
    
    # Find the most recent results directory for this temperature
    temp_str = f"temp_{temperature:.2f}".replace('.', '_')
    result_dirs = list(Path(output_dir).glob(f"llm_gen_results_*{temp_str}*"))
    
    if not result_dirs:
        raise FileNotFoundError(f"No results found for temperature {temperature}")
    
    # Sort by creation time (most recent first)
    result_dir = sorted(result_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    results_file = result_dir / "all_results.json"
    
    return str(results_file)


def run_experiment(args):
    """
    Run the temperature-standard deviation correlation experiment.

    Args:
        args: Command-line arguments
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_id.split(":")[-1] if ":" in args.model_id else args.model_id.split("/")[-1]
    generations_dir = os.path.join(args.output_dir, f"generations-{model_short}-{current_time}")
    os.makedirs(generations_dir, exist_ok=True)

    # Define temperatures to test
    if args.temperatures:
        temperatures = args.temperatures
    else:
        temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("=" * 70)
    print(f"Temperature Experiment - Mode: {args.mode}")
    print("=" * 70)
    print(f"Model: {args.model_id}")
    print(f"Temperatures: {temperatures}")
    print(f"Runs per temperature: {args.run_num}")
    print(f"Samples: {args.sample_limit}")
    print(f"Output: {generations_dir}")
    print("=" * 70)

    # Run generation for each temperature
    for temp in tqdm(temperatures, desc="Processing temperatures"):
        print(f"\n=== Processing Temperature {temp} ===")

        # Check if we already have generation results for this temperature
        temp_str = f"temp_{temp:.2f}".replace('.', '_')

        if args.mode == "tool-calling":
            existing_results = list(Path(generations_dir).glob(f"run_*{temp_str}*"))
        else:
            existing_results = list(Path(generations_dir).glob(f"llm_gen_results_*{temp_str}*"))

        gen_results_file = None

        if existing_results and not args.force_regenerate:
            result_dir = sorted(existing_results, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            print(f"Found existing results for temperature {temp}. Using {result_dir}")
            gen_results_file = str(result_dir / "all_results.json")
            print(f"gen_results_file: {gen_results_file}")

        if gen_results_file is None or not os.path.exists(gen_results_file) or args.force_regenerate:
            # Run generation based on mode
            if args.mode == "tool-calling":
                run_tool_calling_generation(
                    dataset_path=args.dataset_path,
                    output_dir=generations_dir,
                    temperature=temp,
                    run_num=args.run_num,
                    model_id=args.model_id,
                    sample_limit=args.sample_limit,
                    max_tokens=args.max_tokens,
                    start_idx=args.start_idx,
                    max_workers=args.max_workers
                )
            else:
                run_generation(
                    data_dir=args.data_dir,
                    output_dir=generations_dir,
                    temperature=temp,
                    run_num=args.run_num,
                    include_schema=args.include_schema,
                    model_id=args.model_id,
                    sample_limit=args.sample_limit,
                    max_tokens=args.max_tokens
                )

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print(f"Results saved to: {generations_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run temperature vs standard deviation correlation experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Structured output generation (ShareGPT-style data)
    python run_temperature_experiment.py --mode structured --data-dir sharegpt_data --output-dir ./temperature_experiment

    # Tool calling generation (Toucan dataset)
    python run_temperature_experiment.py --mode tool-calling --dataset-path toucan_data/toucan_tool_calls.json --output-dir ./tool_calling_experiment

    # Quick test with specific temperatures
    python run_temperature_experiment.py --mode tool-calling --dataset-path toucan_data/toucan_tool_calls.json --temperatures 0.0 0.5 1.0 --sample-limit 10 --run-num 5
        """
    )

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["structured", "tool-calling"], default="structured",
                        help="Generation mode: 'structured' for ShareGPT data, 'tool-calling' for Toucan data")

    # Shared arguments
    parser.add_argument("--output-dir", type=str, default="./temperature_experiment",
                        help="Directory to save experiment results.")
    parser.add_argument("--temperatures", type=float, nargs="+",
                        help="List of temperatures to test. Default: 0.0 to 1.0 in 0.1 increments.")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Force regeneration even if results already exist.")
    parser.add_argument("--run-num", type=int, default=10,
                        help="Number of runs per temperature.")
    parser.add_argument("--model-id", type=str, default="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                        help="Model ID to use.")
    parser.add_argument("--sample-limit", type=int, default=100,
                        help="Maximum number of samples to process.")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens for LLM generation.")

    # Structured output mode arguments
    parser.add_argument("--data-dir", type=str,
                        help="Directory containing ShareGPT data files (required for structured mode).")
    parser.add_argument("--include-schema", action="store_true", default=True,
                        help="Include JSON schema in the prompt (structured mode only). Default: True.")

    # Tool calling mode arguments
    parser.add_argument("--dataset-path", type=str, default="toucan_data/toucan_tool_calls.json",
                        help="Path to Toucan dataset JSON file (for tool-calling mode).")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting index in dataset (for tool-calling mode).")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Maximum parallel workers for inference (for tool-calling mode).")

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == "structured" and not args.data_dir:
        parser.error("--data-dir is required for structured mode")

    if args.mode == "tool-calling" and not args.dataset_path:
        parser.error("--dataset-path is required for tool-calling mode")

    run_experiment(args)


if __name__ == "__main__":
    main()
