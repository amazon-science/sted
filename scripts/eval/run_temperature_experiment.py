#!/usr/bin/env python
"""
Temperature-Stability Correlation Experiment

This script runs a comprehensive experiment to analyze the relationship between
temperature settings and standard deviation of mean similarity in LLM generations.
It focuses on how temperature affects the variability of similarity scores.

Usage:
    python run_temperature_experiment.py --data-dir extracted_sharegpt_data --output-dir ./temperature_experiment
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Any, Tuple
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from datetime import datetime


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

def run_generation(data_dir: str, output_dir: str, temperature: float, run_num: int, include_schema: bool, model_id: str, sample_limit: int=40, max_tokens: int=3000) -> str:
    """
    Run LLM generation with specified parameters.
    
    Args:
        data_dir: Directory containing the data files
        output_dir: Directory to save generation results
        temperature: Temperature setting for generation
        run_num: Number of runs to perform
        include_schema: Whether to include schema in the prompt
        
    Returns:
        Path to the generated results file
    """
    cmd = [
        "python", "scripts/eval/generate_structured_outputs.py",
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
    
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=project_root, env=env)
    
    if result.returncode != 0:
        print(f"Subprocess failed with return code {result.returncode}")
        print(f"Subprocess stdout: {result.stdout}")
        print(f"Subprocess stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    
    print(f"Subprocess stdout: {result.stdout}")
    print(f"Subprocess stderr: {result.stderr}")
    
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
    
    # get the string of current time 
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generations_dir = os.path.join(args.output_dir, f"generations-{current_time}")
    visualizations_dir = os.path.join(args.output_dir, f"visualizations-{current_time}")
    os.makedirs(generations_dir, exist_ok=True)
    
    # Define temperatures to test
    if args.temperatures:
        temperatures = args.temperatures
    else:
        temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        #temperatures = np.arange(0.0, 1.0, 0.05)
    
    # Define similarity methods to use
    methods = ["ted", "bertscore", "deepdiff"]
    
    results = []
    
    # Run generation and evaluation for each temperature
    for temp in tqdm(temperatures, desc="Processing tempeyratures"):
        print(f"\n=== Processing Temperature {temp} ===")
        
        # Check if we already have generation results for this temperature and this model
        temp_str = f"temp_{temp:.2f}".replace('.', '_')
        existing_results = list(Path(generations_dir).glob(f"llm_gen_results_*{temp_str}*"))
        
        gen_results_file = None
                
        if existing_results:
            result_dir = sorted(existing_results, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            print(f"Found existing results for temperature {temp}. Using {result_dir}")
            gen_results_file = str(result_dir / "all_results.json")
            print(f"gen_results_file: {gen_results_file}")
            
        
        if gen_results_file is None or not os.path.exists(gen_results_file) or args.force_regenerate:
            # Run generation
            gen_results_file = run_generation(
                data_dir=args.data_dir,
                output_dir=generations_dir,
                temperature=temp,
                run_num=args.run_num,
                include_schema=args.include_schema,
                model_id=args.model_id,
                sample_limit=args.sample_limit,
                max_tokens=args.max_tokens
            )
            
def main():
    parser = argparse.ArgumentParser(description="Run temperature vs standard deviation correlation experiment.")
    
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the data files.")
    parser.add_argument("--output-dir", type=str, default="./temperature_experiment", help="Directory to save experiment results.")
    parser.add_argument("--temperatures", type=float, nargs="+", help="List of temperatures to test. Default: 0.0 to 1.0 in 0.1 increments.")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regeneration even if results already exist.")
    parser.add_argument("--run-num", type=int, default=10, help="Number of runs per temperature.")
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt.")
    parser.add_argument("--model-id", type=str, default="us.anthropic.claude-3-5-sonnet-20241022-v2:0", help="Model id")
    parser.add_argument("--sample-limit", type=int, default=0, help="Limit the number of samples to process.")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Limit the number of samples to process.")
    
    args = parser.parse_args()
    run_experiment(args)
    
if __name__ == "__main__":
    main()