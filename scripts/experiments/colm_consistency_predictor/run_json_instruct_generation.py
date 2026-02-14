#!/usr/bin/env python3
"""
Run LLM generation on json-instruct dataset for COLM consistency predictor experiments.

Cost-efficient configuration:
- 3 temperatures (0.0, 0.5, 1.0) instead of 11
- 5 runs instead of 10
- 3 models (haiku, gpt-4.1-mini, llama) instead of 21
- 500 samples with high schema complexity

Usage:
    python run_json_instruct_generation.py --num-samples 500 --dry-run
    python run_json_instruct_generation.py --num-samples 500
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from sted.model_config import MODEL_REGISTRY, get_display_name


def load_json_instruct(dataset_path: str) -> List[Dict]:
    """Load json-instruct dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def filter_complex_samples(data: List[Dict], min_complexity: int = 2) -> List[Dict]:
    """Filter samples with complex schemas (nested objects, arrays)."""
    complex_samples = []

    for sample in data:
        schema_str = sample.get('schema', '')
        complexity = 0

        # Count complexity indicators
        if '"type": "array"' in schema_str or '"type":"array"' in schema_str:
            complexity += 1
        if schema_str.count('"type": "object"') > 1 or schema_str.count('"type":"object"') > 1:
            complexity += 1
        if schema_str.count('"properties"') > 1:
            complexity += 1
        if '"items"' in schema_str:
            complexity += 1

        if complexity >= min_complexity:
            complex_samples.append(sample)

    return complex_samples


def convert_to_sharegpt_format(samples: List[Dict], output_dir: str) -> str:
    """Convert json-instruct samples to ShareGPT format for existing pipeline."""
    converted = []

    for i, sample in enumerate(samples):
        # Create ShareGPT-style conversation
        system_prompt = """You are a helpful assistant that generates valid JSON data conforming to provided schemas.
Always respond with ONLY valid JSON that matches the schema exactly. No explanations or markdown."""

        user_prompt = sample.get('input', '')
        schema = sample.get('schema', '')
        expected_output = sample.get('output', '')

        # Combine prompt with schema
        full_user_prompt = f"{user_prompt}\n\nJSON Schema:\n{schema}"

        converted.append({
            'id': f'json_instruct_{i}',
            'conversations': [
                {'from': 'system', 'value': system_prompt},
                {'from': 'human', 'value': full_user_prompt},
                {'from': 'gpt', 'value': expected_output}
            ],
            'original_schema': schema,
            'original_input': sample.get('input', ''),
            'task': sample.get('task', 'generation')
        })

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'all_conversations.json')

    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {len(converted)} samples to {output_path}")
    return output_dir


def run_temperature_experiment(
    data_dir: str,
    output_dir: str,
    model_id: str,
    temperatures: List[float],
    num_runs: int,
    num_samples: int,
    dry_run: bool = False
):
    """Run temperature experiment using existing generate_structured_outputs.py."""

    script_path = Path(__file__).parent.parent.parent / 'eval' / 'generate_structured_outputs.py'

    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Running {model_id} at temperature {temp}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(script_path),
            '--data-dir', data_dir,
            '--output-dir', output_dir,
            '--model-id', model_id,
            '--temperature', str(temp),
            '--run-num', str(num_runs),
            '--sample-limit', str(num_samples),
            '--include-schema',
            '--max-tokens', '4096'
        ]

        print(f"Command: {' '.join(cmd)}")

        if not dry_run:
            subprocess.run(cmd)
        else:
            print("[DRY RUN] Would execute above command")


def main():
    parser = argparse.ArgumentParser(description="Run json-instruct generation experiment")
    parser.add_argument('--dataset-path', type=str,
                        default='research/datasets/Maxscha_json-instruct-generation.json',
                        help='Path to json-instruct dataset')
    parser.add_argument('--output-dir', type=str, default='llm_gen_results/json_instruct',
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of samples to process')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of runs per sample per temperature')
    parser.add_argument('--min-complexity', type=int, default=2,
                        help='Minimum schema complexity score to include')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['us.anthropic.claude-3-5-haiku-20241022-v1:0'],
                        help='Models to use')
    parser.add_argument('--temperatures', type=float, nargs='+',
                        default=[0.0, 0.5, 1.0],
                        help='Temperatures to test')

    args = parser.parse_args()

    # Load and filter dataset
    print(f"Loading dataset from {args.dataset_path}...")
    data = load_json_instruct(args.dataset_path)
    print(f"Total samples: {len(data)}")

    # Filter for complex schemas
    complex_data = filter_complex_samples(data, args.min_complexity)
    print(f"Samples with complexity >= {args.min_complexity}: {len(complex_data)}")

    # Limit samples
    samples_to_use = complex_data[:args.num_samples]
    print(f"Using {len(samples_to_use)} samples")

    # Convert to ShareGPT format
    converted_dir = os.path.join(args.output_dir, 'converted_data', 'json_instruct')
    convert_to_sharegpt_format(samples_to_use, converted_dir)

    # Calculate API calls
    total_calls = len(samples_to_use) * len(args.temperatures) * args.num_runs * len(args.models)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Samples: {len(samples_to_use)}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Runs per temp: {args.num_runs}")
    print(f"Models: {args.models}")
    print(f"Total API calls: {total_calls:,}")
    print(f"Estimated cost: ~${total_calls * 0.001:.2f} (assuming $0.001/call avg)")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[DRY RUN MODE] No API calls will be made")

    # Run experiments for each model
    for model_id in args.models:
        model_name = get_display_name(model_id)
        print(f"\nStarting experiment for {model_name}...")

        run_temperature_experiment(
            data_dir=os.path.dirname(converted_dir),
            output_dir=args.output_dir,
            model_id=model_id,
            temperatures=args.temperatures,
            num_runs=args.num_runs,
            num_samples=len(samples_to_use),
            dry_run=args.dry_run
        )

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
