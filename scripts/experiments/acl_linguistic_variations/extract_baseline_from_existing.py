#!/usr/bin/env python3
"""
ACL 2026 Paper: Extract Baseline Data from Existing Results

Leverages existing llm_gen_results to skip Phase 1 entirely.
Extracts baseline consistency metrics for stratified samples.

Existing data:
- 20 models × 11 temperatures × 1006 samples × 10 runs
- Can derive all Phase 1 baseline estimates without new API calls

This saves ~51,300 API calls!

Usage:
    python extract_baseline_from_existing.py --samples data/acl_stratified/stratified_samples.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any
import statistics

import numpy as np


# Model mapping: ACL model IDs to existing result directory names
MODEL_MAPPING = {
    "us.anthropic.claude-sonnet-4-20250514-v1:0": "claude-sonnet-4",
    "openai/gpt-4.1-mini": "gpt-4.1-mini",
    "us.meta.llama3-3-70b-instruct-v1:0": "llama-3.3-70b",
    "qwen.qwen3-235b-a22b-2507-v1:0": "qwen3-235b-a22b",
    "google/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    # Additional models available
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": "claude-3.5-sonnet",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": "claude-3.5-haiku",
    "us.amazon.nova-lite-v1:0": "nova2-lite",
}


def find_model_results_dir(results_base: Path, model_pattern: str) -> Path:
    """Find the results directory matching the model pattern."""
    for d in results_base.iterdir():
        if d.is_dir() and model_pattern in d.name:
            return d
    return None


def load_existing_results(results_dir: Path, temperature: float) -> Dict[str, Dict]:
    """
    Load existing results for a specific temperature.

    Returns dict mapping sample_id -> sample data with generated_runs
    """
    # Find the run directory for this temperature
    temp_str = f"temp_{int(temperature)}_{int((temperature % 1) * 100):02d}"

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if temp_str in run_dir.name:
            results_file = run_dir / "all_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)

                # Index by sample_id
                results_by_id = {}
                for sample in data.get('results', []):
                    sample_id = sample.get('sample_id')
                    if sample_id:
                        results_by_id[sample_id] = sample
                return results_by_id

    return {}


def compute_consistency(runs: List[List[Dict]]) -> Dict[str, float]:
    """Compute consistency metrics from generated runs."""
    valid_runs = [r for r in runs if r]

    if len(valid_runs) < 2:
        return {
            'validity_rate': len(valid_runs) / len(runs) if runs else 0,
            'c_mean': 0.0 if len(valid_runs) < 2 else 1.0,
            'c_std': 0.0,
            'num_valid': len(valid_runs)
        }

    similarities = []
    for i in range(len(valid_runs)):
        for j in range(i + 1, len(valid_runs)):
            tools1 = set(tc.get('name', '') for tc in valid_runs[i])
            tools2 = set(tc.get('name', '') for tc in valid_runs[j])

            if not tools1 and not tools2:
                sim = 1.0
            elif not tools1 or not tools2:
                sim = 0.0
            else:
                sim = len(tools1 & tools2) / len(tools1 | tools2)
            similarities.append(sim)

    return {
        'validity_rate': len(valid_runs) / len(runs),
        'c_mean': statistics.mean(similarities),
        'c_std': statistics.stdev(similarities) if len(similarities) > 1 else 0.0,
        'num_valid': len(valid_runs)
    }


def extract_baseline(model_id: str, samples_file: str, results_base: str,
                     temperatures: List[float] = None, output_dir: str = None) -> Dict:
    """
    Extract baseline consistency from existing results.

    No API calls needed - purely data extraction!
    """
    if temperatures is None:
        temperatures = [0.3, 0.7, 1.0]

    if output_dir is None:
        output_dir = "results/acl_linguistic/phase1_baseline"

    results_base = Path(results_base)

    # Find model pattern
    model_pattern = MODEL_MAPPING.get(model_id)
    if not model_pattern:
        # Try to extract from model_id
        model_pattern = model_id.split("/")[-1].split(":")[0].lower()

    print("=" * 70)
    print(f"EXTRACTING BASELINE: {model_pattern}")
    print("=" * 70)

    # Find results directory
    model_results_dir = find_model_results_dir(results_base / "toucan", model_pattern)
    if not model_results_dir:
        print(f"ERROR: No results found for {model_pattern}")
        print(f"Available: {[d.name for d in (results_base / 'toucan').iterdir() if d.is_dir()]}")
        return None

    print(f"Found results: {model_results_dir.name}")

    # Load stratified samples
    with open(samples_file) as f:
        samples_data = json.load(f)

    # Get sample IDs we need
    target_sample_ids = set()
    samples_by_id = {}
    for level in ['simple', 'medium', 'complex']:
        for sample in samples_data['samples'].get(level, []):
            target_sample_ids.add(sample['id'])
            samples_by_id[sample['id']] = sample

    print(f"Target samples: {len(target_sample_ids)}")

    results = []
    baseline_estimates = {}

    for temp in temperatures:
        print(f"\nLoading T={temp}...")
        existing_results = load_existing_results(model_results_dir, temp)
        print(f"  Found {len(existing_results)} samples in existing results")

        matched = 0
        for sample_id in target_sample_ids:
            if sample_id not in existing_results:
                continue

            matched += 1
            existing_sample = existing_results[sample_id]
            runs = existing_sample.get('generated_runs', [])

            # Compute metrics
            metrics = compute_consistency(runs)

            # Get sample metadata
            sample_meta = samples_by_id.get(sample_id, {})
            complexity = sample_meta.get('schema_complexity', {})

            results.append({
                'sample_id': sample_id,
                'temperature': temp,
                'schema_complexity': complexity,
                'question': existing_sample.get('query', ''),
                'metrics': metrics,
                'num_runs': len(runs),
                'source': 'existing_results'
            })

            # Store baseline estimate at T=0.7 for Phase 2 stratification
            if temp == 0.7:
                baseline_estimates[sample_id] = metrics['c_mean']

        print(f"  Matched: {matched}/{len(target_sample_ids)}")

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    # Group by complexity
    by_complexity = defaultdict(list)
    for r in results:
        if 'metrics' in r:
            level = r['schema_complexity'].get('complexity_level', 'unknown')
            by_complexity[level].append(r['metrics']['c_mean'])

    print("\nConsistency by Schema Complexity:")
    for level in ['simple', 'medium', 'complex']:
        values = by_complexity[level]
        if values:
            print(f"  {level}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, n={len(values)}")

    # Group by temperature
    by_temp = defaultdict(list)
    for r in results:
        if 'metrics' in r:
            by_temp[r['temperature']].append(r['metrics']['c_mean'])

    print("\nConsistency by Temperature:")
    for temp in temperatures:
        values = by_temp[temp]
        if values:
            print(f"  T={temp}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

    # Classify into difficulty strata (for Phase 2)
    difficulty_strata = {'difficult': [], 'medium': [], 'easy': []}
    for sample_id, baseline in baseline_estimates.items():
        if baseline < 0.7:
            difficulty_strata['difficult'].append(sample_id)
        elif baseline >= 0.9:
            difficulty_strata['easy'].append(sample_id)
        else:
            difficulty_strata['medium'].append(sample_id)

    print("\nBaseline Difficulty Distribution (T=0.7):")
    for level in ['difficult', 'medium', 'easy']:
        print(f"  {level}: {len(difficulty_strata[level])} samples")

    # Save results
    display_name = model_pattern.replace("-", "_").lower()
    output_path = Path(output_dir) / f"{display_name}_baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'model_id': model_id,
            'model_pattern': model_pattern,
            'source_dir': str(model_results_dir),
            'temperatures': temperatures,
            'num_samples': len(target_sample_ids),
            'timestamp': datetime.now().isoformat(),
            'purpose': 'Phase 1 baseline extracted from existing results (NO API CALLS)',
            'api_calls_saved': len(target_sample_ids) * len(temperatures) * 10
        },
        'summary': {
            'by_complexity': {
                level: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'n': len(values)
                }
                for level, values in by_complexity.items() if values
            },
            'by_temperature': {
                str(temp): {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
                for temp, values in by_temp.items() if values
            }
        },
        'baseline_estimates': baseline_estimates,
        'difficulty_strata': difficulty_strata,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved baseline results to {output_path}")
    print(f"API calls saved: {output_data['metadata']['api_calls_saved']:,}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Extract baseline from existing llm_gen_results (saves Phase 1 API calls)'
    )
    parser.add_argument('--model', type=str, help='Model ID to extract')
    parser.add_argument('--all-models', action='store_true',
                        help='Extract for all ACL models')
    parser.add_argument('--samples', type=str,
                        default='data/acl_stratified/stratified_samples.json')
    parser.add_argument('--results-base', type=str,
                        default='llm_gen_results',
                        help='Base directory for existing results')
    parser.add_argument('--temperatures', type=float, nargs='+',
                        default=[0.3, 0.7, 1.0])
    parser.add_argument('--output-dir', type=str,
                        default='results/acl_linguistic/phase1_baseline')

    args = parser.parse_args()

    if not Path(args.samples).exists():
        print(f"Error: Samples file not found: {args.samples}")
        print("Run prepare_stratified_data.py first.")
        sys.exit(1)

    if args.all_models:
        models = list(MODEL_MAPPING.keys())[:5]  # Primary ACL models
    elif args.model:
        models = [args.model]
    else:
        print("Please specify --model or --all-models")
        sys.exit(1)

    total_saved = 0
    for model_id in models:
        result = extract_baseline(
            model_id=model_id,
            samples_file=args.samples,
            results_base=args.results_base,
            temperatures=args.temperatures,
            output_dir=args.output_dir
        )
        if result:
            total_saved += result['metadata']['api_calls_saved']

    print("\n" + "=" * 70)
    print(f"TOTAL API CALLS SAVED: {total_saved:,}")
    print("=" * 70)


if __name__ == '__main__':
    main()
