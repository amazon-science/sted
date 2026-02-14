#!/usr/bin/env python3
"""
ACL 2026 Paper: Phase 1 - Multi-Model Baseline Collection

Collects baseline consistency metrics across multiple models WITHOUT linguistic
modifications. This establishes:
1. Per-prompt baseline difficulty (for ceiling effect analysis in Phase 2)
2. Cross-model consistency patterns (for LOMO analysis)
3. Model-specific consistency distributions

Building on KDD findings:
- KDD showed per-model R2=0.67 vs pooled R2=0.10 (model heterogeneity)
- GPT-4.1-Mini is a major outlier (SHAP correlation rho=0.07 with others)
- Schema complexity dominates (19% SHAP importance)

Usage:
    python phase1_baseline_collection.py --model MODEL_ID
    python phase1_baseline_collection.py --all-models
"""

import argparse
import json
import sys
import os
import time
import random
import concurrent.futures
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
import statistics

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sted.model_config import MODEL_REGISTRY, get_provider, get_display_name, get_max_workers

import boto3
from botocore.config import Config


# =============================================================================
# Model Configuration for ACL Paper
# =============================================================================

# Primary models (full evaluation) - selected based on KDD findings
ACL_PRIMARY_MODELS = [
    "us.anthropic.claude-sonnet-4-20250514-v1:0",      # High consistency baseline
    "openai/gpt-4.1-mini",                              # SHAP outlier (rho=0.07)
    "us.meta.llama3-3-70b-instruct-v1:0",              # Good SHAP correlation (0.53)
]

# Secondary models (validation)
ACL_SECONDARY_MODELS = [
    "qwen.qwen3-235b-a22b-2507-v1:0",                  # Good generalization
    "google/gemini-2.5-flash-lite",                     # Different architecture
]

ALL_ACL_MODELS = ACL_PRIMARY_MODELS + ACL_SECONDARY_MODELS


# =============================================================================
# API Functions (reused from run_comprehensive.py)
# =============================================================================

def get_bedrock_client(region: str = "us-east-1"):
    """Get AWS Bedrock client with retry configuration."""
    boto_config = Config(
        retries={'max_attempts': 20, 'mode': 'adaptive'},
        max_pool_connections=100,
        connect_timeout=30,
        read_timeout=120
    )
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=boto_config
    )


def clean_schema_for_bedrock(obj):
    """Remove unsupported JSON Schema keys."""
    if isinstance(obj, dict):
        return {k: clean_schema_for_bedrock(v) for k, v in obj.items() if not k.startswith('$')}
    elif isinstance(obj, list):
        return [clean_schema_for_bedrock(item) for item in obj]
    return obj


def convert_openai_to_xlam(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI format tools to xLAM format."""
    xlam_tools = []
    for t in tools:
        if isinstance(t, dict) and 'function' in t:
            func = t['function']
            params = clean_schema_for_bedrock(func.get('parameters', {}))
            xlam_tools.append({
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'parameters': params,
            })
    return xlam_tools


def xlam_tool_to_bedrock_tool(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """Convert xLAM tool to Bedrock format."""
    return {
        "toolSpec": {
            "name": tool_def.get("name", "unknown"),
            "description": tool_def.get("description", ""),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": tool_def.get("parameters", {}).get('properties', {}),
                    "required": tool_def.get("parameters", {}).get('required', []),
                }
            }
        }
    }


def call_bedrock(client, model_id: str, messages: List[Dict], tools: List[Dict],
                 temperature: float, max_tokens: int = 4000) -> List[Dict]:
    """Call Bedrock Converse API."""
    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            toolConfig={"tools": tools},
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
        )

        tool_calls = []
        content = response.get('output', {}).get('message', {}).get('content', [])
        for item in content:
            if 'toolUse' in item:
                tool_use = item['toolUse']
                tool_calls.append({
                    'name': tool_use.get('name', ''),
                    'arguments': tool_use.get('input', {})
                })
        return tool_calls
    except Exception as e:
        return []


def call_openai_compatible(model_id: str, messages: List[Dict], tools: List[Dict],
                          temperature: float, max_tokens: int = 4000) -> List[Dict]:
    """Call OpenAI-compatible API (via OpenRouter)."""
    import openai

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        return []

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    )

    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {})
            }
        })

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens
        )

        tool_calls = []
        if response.choices and response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                tool_calls.append({
                    'name': tc.function.name,
                    'arguments': json.loads(tc.function.arguments) if tc.function.arguments else {}
                })
        return tool_calls
    except Exception as e:
        return []


def generate_tool_calls(model_id: str, query: str, tools: List[Dict],
                       num_runs: int = 10, temperature: float = 0.7,
                       bedrock_client=None) -> List[List[Dict]]:
    """Generate tool calls with parallel execution."""
    provider = get_provider(model_id)
    max_workers = min(num_runs, get_max_workers(model_id))

    if provider == "bedrock":
        if bedrock_client is None:
            bedrock_client = get_bedrock_client()
        bedrock_tools = [xlam_tool_to_bedrock_tool(t) for t in tools]
        messages = [{"role": "user", "content": [{"text": query}]}]

        def single_call():
            return call_bedrock(bedrock_client, model_id, messages, bedrock_tools, temperature)
    else:
        messages = [{"role": "user", "content": query}]

        def single_call():
            return call_openai_compatible(model_id, messages, tools, temperature)

    all_runs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_call) for _ in range(num_runs)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_runs.append(result or [])

    return all_runs


def compute_consistency(outputs: List[List[Dict]]) -> Dict[str, float]:
    """Compute consistency metrics (Jaccard similarity of tool names)."""
    valid_outputs = [o for o in outputs if o]

    if len(valid_outputs) < 2:
        return {
            'validity_rate': len(valid_outputs) / len(outputs) if outputs else 0,
            'c_mean': 0.0 if len(valid_outputs) < 2 else 1.0,
            'c_std': 0.0,
            'num_valid': len(valid_outputs)
        }

    similarities = []
    for i in range(len(valid_outputs)):
        for j in range(i + 1, len(valid_outputs)):
            tools1 = set(tc.get('name', '') for tc in valid_outputs[i])
            tools2 = set(tc.get('name', '') for tc in valid_outputs[j])

            if not tools1 and not tools2:
                sim = 1.0
            elif not tools1 or not tools2:
                sim = 0.0
            else:
                sim = len(tools1 & tools2) / len(tools1 | tools2)
            similarities.append(sim)

    return {
        'validity_rate': len(valid_outputs) / len(outputs),
        'c_mean': statistics.mean(similarities),
        'c_std': statistics.stdev(similarities) if len(similarities) > 1 else 0.0,
        'num_valid': len(valid_outputs)
    }


# =============================================================================
# Phase 1: Baseline Collection
# =============================================================================

def collect_baseline(model_id: str, samples: List[Dict], num_runs: int = 15,
                     temperatures: List[float] = None, output_dir: str = None) -> Dict:
    """
    Collect baseline consistency for a single model across all samples.

    Uses 15 runs per condition to enable cross-fitting in Phase 2:
    - Runs 1-5: baseline estimation (for stratification)
    - Runs 6-15: effect measurement (independent)

    Args:
        model_id: Model identifier
        samples: List of samples with 'question', 'tools', 'schema_complexity'
        num_runs: Runs per condition (default 15 for cross-fitting)
        temperatures: Temperatures to test (default [0.3, 0.7, 1.0])
        output_dir: Output directory

    Returns:
        Results dictionary with per-sample baselines
    """
    if temperatures is None:
        temperatures = [0.3, 0.7, 1.0]

    if output_dir is None:
        output_dir = "results/acl_linguistic/phase1_baseline"

    display_name = get_display_name(model_id)
    provider = get_provider(model_id)

    print("=" * 70)
    print(f"PHASE 1: Baseline Collection for {display_name}")
    print("=" * 70)

    # Setup client
    bedrock_client = None
    if provider == "bedrock":
        bedrock_client = get_bedrock_client()

    total_conditions = len(samples) * len(temperatures)
    total_calls = total_conditions * num_runs

    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Samples: {len(samples)}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Runs per condition: {num_runs}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Estimated time: ~{total_calls * 0.5 / 60:.0f} minutes")

    results = []
    start_time = time.time()
    completed = 0

    for sample in samples:
        sample_id = sample['id']
        question = sample['question']
        tools = convert_openai_to_xlam(sample['tools'])
        complexity = sample.get('schema_complexity', {})

        for temp in temperatures:
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_conditions - completed) / rate / 60 if rate > 0 else 0

            print(f"\r[{completed}/{total_conditions}] T={temp} {sample_id[:20]:20s} | "
                  f"ETA: {eta:.1f}min", end='', flush=True)

            try:
                outputs = generate_tool_calls(
                    model_id=model_id,
                    query=question,
                    tools=tools,
                    num_runs=num_runs,
                    temperature=temp,
                    bedrock_client=bedrock_client
                )

                metrics = compute_consistency(outputs)

                # Compute metrics for cross-fitting split
                # First 5 runs for stratification
                metrics_first5 = compute_consistency(outputs[:5])
                # Remaining runs for effect measurement
                metrics_last10 = compute_consistency(outputs[5:])

                results.append({
                    'sample_id': sample_id,
                    'temperature': temp,
                    'schema_complexity': complexity,
                    'question': question,
                    'metrics': metrics,
                    'metrics_first5': metrics_first5,
                    'metrics_last10': metrics_last10,
                    'all_tool_calls': [
                        [tc.get('name') for tc in run] for run in outputs
                    ]
                })

            except Exception as e:
                print(f"\n  ERROR: {e}")
                results.append({
                    'sample_id': sample_id,
                    'temperature': temp,
                    'schema_complexity': complexity,
                    'error': str(e)
                })

    elapsed_total = time.time() - start_time
    print(f"\n\nCompleted in {elapsed_total / 60:.1f} minutes")

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    # Group by complexity level
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

    # Determine baseline difficulty strata
    # Using first 5 runs at T=0.7 as baseline estimate (cross-fitting)
    baseline_estimates = {}
    for r in results:
        if 'metrics_first5' in r and r['temperature'] == 0.7:
            baseline_estimates[r['sample_id']] = r['metrics_first5']['c_mean']

    # Classify into difficulty strata
    difficulty_strata = {'difficult': [], 'medium': [], 'easy': []}
    for sample_id, baseline in baseline_estimates.items():
        if baseline < 0.7:
            difficulty_strata['difficult'].append(sample_id)
        elif baseline >= 0.9:
            difficulty_strata['easy'].append(sample_id)
        else:
            difficulty_strata['medium'].append(sample_id)

    print("\nBaseline Difficulty Distribution (T=0.7, first 5 runs):")
    for level in ['difficult', 'medium', 'easy']:
        print(f"  {level}: {len(difficulty_strata[level])} samples")

    # Save results
    output_path = Path(output_dir) / f"{display_name.replace(' ', '_').lower()}_baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'model_id': model_id,
            'display_name': display_name,
            'provider': provider,
            'temperatures': temperatures,
            'num_runs': num_runs,
            'num_samples': len(samples),
            'timestamp': datetime.now().isoformat(),
            'elapsed_minutes': elapsed_total / 60,
            'purpose': 'Phase 1 baseline collection for ceiling effect analysis'
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

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='ACL Phase 1: Multi-Model Baseline Collection'
    )
    parser.add_argument('--model', type=str, help='Model ID to evaluate')
    parser.add_argument('--all-models', action='store_true',
                        help='Run on all ACL models')
    parser.add_argument('--primary-only', action='store_true',
                        help='Run on primary models only')
    parser.add_argument('--samples', type=str,
                        default='data/acl_stratified/stratified_samples.json',
                        help='Stratified samples file')
    parser.add_argument('--num-runs', type=int, default=15,
                        help='Runs per condition (default 15 for cross-fitting)')
    parser.add_argument('--temperatures', type=float, nargs='+',
                        default=[0.3, 0.7, 1.0])
    parser.add_argument('--output-dir', type=str,
                        default='results/acl_linguistic/phase1_baseline')

    args = parser.parse_args()

    # Load stratified samples
    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"Error: Samples file not found: {samples_path}")
        print("Run prepare_stratified_data.py first.")
        sys.exit(1)

    with open(samples_path) as f:
        data = json.load(f)

    # Flatten samples from all strata
    all_samples = []
    for level in ['simple', 'medium', 'complex']:
        all_samples.extend(data['samples'].get(level, []))

    print(f"Loaded {len(all_samples)} stratified samples")

    # Determine models to run
    if args.all_models:
        models = ALL_ACL_MODELS
    elif args.primary_only:
        models = ACL_PRIMARY_MODELS
    elif args.model:
        models = [args.model]
    else:
        print("Please specify --model, --primary-only, or --all-models")
        sys.exit(1)

    # Run baseline collection for each model
    for model_id in models:
        try:
            collect_baseline(
                model_id=model_id,
                samples=all_samples,
                num_runs=args.num_runs,
                temperatures=args.temperatures,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"Error with {model_id}: {e}")
            continue


if __name__ == '__main__':
    main()
