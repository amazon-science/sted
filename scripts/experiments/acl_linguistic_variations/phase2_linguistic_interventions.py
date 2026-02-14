#!/usr/bin/env python3
"""
ACL 2026 Paper: Phase 2 - Stratified Linguistic Interventions

Applies linguistic modifications stratified by baseline difficulty (from Phase 1).
Tests KDD's ceiling effect hypothesis:
- Interventions should help difficult prompts (baseline < 0.7)
- Interventions should be neutral/harmful for easy prompts (baseline >= 0.9)

Key linguistic features tested (8 variations):
1. baseline (no modification)
2. polite_please (positive politeness)
3. polite_bald (remove politeness)
4. modal_must (strong deontic)
5. modal_might (weak epistemic)
6. hedge_conditional ("if possible")
7. speech_directive (bare command)
8. speech_hint (indirect)

Usage:
    python phase2_linguistic_interventions.py --model MODEL_ID --baseline-file BASELINE.json
    python phase2_linguistic_interventions.py --all-models
"""

import argparse
import json
import sys
import os
import re
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import statistics

import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sted.model_config import MODEL_REGISTRY, get_provider, get_display_name, get_max_workers

import boto3
from botocore.config import Config


# =============================================================================
# Linguistic Transformations (8 key variations)
# =============================================================================

def add_modal_must(text: str) -> str:
    """Strong deontic modal (obligation)."""
    for modal in ['should', 'need to', 'have to', 'can', 'could', 'would like to', 'want to', 'might']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'must', text, count=1, flags=re.IGNORECASE)
    return "You must " + text[0].lower() + text[1:]


def add_modal_might(text: str) -> str:
    """Weakest epistemic modal (tentative possibility)."""
    for modal in ['must', 'should', 'need to', 'have to', 'can', 'could']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'might want to', text, count=1, flags=re.IGNORECASE)
    return "You might want to " + text[0].lower() + text[1:]


def add_please(text: str) -> str:
    """Positive politeness with 'please'."""
    if re.search(r'\bplease\b', text, re.IGNORECASE):
        return text
    return "Please " + text[0].lower() + text[1:]


def remove_please(text: str) -> str:
    """Bald on-record (remove politeness markers)."""
    result = re.sub(r'\bplease\s*', '', text, flags=re.IGNORECASE)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result


def add_conditional(text: str) -> str:
    """Conditional hedge ('if possible')."""
    if re.search(r'\bif possible\b', text, re.IGNORECASE):
        return text
    return text.rstrip('.') + ", if possible."


def make_directive(text: str) -> str:
    """Direct imperative (bare command)."""
    text = re.sub(r'^(I would like|I want|I need|Can you|Could you|Would you mind|Please)\s*(to|you to)?\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def add_hint(text: str) -> str:
    """Indirect speech act (hint)."""
    text = re.sub(r'^(I need to|I want to|I would like to|Can you|Could you)\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return "I wonder if you could " + text.rstrip('.?') + "."


def identity(text: str) -> str:
    """No modification (baseline)."""
    return text


# Key 8 variations for Phase 2
VARIATIONS = {
    'baseline': {
        'transform': identity,
        'features': {'baseline': True},
        'category': 'baseline'
    },
    'polite_please': {
        'transform': add_please,
        'features': {'politeness': 'please', 'strategy': 'positive'},
        'category': 'politeness'
    },
    'polite_bald': {
        'transform': remove_please,
        'features': {'politeness': 'bald', 'strategy': 'bald_on_record'},
        'category': 'politeness'
    },
    'modal_must': {
        'transform': add_modal_must,
        'features': {'modal': 'must', 'modal_strength': 'strong', 'modal_type': 'deontic'},
        'category': 'modal'
    },
    'modal_might': {
        'transform': add_modal_might,
        'features': {'modal': 'might', 'modal_strength': 'weak', 'modal_type': 'epistemic'},
        'category': 'modal'
    },
    'hedge_conditional': {
        'transform': add_conditional,
        'features': {'hedge': 'conditional'},
        'category': 'hedge'
    },
    'speech_directive': {
        'transform': make_directive,
        'features': {'speech_act': 'directive', 'directness': 'direct'},
        'category': 'speech_act'
    },
    'speech_hint': {
        'transform': add_hint,
        'features': {'speech_act': 'hint', 'directness': 'indirect'},
        'category': 'speech_act'
    },
}


# =============================================================================
# API Functions (same as Phase 1)
# =============================================================================

def get_bedrock_client(region: str = "us-east-1"):
    boto_config = Config(
        retries={'max_attempts': 20, 'mode': 'adaptive'},
        max_pool_connections=100,
        connect_timeout=30,
        read_timeout=120
    )
    return boto3.client("bedrock-runtime", region_name=region, config=boto_config)


def clean_schema_for_bedrock(obj):
    if isinstance(obj, dict):
        return {k: clean_schema_for_bedrock(v) for k, v in obj.items() if not k.startswith('$')}
    elif isinstance(obj, list):
        return [clean_schema_for_bedrock(item) for item in obj]
    return obj


def convert_openai_to_xlam(tools: List[Dict]) -> List[Dict]:
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


def xlam_tool_to_bedrock_tool(tool_def: Dict) -> Dict:
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
    except:
        return []


def call_openai_compatible(model_id: str, messages: List[Dict], tools: List[Dict],
                          temperature: float, max_tokens: int = 4000) -> List[Dict]:
    import openai
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        return []

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    )

    openai_tools = [{"type": "function", "function": {
        "name": t.get("name", ""),
        "description": t.get("description", ""),
        "parameters": t.get("parameters", {})
    }} for t in tools]

    try:
        response = client.chat.completions.create(
            model=model_id, messages=messages, tools=openai_tools,
            tool_choice="auto", temperature=temperature, max_tokens=max_tokens
        )
        tool_calls = []
        if response.choices and response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                tool_calls.append({
                    'name': tc.function.name,
                    'arguments': json.loads(tc.function.arguments) if tc.function.arguments else {}
                })
        return tool_calls
    except:
        return []


def generate_tool_calls(model_id: str, query: str, tools: List[Dict],
                       num_runs: int = 10, temperature: float = 0.7,
                       bedrock_client=None) -> List[List[Dict]]:
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
            all_runs.append(future.result() or [])
    return all_runs


def compute_consistency(outputs: List[List[Dict]]) -> Dict[str, float]:
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
# Statistical Analysis
# =============================================================================

def cohens_d(group1: List[float], group2: List[float]) -> float:
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    if len(data) < 2:
        return (data[0] if data else 0.0, 0.0, 0.0)
    data = np.array(data)
    bootstrap_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    mean = np.mean(data)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    return (mean, lower, upper)


# =============================================================================
# Phase 2: Stratified Interventions
# =============================================================================

def run_stratified_interventions(model_id: str, baseline_file: str,
                                  samples_file: str, num_runs: int = 10,
                                  temperatures: List[float] = None,
                                  output_dir: str = None) -> Dict:
    """
    Run linguistic interventions stratified by baseline difficulty.

    Uses baseline_estimates from Phase 1 to stratify samples:
    - Difficult: baseline < 0.7
    - Medium: 0.7 <= baseline < 0.9
    - Easy: baseline >= 0.9

    Tests KDD's ceiling effect hypothesis.
    """
    if temperatures is None:
        temperatures = [0.7]  # Focus on T=0.7 for main analysis

    if output_dir is None:
        output_dir = "results/acl_linguistic/phase2_interventions"

    display_name = get_display_name(model_id)
    provider = get_provider(model_id)

    print("=" * 70)
    print(f"PHASE 2: Stratified Linguistic Interventions for {display_name}")
    print("=" * 70)

    # Load baseline data
    with open(baseline_file) as f:
        baseline_data = json.load(f)

    baseline_estimates = baseline_data.get('baseline_estimates', {})
    difficulty_strata = baseline_data.get('difficulty_strata', {})

    print(f"\nBaseline difficulty distribution:")
    for level in ['difficult', 'medium', 'easy']:
        print(f"  {level}: {len(difficulty_strata.get(level, []))} samples")

    # Load samples
    with open(samples_file) as f:
        samples_data = json.load(f)

    # Flatten samples
    all_samples = {}
    for level in ['simple', 'medium', 'complex']:
        for sample in samples_data['samples'].get(level, []):
            all_samples[sample['id']] = sample

    # Setup client
    bedrock_client = None
    if provider == "bedrock":
        bedrock_client = get_bedrock_client()

    total_conditions = len(baseline_estimates) * len(VARIATIONS) * len(temperatures)
    total_calls = total_conditions * num_runs

    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Samples: {len(baseline_estimates)}")
    print(f"  Variations: {len(VARIATIONS)}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Runs per condition: {num_runs}")
    print(f"  Total API calls: {total_calls}")

    results = []
    start_time = time.time()
    completed = 0

    for sample_id, baseline in baseline_estimates.items():
        if sample_id not in all_samples:
            continue

        sample = all_samples[sample_id]
        question = sample['question']
        tools = convert_openai_to_xlam(sample['tools'])
        complexity = sample.get('schema_complexity', {})

        # Determine difficulty stratum
        if baseline < 0.7:
            difficulty = 'difficult'
        elif baseline >= 0.9:
            difficulty = 'easy'
        else:
            difficulty = 'medium'

        for var_name, var_config in VARIATIONS.items():
            varied_prompt = var_config['transform'](question)

            for temp in temperatures:
                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_conditions - completed) / rate / 60 if rate > 0 else 0

                print(f"\r[{completed}/{total_conditions}] {difficulty[:4]} {var_name[:12]:12s} | "
                      f"ETA: {eta:.1f}min", end='', flush=True)

                try:
                    outputs = generate_tool_calls(
                        model_id=model_id,
                        query=varied_prompt,
                        tools=tools,
                        num_runs=num_runs,
                        temperature=temp,
                        bedrock_client=bedrock_client
                    )

                    metrics = compute_consistency(outputs)

                    results.append({
                        'sample_id': sample_id,
                        'variation': var_name,
                        'category': var_config['category'],
                        'features': var_config['features'],
                        'temperature': temp,
                        'baseline_consistency': baseline,
                        'difficulty_stratum': difficulty,
                        'schema_complexity': complexity,
                        'original_prompt': question,
                        'varied_prompt': varied_prompt,
                        'metrics': metrics,
                        'delta_consistency': metrics['c_mean'] - baseline
                    })

                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    results.append({
                        'sample_id': sample_id,
                        'variation': var_name,
                        'difficulty_stratum': difficulty,
                        'error': str(e)
                    })

    elapsed_total = time.time() - start_time
    print(f"\n\nCompleted in {elapsed_total / 60:.1f} minutes")

    # ==========================================================================
    # Ceiling Effect Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CEILING EFFECT ANALYSIS")
    print("=" * 70)

    # Group results by variation and difficulty
    by_var_diff = defaultdict(lambda: defaultdict(list))
    for r in results:
        if 'metrics' in r:
            by_var_diff[r['variation']][r['difficulty_stratum']].append(r['delta_consistency'])

    # Print ceiling effect table
    print("\n{:<15} {:>12} {:>12} {:>12} {:>10}".format(
        "Variation", "Difficult", "Medium", "Easy", "Ceiling?"
    ))
    print("-" * 65)

    ceiling_effects = {}
    for var_name in VARIATIONS.keys():
        var_data = by_var_diff[var_name]

        difficult_delta = np.mean(var_data['difficult']) if var_data['difficult'] else 0
        medium_delta = np.mean(var_data['medium']) if var_data['medium'] else 0
        easy_delta = np.mean(var_data['easy']) if var_data['easy'] else 0

        # Ceiling effect: improvement for difficult, harm for easy
        has_ceiling = difficult_delta > 0 and easy_delta < 0

        print("{:<15} {:>+11.1%} {:>+11.1%} {:>+11.1%} {:>10}".format(
            var_name,
            difficult_delta,
            medium_delta,
            easy_delta,
            "YES" if has_ceiling else "no"
        ))

        ceiling_effects[var_name] = {
            'difficult_delta': difficult_delta,
            'medium_delta': medium_delta,
            'easy_delta': easy_delta,
            'has_ceiling_effect': has_ceiling,
            'n_difficult': len(var_data['difficult']),
            'n_medium': len(var_data['medium']),
            'n_easy': len(var_data['easy'])
        }

    # Statistical tests for key contrasts
    print("\n" + "-" * 65)
    print("Statistical Tests (Difficult vs Easy):")
    print("-" * 65)

    for var_name in ['polite_please', 'modal_might', 'hedge_conditional']:
        var_data = by_var_diff[var_name]
        difficult = var_data['difficult']
        easy = var_data['easy']

        if len(difficult) >= 3 and len(easy) >= 3:
            t_stat, p_value = stats.ttest_ind(difficult, easy)
            effect_d = cohens_d(difficult, easy)
            print(f"  {var_name}: t={t_stat:.2f}, p={p_value:.4f}, d={effect_d:.2f}")

    # Save results
    output_path = Path(output_dir) / f"{display_name.replace(' ', '_').lower()}_interventions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'model_id': model_id,
            'display_name': display_name,
            'provider': provider,
            'temperatures': temperatures,
            'num_runs': num_runs,
            'variations': list(VARIATIONS.keys()),
            'timestamp': datetime.now().isoformat(),
            'elapsed_minutes': elapsed_total / 60,
            'purpose': 'Phase 2 stratified linguistic interventions - ceiling effect test'
        },
        'ceiling_effects': ceiling_effects,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved intervention results to {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='ACL Phase 2: Stratified Linguistic Interventions'
    )
    parser.add_argument('--model', type=str, help='Model ID to evaluate')
    parser.add_argument('--baseline-file', type=str, help='Phase 1 baseline results file')
    parser.add_argument('--samples', type=str,
                        default='data/acl_stratified/stratified_samples.json')
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.7])
    parser.add_argument('--output-dir', type=str,
                        default='results/acl_linguistic/phase2_interventions')

    args = parser.parse_args()

    if not args.model or not args.baseline_file:
        print("Please specify --model and --baseline-file")
        print("Example: python phase2_linguistic_interventions.py "
              "--model us.anthropic.claude-sonnet-4-20250514-v1:0 "
              "--baseline-file results/acl_linguistic/phase1_baseline/claude_sonnet_4_baseline.json")
        sys.exit(1)

    if not Path(args.baseline_file).exists():
        print(f"Error: Baseline file not found: {args.baseline_file}")
        print("Run phase1_baseline_collection.py first.")
        sys.exit(1)

    run_stratified_interventions(
        model_id=args.model,
        baseline_file=args.baseline_file,
        samples_file=args.samples,
        num_runs=args.num_runs,
        temperatures=args.temperatures,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
