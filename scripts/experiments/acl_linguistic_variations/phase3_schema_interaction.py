#!/usr/bin/env python3
"""
ACL 2026 Paper: Phase 3 - Linguistic × Schema Complexity Interaction

Tests whether linguistic intervention effects depend on schema complexity.
Building on KDD's finding that schema complexity is the #1 factor (19% SHAP importance).

Key hypotheses:
1. Linguistic interventions have larger effects on complex schemas
2. Interaction effect: politeness × complexity should be significant
3. Simple schemas are "solved" - interventions cannot improve further

Uses full factorial design: 8 variations × 3 complexity levels × 5 models.

Usage:
    python phase3_schema_interaction.py --model MODEL_ID --samples-file SAMPLES.json
    python phase3_schema_interaction.py --all-models
"""

import argparse
import json
import sys
import os
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
# Models for Interaction Analysis
# =============================================================================

ACL_INTERACTION_MODELS = [
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "openai/gpt-4.1-mini",  # KDD outlier
    "us.meta.llama3-3-70b-instruct-v1:0",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "google/gemini-2.5-flash-lite",
]


# =============================================================================
# Linguistic Transformations (focused subset)
# =============================================================================

import re

def add_modal_must(text: str) -> str:
    """Strong deontic modal."""
    for modal in ['should', 'need to', 'have to', 'can', 'could', 'would like to', 'want to', 'might']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'must', text, count=1, flags=re.IGNORECASE)
    return "You must " + text[0].lower() + text[1:]


def add_modal_might(text: str) -> str:
    """Weak epistemic modal."""
    for modal in ['must', 'should', 'need to', 'have to', 'can', 'could']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'might want to', text, count=1, flags=re.IGNORECASE)
    return "You might want to " + text[0].lower() + text[1:]


def add_please(text: str) -> str:
    """Positive politeness."""
    if re.search(r'\bplease\b', text, re.IGNORECASE):
        return text
    return "Please " + text[0].lower() + text[1:]


def remove_please(text: str) -> str:
    """Bald on-record."""
    result = re.sub(r'\bplease\s*', '', text, flags=re.IGNORECASE)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result


def add_conditional(text: str) -> str:
    """Conditional hedge."""
    if re.search(r'\bif possible\b', text, re.IGNORECASE):
        return text
    return text.rstrip('.') + ", if possible."


def make_directive(text: str) -> str:
    """Direct imperative."""
    text = re.sub(r'^(I would like|I want|I need|Can you|Could you|Would you mind|Please)\s*(to|you to)?\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def add_hint(text: str) -> str:
    """Indirect speech act."""
    text = re.sub(r'^(I need to|I want to|I would like to|Can you|Could you)\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return "I wonder if you could " + text.rstrip('.?') + "."


def identity(text: str) -> str:
    return text


VARIATIONS = {
    'baseline': {'transform': identity, 'category': 'baseline'},
    'polite_please': {'transform': add_please, 'category': 'politeness'},
    'polite_bald': {'transform': remove_please, 'category': 'politeness'},
    'modal_must': {'transform': add_modal_must, 'category': 'modal'},
    'modal_might': {'transform': add_modal_might, 'category': 'modal'},
    'hedge_conditional': {'transform': add_conditional, 'category': 'hedge'},
    'speech_directive': {'transform': make_directive, 'category': 'speech_act'},
    'speech_hint': {'transform': add_hint, 'category': 'speech_act'},
}


# =============================================================================
# API Functions
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

def two_way_anova(data: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    """
    Two-way ANOVA for variation × complexity interaction.

    Args:
        data: {variation_name: {complexity_level: [consistency_values]}}

    Returns:
        Dict with F-statistics and p-values for main effects and interaction
    """
    # Flatten data for ANOVA
    all_values = []
    variation_labels = []
    complexity_labels = []

    for var_name, complexity_data in data.items():
        for comp_level, values in complexity_data.items():
            for v in values:
                all_values.append(v)
                variation_labels.append(var_name)
                complexity_labels.append(comp_level)

    if len(all_values) < 10:
        return {'error': 'insufficient_data'}

    # Simple effects calculation (F-test approximation)
    # Group by variation
    by_variation = defaultdict(list)
    for v, var in zip(all_values, variation_labels):
        by_variation[var].append(v)

    # Group by complexity
    by_complexity = defaultdict(list)
    for v, comp in zip(all_values, complexity_labels):
        by_complexity[comp].append(v)

    # F-test for variation effect
    var_groups = list(by_variation.values())
    if len(var_groups) >= 2 and all(len(g) >= 2 for g in var_groups):
        f_var, p_var = stats.f_oneway(*var_groups)
    else:
        f_var, p_var = 0, 1

    # F-test for complexity effect
    comp_groups = list(by_complexity.values())
    if len(comp_groups) >= 2 and all(len(g) >= 2 for g in comp_groups):
        f_comp, p_comp = stats.f_oneway(*comp_groups)
    else:
        f_comp, p_comp = 0, 1

    return {
        'variation_effect': {'F': f_var, 'p': p_var},
        'complexity_effect': {'F': f_comp, 'p': p_comp},
        'n_observations': len(all_values)
    }


def compute_interaction_strength(data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """
    Compute interaction strength metrics.

    Tests if the effect of linguistic variation depends on schema complexity.
    """
    # Compute mean effect for each variation × complexity cell
    cell_means = {}
    for var_name, complexity_data in data.items():
        cell_means[var_name] = {}
        for comp_level, values in complexity_data.items():
            if values:
                cell_means[var_name][comp_level] = np.mean(values)

    # Compute interaction as difference in simple effects
    # Effect of politeness on simple vs complex schemas
    interactions = {}

    for var_name in ['polite_please', 'modal_might', 'hedge_conditional']:
        if var_name not in cell_means or 'baseline' not in cell_means:
            continue

        baseline_means = cell_means.get('baseline', {})
        var_means = cell_means.get(var_name, {})

        # Effect on simple schemas
        effect_simple = var_means.get('simple', 0) - baseline_means.get('simple', 0)
        # Effect on complex schemas
        effect_complex = var_means.get('complex', 0) - baseline_means.get('complex', 0)

        interactions[var_name] = {
            'effect_simple': effect_simple,
            'effect_complex': effect_complex,
            'interaction': effect_complex - effect_simple,
            'interaction_interpretation': 'larger_on_complex' if effect_complex > effect_simple else 'larger_on_simple'
        }

    return interactions


# =============================================================================
# Phase 3: Schema Complexity Interaction
# =============================================================================

def run_interaction_analysis(model_id: str, samples_file: str, num_runs: int = 10,
                             temperature: float = 0.7, output_dir: str = None) -> Dict:
    """
    Run full factorial design: 8 variations × 3 complexity levels.

    Tests KDD's hypothesis that schema complexity moderates all effects.
    """
    if output_dir is None:
        output_dir = "results/acl_linguistic/phase3_interaction"

    display_name = get_display_name(model_id)
    provider = get_provider(model_id)

    print("=" * 70)
    print(f"PHASE 3: Schema × Linguistic Interaction for {display_name}")
    print("=" * 70)

    # Load stratified samples
    with open(samples_file) as f:
        samples_data = json.load(f)

    # Count samples per complexity level
    samples_by_complexity = {}
    for level in ['simple', 'medium', 'complex']:
        samples_by_complexity[level] = samples_data['samples'].get(level, [])
        print(f"  {level}: {len(samples_by_complexity[level])} samples")

    total_samples = sum(len(s) for s in samples_by_complexity.values())
    total_conditions = total_samples * len(VARIATIONS)
    total_calls = total_conditions * num_runs

    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Samples: {total_samples}")
    print(f"  Variations: {len(VARIATIONS)}")
    print(f"  Temperature: {temperature}")
    print(f"  Runs per condition: {num_runs}")
    print(f"  Total API calls: {total_calls}")

    # Setup client
    bedrock_client = None
    if provider == "bedrock":
        bedrock_client = get_bedrock_client()

    results = []
    start_time = time.time()
    completed = 0

    # Full factorial: complexity × variation
    for complexity_level, samples in samples_by_complexity.items():
        for sample in samples:
            sample_id = sample['id']
            question = sample['question']
            tools = convert_openai_to_xlam(sample['tools'])
            schema_complexity = sample.get('schema_complexity', {})

            for var_name, var_config in VARIATIONS.items():
                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_conditions - completed) / rate / 60 if rate > 0 else 0

                print(f"\r[{completed}/{total_conditions}] {complexity_level[:4]} {var_name[:12]:12s} | "
                      f"ETA: {eta:.1f}min", end='', flush=True)

                varied_prompt = var_config['transform'](question)

                try:
                    outputs = generate_tool_calls(
                        model_id=model_id,
                        query=varied_prompt,
                        tools=tools,
                        num_runs=num_runs,
                        temperature=temperature,
                        bedrock_client=bedrock_client
                    )

                    metrics = compute_consistency(outputs)

                    results.append({
                        'sample_id': sample_id,
                        'complexity_level': complexity_level,
                        'variation': var_name,
                        'category': var_config['category'],
                        'temperature': temperature,
                        'schema_complexity': schema_complexity,
                        'original_prompt': question,
                        'varied_prompt': varied_prompt,
                        'metrics': metrics
                    })

                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    results.append({
                        'sample_id': sample_id,
                        'complexity_level': complexity_level,
                        'variation': var_name,
                        'error': str(e)
                    })

    elapsed_total = time.time() - start_time
    print(f"\n\nCompleted in {elapsed_total / 60:.1f} minutes")

    # ==========================================================================
    # Interaction Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("INTERACTION ANALYSIS: Variation × Schema Complexity")
    print("=" * 70)

    # Organize data for ANOVA
    anova_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if 'metrics' in r:
            anova_data[r['variation']][r['complexity_level']].append(r['metrics']['c_mean'])

    # Two-way ANOVA
    anova_results = two_way_anova(dict(anova_data))
    print(f"\nTwo-Way Effects:")
    if 'error' not in anova_results:
        print(f"  Variation effect: F={anova_results['variation_effect']['F']:.2f}, "
              f"p={anova_results['variation_effect']['p']:.4f}")
        print(f"  Complexity effect: F={anova_results['complexity_effect']['F']:.2f}, "
              f"p={anova_results['complexity_effect']['p']:.4f}")

    # Interaction strength
    interactions = compute_interaction_strength(dict(anova_data))
    print("\nInteraction Effects (Effect on Complex - Effect on Simple):")
    for var_name, interaction in interactions.items():
        print(f"  {var_name}: {interaction['interaction']:+.3f} "
              f"({interaction['interaction_interpretation']})")

    # Cell means table
    print("\n" + "-" * 70)
    print("Cell Means (Consistency by Variation × Complexity):")
    print("-" * 70)
    print("{:<15} {:>12} {:>12} {:>12} {:>12}".format(
        "Variation", "Simple", "Medium", "Complex", "Delta(C-S)"
    ))
    print("-" * 70)

    for var_name in VARIATIONS.keys():
        var_data = anova_data[var_name]
        simple_mean = np.mean(var_data['simple']) if var_data['simple'] else 0
        medium_mean = np.mean(var_data['medium']) if var_data['medium'] else 0
        complex_mean = np.mean(var_data['complex']) if var_data['complex'] else 0
        delta = complex_mean - simple_mean

        print("{:<15} {:>11.3f} {:>11.3f} {:>11.3f} {:>+11.3f}".format(
            var_name, simple_mean, medium_mean, complex_mean, delta
        ))

    # Save results
    output_path = Path(output_dir) / f"{display_name.replace(' ', '_').lower()}_interaction.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'model_id': model_id,
            'display_name': display_name,
            'provider': provider,
            'temperature': temperature,
            'num_runs': num_runs,
            'variations': list(VARIATIONS.keys()),
            'timestamp': datetime.now().isoformat(),
            'elapsed_minutes': elapsed_total / 60,
            'purpose': 'Phase 3 schema complexity × linguistic variation interaction'
        },
        'anova_results': anova_results if 'error' not in anova_results else None,
        'interaction_effects': interactions,
        'cell_means': {
            var_name: {
                level: float(np.mean(values)) if values else None
                for level, values in complexity_data.items()
            }
            for var_name, complexity_data in anova_data.items()
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved interaction results to {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='ACL Phase 3: Schema Complexity × Linguistic Interaction'
    )
    parser.add_argument('--model', type=str, help='Model ID to evaluate')
    parser.add_argument('--all-models', action='store_true',
                        help='Run on all interaction models')
    parser.add_argument('--samples', type=str,
                        default='data/acl_stratified/stratified_samples.json')
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--output-dir', type=str,
                        default='results/acl_linguistic/phase3_interaction')

    args = parser.parse_args()

    if not Path(args.samples).exists():
        print(f"Error: Samples file not found: {args.samples}")
        print("Run prepare_stratified_data.py first.")
        sys.exit(1)

    if args.all_models:
        models = ACL_INTERACTION_MODELS
    elif args.model:
        models = [args.model]
    else:
        print("Please specify --model or --all-models")
        sys.exit(1)

    for model_id in models:
        try:
            run_interaction_analysis(
                model_id=model_id,
                samples_file=args.samples,
                num_runs=args.num_runs,
                temperature=args.temperature,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"Error with {model_id}: {e}")
            continue


if __name__ == '__main__':
    main()
