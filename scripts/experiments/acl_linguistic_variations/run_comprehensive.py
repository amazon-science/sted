#!/usr/bin/env python3
"""
ACL 2026 Paper: Comprehensive Linguistic Experiments

Multi-model evaluation of linguistic features affecting LLM tool calling consistency.

This script runs experiments across:
- Multiple LLM models (Claude, GPT, Gemini, Llama, Qwen, etc.)
- Extended linguistic variations (18 total)
- Statistical significance testing with bootstrap confidence intervals
- Effect size calculations (Cohen's d)

Usage:
    python run_comprehensive.py --model MODEL_ID [--num-prompts N] [--num-runs N]
    python run_comprehensive.py --all-models  # Run on all supported models
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures
import time
import random
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sted.model_config import MODEL_REGISTRY, get_provider, get_display_name, get_max_workers

import boto3
from botocore.config import Config

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


# =============================================================================
# Extended Linguistic Variations (18 total)
# =============================================================================

def add_modal_must(text: str) -> str:
    """Strong deontic modal (obligation)."""
    for modal in ['should', 'need to', 'have to', 'can', 'could', 'would like to', 'want to', 'might']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'must', text, count=1, flags=re.IGNORECASE)
    return "You must " + text[0].lower() + text[1:]


def add_modal_should(text: str) -> str:
    """Medium deontic modal (recommendation)."""
    for modal in ['must', 'need to', 'have to', 'would like to', 'want to', 'might']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'should', text, count=1, flags=re.IGNORECASE)
    return "You should " + text[0].lower() + text[1:]


def add_modal_could(text: str) -> str:
    """Weak epistemic modal (possibility)."""
    for modal in ['must', 'should', 'need to', 'have to', 'can', 'would like to', 'want to']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'could', text, count=1, flags=re.IGNORECASE)
    return "You could " + text[0].lower() + text[1:]


def add_modal_might(text: str) -> str:
    """Weakest epistemic modal (tentative possibility)."""
    for modal in ['must', 'should', 'need to', 'have to', 'can', 'could']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'might want to', text, count=1, flags=re.IGNORECASE)
    return "You might want to " + text[0].lower() + text[1:]


def add_modal_need_to(text: str) -> str:
    """Necessity modal (requirement)."""
    for modal in ['must', 'should', 'can', 'could', 'would like to', 'want to', 'might']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'need to', text, count=1, flags=re.IGNORECASE)
    return "I need to " + text[0].lower() + text[1:]


def add_modal_would(text: str) -> str:
    """Hypothetical modal (conditional)."""
    for modal in ['must', 'should', 'can', 'could', 'need to', 'want to', 'might']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'would like to', text, count=1, flags=re.IGNORECASE)
    return "I would like to " + text[0].lower() + text[1:]


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


def add_can_you(text: str) -> str:
    """Conventional indirect request (negative politeness)."""
    if re.search(r'\bcan you\b', text, re.IGNORECASE):
        return text
    text = re.sub(r'^(I need to|I want to|I would like to)\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return "Can you " + text.rstrip('.?') + "?"


def add_could_you(text: str) -> str:
    """More polite conventional indirect request."""
    if re.search(r'\bcould you\b', text, re.IGNORECASE):
        return text
    text = re.sub(r'^(I need to|I want to|I would like to|Can you)\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return "Could you " + text.rstrip('.?') + "?"


def add_would_you_mind(text: str) -> str:
    """Strong negative politeness (maximum deference)."""
    if re.search(r'\bwould you mind\b', text, re.IGNORECASE):
        return text
    text = re.sub(r'^(I need to|I want to|I would like to|Can you|Could you)\s*', '', text, flags=re.IGNORECASE)
    words = text.split()
    if words:
        verb = words[0].lower()
        if verb.endswith('e') and not verb.endswith('ee'):
            verb = verb[:-1] + 'ing'
        else:
            verb = verb + 'ing'
        words[0] = verb
        text = ' '.join(words)
    return "Would you mind " + text.rstrip('.?') + "?"


def add_i_wonder(text: str) -> str:
    """Indirect speech act (hint)."""
    text = re.sub(r'^(I need to|I want to|I would like to|Can you|Could you)\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return "I wonder if you could " + text.rstrip('.?') + "."


def make_directive(text: str) -> str:
    """Direct imperative (bare command)."""
    text = re.sub(r'^(I would like|I want|I need|Can you|Could you|Would you mind|Please)\s*(to|you to)?\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def add_conditional(text: str) -> str:
    """Conditional hedge ('if possible')."""
    if re.search(r'\bif possible\b', text, re.IGNORECASE):
        return text
    return text.rstrip('.') + ", if possible."


def add_when_you_can(text: str) -> str:
    """Temporal hedge ('when you have a chance')."""
    if re.search(r'\bwhen you\b', text, re.IGNORECASE):
        return text
    return text.rstrip('.') + ", when you have a chance."


def add_grateful(text: str) -> str:
    """Gratitude marker ('I would be grateful')."""
    text = re.sub(r'^(I need to|I want to|I would like to)\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return "I would be grateful if you could " + text.rstrip('.?') + "."


def add_appreciate(text: str) -> str:
    """Appreciation marker ('I would appreciate')."""
    text = re.sub(r'^(I need to|I want to|I would like to)\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return "I would appreciate it if you could " + text.rstrip('.?') + "."


def identity(text: str) -> str:
    """No modification (baseline)."""
    return text


# Complete variation definitions with linguistic annotations
VARIATIONS = {
    # === MODAL VERBS (Deontic-Epistemic Continuum) ===
    'modal_must': {
        'transform': add_modal_must,
        'features': {'modal': 'must', 'modal_strength': 'strong', 'modal_type': 'deontic', 'force': 0.95},
        'category': 'modal'
    },
    'modal_need_to': {
        'transform': add_modal_need_to,
        'features': {'modal': 'need_to', 'modal_strength': 'strong', 'modal_type': 'deontic', 'force': 0.85},
        'category': 'modal'
    },
    'modal_should': {
        'transform': add_modal_should,
        'features': {'modal': 'should', 'modal_strength': 'medium', 'modal_type': 'deontic', 'force': 0.70},
        'category': 'modal'
    },
    'modal_would': {
        'transform': add_modal_would,
        'features': {'modal': 'would', 'modal_strength': 'medium', 'modal_type': 'epistemic', 'force': 0.55},
        'category': 'modal'
    },
    'modal_could': {
        'transform': add_modal_could,
        'features': {'modal': 'could', 'modal_strength': 'weak', 'modal_type': 'epistemic', 'force': 0.40},
        'category': 'modal'
    },
    'modal_might': {
        'transform': add_modal_might,
        'features': {'modal': 'might', 'modal_strength': 'weakest', 'modal_type': 'epistemic', 'force': 0.25},
        'category': 'modal'
    },

    # === POLITENESS STRATEGIES (Brown & Levinson Framework) ===
    'polite_bald': {
        'transform': remove_please,
        'features': {'politeness': 'bald', 'strategy': 'bald_on_record', 'face_threat': 1.0},
        'category': 'politeness'
    },
    'polite_please': {
        'transform': add_please,
        'features': {'politeness': 'please', 'strategy': 'positive', 'face_threat': 0.7},
        'category': 'politeness'
    },
    'polite_can_you': {
        'transform': add_can_you,
        'features': {'politeness': 'can_you', 'strategy': 'negative_weak', 'face_threat': 0.5},
        'category': 'politeness'
    },
    'polite_could_you': {
        'transform': add_could_you,
        'features': {'politeness': 'could_you', 'strategy': 'negative_medium', 'face_threat': 0.4},
        'category': 'politeness'
    },
    'polite_would_mind': {
        'transform': add_would_you_mind,
        'features': {'politeness': 'would_mind', 'strategy': 'negative_strong', 'face_threat': 0.2},
        'category': 'politeness'
    },
    'polite_grateful': {
        'transform': add_grateful,
        'features': {'politeness': 'grateful', 'strategy': 'positive_strong', 'face_threat': 0.3},
        'category': 'politeness'
    },

    # === SPEECH ACT DIRECTNESS (Searle's Taxonomy) ===
    'speech_directive': {
        'transform': make_directive,
        'features': {'speech_act': 'directive', 'directness': 'direct', 'illocution': 1.0},
        'category': 'speech_act'
    },
    'speech_indirect': {
        'transform': add_can_you,
        'features': {'speech_act': 'request', 'directness': 'conventional_indirect', 'illocution': 0.6},
        'category': 'speech_act'
    },
    'speech_hint': {
        'transform': add_i_wonder,
        'features': {'speech_act': 'hint', 'directness': 'nonconventional_indirect', 'illocution': 0.3},
        'category': 'speech_act'
    },

    # === HEDGING/MITIGATION ===
    'hedge_conditional': {
        'transform': add_conditional,
        'features': {'hedge': 'conditional', 'mitigation': 0.5},
        'category': 'hedge'
    },
    'hedge_temporal': {
        'transform': add_when_you_can,
        'features': {'hedge': 'temporal', 'mitigation': 0.4},
        'category': 'hedge'
    },

    # === BASELINE ===
    'baseline': {
        'transform': identity,
        'features': {'baseline': True},
        'category': 'baseline'
    },
}


# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns: (mean, lower_ci, upper_ci)
    """
    if len(data) < 2:
        return (data[0] if data else 0.0, 0.0, 0.0)

    data = np.array(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    mean = np.mean(data)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (mean, lower, upper)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def mann_whitney_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """Perform Mann-Whitney U test (non-parametric).

    Returns: (U statistic, p-value)
    """
    if len(group1) < 3 or len(group2) < 3:
        return (0.0, 1.0)

    try:
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return (stat, p_value)
    except Exception:
        return (0.0, 1.0)


# =============================================================================
# API Functions
# =============================================================================

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

    # Convert tools to OpenAI format
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
        # OpenAI-compatible
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
# Main Experiment Runner
# =============================================================================

def run_experiment(model_id: str, num_prompts: int = 30, num_runs: int = 10,
                   temperatures: List[float] = None, output_dir: str = None,
                   variations_subset: List[str] = None) -> Dict:
    """Run comprehensive linguistic experiment for a single model.

    Args:
        model_id: Model identifier (from MODEL_REGISTRY)
        num_prompts: Number of base prompts to use
        num_runs: Number of runs per condition
        temperatures: List of temperatures to test
        output_dir: Output directory for results
        variations_subset: List of variation names to test (None = all)

    Returns:
        Dictionary with results and statistical analysis
    """
    if temperatures is None:
        temperatures = [0.3, 0.5, 0.7, 1.0]

    if output_dir is None:
        output_dir = "results/acl_linguistic/comprehensive"

    display_name = get_display_name(model_id)
    provider = get_provider(model_id)

    print("=" * 70)
    print(f"ACL Linguistic Experiment: {display_name}")
    print("=" * 70)

    # Load Toucan data
    data_path = Path("data/toucan/toucan_tool_calls_1006.json")
    with open(data_path) as f:
        toucan_data = json.load(f)

    # Filter suitable prompts
    suitable = []
    for item in toucan_data:
        q = item.get('question', '')
        if q.isascii() and item.get('num_tool_calls') == 1 and 50 < len(q) < 500:
            suitable.append(item)

    # Randomly sample base prompts (with seed for reproducibility)
    random.seed(42)
    base_prompts = random.sample(suitable, min(num_prompts, len(suitable)))
    print(f"Using {len(base_prompts)} base prompts")

    # Select variations
    if variations_subset:
        variations = {k: v for k, v in VARIATIONS.items() if k in variations_subset}
    else:
        variations = VARIATIONS

    # Setup client
    bedrock_client = None
    if provider == "bedrock":
        bedrock_client = get_bedrock_client()

    # Calculate totals
    total_calls = len(base_prompts) * len(variations) * len(temperatures) * num_runs
    total_conditions = len(base_prompts) * len(variations) * len(temperatures)

    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Provider: {provider}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Runs per condition: {num_runs}")
    print(f"  Variations: {len(variations)}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Estimated time: ~{total_calls * 0.5 / 60:.0f} minutes")

    # Run evaluation
    results = []
    start_time = time.time()
    completed = 0

    for base_idx, base in enumerate(base_prompts):
        original_prompt = base['question']
        tools = convert_openai_to_xlam(base['tools'])

        for var_name, var_config in variations.items():
            varied_prompt = var_config['transform'](original_prompt)

            for temp in temperatures:
                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_conditions - completed) / rate / 60 if rate > 0 else 0

                print(f"\r[{completed}/{total_conditions}] T={temp} {var_name[:15]:15s} | "
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
                        'base_id': base['id'],
                        'base_idx': base_idx,
                        'variation_name': var_name,
                        'category': var_config['category'],
                        'linguistic_features': var_config['features'],
                        'temperature': temp,
                        'original_prompt': original_prompt,
                        'varied_prompt': varied_prompt,
                        'metrics': metrics,
                        'tool_calls_summary': [
                            [tc.get('name') for tc in run] for run in outputs
                        ]
                    })

                except Exception as e:
                    print(f"\n  ERROR at T={temp} {var_name}: {e}")
                    results.append({
                        'base_id': base['id'],
                        'base_idx': base_idx,
                        'variation_name': var_name,
                        'category': var_config['category'],
                        'temperature': temp,
                        'error': str(e)
                    })

    elapsed_total = time.time() - start_time
    print(f"\n\nCompleted in {elapsed_total / 60:.1f} minutes")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Group results by variation and temperature
    by_variation_temp = defaultdict(list)
    for r in results:
        if 'metrics' in r:
            key = (r['variation_name'], r['temperature'])
            by_variation_temp[key].append(r['metrics']['c_mean'])

    # Compute statistics for each variation at T=1.0 (highest variance)
    stats_results = {}
    baseline_values = by_variation_temp.get(('baseline', 1.0), [])

    for var_name in variations.keys():
        values = by_variation_temp.get((var_name, 1.0), [])
        if values:
            mean, ci_lower, ci_upper = bootstrap_ci(values)

            # Compare to baseline
            if baseline_values and var_name != 'baseline':
                u_stat, p_value = mann_whitney_test(values, baseline_values)
                effect_size = cohens_d(values, baseline_values)
            else:
                u_stat, p_value, effect_size = 0.0, 1.0, 0.0

            stats_results[var_name] = {
                'mean': mean,
                'ci_95': (ci_lower, ci_upper),
                'std': np.std(values),
                'n': len(values),
                'vs_baseline_p': p_value,
                'vs_baseline_d': effect_size
            }

    # Print results table
    print("\nVariation Statistics at T=1.0 (vs Baseline):")
    print("-" * 80)
    print(f"{'Variation':<20} {'Mean':>8} {'95% CI':>18} {'p-value':>10} {'Cohen d':>10}")
    print("-" * 80)

    for var_name, stat in sorted(stats_results.items(), key=lambda x: -x[1]['mean']):
        ci_str = f"[{stat['ci_95'][0]:.3f}, {stat['ci_95'][1]:.3f}]"
        p_str = f"{stat['vs_baseline_p']:.4f}" if stat['vs_baseline_p'] < 1.0 else "N/A"
        d_str = f"{stat['vs_baseline_d']:.3f}" if stat['vs_baseline_d'] != 0 else "N/A"
        sig = "*" if stat['vs_baseline_p'] < 0.05 else ""
        print(f"{var_name:<20} {stat['mean']:>8.3f} {ci_str:>18} {p_str:>10} {d_str:>10} {sig}")

    # Save results
    output_path = Path(output_dir) / f"{display_name.replace(' ', '_').lower()}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'model_id': model_id,
            'display_name': display_name,
            'provider': provider,
            'temperatures': temperatures,
            'num_runs': num_runs,
            'num_base_prompts': len(base_prompts),
            'variations': list(variations.keys()),
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'elapsed_minutes': elapsed_total / 60
        },
        'results': results,
        'statistics': {k: {
            'mean': v['mean'],
            'ci_95_lower': v['ci_95'][0],
            'ci_95_upper': v['ci_95'][1],
            'std': v['std'],
            'n': v['n'],
            'vs_baseline_p': v['vs_baseline_p'],
            'vs_baseline_d': v['vs_baseline_d']
        } for k, v in stats_results.items()}
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved results to {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="ACL 2026 Comprehensive Linguistic Experiments")
    parser.add_argument("--model", type=str, help="Model ID to evaluate")
    parser.add_argument("--all-models", action="store_true", help="Run on all supported models")
    parser.add_argument("--num-prompts", type=int, default=30, help="Number of base prompts")
    parser.add_argument("--num-runs", type=int, default=10, help="Runs per condition")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--output-dir", type=str, default="results/acl_linguistic/comprehensive")
    parser.add_argument("--variations", type=str, nargs="+", help="Subset of variations to test")

    args = parser.parse_args()

    if args.all_models:
        # Run on key models for ACL paper
        acl_models = [
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "qwen.qwen3-235b-a22b-2507-v1:0",
            "openai/gpt-4.1-mini",
            "google/gemini-2.5-flash-lite",
        ]

        for model_id in acl_models:
            try:
                run_experiment(
                    model_id=model_id,
                    num_prompts=args.num_prompts,
                    num_runs=args.num_runs,
                    temperatures=args.temperatures,
                    output_dir=args.output_dir,
                    variations_subset=args.variations
                )
            except Exception as e:
                print(f"Error with {model_id}: {e}")
                continue

    elif args.model:
        run_experiment(
            model_id=args.model,
            num_prompts=args.num_prompts,
            num_runs=args.num_runs,
            temperatures=args.temperatures,
            output_dir=args.output_dir,
            variations_subset=args.variations
        )

    else:
        print("Please specify --model MODEL_ID or --all-models")
        sys.exit(1)


if __name__ == '__main__':
    main()
