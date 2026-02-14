#!/usr/bin/env python3
"""
ACL Paper: Full Verification Experiment with Claude-Sonnet-4

Runs comprehensive evaluation across multiple temperatures and linguistic variations.

Usage:
    python run_verification.py
"""

import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics
from typing import List, Dict, Any
import concurrent.futures
import time

import boto3
from botocore.config import Config


def get_bedrock_client(region: str = "us-east-1"):
    """Get AWS Bedrock client."""
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
# Linguistic Modification Functions
# =============================================================================

def add_modal_must(text: str) -> str:
    """Add 'must' modal verb (strong deontic)."""
    if re.search(r'\bmust\b', text, re.IGNORECASE):
        return text
    for modal in ['should', 'need to', 'have to', 'can', 'could', 'would like to', 'want to']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'must', text, count=1, flags=re.IGNORECASE)
    return "You must " + text[0].lower() + text[1:]


def add_modal_should(text: str) -> str:
    """Add 'should' modal verb (medium deontic)."""
    if re.search(r'\bshould\b', text, re.IGNORECASE):
        return text
    for modal in ['must', 'need to', 'have to', 'would like to', 'want to']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'should', text, count=1, flags=re.IGNORECASE)
    return "You should " + text[0].lower() + text[1:]


def add_modal_could(text: str) -> str:
    """Add 'could' modal verb (weak epistemic)."""
    if re.search(r'\bcould\b', text, re.IGNORECASE):
        return text
    for modal in ['must', 'should', 'need to', 'have to', 'can', 'would like to', 'want to']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'could', text, count=1, flags=re.IGNORECASE)
    return "You could " + text[0].lower() + text[1:]


def add_modal_might(text: str) -> str:
    """Add 'might' modal verb (weakest epistemic)."""
    if re.search(r'\bmight\b', text, re.IGNORECASE):
        return text
    for modal in ['must', 'should', 'need to', 'have to', 'can', 'could']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'might want to', text, count=1, flags=re.IGNORECASE)
    return "You might want to " + text[0].lower() + text[1:]


def add_please(text: str) -> str:
    """Add 'please' for positive politeness."""
    if re.search(r'\bplease\b', text, re.IGNORECASE):
        return text
    return "Please " + text[0].lower() + text[1:]


def remove_please(text: str) -> str:
    """Remove 'please' for bald on-record."""
    result = re.sub(r'\bplease\s*', '', text, flags=re.IGNORECASE)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result


def add_can_you(text: str) -> str:
    """Add 'Can you' for negative politeness (question form)."""
    if re.search(r'\bcan you\b', text, re.IGNORECASE):
        return text
    # Remove existing request phrases
    text = re.sub(r'^(I need to|I want to|I would like to)\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return "Can you " + text.rstrip('.?') + "?"


def add_would_you_mind(text: str) -> str:
    """Add 'Would you mind' for strong negative politeness."""
    if re.search(r'\bwould you mind\b', text, re.IGNORECASE):
        return text
    # Remove existing request phrases and convert to gerund
    text = re.sub(r'^(I need to|I want to|I would like to|Can you|Could you)\s*', '', text, flags=re.IGNORECASE)
    # Try to convert first verb to gerund
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


def add_conditional(text: str) -> str:
    """Add 'if possible' conditional."""
    if re.search(r'\bif possible\b', text, re.IGNORECASE):
        return text
    return text.rstrip('.') + ", if possible."


def add_when_you_can(text: str) -> str:
    """Add 'when you have a chance' hedge."""
    if re.search(r'\bwhen you\b', text, re.IGNORECASE):
        return text
    return text.rstrip('.') + ", when you have a chance."


def make_directive(text: str) -> str:
    """Convert to direct imperative."""
    text = re.sub(r'^(I would like|I want|I need|Can you|Could you|Would you mind|Please)\s*(to|you to)?\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def identity(text: str) -> str:
    """No modification (baseline)."""
    return text


# Variation definitions with linguistic feature annotations
VARIATIONS = {
    # Modal verbs (strength hierarchy)
    'modal_must': {
        'transform': add_modal_must,
        'features': {'modal': 'must', 'modal_strength': 'strong', 'modal_type': 'deontic'},
        'category': 'modal'
    },
    'modal_should': {
        'transform': add_modal_should,
        'features': {'modal': 'should', 'modal_strength': 'medium', 'modal_type': 'deontic'},
        'category': 'modal'
    },
    'modal_could': {
        'transform': add_modal_could,
        'features': {'modal': 'could', 'modal_strength': 'weak', 'modal_type': 'epistemic'},
        'category': 'modal'
    },
    'modal_might': {
        'transform': add_modal_might,
        'features': {'modal': 'might', 'modal_strength': 'weakest', 'modal_type': 'epistemic'},
        'category': 'modal'
    },

    # Politeness strategies (Brown & Levinson)
    'polite_bald': {
        'transform': remove_please,
        'features': {'politeness': 'bald', 'strategy': 'bald_on_record', 'face_threat': 'high'},
        'category': 'politeness'
    },
    'polite_please': {
        'transform': add_please,
        'features': {'politeness': 'please', 'strategy': 'positive', 'face_threat': 'medium'},
        'category': 'politeness'
    },
    'polite_can_you': {
        'transform': add_can_you,
        'features': {'politeness': 'can_you', 'strategy': 'negative', 'face_threat': 'low'},
        'category': 'politeness'
    },
    'polite_would_mind': {
        'transform': add_would_you_mind,
        'features': {'politeness': 'would_mind', 'strategy': 'negative_strong', 'face_threat': 'minimal'},
        'category': 'politeness'
    },

    # Syntactic complexity
    'syntax_directive': {
        'transform': make_directive,
        'features': {'syntax': 'imperative', 'complexity': 'simple'},
        'category': 'syntax'
    },
    'syntax_conditional': {
        'transform': add_conditional,
        'features': {'syntax': 'conditional', 'complexity': 'complex'},
        'category': 'syntax'
    },
    'syntax_hedge': {
        'transform': add_when_you_can,
        'features': {'syntax': 'hedged', 'complexity': 'complex'},
        'category': 'syntax'
    },

    # Baseline
    'baseline': {
        'transform': identity,
        'features': {'baseline': True},
        'category': 'baseline'
    },
}


# =============================================================================
# Bedrock API Functions
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


def single_inference(client, model_id, messages, tools, temperature, max_tokens):
    """Run single inference."""
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


def generate_tool_calls(client, model_id, query, tools, num_runs=10, temperature=0.7):
    """Generate tool calls with parallel execution."""
    bedrock_tools = [xlam_tool_to_bedrock_tool(t) for t in tools]
    messages = [{"role": "user", "content": [{"text": query}]}]

    all_runs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_runs, 10)) as executor:
        futures = [
            executor.submit(single_inference, client, model_id, messages, bedrock_tools, temperature, 4000)
            for _ in range(num_runs)
        ]
        for future in concurrent.futures.as_completed(futures):
            all_runs.append(future.result() or [])

    return all_runs


def compute_consistency(outputs: list) -> dict:
    """Compute consistency metrics."""
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


def run_verification():
    """Run full verification experiment."""
    print("=" * 70)
    print("ACL Linguistic Variation Verification - Claude-Sonnet-4")
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

    print(f"Found {len(suitable)} suitable prompts")

    # Select base prompts
    NUM_BASE = 20
    base_prompts = suitable[:NUM_BASE]
    print(f"Using {len(base_prompts)} base prompts")

    # Configuration
    client = get_bedrock_client(region="us-east-1")
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    temperatures = [0.3, 0.5, 0.7, 1.0]
    num_runs = 10

    total_calls = len(base_prompts) * len(VARIATIONS) * len(temperatures) * num_runs
    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Runs per condition: {num_runs}")
    print(f"  Variations: {len(VARIATIONS)}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Estimated time: ~{total_calls * 0.5 / 60:.0f} minutes")

    # Run evaluation
    results = []
    start_time = time.time()
    completed = 0
    total_conditions = len(base_prompts) * len(VARIATIONS) * len(temperatures)

    for base_idx, base in enumerate(base_prompts):
        original_prompt = base['question']
        tools = convert_openai_to_xlam(base['tools'])

        for var_name, var_config in VARIATIONS.items():
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
                        client=client,
                        model_id=model_id,
                        query=varied_prompt,
                        tools=tools,
                        num_runs=num_runs,
                        temperature=temp,
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
                        'variation_name': var_name,
                        'temperature': temp,
                        'error': str(e)
                    })

    print(f"\n\nCompleted in {(time.time() - start_time) / 60:.1f} minutes")

    # Save results
    output_path = Path("results/acl_linguistic/verification_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'model': model_id,
                'temperatures': temperatures,
                'num_runs': num_runs,
                'num_base_prompts': len(base_prompts),
                'variations': list(VARIATIONS.keys()),
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results)
            },
            'results': results
        }, f, indent=2)

    print(f"Saved {len(results)} results to {output_path}")

    # Generate summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # By variation and temperature
    summary = defaultdict(lambda: defaultdict(list))
    for r in results:
        if 'metrics' in r:
            key = (r['variation_name'], r['temperature'])
            summary[key]['c_mean'].append(r['metrics']['c_mean'])
            summary[key]['validity'].append(r['metrics']['validity_rate'])

    # Print table
    print("\n{:<20} {:>8} {:>8} {:>8} {:>8}".format("Variation", "T=0.3", "T=0.5", "T=0.7", "T=1.0"))
    print("-" * 60)

    for var_name in VARIATIONS.keys():
        row = [var_name]
        for temp in temperatures:
            key = (var_name, temp)
            if key in summary:
                mean = statistics.mean(summary[key]['c_mean'])
                row.append(f"{mean:.3f}")
            else:
                row.append("N/A")
        print("{:<20} {:>8} {:>8} {:>8} {:>8}".format(*row))

    # By category
    print("\n" + "-" * 60)
    print("By Category (averaged across temperatures):")

    by_category = defaultdict(list)
    for r in results:
        if 'metrics' in r:
            cat = r.get('category', 'unknown')
            by_category[cat].append(r['metrics']['c_mean'])

    for cat, values in sorted(by_category.items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        print(f"  {cat}: mean={mean:.3f}, std={std:.3f}, n={len(values)}")

    # Modal strength comparison
    print("\n" + "-" * 60)
    print("Modal Strength Effect (T=1.0 only):")

    modal_by_strength = defaultdict(list)
    for r in results:
        if 'metrics' in r and r['temperature'] == 1.0:
            features = r.get('linguistic_features', {})
            if 'modal_strength' in features:
                modal_by_strength[features['modal_strength']].append(r['metrics']['c_mean'])

    for strength in ['strong', 'medium', 'weak', 'weakest']:
        if strength in modal_by_strength:
            values = modal_by_strength[strength]
            mean = statistics.mean(values)
            print(f"  {strength}: mean c_mean = {mean:.3f} (n={len(values)})")

    # Politeness strategy comparison
    print("\n" + "-" * 60)
    print("Politeness Strategy Effect (T=1.0 only):")

    pol_by_strategy = defaultdict(list)
    for r in results:
        if 'metrics' in r and r['temperature'] == 1.0:
            features = r.get('linguistic_features', {})
            if 'strategy' in features:
                pol_by_strategy[features['strategy']].append(r['metrics']['c_mean'])

    for strategy in ['bald_on_record', 'positive', 'negative', 'negative_strong']:
        if strategy in pol_by_strategy:
            values = pol_by_strategy[strategy]
            mean = statistics.mean(values)
            print(f"  {strategy}: mean c_mean = {mean:.3f} (n={len(values)})")


if __name__ == '__main__':
    run_verification()
