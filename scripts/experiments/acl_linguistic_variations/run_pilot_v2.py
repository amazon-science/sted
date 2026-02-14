#!/usr/bin/env python3
"""
ACL Paper: Pilot Evaluation Script v2

Uses original prompts with linguistic modifications (similar to KDD causal intervention).

Usage:
    python run_pilot_v2.py
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

import boto3
from botocore.config import Config


def get_bedrock_client(region: str = "us-west-2"):
    """Get AWS Bedrock client."""
    boto_config = Config(
        retries={'max_attempts': 20, 'mode': 'adaptive'},
        max_pool_connections=50,
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
    """Add 'must' modal verb."""
    if re.search(r'\bmust\b', text, re.IGNORECASE):
        return text
    # Replace existing modals
    for modal in ['should', 'need to', 'have to', 'can', 'could']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'must', text, count=1, flags=re.IGNORECASE)
    # Add at beginning
    return "You must " + text[0].lower() + text[1:]


def add_modal_should(text: str) -> str:
    """Add 'should' modal verb."""
    if re.search(r'\bshould\b', text, re.IGNORECASE):
        return text
    for modal in ['must', 'need to', 'have to']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'should', text, count=1, flags=re.IGNORECASE)
    return "You should " + text[0].lower() + text[1:]


def add_modal_could(text: str) -> str:
    """Add 'could' modal verb."""
    if re.search(r'\bcould\b', text, re.IGNORECASE):
        return text
    for modal in ['must', 'should', 'need to', 'have to', 'can']:
        if re.search(rf'\b{modal}\b', text, re.IGNORECASE):
            return re.sub(rf'\b{modal}\b', 'could', text, count=1, flags=re.IGNORECASE)
    return "You could " + text[0].lower() + text[1:]


def add_please(text: str) -> str:
    """Add 'please' for politeness."""
    if re.search(r'\bplease\b', text, re.IGNORECASE):
        return text
    return "Please " + text[0].lower() + text[1:]


def remove_please(text: str) -> str:
    """Remove 'please'."""
    result = re.sub(r'\bplease\s*', '', text, flags=re.IGNORECASE)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result


def add_can_you(text: str) -> str:
    """Add polite 'Can you' question form."""
    if re.search(r'\bcan you\b', text, re.IGNORECASE):
        return text
    return "Can you " + text[0].lower() + text[1:].rstrip('.') + "?"


def add_conditional(text: str) -> str:
    """Add conditional 'if possible' clause."""
    if re.search(r'\bif possible\b', text, re.IGNORECASE):
        return text
    return text.rstrip('.') + ", if possible."


def make_directive(text: str) -> str:
    """Convert to direct imperative form."""
    # Remove hedging phrases
    text = re.sub(r'^(I would like|I want|I need|Can you|Could you|Please)\s*(to|you to)?\s*', '', text, flags=re.IGNORECASE)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


# Variation definitions
VARIATIONS = {
    'modal_must': {'transform': add_modal_must, 'feature': {'modal': 'must', 'modal_strength': 'strong'}},
    'modal_should': {'transform': add_modal_should, 'feature': {'modal': 'should', 'modal_strength': 'medium'}},
    'modal_could': {'transform': add_modal_could, 'feature': {'modal': 'could', 'modal_strength': 'weak'}},
    'polite_please': {'transform': add_please, 'feature': {'politeness': 'please', 'strategy': 'positive'}},
    'polite_none': {'transform': remove_please, 'feature': {'politeness': 'none', 'strategy': 'bald'}},
    'polite_can_you': {'transform': add_can_you, 'feature': {'politeness': 'can_you', 'strategy': 'negative'}},
    'syntax_conditional': {'transform': add_conditional, 'feature': {'syntax': 'conditional', 'complexity': 'complex'}},
    'syntax_directive': {'transform': make_directive, 'feature': {'syntax': 'directive', 'complexity': 'simple'}},
}


# =============================================================================
# Bedrock API Functions
# =============================================================================

def clean_schema_for_bedrock(obj):
    """Remove JSON Schema meta-keys that Bedrock doesn't support."""
    if isinstance(obj, dict):
        return {k: clean_schema_for_bedrock(v) for k, v in obj.items() if not k.startswith('$')}
    elif isinstance(obj, list):
        return [clean_schema_for_bedrock(item) for item in obj]
    else:
        return obj


def convert_openai_to_xlam(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI format tools to xLAM format."""
    xlam_tools = []
    for t in tools:
        if isinstance(t, dict):
            if 'function' in t:
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
    name = tool_def.get("name", "unknown")
    description = tool_def.get("description", "")
    params = tool_def.get("parameters", {})

    properties = params.get('properties', {})
    required = params.get('required', [])

    return {
        "toolSpec": {
            "name": name,
            "description": description,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    }


def single_inference(client, model_id, messages, tools, temperature, max_tokens, run_idx):
    """Run single inference."""
    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            toolConfig={"tools": tools},
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
        )

        tool_calls = []
        output = response.get('output', {})
        message = output.get('message', {})
        content = message.get('content', [])

        for item in content:
            if 'toolUse' in item:
                tool_use = item['toolUse']
                tool_calls.append({
                    'name': tool_use.get('name', ''),
                    'arguments': tool_use.get('input', {})
                })

        return tool_calls

    except Exception as e:
        print(f"    Run {run_idx + 1} error: {e}")
        return []


def generate_tool_calls(client, model_id, query, tools, num_runs=5, temperature=0.7):
    """Generate tool calls with parallel execution."""
    bedrock_tools = [xlam_tool_to_bedrock_tool(t) for t in tools]
    messages = [{"role": "user", "content": [{"text": query}]}]

    all_runs = [None] * num_runs
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_runs, 5)) as executor:
        future_to_idx = {
            executor.submit(single_inference, client, model_id, messages, bedrock_tools, temperature, 4000, i): i
            for i in range(num_runs)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            all_runs[idx] = future.result() or []

    return all_runs


def compute_consistency(outputs: list) -> dict:
    """Compute consistency metrics."""
    valid_outputs = [o for o in outputs if o]

    if len(valid_outputs) < 2:
        return {
            'validity_rate': len(valid_outputs) / len(outputs) if outputs else 0,
            'c_mean': 0.0,
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


def run_pilot():
    """Run pilot with original prompts and linguistic modifications."""
    print("=" * 60)
    print("ACL Linguistic Variation Pilot v2")
    print("=" * 60)

    # Load original Toucan data
    data_path = Path("data/toucan/toucan_tool_calls_1006.json")
    with open(data_path) as f:
        toucan_data = json.load(f)

    # Filter for English, single-tool prompts
    suitable = []
    for item in toucan_data:
        q = item.get('question', '')
        if q.isascii() and item.get('num_tool_calls') == 1 and 50 < len(q) < 500:
            suitable.append(item)

    print(f"Found {len(suitable)} suitable prompts")

    # Select 5 base prompts for pilot
    base_prompts = suitable[:5]
    print(f"Selected {len(base_prompts)} base prompts")

    # Setup
    client = get_bedrock_client(region="us-east-1")
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    temperature = 0.5
    num_runs = 5

    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Temperature: {temperature}")
    print(f"  Runs per variation: {num_runs}")
    print(f"  Variations: {len(VARIATIONS)}")
    print(f"  Total API calls: {len(base_prompts) * len(VARIATIONS) * num_runs}")

    # Run evaluation
    results = []
    total = len(base_prompts) * len(VARIATIONS)
    idx = 0

    for base in base_prompts:
        original_prompt = base['question']
        tools = convert_openai_to_xlam(base['tools'])

        print(f"\n--- Base prompt: {original_prompt[:50]}... ---")

        for var_name, var_config in VARIATIONS.items():
            idx += 1
            # Apply transformation
            varied_prompt = var_config['transform'](original_prompt)

            print(f"\n[{idx}/{total}] {var_name}")
            print(f"  Original: {original_prompt[:50]}...")
            print(f"  Varied:   {varied_prompt[:50]}...")

            try:
                outputs = generate_tool_calls(
                    client=client,
                    model_id=model_id,
                    query=varied_prompt,
                    tools=tools,
                    num_runs=num_runs,
                    temperature=temperature,
                )

                metrics = compute_consistency(outputs)

                results.append({
                    'base_id': base['id'],
                    'variation_name': var_name,
                    'linguistic_features': var_config['feature'],
                    'original_prompt': original_prompt,
                    'varied_prompt': varied_prompt,
                    'metrics': metrics,
                    'outputs': outputs
                })

                print(f"  c_mean: {metrics['c_mean']:.3f}, validity: {metrics['validity_rate']:.2f}, valid: {metrics['num_valid']}/{num_runs}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'base_id': base['id'],
                    'variation_name': var_name,
                    'error': str(e)
                })

    # Save results
    output_path = Path("results/acl_linguistic/pilot_v2_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'model': model_id,
                'temperature': temperature,
                'num_runs': num_runs,
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }, f, indent=2)

    print(f"\nSaved results to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary by Variation")
    print("=" * 60)

    by_var = defaultdict(list)
    for r in results:
        if 'metrics' in r:
            by_var[r['variation_name']].append(r['metrics']['c_mean'])

    for var_name, values in sorted(by_var.items()):
        mean = statistics.mean(values) if values else 0
        print(f"  {var_name}: mean c_mean = {mean:.3f} (n={len(values)})")


if __name__ == '__main__':
    run_pilot()
