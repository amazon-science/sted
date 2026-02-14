#!/usr/bin/env python3
"""
ACL Paper: Pilot Evaluation Script

Runs a small pilot test on 10 linguistic variations to validate the pipeline.

Usage:
    python run_pilot.py
"""

import json
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


def clean_schema_for_bedrock(obj):
    """Remove JSON Schema meta-keys that Bedrock doesn't support."""
    if isinstance(obj, dict):
        return {k: clean_schema_for_bedrock(v) for k, v in obj.items() if not k.startswith('$')}
    elif isinstance(obj, list):
        return [clean_schema_for_bedrock(item) for item in obj]
    else:
        return obj


def convert_openai_to_xlam(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI format tools to xLAM format for Bedrock API."""
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
            else:
                params = clean_schema_for_bedrock(t.get('parameters', {}))
                xlam_tools.append({
                    'name': t.get('name', ''),
                    'description': t.get('description', ''),
                    'parameters': params,
                })
    return xlam_tools


def xlam_tool_to_bedrock_tool(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """Convert xLAM tool definition to Bedrock Converse API format."""
    name = tool_def.get("name", "unknown_function")
    description = tool_def.get("description", "No description provided")
    params = tool_def.get("parameters", {})

    # Build JSON schema for parameters
    properties = {}
    required = []

    if 'properties' in params:
        # Already in JSON Schema format
        properties = params.get('properties', {})
        required = params.get('required', [])
    else:
        # Convert from xLAM parameter format
        for param_name, param_info in params.items():
            if isinstance(param_info, dict):
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                type_map = {
                    "str": "string", "string": "string",
                    "int": "integer", "integer": "integer",
                    "float": "number", "number": "number",
                    "bool": "boolean", "boolean": "boolean",
                    "list": "array", "array": "array",
                    "dict": "object", "object": "object",
                }
                json_type = type_map.get(param_type.lower(), "string")
                properties[param_name] = {"type": json_type, "description": param_desc}
                if param_info.get("default") is None:
                    required.append(param_name)

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


def single_bedrock_inference(client, model_id, messages, tools, temperature, max_tokens, run_idx):
    """Run a single inference with Bedrock Converse API."""
    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            toolConfig={"tools": tools},
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
        )

        # Extract tool calls from response
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


def generate_tool_calls(client, model_id, query, tools, num_runs=5, temperature=0.7, max_tokens=4000):
    """Generate tool calls using Bedrock Converse API with parallel execution."""
    bedrock_tools = [xlam_tool_to_bedrock_tool(t) for t in tools]
    messages = [{"role": "user", "content": [{"text": query}]}]

    max_workers = min(num_runs, 5)
    all_runs = [None] * num_runs

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                single_bedrock_inference,
                client, model_id, messages, bedrock_tools,
                temperature, max_tokens, i
            ): i for i in range(num_runs)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_runs[idx] = future.result()
            except Exception as e:
                print(f"    Run {idx + 1} failed: {e}")
                all_runs[idx] = []

    return all_runs


def compute_consistency(outputs: list) -> dict:
    """Compute consistency metrics from multiple outputs."""
    valid_outputs = [o for o in outputs if o]  # Filter empty outputs

    if len(valid_outputs) < 2:
        return {
            'validity_rate': len(valid_outputs) / len(outputs) if outputs else 0,
            'c_mean': 0.0,
            'c_std': 0.0,
            'num_valid': len(valid_outputs)
        }

    # Compute pairwise Jaccard similarity of tool names
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
                intersection = len(tools1 & tools2)
                union = len(tools1 | tools2)
                sim = intersection / union if union > 0 else 0.0
            similarities.append(sim)

    c_mean = statistics.mean(similarities) if similarities else 0.0
    c_std = statistics.stdev(similarities) if len(similarities) > 1 else 0.0

    return {
        'validity_rate': len(valid_outputs) / len(outputs),
        'c_mean': c_mean,
        'c_std': c_std,
        'num_valid': len(valid_outputs)
    }


def run_pilot():
    """Run pilot evaluation on 10 variations."""
    print("=" * 60)
    print("ACL Linguistic Variation Pilot Test")
    print("=" * 60)

    # Load variations
    variations_path = Path("data/acl_variations/linguistic_variations_full.json")
    if not variations_path.exists():
        print(f"Error: Variations file not found: {variations_path}")
        return

    with open(variations_path) as f:
        all_variations = json.load(f)

    print(f"Loaded {len(all_variations)} total variations")

    # Select 12 diverse variations for pilot (3 from each type)
    selected = []
    by_type = defaultdict(list)
    for v in all_variations:
        by_type[v['variation_type']].append(v)

    for vtype, vars in by_type.items():
        selected.extend(vars[:3])  # Take first 3 of each type

    selected = selected[:12]  # Limit to 12 total
    print(f"Selected {len(selected)} variations for pilot")

    # Setup
    client = get_bedrock_client(region="us-east-1")
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    temperature = 0.5
    num_runs = 5

    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Temperature: {temperature}")
    print(f"  Runs per variation: {num_runs}")
    print(f"  Total API calls: {len(selected) * num_runs}")

    # Run evaluation
    results = []

    for idx, var in enumerate(selected):
        print(f"\n[{idx+1}/{len(selected)}] {var['variation_type']}:{var['variation_subtype']}")
        print(f"  Prompt: {var['varied_prompt'][:60]}...")

        # Convert Toucan tools to xLAM format
        tools = var['tools']
        xlam_tools = convert_openai_to_xlam(tools)

        # Generate outputs
        try:
            outputs = generate_tool_calls(
                client=client,
                model_id=model_id,
                query=var['varied_prompt'],
                tools=xlam_tools,
                num_runs=num_runs,
                temperature=temperature,
            )

            # Compute metrics
            metrics = compute_consistency(outputs)

            result = {
                'variation_id': var['variation_id'],
                'variation_type': var['variation_type'],
                'variation_subtype': var['variation_subtype'],
                'linguistic_features': var['linguistic_features'],
                'varied_prompt': var['varied_prompt'],
                'metrics': metrics,
                'outputs': outputs
            }
            results.append(result)

            print(f"  c_mean: {metrics['c_mean']:.3f}, validity: {metrics['validity_rate']:.2f}, valid: {metrics['num_valid']}/{num_runs}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'variation_id': var['variation_id'],
                'variation_type': var['variation_type'],
                'variation_subtype': var['variation_subtype'],
                'error': str(e)
            })

    # Save results
    output_path = Path("results/acl_linguistic/pilot_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'model': model_id,
            'temperature': temperature,
            'num_runs': num_runs,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nSaved results to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Pilot Summary")
    print("=" * 60)

    by_type = defaultdict(list)
    for r in results:
        if 'metrics' in r:
            by_type[r['variation_type']].append(r['metrics']['c_mean'])

    print("\nConsistency by variation type:")
    for vtype, values in sorted(by_type.items()):
        mean = statistics.mean(values) if values else 0
        print(f"  {vtype}: mean c_mean = {mean:.3f} (n={len(values)})")

    # Overall
    all_cmeans = [r['metrics']['c_mean'] for r in results if 'metrics' in r]
    if all_cmeans:
        print(f"\nOverall: mean c_mean = {statistics.mean(all_cmeans):.3f}")


if __name__ == '__main__':
    run_pilot()
