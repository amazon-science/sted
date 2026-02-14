#!/usr/bin/env python3
"""
Scaled-up causal intervention experiment for KDD paper.
Runs bidirectional feature interventions on Toucan and ShareGPT datasets.

Usage:
    python causal_intervention_scaled.py --dataset toucan --samples-per-feature 100
    python causal_intervention_scaled.py --dataset sharegpt --samples-per-feature 100
"""

import json
import sys
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.eval.generate_tool_calls import (
    get_bedrock_client,
    generate_tool_calls_bedrock,
)


# =============================================================================
# Feature Detection Functions
# =============================================================================

def has_should(text: str) -> bool:
    return bool(re.search(r'\bshould\b', text, re.IGNORECASE))

def has_must(text: str) -> bool:
    return bool(re.search(r'\bmust\b', text, re.IGNORECASE))

def has_if(text: str) -> bool:
    return bool(re.search(r'\bif\b', text, re.IGNORECASE))

def has_can_you(text: str) -> bool:
    return bool(re.search(r'\bcan you\b', text, re.IGNORECASE))

def has_please(text: str) -> bool:
    return bool(re.search(r'\bplease\b', text, re.IGNORECASE))

def is_short(text: str, threshold: int = 100) -> bool:
    return len(text.split()) < threshold


# =============================================================================
# Rule-Based Rewriting Functions
# =============================================================================

def add_should(text: str) -> str:
    """Add 'should' to text by replacing 'need to' or 'have to'."""
    replacements = [
        (r'\bneed to\b', 'should'),
        (r'\bhave to\b', 'should'),
        (r'\bmust\b', 'should'),
    ]
    result = text
    for pattern, replacement in replacements:
        if re.search(pattern, result, re.IGNORECASE):
            result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
            return result
    # If no replacement found, add "You should" at beginning
    if text.strip():
        return "You should " + text[0].lower() + text[1:]
    return text

def remove_should(text: str) -> str:
    """Remove 'should' by replacing with 'need to'."""
    return re.sub(r'\bshould\b', 'need to', text, flags=re.IGNORECASE)

def add_must(text: str) -> str:
    """Add 'must' by replacing 'should' or 'need to'."""
    replacements = [
        (r'\bshould\b', 'must'),
        (r'\bneed to\b', 'must'),
        (r'\bhave to\b', 'must'),
    ]
    result = text
    for pattern, replacement in replacements:
        if re.search(pattern, result, re.IGNORECASE):
            result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
            return result
    return text

def remove_must(text: str) -> str:
    """Remove 'must' by replacing with 'should'."""
    return re.sub(r'\bmust\b', 'should', text, flags=re.IGNORECASE)

def add_if(text: str) -> str:
    """Add conditional 'if' clause."""
    if not has_if(text):
        return text + " If possible, provide detailed results."
    return text

def remove_if(text: str) -> str:
    """Remove 'if' clauses (simplified)."""
    # Remove simple "if X, " patterns
    result = re.sub(r'\bif [^,]+,\s*', '', text, flags=re.IGNORECASE)
    # Remove trailing "if possible" etc
    result = re.sub(r',?\s*if (possible|applicable|needed|necessary)[.,]?\s*$', '.', result, flags=re.IGNORECASE)
    return result

def add_can_you(text: str) -> str:
    """Add polite 'Can you' phrasing."""
    if not has_can_you(text):
        # Add "Can you" at the beginning
        if text.strip():
            return "Can you " + text[0].lower() + text[1:]
    return text

def remove_can_you(text: str) -> str:
    """Remove 'Can you' phrasing."""
    result = re.sub(r'^can you\s+', '', text, flags=re.IGNORECASE)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result

def add_please(text: str) -> str:
    """Add 'please' to text."""
    if not has_please(text):
        return "Please " + text[0].lower() + text[1:] if text else text
    return text

def remove_please(text: str) -> str:
    """Remove 'please' from text."""
    result = re.sub(r'\bplease\s+', '', text, flags=re.IGNORECASE)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result

def shorten_query(text: str) -> str:
    """Shorten query by removing filler phrases."""
    fillers = [
        r'\bI would like (you )?to\b',
        r'\bI need (you )?to\b',
        r'\bI want (you )?to\b',
        r'\bCould you (please )?\b',
        r'\bWould you (please )?\b',
        r'\bPlease\b',
        r'\bkindly\b',
        r'\bas soon as possible\b',
        r'\bASAP\b',
        r'\bat your earliest convenience\b',
    ]
    result = text
    for filler in fillers:
        result = re.sub(filler, '', result, flags=re.IGNORECASE)
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result

def lengthen_query(text: str) -> str:
    """Lengthen query by adding context."""
    additions = [
        "I would appreciate if you could ",
        "Please ensure that you ",
        "It would be helpful if you could ",
    ]
    prefix = random.choice(additions)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return prefix + text


# =============================================================================
# LLM-based Semantic-Preserving Rewriting
# =============================================================================

def llm_rewrite_with_verification(
    client,
    query: str,
    feature_name: str,
    intervention_type: str,  # 'add' or 'remove'
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
) -> Tuple[str, bool, str]:
    """
    Use LLM to rewrite query while preserving semantics.
    Returns: (rewritten_query, is_semantically_equivalent, explanation)
    """

    # Define rewrite instructions per feature
    instructions = {
        'has_should': {
            'add': "Rewrite to include the word 'should' naturally (e.g., 'You should...' or 'The output should...'). Keep ALL original meaning intact.",
            'remove': "Rewrite to remove 'should' by using alternatives like 'need to', 'must', or direct imperatives. Keep ALL original meaning intact.",
        },
        'has_must': {
            'add': "Rewrite to include the word 'must' for emphasis. Keep ALL original meaning intact.",
            'remove': "Rewrite to replace 'must' with softer alternatives like 'should' or 'need to'. Keep ALL original meaning intact.",
        },
        'has_if': {
            'add': "Rewrite to include a conditional 'if' clause naturally (e.g., 'If possible...', 'If applicable...'). Keep ALL original meaning intact.",
            'remove': "Rewrite to remove conditional 'if' clauses while preserving all requirements. Keep ALL original meaning intact.",
        },
        'has_can_you': {
            'add': "Rewrite to start with polite phrasing like 'Can you...' or 'Could you...'. Keep ALL original meaning intact.",
            'remove': "Rewrite to use direct imperatives instead of 'Can you...' or 'Could you...'. Keep ALL original meaning intact.",
        },
        'has_please': {
            'add': "Rewrite to include 'please' for politeness. Keep ALL original meaning intact.",
            'remove': "Rewrite to remove 'please' while keeping the request professional. Keep ALL original meaning intact.",
        },
        'word_count': {
            'add': "Make the query more concise by removing filler words and redundant phrases. Keep ALL original requirements and meaning intact.",
            'remove': "Expand the query slightly with more explicit context or polite phrasing. Keep ALL original requirements and meaning intact.",
        },
    }

    instruction = instructions.get(feature_name, {}).get(intervention_type, "Rewrite naturally.")

    # Step 1: Rewrite
    rewrite_prompt = f"""Rewrite the following query according to the instruction.
CRITICAL: Keep the semantic meaning EXACTLY the same - only change the format/style as specified.
Do NOT add, remove, or change any information, requirements, or tasks.
Output ONLY the rewritten query, nothing else.

INSTRUCTION: {instruction}

ORIGINAL QUERY:
{query}

REWRITTEN QUERY:"""

    try:
        messages = [{"role": "user", "content": [{"text": rewrite_prompt}]}]
        response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={"maxTokens": 2000, "temperature": 0.0}
        )
        rewritten = response['output']['message']['content'][0]['text'].strip()

        # Step 2: Verify semantic equivalence
        verify_prompt = f"""Compare these two queries and determine if they are semantically equivalent.
They should request the SAME tasks/actions with the SAME requirements - only formatting/style should differ.

ORIGINAL:
{query}

REWRITTEN:
{rewritten}

Answer with JSON format ONLY:
{{"equivalent": true, "reason": "brief explanation"}}
or
{{"equivalent": false, "reason": "what changed"}}"""

        messages = [{"role": "user", "content": [{"text": verify_prompt}]}]
        response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={"maxTokens": 500, "temperature": 0.0}
        )
        verify_text = response['output']['message']['content'][0]['text'].strip()

        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', verify_text)
        if json_match:
            verify_result = json.loads(json_match.group())
            is_equivalent = verify_result.get('equivalent', False)
            reason = verify_result.get('reason', '')
        else:
            is_equivalent = False
            reason = "Could not parse verification"

        return rewritten, is_equivalent, reason

    except Exception as e:
        return query, False, f"LLM error: {e}"


# =============================================================================
# Feature Configuration
# =============================================================================

FEATURES = {
    'has_should': {
        'detect': has_should,
        'add': add_should,
        'remove': remove_should,
        'importance': 0.071,
    },
    'has_must': {
        'detect': has_must,
        'add': add_must,
        'remove': remove_must,
        'importance': 0.040,
    },
    'has_if': {
        'detect': has_if,
        'add': add_if,
        'remove': remove_if,
        'importance': 0.070,
    },
    'has_can_you': {
        'detect': has_can_you,
        'add': add_can_you,
        'remove': remove_can_you,
        'importance': 0.065,
    },
    'has_please': {
        'detect': has_please,
        'add': add_please,
        'remove': remove_please,
        'importance': 0.030,
    },
    'word_count': {
        'detect': lambda x: is_short(x, 50),  # "short" = has feature
        'add': shorten_query,  # make short
        'remove': lengthen_query,  # make long
        'importance': 0.673,
    },
}


# =============================================================================
# Tool Format Conversion
# =============================================================================

def clean_schema_for_bedrock(obj):
    """Remove JSON Schema meta-keys that Bedrock doesn't support (keys starting with $)."""
    if isinstance(obj, dict):
        return {k: clean_schema_for_bedrock(v) for k, v in obj.items() if not k.startswith('$')}
    elif isinstance(obj, list):
        return [clean_schema_for_bedrock(item) for item in obj]
    else:
        return obj


def convert_json_schema_to_xlam_params(json_schema_params: Dict) -> Dict:
    """Convert JSON Schema parameters to xLAM parameter format.

    JSON Schema format:
    {
        "type": "object",
        "properties": {"param1": {"type": "string", "description": "..."}},
        "required": ["param1"]
    }

    xLAM format:
    {
        "param1": {"type": "string", "description": "..."},
        ...
    }
    """
    if not json_schema_params:
        return {}

    # If it's already xLAM format (no "properties" key), return as-is
    if "properties" not in json_schema_params:
        return json_schema_params

    xlam_params = {}
    properties = json_schema_params.get("properties", {})
    required = json_schema_params.get("required", [])

    for param_name, param_info in properties.items():
        if isinstance(param_info, dict):
            xlam_params[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_name not in required:
                xlam_params[param_name]["default"] = None
        else:
            xlam_params[param_name] = {"type": "string"}
            if param_name not in required:
                xlam_params[param_name]["default"] = None

    return xlam_params


def convert_openai_to_xlam(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI format tools to xLAM format for Bedrock API.

    OpenAI format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    xLAM format: {"name": ..., "description": ..., "parameters": {...param_name: {type, description}...}}

    Also cleans up JSON Schema meta-keys (like $schema, $ref) that Bedrock doesn't support.
    """
    xlam_tools = []
    for t in tools:
        if isinstance(t, dict):
            if 'function' in t:
                # OpenAI format - extract from nested function object
                func = t['function']
                raw_params = clean_schema_for_bedrock(func.get('parameters', {}))
                xlam_params = convert_json_schema_to_xlam_params(raw_params)
                xlam_tools.append({
                    'name': func.get('name', ''),
                    'description': func.get('description', ''),
                    'parameters': xlam_params,
                })
            else:
                # Already in xlam format or similar
                raw_params = clean_schema_for_bedrock(t.get('parameters', {}))
                xlam_params = convert_json_schema_to_xlam_params(raw_params)
                xlam_tools.append({
                    'name': t.get('name', ''),
                    'description': t.get('description', ''),
                    'parameters': xlam_params,
                })
    return xlam_tools


# =============================================================================
# Dataset Loading
# =============================================================================

def load_toucan_dataset(path: str = "toucan_data/toucan_tool_calls_1006.json") -> List[Dict]:
    """Load Toucan dataset."""
    with open(path) as f:
        data = json.load(f)
    # Filter for valid tools
    valid = []
    skipped_duplicate = 0
    skipped_no_desc = 0
    for sample in data:
        tools = sample.get('tools', [])
        # Tools are in OpenAI format: {"type": "function", "function": {"name": ..., "description": ...}}
        def get_tool_name(t):
            if isinstance(t, dict):
                if 'function' in t:
                    return t['function'].get('name', '')
                return t.get('name', '')
            return ''
        def get_description(t):
            if isinstance(t, dict):
                if 'function' in t:
                    return t['function'].get('description', '').strip()
                return t.get('description', '').strip()
            return ''
        # Check for duplicate tool names
        tool_names = [get_tool_name(t) for t in tools]
        if len(tool_names) != len(set(tool_names)):
            skipped_duplicate += 1
            continue
        if tools and all(get_description(t) for t in tools):
            # Normalize: add 'query' key from 'question' and convert tools to xlam format
            sample_copy = sample.copy()
            sample_copy['query'] = sample.get('question', sample.get('query', ''))
            sample_copy['tools'] = convert_openai_to_xlam(tools)  # Convert to xlam format for Bedrock
            valid.append(sample_copy)
        else:
            skipped_no_desc += 1
    print(f"Toucan: Loaded {len(valid)} samples (skipped {skipped_duplicate} with duplicate tools, {skipped_no_desc} with missing descriptions)")
    sys.stdout.flush()
    return valid


def load_sharegpt_dataset(dataset_dir: str = "data/sharegpt") -> List[Dict]:
    """Load ShareGPT dataset following generate_structured_outputs.py approach."""
    import os

    # Try multiple base paths
    paths_to_try = [
        dataset_dir,
        "data/sharegpt",
        "/home/ubuntu/sted-internal/data/sharegpt",
    ]

    for base_dir in paths_to_try:
        if not os.path.exists(base_dir):
            continue

        # Filter out macOS metadata files
        dataset_list = sorted([d for d in os.listdir(base_dir)
                               if not d.startswith('._') and d != '.DS_Store'
                               and os.path.isdir(os.path.join(base_dir, d))])

        if not dataset_list:
            continue

        samples = []
        for dataset_name in dataset_list:
            data_path = os.path.join(base_dir, dataset_name, "all_conversations.json")
            if not os.path.exists(data_path):
                continue

            with open(data_path) as f:
                data = json.load(f)

            for i, item in enumerate(data):
                # Extract human query from conversations
                conversations = item.get('conversations', [])
                query = ""
                system_prompt = ""
                for conv in conversations:
                    if conv.get('from') == 'system':
                        system_prompt = conv.get('value', '')
                    elif conv.get('from') == 'human':
                        query = conv.get('value', '')
                        break

                if query and len(query) > 20:
                    samples.append({
                        'id': f"{dataset_name}_{i}",
                        'query': query,
                        'system_prompt': system_prompt,
                        'tools': [],  # ShareGPT doesn't have tools
                    })

        if samples:
            print(f"Loaded {len(samples)} samples from {base_dir}")
            return samples

    raise FileNotFoundError(f"ShareGPT dataset not found in: {paths_to_try}")


# =============================================================================
# Consistency Calculation
# =============================================================================

def calculate_consistency(runs: List[List[Dict]]) -> float:
    """Calculate mean pairwise Jaccard similarity of tool names."""
    valid_runs = [r for r in runs if r is not None and len(r) > 0]
    if len(valid_runs) < 2:
        return 1.0 if len(valid_runs) == 1 else 0.0

    similarities = []
    n = len(valid_runs)
    for i in range(n):
        for j in range(i + 1, n):
            sim = calculate_run_similarity(valid_runs[i], valid_runs[j])
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def calculate_run_similarity(run1: List[Dict], run2: List[Dict]) -> float:
    """Calculate Jaccard similarity between two runs."""
    if not run1 and not run2:
        return 1.0
    if not run1 or not run2:
        return 0.0

    # Exact match
    if json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True):
        return 1.0

    # Jaccard on tool names
    names1 = set(tc.get('name', '') for tc in run1 if isinstance(tc, dict))
    names2 = set(tc.get('name', '') for tc in run2 if isinstance(tc, dict))

    if not names1 and not names2:
        return 1.0
    if not names1 or not names2:
        return 0.0

    return len(names1 & names2) / len(names1 | names2)


# =============================================================================
# Intervention Experiment
# =============================================================================

def _single_json_inference(client, model_id: str, query: str, temperature: float, max_tokens: int) -> Optional[str]:
    """Single JSON inference call for parallel execution."""
    system_prompt = "You are a JSON generation assistant. Generate valid JSON output based on the user's request."
    try:
        messages = [{"role": "user", "content": [{"text": query}]}]
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
        )
        return response['output']['message']['content'][0]['text']
    except Exception as e:
        return None


def generate_json_outputs(
    client,
    model_id: str,
    query: str,
    num_runs: int,
    temperature: float,
    max_tokens: int = 4000,
    max_workers: int = 3,
) -> List[str]:
    """Generate JSON outputs for ShareGPT (no tools) with parallel execution."""
    results = [None] * num_runs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_single_json_inference, client, model_id, query, temperature, max_tokens): i
            for i in range(num_runs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = None

    return results


def calculate_text_consistency(outputs: List[str]) -> float:
    """Calculate consistency for text outputs (ShareGPT)."""
    valid = [o for o in outputs if o is not None]
    if len(valid) < 2:
        return 1.0 if len(valid) == 1 else 0.0

    # Use exact match ratio
    similarities = []
    n = len(valid)
    for i in range(n):
        for j in range(i + 1, n):
            # Normalize and compare
            s1 = valid[i].strip().lower()
            s2 = valid[j].strip().lower()
            if s1 == s2:
                similarities.append(1.0)
            else:
                # Use word overlap as similarity
                words1 = set(s1.split())
                words2 = set(s2.split())
                if words1 or words2:
                    jaccard = len(words1 & words2) / len(words1 | words2)
                    similarities.append(jaccard)
                else:
                    similarities.append(0.0)

    return float(np.mean(similarities)) if similarities else 0.0


def run_single_intervention(
    client,
    sample: Dict,
    feature_name: str,
    intervention_type: str,  # 'add' or 'remove'
    model_id: str,
    temperature: float,
    num_runs: int = 10,
    dataset_type: str = 'toucan',
    use_llm_rewrite: bool = False,
) -> Optional[Dict]:
    """Run a single intervention experiment."""

    feature_config = FEATURES[feature_name]
    original_query = sample.get('query', '')
    tools = sample.get('tools', [])
    sample_id = sample.get('id', 'unknown')
    system_prompt = sample.get('system_prompt', '')

    # Apply intervention
    if use_llm_rewrite:
        rewritten_query, is_equivalent, reason = llm_rewrite_with_verification(
            client, original_query, feature_name, intervention_type, model_id
        )
        if not is_equivalent:
            return None  # Skip non-semantic-preserving rewrites
    else:
        if intervention_type == 'add':
            rewritten_query = feature_config['add'](original_query)
        else:
            rewritten_query = feature_config['remove'](original_query)

    # Skip if no change
    if original_query.strip() == rewritten_query.strip():
        return None

    # Skip if too short after rewriting
    if len(rewritten_query.strip()) < 10:
        return None

    try:
        if dataset_type == 'toucan' and tools:
            # Tool calling for Toucan
            original_runs = generate_tool_calls_bedrock(
                client=client,
                model_id=model_id,
                query=original_query,
                tools=tools,
                num_runs=num_runs,
                temperature=temperature,
                max_tokens=4000,
                max_workers=3,
            )
            rewritten_runs = generate_tool_calls_bedrock(
                client=client,
                model_id=model_id,
                query=rewritten_query,
                tools=tools,
                num_runs=num_runs,
                temperature=temperature,
                max_tokens=4000,
                max_workers=3,
            )
            orig_cons = calculate_consistency(original_runs)
            rewr_cons = calculate_consistency(rewritten_runs)
        else:
            # JSON generation for ShareGPT
            original_outputs = generate_json_outputs(
                client, model_id, original_query, num_runs, temperature
            )
            rewritten_outputs = generate_json_outputs(
                client, model_id, rewritten_query, num_runs, temperature
            )
            orig_cons = calculate_text_consistency(original_outputs)
            rewr_cons = calculate_text_consistency(rewritten_outputs)

        return {
            'sample_id': sample_id,
            'feature': feature_name,
            'intervention_type': intervention_type,
            'temperature': temperature,
            'original_query': original_query[:500],
            'rewritten_query': rewritten_query[:500],
            'original_consistency': orig_cons,
            'rewritten_consistency': rewr_cons,
            'delta_consistency': rewr_cons - orig_cons,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return None


def save_result_incremental(result: Dict, output_file: Path):
    """Save a single result incrementally to JSONL file."""
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + '\n')


def load_existing_results(output_file: Path) -> Tuple[List[Dict], set]:
    """Load existing results and return set of completed (feature, temp, intervention_type, sample_id)."""
    results = []
    completed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    results.append(r)
                    # Track what's completed
                    key = (r['feature'], r['temperature'], r['intervention_type'], r['sample_id'])
                    completed.add(key)
                except:
                    pass
    return results, completed


def run_feature_experiment(
    client,
    dataset: List[Dict],
    feature_name: str,
    model_id: str,
    temperatures: List[float],
    samples_per_intervention: int = 50,
    num_runs: int = 10,
    dataset_type: str = 'toucan',
    use_llm_rewrite: bool = False,
    output_file: Path = None,
    completed_keys: set = None,
) -> List[Dict]:
    """Run intervention experiment for a single feature with incremental saving."""

    feature_config = FEATURES[feature_name]
    detect_fn = feature_config['detect']

    # Split dataset by feature presence
    has_feature = [s for s in dataset if detect_fn(s.get('query', ''))]
    no_feature = [s for s in dataset if not detect_fn(s.get('query', ''))]

    print(f"\n{'='*60}")
    print(f"Feature: {feature_name} (importance: {feature_config['importance']})")
    print(f"  With feature: {len(has_feature)}, Without: {len(no_feature)}")
    print(f"  Rewriting: {'LLM-based' if use_llm_rewrite else 'Rule-based'}")
    print(f"{'='*60}")
    sys.stdout.flush()

    results = []
    completed_keys = completed_keys or set()

    for temp in temperatures:
        print(f"\n  Temperature: {temp}")
        sys.stdout.flush()

        # ADD intervention (add feature to samples without it)
        # Sample more if using LLM rewrite (some will be rejected)
        sample_multiplier = 2 if use_llm_rewrite else 1
        add_samples = random.sample(no_feature, min(samples_per_intervention * sample_multiplier, len(no_feature)))
        add_count = 0
        for i, sample in enumerate(add_samples):
            if add_count >= samples_per_intervention:
                break

            # Check if already completed (for resume)
            sample_id = sample.get('id', f'unknown_{i}')
            key = (feature_name, temp, 'add', sample_id)
            if key in completed_keys:
                add_count += 1
                continue

            result = run_single_intervention(
                client, sample, feature_name, 'add', model_id, temp, num_runs, dataset_type, use_llm_rewrite
            )
            if result:
                results.append(result)
                add_count += 1
                # Save incrementally
                if output_file:
                    save_result_incremental(result, output_file)
                if add_count % 10 == 0:
                    print(f"    ADD: {add_count}/{samples_per_intervention}")
                    sys.stdout.flush()

        # REMOVE intervention (remove feature from samples with it)
        remove_samples = random.sample(has_feature, min(samples_per_intervention * sample_multiplier, len(has_feature)))
        remove_count = 0
        for i, sample in enumerate(remove_samples):
            if remove_count >= samples_per_intervention:
                break

            # Check if already completed (for resume)
            sample_id = sample.get('id', f'unknown_{i}')
            key = (feature_name, temp, 'remove', sample_id)
            if key in completed_keys:
                remove_count += 1
                continue

            result = run_single_intervention(
                client, sample, feature_name, 'remove', model_id, temp, num_runs, dataset_type, use_llm_rewrite
            )
            if result:
                results.append(result)
                remove_count += 1
                # Save incrementally
                if output_file:
                    save_result_incremental(result, output_file)
                if remove_count % 10 == 0:
                    print(f"    REMOVE: {remove_count}/{samples_per_intervention}")
                    sys.stdout.flush()

        print(f"    Completed: ADD={add_count}, REMOVE={remove_count}")
        sys.stdout.flush()

    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze intervention results."""

    analysis = {
        'total_samples': len(results),
        'by_feature': {},
        'aggregate': {},
    }

    # Group by feature
    features = set(r['feature'] for r in results)

    for feature in features:
        feature_results = [r for r in results if r['feature'] == feature]
        deltas = [r['delta_consistency'] for r in feature_results]
        orig_cons = [r['original_consistency'] for r in feature_results]

        # Aggregate effect
        if len(deltas) > 1:
            t_stat, p_value = stats.ttest_1samp(deltas, 0)
        else:
            t_stat, p_value = 0, 1

        # Conditional effects (low vs high baseline)
        low_baseline = [r for r in feature_results if r['original_consistency'] < 0.8]
        high_baseline = [r for r in feature_results if r['original_consistency'] >= 0.8]

        low_deltas = [r['delta_consistency'] for r in low_baseline]
        high_deltas = [r['delta_consistency'] for r in high_baseline]

        # Test difference between low and high
        if len(low_deltas) > 1 and len(high_deltas) > 1:
            t_cond, p_cond = stats.ttest_ind(low_deltas, high_deltas)
        else:
            t_cond, p_cond = 0, 1

        # Correlation between delta and original
        if len(deltas) > 2:
            corr, corr_p = stats.pearsonr(deltas, orig_cons)
        else:
            corr, corr_p = 0, 1

        analysis['by_feature'][feature] = {
            'n': len(feature_results),
            'mean_delta': np.mean(deltas) if deltas else 0,
            'std_delta': np.std(deltas) if deltas else 0,
            't_stat': t_stat,
            'p_value': p_value,
            'low_baseline_n': len(low_baseline),
            'low_baseline_delta': np.mean(low_deltas) if low_deltas else 0,
            'high_baseline_n': len(high_baseline),
            'high_baseline_delta': np.mean(high_deltas) if high_deltas else 0,
            'conditional_t': t_cond,
            'conditional_p': p_cond,
            'delta_orig_corr': corr,
            'delta_orig_corr_p': corr_p,
        }

    # Overall aggregate
    all_deltas = [r['delta_consistency'] for r in results]
    if len(all_deltas) > 1:
        t_all, p_all = stats.ttest_1samp(all_deltas, 0)
    else:
        t_all, p_all = 0, 1

    analysis['aggregate'] = {
        'mean_delta': np.mean(all_deltas) if all_deltas else 0,
        'std_delta': np.std(all_deltas) if all_deltas else 0,
        't_stat': t_all,
        'p_value': p_all,
    }

    return analysis


def print_analysis(analysis: Dict):
    """Print analysis results."""

    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)

    print(f"\nTotal samples: {analysis['total_samples']}")
    print(f"Aggregate: Δ = {analysis['aggregate']['mean_delta']*100:+.2f}% ± {analysis['aggregate']['std_delta']*100:.2f}%")
    print(f"           t = {analysis['aggregate']['t_stat']:.2f}, p = {analysis['aggregate']['p_value']:.4f}")

    print("\n" + "-"*70)
    print("BY FEATURE:")
    print("-"*70)

    for feature, stats in sorted(analysis['by_feature'].items()):
        print(f"\n{feature} (n={stats['n']}):")
        print(f"  Aggregate: Δ = {stats['mean_delta']*100:+.2f}% ± {stats['std_delta']*100:.2f}%, p = {stats['p_value']:.4f}")
        print(f"  Low baseline (<0.8):  Δ = {stats['low_baseline_delta']*100:+.2f}% (n={stats['low_baseline_n']})")
        print(f"  High baseline (≥0.8): Δ = {stats['high_baseline_delta']*100:+.2f}% (n={stats['high_baseline_n']})")
        print(f"  Conditional diff: p = {stats['conditional_p']:.4f}")
        print(f"  Δ-Original corr: r = {stats['delta_orig_corr']:.3f}, p = {stats['delta_orig_corr_p']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Scaled causal intervention experiment')
    parser.add_argument('--dataset', choices=['toucan', 'sharegpt'], required=True)
    parser.add_argument('--samples-per-feature', type=int, default=100)
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--model', default='us.anthropic.claude-sonnet-4-20250514-v1:0')
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument('--features', nargs='+', default=list(FEATURES.keys()))
    parser.add_argument('--output-dir', default='/tmp/causal_intervention')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-llm-rewrite', action='store_true',
                        help='Use LLM-based rewriting with semantic verification (slower but more accurate)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results file')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Incremental results file (JSONL format)
    incremental_file = output_dir / f"{args.dataset}_results.jsonl"

    # Load existing results if resuming
    existing_results = []
    completed_keys = set()
    if args.resume and incremental_file.exists():
        existing_results, completed_keys = load_existing_results(incremental_file)
        print(f"Resuming: loaded {len(existing_results)} existing results")
        sys.stdout.flush()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*70)
    print(f"CAUSAL INTERVENTION EXPERIMENT - {args.dataset.upper()}")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Samples per feature: {args.samples_per_feature}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Features: {args.features}")
    print(f"Runs per sample: {args.num_runs}")
    print(f"Rewriting method: {'LLM-based' if args.use_llm_rewrite else 'Rule-based'}")
    print(f"Incremental save: {incremental_file}")
    sys.stdout.flush()

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    sys.stdout.flush()
    if args.dataset == 'toucan':
        dataset = load_toucan_dataset()
    else:
        dataset = load_sharegpt_dataset()
    print(f"Loaded {len(dataset)} samples")
    sys.stdout.flush()

    # Initialize client
    client = get_bedrock_client()

    # Run experiments
    all_results = list(existing_results)

    for feature in args.features:
        if feature not in FEATURES:
            print(f"Warning: Unknown feature {feature}, skipping")
            sys.stdout.flush()
            continue

        results = run_feature_experiment(
            client=client,
            dataset=dataset,
            feature_name=feature,
            model_id=args.model,
            temperatures=args.temperatures,
            samples_per_intervention=args.samples_per_feature,
            num_runs=args.num_runs,
            dataset_type=args.dataset,
            use_llm_rewrite=args.use_llm_rewrite,
            output_file=incremental_file,
            completed_keys=completed_keys,
        )
        all_results.extend(results)
        print(f"  Feature {feature}: {len(results)} new results (total: {len(all_results)})")
        sys.stdout.flush()

    # Analyze
    analysis = analyze_results(all_results)
    print_analysis(analysis)

    # Save final results
    final_file = output_dir / f"{args.dataset}_final_{timestamp}.json"
    with open(final_file, 'w') as f:
        json.dump({
            'metadata': {
                'dataset': args.dataset,
                'model': args.model,
                'samples_per_feature': args.samples_per_feature,
                'temperatures': args.temperatures,
                'features': args.features,
                'num_runs': args.num_runs,
                'timestamp': timestamp,
            },
            'results': all_results,
            'analysis': analysis,
        }, f, indent=2)
    print(f"\nFinal results saved to {final_file}")


if __name__ == "__main__":
    main()
