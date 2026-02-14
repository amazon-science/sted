#!/usr/bin/env python3
"""
Small-scale test of LLM-based feature intervention.
Tests numbered_list feature with proper LLM rewriting.
"""

import json
import sys
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.eval.generate_tool_calls import (
    get_bedrock_client,
    generate_tool_calls_bedrock,
    load_toucan_dataset,
)


def has_numbered_list(text: str) -> bool:
    return bool(re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))


def llm_rewrite(client, query: str, instruction: str, model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0") -> str:
    """Use LLM to rewrite a query according to instruction."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"""Rewrite the following query according to the instruction.
Keep the semantic meaning EXACTLY the same - only change the format/style as specified.
Output ONLY the rewritten query, nothing else.

INSTRUCTION: {instruction}

ORIGINAL QUERY:
{query}

REWRITTEN QUERY:"""
                }
            ]
        }
    ]

    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={"maxTokens": 2000, "temperature": 0.0}
    )

    return response['output']['message']['content'][0]['text'].strip()


def calculate_consistency(runs: List[List[Dict]]) -> float:
    """Calculate mean pairwise similarity across runs."""
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
    if not run1 and not run2:
        return 1.0
    if not run1 or not run2:
        return 0.0
    if json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True):
        return 1.0

    names1 = set(tc.get('name', '') for tc in run1 if isinstance(tc, dict))
    names2 = set(tc.get('name', '') for tc in run2 if isinstance(tc, dict))

    if not names1 and not names2:
        return 1.0
    if not names1 or not names2:
        return 0.0

    jaccard = len(names1 & names2) / len(names1 | names2)
    return jaccard


def run_intervention_test(
    client,
    sample: Dict,
    intervention_type: str,  # "add" or "remove"
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    num_runs: int = 5,
    temperature: float = 0.5,
) -> Optional[Dict]:
    """Run a single intervention test with LLM rewriting."""

    original_query = sample.get('query', '')
    tools = sample.get('tools', [])
    sample_id = sample.get('id', 'unknown')

    # LLM-based rewriting
    if intervention_type == "add":
        instruction = "Convert the request into a numbered list format, clearly separating each distinct task or requirement. Keep all information intact."
        rewritten_query = llm_rewrite(client, original_query, instruction)
    else:
        instruction = "Convert the numbered list into natural prose/paragraph format. Keep ALL information and context intact."
        rewritten_query = llm_rewrite(client, original_query, instruction)

    # Skip if no meaningful change
    if original_query.strip() == rewritten_query.strip():
        return None

    print(f"\n  Sample: {sample_id[:8]}...")
    print(f"  Original ({len(original_query)} chars): {original_query[:80]}...")
    print(f"  Rewritten ({len(rewritten_query)} chars): {rewritten_query[:80]}...")

    # Generate with original
    try:
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
    except Exception as e:
        print(f"    Original failed: {e}")
        original_runs = [[] for _ in range(num_runs)]

    # Generate with rewritten
    try:
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
    except Exception as e:
        print(f"    Rewritten failed: {e}")
        rewritten_runs = [[] for _ in range(num_runs)]

    orig_cons = calculate_consistency(original_runs)
    rewr_cons = calculate_consistency(rewritten_runs)

    print(f"  Consistency: {orig_cons:.4f} → {rewr_cons:.4f} (Δ={rewr_cons - orig_cons:+.4f})")

    return {
        "sample_id": sample_id,
        "intervention_type": intervention_type,
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "original_consistency": orig_cons,
        "rewritten_consistency": rewr_cons,
        "delta_consistency": rewr_cons - orig_cons,
    }


def main():
    print("=" * 70)
    print("LLM-BASED FEATURE INTERVENTION TEST: has_numbered_list")
    print("=" * 70)

    # Load dataset
    dataset = load_toucan_dataset("toucan_data/toucan_tool_calls_1006.json")
    print(f"Loaded {len(dataset)} samples")

    # Split by feature
    has_list = [s for s in dataset if has_numbered_list(s.get('query', ''))]
    no_list = [s for s in dataset if not has_numbered_list(s.get('query', ''))]
    print(f"With numbered list: {len(has_list)}")
    print(f"Without numbered list: {len(no_list)}")

    client = get_bedrock_client()

    # Small sample test
    random.seed(42)
    n_samples = 3  # Small test

    results = []

    # Test ADD (add list to prompts without it)
    print(f"\n{'='*70}")
    print(f"ADD NUMBERED LIST (n={n_samples})")
    print(f"{'='*70}")
    add_samples = random.sample(no_list, min(n_samples, len(no_list)))
    for sample in add_samples:
        result = run_intervention_test(client, sample, "add", temperature=0.5, num_runs=5)
        if result:
            results.append(result)

    # Test REMOVE (remove list from prompts with it)
    print(f"\n{'='*70}")
    print(f"REMOVE NUMBERED LIST (n={n_samples})")
    print(f"{'='*70}")
    remove_samples = random.sample(has_list, min(n_samples, len(has_list)))
    for sample in remove_samples:
        result = run_intervention_test(client, sample, "remove", temperature=0.5, num_runs=5)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    add_results = [r for r in results if r['intervention_type'] == 'add']
    remove_results = [r for r in results if r['intervention_type'] == 'remove']

    if add_results:
        add_delta = np.mean([r['delta_consistency'] for r in add_results])
        print(f"ADD: Δ Consistency = {add_delta:+.4f} (n={len(add_results)})")

    if remove_results:
        remove_delta = np.mean([r['delta_consistency'] for r in remove_results])
        print(f"REMOVE: Δ Consistency = {remove_delta:+.4f} (n={len(remove_results)})")

    if results:
        overall_delta = np.mean([r['delta_consistency'] for r in results])
        print(f"OVERALL: Δ Consistency = {overall_delta:+.4f} (n={len(results)})")

    # Save results
    with open('/tmp/llm_intervention_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/llm_intervention_test_results.json")


if __name__ == "__main__":
    main()
