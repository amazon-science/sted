#!/usr/bin/env python3
"""
LLM-based feature intervention with semantic preservation verification.
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


def has_valid_tools(sample: Dict) -> bool:
    """Check if sample has valid tools (non-empty descriptions)."""
    tools = sample.get('tools', [])
    if not tools:
        return False
    for tool in tools:
        desc = tool.get('description', '')
        if not desc or len(desc.strip()) == 0:
            return False
    return True


def llm_rewrite_with_verification(
    client,
    query: str,
    instruction: str,
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
) -> tuple[str, bool, str]:
    """
    Use LLM to rewrite query with semantic preservation verification.
    Returns: (rewritten_query, is_semantically_equivalent, explanation)
    """

    # Step 1: Rewrite
    rewrite_prompt = f"""Rewrite the following query according to the instruction.
Keep the semantic meaning EXACTLY the same - only change the format/style as specified.
Do NOT add, remove, or change any information, requirements, or tasks.
Output ONLY the rewritten query, nothing else.

INSTRUCTION: {instruction}

ORIGINAL QUERY:
{query}

REWRITTEN QUERY:"""

    messages = [{"role": "user", "content": [{"text": rewrite_prompt}]}]
    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={"maxTokens": 2000, "temperature": 0.0}
    )
    rewritten = response['output']['message']['content'][0]['text'].strip()

    # Step 2: Verify semantic equivalence
    verify_prompt = f"""Compare these two queries and determine if they are semantically equivalent.
They should request the SAME tasks/actions with the SAME requirements - only formatting should differ.

ORIGINAL:
{query}

REWRITTEN:
{rewritten}

Answer with JSON format:
{{"equivalent": true/false, "reason": "brief explanation"}}"""

    messages = [{"role": "user", "content": [{"text": verify_prompt}]}]
    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={"maxTokens": 500, "temperature": 0.0}
    )
    verify_text = response['output']['message']['content'][0]['text'].strip()

    try:
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', verify_text)
        if json_match:
            verify_result = json.loads(json_match.group())
            is_equivalent = verify_result.get('equivalent', False)
            reason = verify_result.get('reason', '')
        else:
            is_equivalent = False
            reason = "Could not parse verification"
    except:
        is_equivalent = False
        reason = "Verification failed"

    return rewritten, is_equivalent, reason


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
    intervention_type: str,
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    num_runs: int = 5,
    temperature: float = 0.5,
) -> Optional[Dict]:
    """Run a single intervention test with LLM rewriting and verification."""

    original_query = sample.get('query', '')
    tools = sample.get('tools', [])
    sample_id = sample.get('id', 'unknown')

    # LLM-based rewriting with verification
    if intervention_type == "add":
        instruction = """Convert the request into a numbered list format.
- Identify each distinct task, action, or requirement
- Number them sequentially (1., 2., 3., etc.)
- Keep ALL context, constraints, and details intact
- Preserve the intro/context before the list if applicable"""
    else:
        instruction = """Convert the numbered list into natural prose/paragraph format.
- Combine the numbered items into flowing sentences
- Keep ALL information, context, and requirements intact
- Use connectors like 'and', 'then', 'also' to join items
- Preserve any intro text before the list"""

    rewritten_query, is_equivalent, reason = llm_rewrite_with_verification(
        client, original_query, instruction
    )

    # Skip if semantically different
    if not is_equivalent:
        print(f"\n  Sample {sample_id[:8]}: SKIPPED - not semantically equivalent")
        print(f"    Reason: {reason}")
        return None

    # Skip if no meaningful change
    if original_query.strip() == rewritten_query.strip():
        print(f"\n  Sample {sample_id[:8]}: SKIPPED - no change")
        return None

    print(f"\n  Sample: {sample_id[:8]}... ({intervention_type})")
    print(f"  Original ({len(original_query)} chars): {original_query[:60]}...")
    print(f"  Rewritten ({len(rewritten_query)} chars): {rewritten_query[:60]}...")
    print(f"  Semantic check: PASSED - {reason}")

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
        print(f"    Original generation failed: {e}")
        return None

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
        print(f"    Rewritten generation failed: {e}")
        return None

    orig_cons = calculate_consistency(original_runs)
    rewr_cons = calculate_consistency(rewritten_runs)

    print(f"  Consistency: {orig_cons:.4f} → {rewr_cons:.4f} (Δ={rewr_cons - orig_cons:+.4f})")

    return {
        "sample_id": sample_id,
        "intervention_type": intervention_type,
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "semantic_verification": reason,
        "original_consistency": orig_cons,
        "rewritten_consistency": rewr_cons,
        "delta_consistency": rewr_cons - orig_cons,
    }


def main():
    print("=" * 70)
    print("LLM-BASED FEATURE INTERVENTION (with semantic verification)")
    print("Feature: has_numbered_list")
    print("=" * 70)

    # Load dataset
    dataset = load_toucan_dataset("toucan_data/toucan_tool_calls_1006.json")
    print(f"Loaded {len(dataset)} samples")

    # Filter for valid tools (non-empty descriptions)
    valid_dataset = [s for s in dataset if has_valid_tools(s)]
    print(f"Samples with valid tools: {len(valid_dataset)}")

    # Split by feature
    has_list = [s for s in valid_dataset if has_numbered_list(s.get('query', ''))]
    no_list = [s for s in valid_dataset if not has_numbered_list(s.get('query', ''))]
    print(f"With numbered list (valid): {len(has_list)}")
    print(f"Without numbered list (valid): {len(no_list)}")

    client = get_bedrock_client()

    # Test parameters
    random.seed(42)
    n_samples = 10  # Per intervention type

    results = []

    # Test ADD (add list to prompts without it)
    print(f"\n{'='*70}")
    print(f"ADD NUMBERED LIST (target n={n_samples})")
    print(f"{'='*70}")
    add_samples = random.sample(no_list, min(n_samples * 2, len(no_list)))  # Extra to account for skips
    add_count = 0
    for sample in add_samples:
        if add_count >= n_samples:
            break
        result = run_intervention_test(client, sample, "add", temperature=0.5, num_runs=5)
        if result:
            results.append(result)
            add_count += 1

    # Test REMOVE (remove list from prompts with it)
    print(f"\n{'='*70}")
    print(f"REMOVE NUMBERED LIST (target n={n_samples})")
    print(f"{'='*70}")
    remove_samples = random.sample(has_list, min(n_samples * 2, len(has_list)))
    remove_count = 0
    for sample in remove_samples:
        if remove_count >= n_samples:
            break
        result = run_intervention_test(client, sample, "remove", temperature=0.5, num_runs=5)
        if result:
            results.append(result)
            remove_count += 1

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (Semantically Verified Rewrites Only)")
    print(f"{'='*70}")

    add_results = [r for r in results if r['intervention_type'] == 'add']
    remove_results = [r for r in results if r['intervention_type'] == 'remove']

    if add_results:
        add_deltas = [r['delta_consistency'] for r in add_results]
        print(f"ADD:    Δ Consistency = {np.mean(add_deltas):+.4f} ± {np.std(add_deltas):.4f} (n={len(add_results)})")
        print(f"        Range: [{min(add_deltas):+.4f}, {max(add_deltas):+.4f}]")

    if remove_results:
        remove_deltas = [r['delta_consistency'] for r in remove_results]
        print(f"REMOVE: Δ Consistency = {np.mean(remove_deltas):+.4f} ± {np.std(remove_deltas):.4f} (n={len(remove_results)})")
        print(f"        Range: [{min(remove_deltas):+.4f}, {max(remove_deltas):+.4f}]")

    if results:
        all_deltas = [r['delta_consistency'] for r in results]
        print(f"\nOVERALL: Δ Consistency = {np.mean(all_deltas):+.4f} ± {np.std(all_deltas):.4f} (n={len(results)})")

        # Statistical test
        from scipy import stats
        if len(all_deltas) > 1:
            t_stat, p_value = stats.ttest_1samp(all_deltas, 0)
            print(f"         t={t_stat:.2f}, p={p_value:.4f}")

    # Save results
    output_file = '/tmp/llm_intervention_verified_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "feature": "has_numbered_list",
                "n_add": len(add_results),
                "n_remove": len(remove_results),
                "verification": "semantic_equivalence_check"
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
