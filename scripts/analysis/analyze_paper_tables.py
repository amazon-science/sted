#!/usr/bin/env python3
"""
Analyze ShareGPT prompt factors and Toucan parameter complexity across all temperatures.
Used to generate accurate data for ICML paper tables.
"""

import json
import os
import re
from scipy import stats
from collections import defaultdict

# Final models to include in analysis
FINAL_MODELS = [
    "Claude-3.5-Sonnet", "Claude-3.7-Sonnet", "Claude-3.5-Haiku",
    "Claude-Haiku-4.5", "Claude-Opus-4", "Claude-Opus-4.5",
    "Claude-Sonnet-4", "Claude-Sonnet-4.5", "Qwen3-235B-A22B",
    "Qwen3-32B", "Llama-3.3-70B", "Nova-2-Lite", "Grok-4.1-Fast",
    "Minimax-M2", "Mimo-V2-Flash", "GPT-OSS-120B", "GPT-4.1-Mini",
    "Gemini-2.5-Flash-Lite"
]

# Model name mapping for Toucan (some models have different names)
TOUCAN_MODEL_MAPPING = {
    "Claude-3.5-Sonnet": "Claude-3.5-Sonnet",
    "Claude-3.7-Sonnet": "Claude-3.7-Sonnet",
    "Claude-3.5-Haiku": "Claude-3.5-Haiku",
    "Claude-Haiku-4.5": "Claude-Haiku-4.5",
    "Claude-Opus-4": "us.anthropic.claude-opus-4-20250514-v1",
    "Claude-Opus-4.5": "Claude-Opus-4.5",
    "Claude-Sonnet-4": "Claude-Sonnet-4",
    "Claude-Sonnet-4.5": "Claude-Sonnet-4.5",
    "Qwen3-235B-A22B": "Qwen3-235B-A22B",
    "Qwen3-32B": "Qwen3-32B",
    "Llama-3.3-70B": "Llama-3.3-70B",
    "Nova-2-Lite": "us.amazon.nova-lite-v1",
    "Grok-4.1-Fast": "Grok-4.1-Fast",
    "Minimax-M2": "Minimax-M2",
    "Mimo-V2-Flash": "Mimo-V2-Flash",
    "GPT-OSS-120B": "GPT-OSS-120B",
    "GPT-4.1-Mini": "GPT-4.1-Mini",
    "Gemini-2.5-Flash-Lite": "Gemini-2.5-Flash-Lite"
}


def analyze_sharegpt_prompt_factors(results_path: str, data_path: str):
    """Analyze ShareGPT prompt factors vs consistency across all temperatures."""

    print("=" * 60)
    print("ShareGPT Prompt Factors Analysis (All Temperatures)")
    print("=" * 60)

    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Find matched models
    matched_models = []
    for model_key in results.keys():
        for fm in FINAL_MODELS:
            if fm in model_key:
                matched_models.append((fm, model_key))
                break

    print(f"Matched models: {len(matched_models)}")

    # Calculate stability per sample across ALL temperatures
    sample_stability = defaultdict(list)

    for final_model, result_key in matched_models:
        for entry in results[result_key]:
            sample_idx = entry['sample_idx']
            if entry.get('stability_score') is not None:
                sample_stability[sample_idx].append(entry['stability_score'])

    # Average stability per sample
    sample_avg_stability = {}
    for idx, scores in sample_stability.items():
        if scores:
            sample_avg_stability[idx] = sum(scores) / len(scores)

    print(f"Samples with stability data: {len(sample_avg_stability)}")

    # Load prompts and extract features
    prompt_features = {}
    sample_idx = 0

    for subdir in ['sharegpt-quizz-generation-json-output', 'sharegpt-structured-output-json']:
        dir_path = os.path.join(data_path, subdir)
        if not os.path.exists(dir_path):
            continue

        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
        for f in files:
            with open(os.path.join(dir_path, f), 'r') as file:
                try:
                    data = json.load(file)
                    prompt = ""
                    if 'conversations' in data:
                        for conv in data['conversations']:
                            if conv.get('from') == 'human':
                                prompt = conv.get('value', '')
                                break
                    elif 'prompt' in data:
                        prompt = data['prompt']

                    if not prompt:
                        sample_idx += 1
                        continue

                    # Extract features
                    has_json_example = bool(re.search(r'\{[^}]*"[^"]*":', prompt))
                    has_table = bool(re.search(r'\|.*\|.*\|', prompt))
                    has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s', prompt, re.MULTILINE))
                    has_bullets = bool(re.search(r'^\s*[-*•]\s', prompt, re.MULTILINE))

                    sentences = re.split(r'[.!?]+', prompt)
                    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
                    has_structure = has_json_example or has_table or has_numbered_list or has_bullets
                    prose_heavy = avg_sentence_len > 20 and not has_structure

                    prompt_features[sample_idx] = {
                        'has_json_example': has_json_example,
                        'has_any_structure': has_structure,
                        'prose_heavy': prose_heavy,
                        'question_marks': prompt.count('?'),
                        'prompt_length': len(prompt)
                    }
                    sample_idx += 1
                except Exception as e:
                    sample_idx += 1
                    continue

    print(f"Prompts analyzed: {len(prompt_features)}")

    # Merge stability with features
    merged = []
    for sample_idx in sample_avg_stability:
        if sample_idx in prompt_features:
            merged.append({
                'sample_idx': sample_idx,
                'stability': sample_avg_stability[sample_idx],
                **prompt_features[sample_idx]
            })

    print(f"Merged samples: {len(merged)}")

    if len(merged) == 0:
        print("ERROR: No merged samples!")
        return

    # Quartile-based thresholds
    stabilities = sorted([m['stability'] for m in merged])
    q1 = stabilities[len(stabilities) // 4]
    q3 = stabilities[3 * len(stabilities) // 4]

    consistent = [m for m in merged if m['stability'] >= q3]
    inconsistent = [m for m in merged if m['stability'] <= q1]

    print(f"\nThresholds: Q1={q1:.3f}, Q3={q3:.3f}")
    print(f"Consistent (S_α >= {q3:.2f}): N={len(consistent)}")
    print(f"Inconsistent (S_α <= {q1:.2f}): N={len(inconsistent)}")

    # Helper functions
    def pct(samples, feature):
        if not samples:
            return 0
        return sum(1 for s in samples if s[feature]) / len(samples) * 100

    def avg_val(samples, feature):
        if not samples:
            return 0
        return sum(s[feature] for s in samples) / len(samples)

    # Print results table
    print(f"\n{'Factor':<25} {'Consistent':>12} {'Inconsistent':>12} {'Ratio':>10}")
    print("-" * 60)

    cons_q = avg_val(consistent, 'question_marks')
    incons_q = avg_val(inconsistent, 'question_marks')
    ratio_q = f"{incons_q/cons_q:.1f}x" if cons_q > 0 else "N/A"
    print(f"{'Question marks (avg)':<25} {cons_q:>12.1f} {incons_q:>12.1f} {ratio_q:>10}")

    cons_len = avg_val(consistent, 'prompt_length')
    incons_len = avg_val(inconsistent, 'prompt_length')
    ratio_len = f"{incons_len/cons_len:.1f}x" if cons_len > 0 else "N/A"
    print(f"{'Prompt length (chars)':<25} {cons_len:>12.0f} {incons_len:>12.0f} {ratio_len:>10}")

    cons_prose = pct(consistent, 'prose_heavy')
    incons_prose = pct(inconsistent, 'prose_heavy')
    ratio_prose = f"{incons_prose/cons_prose:.1f}x" if cons_prose > 0 else "N/A"
    print(f"{'Prose-heavy':<25} {cons_prose:>11.1f}% {incons_prose:>11.1f}% {ratio_prose:>10}")

    # Correlations
    stability_vals = [m['stability'] for m in merged]
    length_vals = [m['prompt_length'] for m in merged]
    r, p = stats.pearsonr(length_vals, stability_vals)
    print(f"\nCorrelation (prompt length vs stability): r={r:.2f}, p={p:.3f}")

    # Constraint score
    for m in merged:
        score = 0
        if m['has_json_example']:
            score += 3
        if m['has_any_structure']:
            score += 1
        if m['prose_heavy']:
            score -= 1.5
        score -= 0.2 * min(m['question_marks'], 5)
        m['constraint_score'] = score

    constraint_vals = [m['constraint_score'] for m in merged]
    r2, p2 = stats.pearsonr(constraint_vals, stability_vals)
    print(f"Correlation (constraint score vs stability): r={r2:.2f}, p={p2:.3f}")

    return {
        'n_consistent': len(consistent),
        'n_inconsistent': len(inconsistent),
        'q1': q1,
        'q3': q3,
        'question_marks': (cons_q, incons_q),
        'prompt_length': (cons_len, incons_len),
        'prose_heavy': (cons_prose, incons_prose),
        'correlation_length': (r, p),
        'correlation_constraint': (r2, p2)
    }


def analyze_toucan_parameter_complexity(results_path: str, data_path: str):
    """Analyze Toucan parameter complexity vs consistency across all temperatures."""

    print("\n" + "=" * 60)
    print("Toucan Parameter Complexity Analysis (All Temperatures)")
    print("=" * 60)

    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Find matched models using mapping
    matched_models = []
    for final_model in FINAL_MODELS:
        mapped_name = TOUCAN_MODEL_MAPPING.get(final_model, final_model)

        # Try exact match first
        if mapped_name in results:
            matched_models.append((final_model, mapped_name))
            continue

        # Try partial match
        for model_key in results.keys():
            if final_model in model_key or mapped_name in model_key:
                matched_models.append((final_model, model_key))
                break

    print(f"Matched models: {len(matched_models)}")
    for fm, mk in matched_models:
        print(f"  {fm} -> {mk}")

    # Calculate stability per sample across ALL temperatures
    sample_stability = defaultdict(list)

    for final_model, result_key in matched_models:
        for entry in results[result_key]:
            sample_idx = entry['sample_idx']
            if entry.get('stability_score') is not None:
                sample_stability[sample_idx].append(entry['stability_score'])

    # Average stability per sample
    sample_avg_stability = {}
    for idx, scores in sample_stability.items():
        if scores:
            sample_avg_stability[idx] = sum(scores) / len(scores)

    print(f"Samples with stability data: {len(sample_avg_stability)}")

    # Load Toucan ground truth data from single JSON file
    toucan_file = os.path.join(data_path, 'toucan_tool_calls_1006.json')
    sample_complexity = {}

    with open(toucan_file, 'r') as f:
        toucan_data = json.load(f)

    for i, sample in enumerate(toucan_data):
        tool_calls = sample.get('tool_calls', [])

        has_nested = False
        has_list = False
        has_dict = False
        max_depth = 0

        for tc in tool_calls:
            # Use 'arguments' field (not 'parameters')
            params = tc.get('arguments', tc.get('parameters', {}))
            if isinstance(params, dict):
                for key, val in params.items():
                    if isinstance(val, list):
                        has_list = True
                        if any(isinstance(item, (dict, list)) for item in val):
                            has_nested = True
                            max_depth = max(max_depth, 2)
                        else:
                            max_depth = max(max_depth, 1)
                    elif isinstance(val, dict):
                        has_dict = True
                        has_nested = True
                        max_depth = max(max_depth, 2)
                    else:
                        max_depth = max(max_depth, 1)

        sample_complexity[i] = {
            'has_nested': has_nested,
            'has_list': has_list,
            'has_dict': has_dict,
            'has_any_complex': has_nested or has_list or has_dict,
            'max_depth': max_depth
        }

    print(f"Samples with complexity data: {len(sample_complexity)}")

    # Merge
    merged = []
    for sample_idx in sample_avg_stability:
        if sample_idx in sample_complexity:
            merged.append({
                'sample_idx': sample_idx,
                'stability': sample_avg_stability[sample_idx],
                **sample_complexity[sample_idx]
            })

    print(f"Merged samples: {len(merged)}")

    if len(merged) == 0:
        print("ERROR: No merged samples!")
        return

    # Quartile-based thresholds
    stabilities = sorted([m['stability'] for m in merged])
    q1 = stabilities[len(stabilities) // 4]
    q3 = stabilities[3 * len(stabilities) // 4]

    consistent = [m for m in merged if m['stability'] >= q3]
    inconsistent = [m for m in merged if m['stability'] <= q1]

    print(f"\nThresholds: Q1={q1:.3f}, Q3={q3:.3f}")
    print(f"Consistent (S_α >= {q3:.2f}): N={len(consistent)}")
    print(f"Inconsistent (S_α <= {q1:.2f}): N={len(inconsistent)}")

    # Helper functions
    def pct(samples, feature):
        if not samples:
            return 0
        return sum(1 for s in samples if s[feature]) / len(samples) * 100

    def avg_val(samples, feature):
        if not samples:
            return 0
        return sum(s[feature] for s in samples) / len(samples)

    # Print results table
    print(f"\n{'Parameter Type':<25} {'Consistent':>12} {'Inconsistent':>12} {'Ratio':>10}")
    print("-" * 60)

    cons_nested = pct(consistent, 'has_nested')
    incons_nested = pct(inconsistent, 'has_nested')
    ratio_nested = f"{incons_nested/cons_nested:.1f}x" if cons_nested > 0 else f">{incons_nested:.0f}%"
    print(f"{'Has Nested Structure':<25} {cons_nested:>11.1f}% {incons_nested:>11.1f}% {ratio_nested:>10}")

    cons_dict = pct(consistent, 'has_dict')
    incons_dict = pct(inconsistent, 'has_dict')
    ratio_dict = f"{incons_dict/cons_dict:.1f}x" if cons_dict > 0 else f">{incons_dict:.0f}%"
    print(f"{'Has Dict Params':<25} {cons_dict:>11.1f}% {incons_dict:>11.1f}% {ratio_dict:>10}")

    cons_list = pct(consistent, 'has_list')
    incons_list = pct(inconsistent, 'has_list')
    ratio_list = f"{incons_list/cons_list:.1f}x" if cons_list > 0 else f">{incons_list:.0f}%"
    print(f"{'Has List Params':<25} {cons_list:>11.1f}% {incons_list:>11.1f}% {ratio_list:>10}")

    cons_complex = pct(consistent, 'has_any_complex')
    incons_complex = pct(inconsistent, 'has_any_complex')
    ratio_complex = f"{incons_complex/cons_complex:.1f}x" if cons_complex > 0 else f">{incons_complex:.0f}%"
    print(f"{'Has Any Complex Type':<25} {cons_complex:>11.1f}% {incons_complex:>11.1f}% {ratio_complex:>10}")

    cons_depth = avg_val(consistent, 'max_depth')
    incons_depth = avg_val(inconsistent, 'max_depth')
    ratio_depth = f"{incons_depth/cons_depth:.1f}x" if cons_depth > 0 else "N/A"
    print(f"{'Mean Max Depth':<25} {cons_depth:>12.2f} {incons_depth:>12.2f} {ratio_depth:>10}")

    return {
        'n_consistent': len(consistent),
        'n_inconsistent': len(inconsistent),
        'q1': q1,
        'q3': q3,
        'has_nested': (cons_nested, incons_nested),
        'has_dict': (cons_dict, incons_dict),
        'has_list': (cons_list, incons_list),
        'has_any_complex': (cons_complex, incons_complex),
        'max_depth': (cons_depth, incons_depth)
    }


def main():
    # Paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sharegpt_results = os.path.join(base_path, 'results/sharegpt/minilm-ec2/combined_consistency_metrics_results.json')
    sharegpt_data = os.path.join(base_path, 'sharegpt_data')

    toucan_results = os.path.join(base_path, 'results/toucan/minilm-ec2/combined_consistency_metrics_results.json')
    toucan_data = os.path.join(base_path, 'toucan_data')

    # Run analyses
    print("\n" + "=" * 70)
    print("PAPER TABLE DATA ANALYSIS")
    print("Aggregating across ALL temperatures and 18 final models")
    print("=" * 70)

    sharegpt_results_data = analyze_sharegpt_prompt_factors(sharegpt_results, sharegpt_data)
    toucan_results_data = analyze_toucan_parameter_complexity(toucan_results, toucan_data)

    # Summary for paper
    print("\n" + "=" * 70)
    print("LATEX TABLE VALUES FOR PAPER")
    print("=" * 70)

    if sharegpt_results_data:
        print("\n% ShareGPT Prompt Factors Table (all temperatures)")
        print(f"% Consistent: S_α >= {sharegpt_results_data['q3']:.2f} (N={sharegpt_results_data['n_consistent']})")
        print(f"% Inconsistent: S_α <= {sharegpt_results_data['q1']:.2f} (N={sharegpt_results_data['n_inconsistent']})")
        q_cons, q_incons = sharegpt_results_data['question_marks']
        l_cons, l_incons = sharegpt_results_data['prompt_length']
        p_cons, p_incons = sharegpt_results_data['prose_heavy']
        print(f"Question marks (avg) & {q_cons:.1f} & {q_incons:.1f} & {q_incons/q_cons:.1f}$\\times$ \\\\")
        print(f"Prompt length (chars) & {l_cons:,.0f} & {l_incons:,.0f} & {l_incons/l_cons:.1f}$\\times$ \\\\")
        print(f"Prose-heavy & {p_cons:.1f}\\% & {p_incons:.1f}\\% & {p_incons/p_cons:.1f}$\\times$ \\\\" if p_cons > 0 else f"Prose-heavy & {p_cons:.1f}\\% & {p_incons:.1f}\\% & -- \\\\")

    if toucan_results_data:
        print("\n% Toucan Parameter Complexity Table (all temperatures)")
        print(f"% Consistent: S_α >= {toucan_results_data['q3']:.2f} (N={toucan_results_data['n_consistent']})")
        print(f"% Inconsistent: S_α <= {toucan_results_data['q1']:.2f} (N={toucan_results_data['n_inconsistent']})")
        n_cons, n_incons = toucan_results_data['has_nested']
        d_cons, d_incons = toucan_results_data['has_dict']
        l_cons, l_incons = toucan_results_data['has_list']
        c_cons, c_incons = toucan_results_data['has_any_complex']
        depth_cons, depth_incons = toucan_results_data['max_depth']

        def ratio_str(cons, incons):
            if cons > 0:
                return f"{incons/cons:.1f}$\\times$"
            return f">{incons:.0f}\\%"

        print(f"Has Nested Structure & {n_cons:.1f}\\% & {n_incons:.1f}\\% & {ratio_str(n_cons, n_incons)} \\\\")
        print(f"Has Dict Params & {d_cons:.1f}\\% & {d_incons:.1f}\\% & {ratio_str(d_cons, d_incons)} \\\\")
        print(f"Has List Params & {l_cons:.1f}\\% & {l_incons:.1f}\\% & {ratio_str(l_cons, l_incons)} \\\\")
        print(f"Has Any Complex Type & {c_cons:.1f}\\% & {c_incons:.1f}\\% & {ratio_str(c_cons, c_incons)} \\\\")
        print(f"Mean Max Depth & {depth_cons:.2f} & {depth_incons:.2f} & {depth_incons/depth_cons:.1f}$\\times$ \\\\" if depth_cons > 0 else f"Mean Max Depth & {depth_cons:.2f} & {depth_incons:.2f} & -- \\\\")


if __name__ == '__main__':
    main()
