#!/usr/bin/env python3
"""
Analyze ShareGPT prompt factors and Toucan parameter complexity BY TEMPERATURE.
Generates visualizations showing how the ratios change across temperatures.
"""

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd

# Final models to include in analysis
FINAL_MODELS = [
    "Claude-3.5-Sonnet", "Claude-3.7-Sonnet", "Claude-3.5-Haiku",
    "Claude-Haiku-4.5", "Claude-Opus-4", "Claude-Opus-4.5",
    "Claude-Sonnet-4", "Claude-Sonnet-4.5", "Qwen3-235B-A22B",
    "Qwen3-32B", "Llama-3.3-70B", "Nova-2-Lite", "Grok-4.1-Fast",
    "Minimax-M2", "Mimo-V2-Flash", "GPT-OSS-120B", "GPT-4.1-Mini",
    "Gemini-2.5-Flash-Lite"
]

# Model name mapping for Toucan
TOUCAN_MODEL_MAPPING = {
    "Claude-Opus-4": "us.anthropic.claude-opus-4-20250514-v1",
    "Nova-2-Lite": "us.amazon.nova-lite-v1",
}

TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_sharegpt_prompt_features(data_path: str):
    """Load and extract features from ShareGPT prompts."""
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
                except Exception:
                    sample_idx += 1
                    continue

    return prompt_features


def load_toucan_complexity(data_path: str):
    """Load and extract complexity features from Toucan samples."""
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

    return sample_complexity


def analyze_by_temperature(results_path: str, features: dict, dataset_name: str):
    """Analyze consistency vs features at each temperature."""

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Find matched models
    matched_models = []
    for model_key in results.keys():
        for fm in FINAL_MODELS:
            mapped = TOUCAN_MODEL_MAPPING.get(fm, fm)
            if fm in model_key or mapped in model_key:
                matched_models.append((fm, model_key))
                break

    print(f"{dataset_name}: Matched {len(matched_models)} models")

    # Collect stability per sample per temperature
    temp_sample_stability = defaultdict(lambda: defaultdict(list))

    for final_model, result_key in matched_models:
        for entry in results[result_key]:
            sample_idx = entry['sample_idx']
            temp = entry.get('temperature', 0.0)
            if entry.get('stability_score') is not None:
                temp_sample_stability[temp][sample_idx].append(entry['stability_score'])

    # Analyze at each temperature
    temp_results = {}

    for temp in TEMPERATURES:
        # Average stability per sample at this temperature
        sample_stability = {}
        for idx, scores in temp_sample_stability[temp].items():
            if scores:
                sample_stability[idx] = sum(scores) / len(scores)

        if not sample_stability:
            continue

        # Merge with features
        merged = []
        for sample_idx in sample_stability:
            if sample_idx in features:
                merged.append({
                    'sample_idx': sample_idx,
                    'stability': sample_stability[sample_idx],
                    **features[sample_idx]
                })

        if len(merged) < 10:
            continue

        # Quartile-based thresholds
        stabilities = sorted([m['stability'] for m in merged])
        q1 = stabilities[len(stabilities) // 4]
        q3 = stabilities[3 * len(stabilities) // 4]

        consistent = [m for m in merged if m['stability'] >= q3]
        inconsistent = [m for m in merged if m['stability'] <= q1]

        temp_results[temp] = {
            'n_samples': len(merged),
            'n_consistent': len(consistent),
            'n_inconsistent': len(inconsistent),
            'q1': q1,
            'q3': q3,
            'consistent': consistent,
            'inconsistent': inconsistent
        }

    return temp_results


def compute_sharegpt_metrics(temp_results):
    """Compute ShareGPT metrics at each temperature."""
    metrics = []

    for temp in TEMPERATURES:
        if temp not in temp_results:
            continue

        data = temp_results[temp]
        consistent = data['consistent']
        inconsistent = data['inconsistent']

        def avg_val(samples, feature):
            if not samples:
                return 0
            return sum(s[feature] for s in samples) / len(samples)

        def pct(samples, feature):
            if not samples:
                return 0
            return sum(1 for s in samples if s[feature]) / len(samples) * 100

        cons_q = avg_val(consistent, 'question_marks')
        incons_q = avg_val(inconsistent, 'question_marks')

        cons_len = avg_val(consistent, 'prompt_length')
        incons_len = avg_val(inconsistent, 'prompt_length')

        cons_prose = pct(consistent, 'prose_heavy')
        incons_prose = pct(inconsistent, 'prose_heavy')

        metrics.append({
            'temperature': temp,
            'n_consistent': data['n_consistent'],
            'n_inconsistent': data['n_inconsistent'],
            'q1': data['q1'],
            'q3': data['q3'],
            'question_marks_consistent': cons_q,
            'question_marks_inconsistent': incons_q,
            'question_marks_ratio': incons_q / cons_q if cons_q > 0 else float('inf'),
            'prompt_length_consistent': cons_len,
            'prompt_length_inconsistent': incons_len,
            'prompt_length_ratio': incons_len / cons_len if cons_len > 0 else 1.0,
            'prose_heavy_consistent': cons_prose,
            'prose_heavy_inconsistent': incons_prose,
            'prose_heavy_ratio': incons_prose / cons_prose if cons_prose > 0 else float('inf'),
        })

    return pd.DataFrame(metrics)


def compute_toucan_metrics(temp_results):
    """Compute Toucan metrics at each temperature."""
    metrics = []

    for temp in TEMPERATURES:
        if temp not in temp_results:
            continue

        data = temp_results[temp]
        consistent = data['consistent']
        inconsistent = data['inconsistent']

        def pct(samples, feature):
            if not samples:
                return 0
            return sum(1 for s in samples if s[feature]) / len(samples) * 100

        def avg_val(samples, feature):
            if not samples:
                return 0
            return sum(s[feature] for s in samples) / len(samples)

        cons_list = pct(consistent, 'has_list')
        incons_list = pct(inconsistent, 'has_list')

        cons_nested = pct(consistent, 'has_nested')
        incons_nested = pct(inconsistent, 'has_nested')

        cons_dict = pct(consistent, 'has_dict')
        incons_dict = pct(inconsistent, 'has_dict')

        cons_complex = pct(consistent, 'has_any_complex')
        incons_complex = pct(inconsistent, 'has_any_complex')

        cons_depth = avg_val(consistent, 'max_depth')
        incons_depth = avg_val(inconsistent, 'max_depth')

        metrics.append({
            'temperature': temp,
            'n_consistent': data['n_consistent'],
            'n_inconsistent': data['n_inconsistent'],
            'q1': data['q1'],
            'q3': data['q3'],
            'has_list_consistent': cons_list,
            'has_list_inconsistent': incons_list,
            'has_list_ratio': incons_list / cons_list if cons_list > 0 else float('inf'),
            'has_nested_consistent': cons_nested,
            'has_nested_inconsistent': incons_nested,
            'has_nested_ratio': incons_nested / cons_nested if cons_nested > 0 else float('inf'),
            'has_dict_consistent': cons_dict,
            'has_dict_inconsistent': incons_dict,
            'has_dict_ratio': incons_dict / cons_dict if cons_dict > 0 else float('inf'),
            'has_any_complex_consistent': cons_complex,
            'has_any_complex_inconsistent': incons_complex,
            'has_any_complex_ratio': incons_complex / cons_complex if cons_complex > 0 else float('inf'),
            'max_depth_consistent': cons_depth,
            'max_depth_inconsistent': incons_depth,
            'max_depth_ratio': incons_depth / cons_depth if cons_depth > 0 else float('inf'),
        })

    return pd.DataFrame(metrics)


def plot_toucan_complexity_by_temperature(df, output_path):
    """Create visualization of parameter complexity vs temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Percentage in inconsistent vs consistent samples
    ax1 = axes[0]
    temps = df['temperature'].values

    ax1.plot(temps, df['has_list_inconsistent'], 'o-', label='List (Inconsistent)', color='#d62728', linewidth=2)
    ax1.plot(temps, df['has_list_consistent'], 's--', label='List (Consistent)', color='#d62728', alpha=0.5, linewidth=1.5)
    ax1.plot(temps, df['has_any_complex_inconsistent'], '^-', label='Any Complex (Inconsistent)', color='#1f77b4', linewidth=2)
    ax1.plot(temps, df['has_any_complex_consistent'], 'v--', label='Any Complex (Consistent)', color='#1f77b4', alpha=0.5, linewidth=1.5)

    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Percentage of Samples (%)', fontsize=12)
    ax1.set_title('Parameter Complexity: Consistent vs Inconsistent', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, max(df['has_any_complex_inconsistent'].max(), df['has_list_inconsistent'].max()) * 1.2)

    # Right: Ratio (inconsistent/consistent)
    ax2 = axes[1]

    # Cap infinite ratios for visualization
    list_ratio = df['has_list_ratio'].replace([float('inf')], [df['has_list_ratio'][df['has_list_ratio'] != float('inf')].max() * 1.5])
    complex_ratio = df['has_any_complex_ratio'].replace([float('inf')], [df['has_any_complex_ratio'][df['has_any_complex_ratio'] != float('inf')].max() * 1.5])

    ax2.bar(temps - 0.02, list_ratio, width=0.04, label='List Params', color='#d62728', alpha=0.8)
    ax2.bar(temps + 0.02, complex_ratio, width=0.04, label='Any Complex', color='#1f77b4', alpha=0.8)

    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Ratio (Inconsistent / Consistent)', fontsize=12)
    ax2.set_title('Complexity Over-representation in Inconsistent Samples', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Equal (1.0)')
    ax2.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sharegpt_factors_by_temperature(df, output_path):
    """Create visualization of prompt factors vs temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Question marks comparison
    ax1 = axes[0]
    temps = df['temperature'].values

    ax1.plot(temps, df['question_marks_inconsistent'], 'o-', label='Inconsistent', color='#d62728', linewidth=2, markersize=8)
    ax1.plot(temps, df['question_marks_consistent'], 's-', label='Consistent', color='#2ca02c', linewidth=2, markersize=8)

    ax1.fill_between(temps, df['question_marks_consistent'], df['question_marks_inconsistent'],
                     alpha=0.2, color='gray')

    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Avg Question Marks per Prompt', fontsize=12)
    ax1.set_title('Question Mark Frequency: Consistent vs Inconsistent', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)

    # Right: Ratio
    ax2 = axes[1]

    # Cap infinite ratios
    q_ratio = df['question_marks_ratio'].replace([float('inf')], [10])
    q_ratio = q_ratio.clip(upper=10)

    colors = ['#d62728' if r > 2 else '#ff7f0e' if r > 1.5 else '#2ca02c' for r in q_ratio]
    ax2.bar(temps, q_ratio, width=0.06, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=2, color='orange', linestyle=':', alpha=0.5, linewidth=1)

    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Ratio (Inconsistent / Consistent)', fontsize=12)
    ax2.set_title('Question Marks Ratio by Temperature', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0, min(q_ratio.max() * 1.2, 12))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_path, 'docs/ICML_paper/figures')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PARAMETER COMPLEXITY BY TEMPERATURE ANALYSIS")
    print("=" * 70)

    # ShareGPT Analysis
    print("\n--- ShareGPT Analysis ---")
    sharegpt_features = load_sharegpt_prompt_features(os.path.join(base_path, 'sharegpt_data'))
    print(f"Loaded {len(sharegpt_features)} prompt features")

    sharegpt_temp_results = analyze_by_temperature(
        os.path.join(base_path, 'results/sharegpt/minilm-ec2/combined_consistency_metrics_results.json'),
        sharegpt_features,
        "ShareGPT"
    )

    sharegpt_df = compute_sharegpt_metrics(sharegpt_temp_results)
    print("\nShareGPT Prompt Factors by Temperature:")
    print(sharegpt_df[['temperature', 'question_marks_consistent', 'question_marks_inconsistent',
                        'question_marks_ratio', 'n_consistent', 'n_inconsistent']].to_string(index=False))

    # Save CSV
    sharegpt_csv = os.path.join(output_dir, 'sharegpt_prompt_factors_by_temperature.csv')
    sharegpt_df.to_csv(sharegpt_csv, index=False)
    print(f"\nSaved: {sharegpt_csv}")

    # Plot
    plot_sharegpt_factors_by_temperature(sharegpt_df, os.path.join(output_dir, 'sharegpt_prompt_factors_by_temp.png'))

    # Toucan Analysis
    print("\n--- Toucan Analysis ---")
    toucan_complexity = load_toucan_complexity(os.path.join(base_path, 'toucan_data'))
    print(f"Loaded {len(toucan_complexity)} complexity features")

    toucan_temp_results = analyze_by_temperature(
        os.path.join(base_path, 'results/toucan/minilm-ec2/combined_consistency_metrics_results.json'),
        toucan_complexity,
        "Toucan"
    )

    toucan_df = compute_toucan_metrics(toucan_temp_results)
    print("\nToucan Parameter Complexity by Temperature:")
    print(toucan_df[['temperature', 'has_list_consistent', 'has_list_inconsistent',
                     'has_list_ratio', 'has_any_complex_ratio', 'n_consistent', 'n_inconsistent']].to_string(index=False))

    # Save CSV
    toucan_csv = os.path.join(output_dir, 'toucan_complexity_by_temperature.csv')
    toucan_df.to_csv(toucan_csv, index=False)
    print(f"\nSaved: {toucan_csv}")

    # Plot
    plot_toucan_complexity_by_temperature(toucan_df, os.path.join(output_dir, 'toucan_complexity_by_temp.png'))

    # Print LaTeX table for Toucan
    print("\n" + "=" * 70)
    print("LATEX TABLE: Toucan Parameter Complexity by Temperature")
    print("=" * 70)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Parameter Complexity vs Inconsistency by Temperature (Toucan)}")
    print("\\label{tab:complexity-by-temp}")
    print("\\scriptsize")
    print("\\begin{tabular}{@{}lcccccc@{}}")
    print("\\toprule")
    print("\\textbf{Temp} & \\textbf{List (C)} & \\textbf{List (I)} & \\textbf{Ratio} & \\textbf{Complex (C)} & \\textbf{Complex (I)} & \\textbf{Ratio} \\\\")
    print("\\midrule")
    for _, row in toucan_df.iterrows():
        list_ratio = f"{row['has_list_ratio']:.0f}$\\times$" if row['has_list_ratio'] < float('inf') else "$>$100$\\times$"
        complex_ratio = f"{row['has_any_complex_ratio']:.0f}$\\times$" if row['has_any_complex_ratio'] < float('inf') else "$>$100$\\times$"
        print(f"{row['temperature']:.1f} & {row['has_list_consistent']:.1f}\\% & {row['has_list_inconsistent']:.1f}\\% & {list_ratio} & {row['has_any_complex_consistent']:.1f}\\% & {row['has_any_complex_inconsistent']:.1f}\\% & {complex_ratio} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == '__main__':
    main()
