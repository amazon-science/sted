#!/usr/bin/env python3
"""
Analyze patterns in inconsistent LLM outputs for ShareGPT structured output task.

Focuses on STRUCTURAL AMBIGUITY rather than lexical features.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats


def load_consistency_metrics(metrics_path: str) -> dict:
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_prompt(sample_idx: int, model_dir: Path) -> str:
    """Load original prompt - handles sample_000.json naming."""
    # Try different naming patterns
    patterns = [
        f'**/sample_{sample_idx:03d}.json',
        f'**/sample_{sample_idx}.json',
    ]
    for pattern in patterns:
        files = list(model_dir.glob(pattern))
        if files:
            with open(files[0], 'r') as f:
                data = json.load(f)
            return data.get('original_prompt', '')
    return ''


def detect_structural_indicators(prompt: str) -> dict:
    """Detect structural indicators that constrain output schema."""
    if not prompt:
        return {}

    prompt_lower = prompt.lower()

    # Schema-constraining indicators
    has_table = bool(re.search(r'\|.*\|.*\|', prompt))
    has_csv_like = bool(re.search(r'[\w\s]+,[\w\s]+,[\w\s]+\n', prompt))
    has_json_example = bool(re.search(r'\{[^{}]*"[^"]+"\s*:', prompt))
    has_schema_spec = any(kw in prompt_lower for kw in [
        'schema:', 'format:', 'fields:', 'columns:', 'output format',
        'json format', 'following format', 'extract the following'
    ])
    has_numbered_list = bool(re.search(r'^\s*\d+[.)]\s+\w', prompt, re.MULTILINE))
    has_bullet_list = bool(re.search(r'^\s*[-*•]\s+\w', prompt, re.MULTILINE))

    # Ambiguity indicators
    open_ended_patterns = [
        r'organize\s+(this|the)\s+information',
        r'structure\s+(this|the)\s+(data|information)',
        r'convert\s+(this|the)\s+.*\s+to\s+json',
    ]
    has_open_ended = any(re.search(p, prompt_lower) for p in open_ended_patterns)

    sentences = re.split(r'[.!?]\s+', prompt)
    avg_sentence_len = np.mean([len(s.split()) for s in sentences if s]) if sentences else 0
    is_prose_heavy = avg_sentence_len > 20 and not (has_table or has_json_example)

    question_marks = prompt.count('?')

    # Constraint score
    constraint_score = 0
    constraint_score += 3.0 if has_json_example else 0
    constraint_score += 2.0 if has_table else 0
    constraint_score += 2.0 if has_schema_spec else 0
    constraint_score += 1.0 if has_numbered_list else 0
    constraint_score += 1.0 if has_bullet_list else 0
    constraint_score -= 2.0 if has_open_ended and not has_schema_spec else 0
    constraint_score -= 1.5 if is_prose_heavy else 0
    constraint_score -= 0.2 * min(question_marks, 5)

    return {
        'has_table': has_table,
        'has_json_example': has_json_example,
        'has_schema_spec': has_schema_spec,
        'has_numbered_list': has_numbered_list,
        'has_open_ended': has_open_ended,
        'is_prose_heavy': is_prose_heavy,
        'question_marks': question_marks,
        'constraint_score': constraint_score,
        'has_any_structure': has_table or has_json_example or has_schema_spec or has_numbered_list,
        'char_len': len(prompt),
    }


def analyze_inconsistency_patterns(metrics_path: str, generations_dir: str, temperature: float = 0.0) -> dict:
    metrics = load_consistency_metrics(metrics_path)

    sample_scores = defaultdict(list)
    for model_name, entries in metrics.items():
        for entry in entries:
            if entry.get('temperature') == temperature:
                sample_idx = entry.get('sample_idx')
                score = entry.get('consistency_coefficient')
                if sample_idx is not None and score is not None:
                    sample_scores[sample_idx].append(score)

    sample_mean = {idx: np.mean(scores) for idx, scores in sample_scores.items()}

    model_dirs = list(Path(generations_dir).glob('generations-*'))
    if not model_dirs:
        raise ValueError(f"No generation directories found in {generations_dir}")
    model_dir = model_dirs[0]

    all_features = []
    for sample_idx, mean_score in sample_mean.items():
        prompt = load_prompt(sample_idx, model_dir)
        if not prompt:
            continue
        features = detect_structural_indicators(prompt)
        features['consistency'] = mean_score
        features['sample_idx'] = sample_idx
        all_features.append(features)

    if len(all_features) < 2:
        return {'error': f'Only {len(all_features)} samples found'}

    consistent = [f for f in all_features if f['consistency'] >= 0.7]
    inconsistent = [f for f in all_features if f['consistency'] < 0.5]

    consistencies = [f['consistency'] for f in all_features]
    constraint_scores = [f['constraint_score'] for f in all_features]
    char_lens = [f['char_len'] for f in all_features]

    constraint_corr = stats.pearsonr(constraint_scores, consistencies)
    length_corr = stats.pearsonr(char_lens, consistencies)

    def pct(lst, key):
        return np.mean([f.get(key, False) for f in lst]) * 100 if lst else 0

    def mean_val(lst, key):
        vals = [f.get(key, 0) for f in lst if key in f]
        return np.mean(vals) if vals else 0

    return {
        'num_samples': len(all_features),
        'num_consistent': len(consistent),
        'num_inconsistent': len(inconsistent),
        'temperature': temperature,
        'correlations': {
            'constraint_score': {'r': constraint_corr[0], 'p': constraint_corr[1]},
            'prompt_length': {'r': length_corr[0], 'p': length_corr[1]},
        },
        'consistent_samples': {
            'pct_has_table': pct(consistent, 'has_table'),
            'pct_has_json_example': pct(consistent, 'has_json_example'),
            'pct_has_schema_spec': pct(consistent, 'has_schema_spec'),
            'pct_has_any_structure': pct(consistent, 'has_any_structure'),
            'pct_is_prose_heavy': pct(consistent, 'is_prose_heavy'),
            'mean_constraint_score': mean_val(consistent, 'constraint_score'),
            'mean_char_len': mean_val(consistent, 'char_len'),
            'mean_question_marks': mean_val(consistent, 'question_marks'),
        },
        'inconsistent_samples': {
            'pct_has_table': pct(inconsistent, 'has_table'),
            'pct_has_json_example': pct(inconsistent, 'has_json_example'),
            'pct_has_schema_spec': pct(inconsistent, 'has_schema_spec'),
            'pct_has_any_structure': pct(inconsistent, 'has_any_structure'),
            'pct_is_prose_heavy': pct(inconsistent, 'is_prose_heavy'),
            'mean_constraint_score': mean_val(inconsistent, 'constraint_score'),
            'mean_char_len': mean_val(inconsistent, 'char_len'),
            'mean_question_marks': mean_val(inconsistent, 'question_marks'),
        },
        'key_finding': f"Structural constraint score correlates with consistency (r={constraint_corr[0]:.2f}, p={constraint_corr[1]:.3f})"
    }


def print_report(results: dict):
    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    print("=" * 70)
    print("STRUCTURAL AMBIGUITY ANALYSIS")
    print("=" * 70)
    print(f"Samples: {results['num_samples']} (consistent: {results['num_consistent']}, inconsistent: {results['num_inconsistent']})")

    print("\nCorrelations:")
    for name, c in results['correlations'].items():
        sig = "***" if c['p'] < 0.001 else "**" if c['p'] < 0.01 else "*" if c['p'] < 0.05 else ""
        print(f"  {name}: r={c['r']:+.3f}, p={c['p']:.4f} {sig}")

    print("\nFeature Comparison:")
    cons, incons = results['consistent_samples'], results['inconsistent_samples']
    print(f"{'Feature':<25} {'Consistent':>12} {'Inconsistent':>12}")
    print("-" * 55)
    for key in ['pct_has_table', 'pct_has_json_example', 'pct_has_schema_spec', 'pct_has_any_structure', 'pct_is_prose_heavy']:
        print(f"{key:<25} {cons[key]:>11.1f}% {incons[key]:>11.1f}%")
    for key in ['mean_constraint_score', 'mean_char_len', 'mean_question_marks']:
        print(f"{key:<25} {cons[key]:>12.1f} {incons[key]:>12.1f}")

    print(f"\nKey Finding: {results['key_finding']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-path', default='results/sharegpt_titan512/combined_consistency_metrics_results.json')
    parser.add_argument('--generations-dir', default='llm_gen_results/sharegpt')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--output-json', default=None)
    args = parser.parse_args()

    results = analyze_inconsistency_patterns(args.metrics_path, args.generations_dir, args.temperature)
    print_report(results)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
