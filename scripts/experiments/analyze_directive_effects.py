#!/usr/bin/env python3
"""
Analyze the effect of directive language (should/must) on consistency and accuracy
using existing generation results.

This is an observational analysis comparing:
- Samples WITH directive language ("should", "must") in queries
- Samples WITHOUT directive language

For each group, we calculate:
- Mean consistency (stability score)
- Mean accuracy (STED similarity to ground truth)
- Statistical tests (t-test, effect size)
"""

import json
import re
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directive patterns
DIRECTIVE_PATTERNS = [
    r'\bshould\b',
    r'\bmust\b',
]


def has_directive_language(text: str) -> bool:
    """Check if text contains directive language (should/must)."""
    text_lower = text.lower()
    for pattern in DIRECTIVE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def get_directive_type(text: str) -> str:
    """Get the type of directive language found."""
    text_lower = text.lower()
    has_should = bool(re.search(r'\bshould\b', text_lower))
    has_must = bool(re.search(r'\bmust\b', text_lower))

    if has_should and has_must:
        return "both"
    elif has_should:
        return "should"
    elif has_must:
        return "must"
    else:
        return "none"


def calculate_stability_score(c_mean_values: List[float], alpha: float = 20) -> float:
    """Calculate stability score from c_mean values."""
    if not c_mean_values:
        return 0.0
    # c_mean is already the mean pairwise similarity
    # Stability score approximation from variance
    mean_cmean = np.mean(c_mean_values)
    return mean_cmean


def calculate_c_mean(runs: List[Any]) -> float:
    """Calculate c_mean (mean pairwise similarity) from runs."""
    # Filter valid runs
    valid_runs = [r for r in runs if r is not None and r != []]
    if len(valid_runs) < 2:
        return 1.0 if len(valid_runs) == 1 else 0.0

    # Simple similarity: check if runs are identical
    # For efficiency, we use exact match comparison
    n = len(valid_runs)
    total_sim = 0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Simple exact match for now
            if json.dumps(valid_runs[i], sort_keys=True) == json.dumps(valid_runs[j], sort_keys=True):
                total_sim += 1.0
            else:
                # Partial similarity based on tool name matches
                names_i = set(t.get('name', '') for t in valid_runs[i] if isinstance(t, dict))
                names_j = set(t.get('name', '') for t in valid_runs[j] if isinstance(t, dict))
                if names_i and names_j:
                    jaccard = len(names_i & names_j) / len(names_i | names_j)
                    total_sim += jaccard
                else:
                    total_sim += 0.5  # Default partial similarity
            count += 1

    return total_sim / count if count > 0 else 0.0


def load_results_from_directory(results_dir: str) -> List[Dict]:
    """Load all results from a model's results directory."""
    results = []
    results_path = Path(results_dir)

    for temp_dir in results_path.glob("run_*"):
        results_file = temp_dir / "all_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)

                metadata = data.get('metadata', {})
                temperature = metadata.get('temperature', 0.0)
                model_name = metadata.get('display_name', metadata.get('model', 'unknown'))

                for result in data.get('results', []):
                    query = result.get('query', '')
                    generated_runs = result.get('generated_runs', [])
                    ground_truth = result.get('ground_truth', [])

                    # Calculate consistency metrics if not present
                    consistency_metrics = result.get('consistency_metrics', {})
                    if not consistency_metrics:
                        c_mean = calculate_c_mean(generated_runs)
                        # Simple ranking score: check if any run matches ground truth tool names
                        ranking_score = 0.0
                        if ground_truth and generated_runs:
                            gt_names = set(t.get('name', '') for t in ground_truth if isinstance(t, dict))
                            for run in generated_runs:
                                if run:
                                    run_names = set(t.get('name', '') for t in run if isinstance(t, dict))
                                    if gt_names and run_names:
                                        score = len(gt_names & run_names) / len(gt_names)
                                        ranking_score = max(ranking_score, score)
                        consistency_metrics = {'c_mean': c_mean, 'ranking_score': ranking_score}

                    results.append({
                        'sample_id': result.get('sample_id'),
                        'query': query,
                        'temperature': temperature,
                        'model': model_name,
                        'ground_truth': ground_truth,
                        'generated_runs': generated_runs,
                        'consistency_metrics': consistency_metrics,
                    })
            except Exception as e:
                logger.warning(f"Error loading {results_file}: {e}")

    return results


def analyze_directive_effects(results: List[Dict]) -> Dict[str, Any]:
    """Analyze the effect of directive language on consistency and accuracy."""

    # Separate samples by directive presence
    with_directive = []
    without_directive = []
    by_directive_type = defaultdict(list)

    for result in results:
        query = result.get('query', '')
        metrics = result.get('consistency_metrics', {})

        c_mean = metrics.get('c_mean', 0.0)
        ranking_score = metrics.get('ranking_score', 0.0)

        if c_mean == 0.0:
            continue

        sample_data = {
            'sample_id': result.get('sample_id'),
            'query': query,
            'temperature': result.get('temperature'),
            'model': result.get('model'),
            'c_mean': c_mean,
            'ranking_score': ranking_score,
        }

        directive_type = get_directive_type(query)
        by_directive_type[directive_type].append(sample_data)

        if has_directive_language(query):
            with_directive.append(sample_data)
        else:
            without_directive.append(sample_data)

    # Calculate statistics
    analysis = {
        'n_with_directive': len(with_directive),
        'n_without_directive': len(without_directive),
    }

    if with_directive and without_directive:
        # C_mean comparison
        cmean_with = [s['c_mean'] for s in with_directive]
        cmean_without = [s['c_mean'] for s in without_directive]

        t_stat, p_val = stats.ttest_ind(cmean_without, cmean_with)
        cohens_d = (np.mean(cmean_without) - np.mean(cmean_with)) / np.sqrt(
            (np.std(cmean_with)**2 + np.std(cmean_without)**2) / 2
        )

        analysis['consistency'] = {
            'with_directive_mean': np.mean(cmean_with),
            'with_directive_std': np.std(cmean_with),
            'without_directive_mean': np.mean(cmean_without),
            'without_directive_std': np.std(cmean_without),
            'delta': np.mean(cmean_without) - np.mean(cmean_with),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
        }

        # Ranking score (accuracy) comparison
        rank_with = [s['ranking_score'] for s in with_directive]
        rank_without = [s['ranking_score'] for s in without_directive]

        t_stat_rank, p_val_rank = stats.ttest_ind(rank_without, rank_with)
        cohens_d_rank = (np.mean(rank_without) - np.mean(rank_with)) / np.sqrt(
            (np.std(rank_with)**2 + np.std(rank_without)**2) / 2
        )

        analysis['accuracy'] = {
            'with_directive_mean': np.mean(rank_with),
            'with_directive_std': np.std(rank_with),
            'without_directive_mean': np.mean(rank_without),
            'without_directive_std': np.std(rank_without),
            'delta': np.mean(rank_without) - np.mean(rank_with),
            't_statistic': t_stat_rank,
            'p_value': p_val_rank,
            'cohens_d': cohens_d_rank,
        }

    # Breakdown by directive type
    analysis['by_directive_type'] = {}
    for dtype, samples in by_directive_type.items():
        if samples:
            analysis['by_directive_type'][dtype] = {
                'n_samples': len(samples),
                'consistency_mean': np.mean([s['c_mean'] for s in samples]),
                'consistency_std': np.std([s['c_mean'] for s in samples]),
                'accuracy_mean': np.mean([s['ranking_score'] for s in samples]),
                'accuracy_std': np.std([s['ranking_score'] for s in samples]),
            }

    # Breakdown by temperature
    by_temp = defaultdict(lambda: {'with': [], 'without': []})
    for s in with_directive:
        by_temp[s['temperature']]['with'].append(s)
    for s in without_directive:
        by_temp[s['temperature']]['without'].append(s)

    analysis['by_temperature'] = {}
    for temp, groups in sorted(by_temp.items()):
        if groups['with'] and groups['without']:
            analysis['by_temperature'][temp] = {
                'with_directive': {
                    'n': len(groups['with']),
                    'consistency_mean': np.mean([s['c_mean'] for s in groups['with']]),
                    'accuracy_mean': np.mean([s['ranking_score'] for s in groups['with']]),
                },
                'without_directive': {
                    'n': len(groups['without']),
                    'consistency_mean': np.mean([s['c_mean'] for s in groups['without']]),
                    'accuracy_mean': np.mean([s['ranking_score'] for s in groups['without']]),
                },
                'delta_consistency': np.mean([s['c_mean'] for s in groups['without']]) - np.mean([s['c_mean'] for s in groups['with']]),
                'delta_accuracy': np.mean([s['ranking_score'] for s in groups['without']]) - np.mean([s['ranking_score'] for s in groups['with']]),
            }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze directive language effects on consistency")
    parser.add_argument("--results-dir", type=str, default="llm_gen_results/toucan",
                        help="Directory containing generation results")
    parser.add_argument("--output-file", type=str, default="results/directive_analysis.json",
                        help="Output file for analysis results")

    args = parser.parse_args()

    # Find all model directories
    results_base = Path(args.results_dir)
    all_results = []

    for model_dir in results_base.glob("generations-*"):
        logger.info(f"Loading results from {model_dir}")
        results = load_results_from_directory(str(model_dir))
        all_results.extend(results)
        logger.info(f"  Loaded {len(results)} samples")

    logger.info(f"\nTotal samples loaded: {len(all_results)}")

    # Analyze
    analysis = analyze_directive_effects(all_results)

    # Print results
    print("\n" + "="*60)
    print("DIRECTIVE LANGUAGE ANALYSIS")
    print("="*60)

    print(f"\nSample counts:")
    print(f"  WITH directive (should/must): {analysis['n_with_directive']}")
    print(f"  WITHOUT directive: {analysis['n_without_directive']}")

    if 'consistency' in analysis:
        print(f"\nCONSISTENCY (C_mean):")
        c = analysis['consistency']
        print(f"  With directive:    {c['with_directive_mean']:.4f} (±{c['with_directive_std']:.4f})")
        print(f"  Without directive: {c['without_directive_mean']:.4f} (±{c['without_directive_std']:.4f})")
        print(f"  Delta:             {c['delta']:+.4f}")
        print(f"  t-statistic:       {c['t_statistic']:.3f}")
        print(f"  p-value:           {c['p_value']:.6f}")
        print(f"  Cohen's d:         {c['cohens_d']:.3f}")

        if c['p_value'] < 0.05:
            print(f"  *** SIGNIFICANT at p<0.05 ***")

    if 'accuracy' in analysis:
        print(f"\nACCURACY (Ranking Score):")
        a = analysis['accuracy']
        print(f"  With directive:    {a['with_directive_mean']:.4f} (±{a['with_directive_std']:.4f})")
        print(f"  Without directive: {a['without_directive_mean']:.4f} (±{a['without_directive_std']:.4f})")
        print(f"  Delta:             {a['delta']:+.4f}")
        print(f"  t-statistic:       {a['t_statistic']:.3f}")
        print(f"  p-value:           {a['p_value']:.6f}")
        print(f"  Cohen's d:         {a['cohens_d']:.3f}")

        if a['p_value'] < 0.05:
            print(f"  *** SIGNIFICANT at p<0.05 ***")

    print(f"\nBREAKDOWN BY DIRECTIVE TYPE:")
    for dtype, stats in analysis.get('by_directive_type', {}).items():
        print(f"  {dtype}: n={stats['n_samples']}, consistency={stats['consistency_mean']:.4f}, accuracy={stats['accuracy_mean']:.4f}")

    print(f"\nBREAKDOWN BY TEMPERATURE:")
    for temp, stats in analysis.get('by_temperature', {}).items():
        print(f"  T={temp:.1f}: Δconsistency={stats['delta_consistency']:+.4f}, Δaccuracy={stats['delta_accuracy']:+.4f}")

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=float)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
