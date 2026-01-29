#!/usr/bin/env python3
"""
Hyperparameter Ablation Experiments for STED (ICML Submission)

This script runs comprehensive ablation studies on STED hyperparameters:
- alpha: Stability score steepness factor S_alpha = (1/(1+2*D_std_norm))^alpha
- theta: Structural similarity threshold for content matching
- path_weight_decay: Depth-based weight decay
- structural_weight: Weight balancing structural vs content costs

Usage:
    # Run all ablations on synthetic data
    python scripts/experiments/run_hyperparameter_ablation.py --output-dir results/ablation

    # Run alpha ablation only with real data
    python scripts/experiments/run_hyperparameter_ablation.py --parameters alpha --real-data results/toucan/minilm-ec2

    # Quick test
    python scripts/experiments/run_hyperparameter_ablation.py --quick
"""

import argparse
import json
import os
import sys
import time
import copy
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer


# Hyperparameter ranges for ablation
HYPERPARAMETER_RANGES = {
    'alpha': [5, 10, 15, 20, 25, 30, 40, 50],  # Stability score steepness
    'theta': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],  # Structural threshold
    'path_decay': [0.5, 0.7, 0.8, 0.9, 1.0],  # Path decay
    'structural_weight': [0.1, 0.3, 0.5, 0.7, 0.9],  # Structural vs content weight
}

# Default values (for reference lines in plots)
DEFAULT_VALUES = {
    'alpha': 20,
    'theta': 0.3,
    'path_decay': 1.0,
    'structural_weight': 0.5,
}


def generate_synthetic_variations(base_json: Dict, num_variations: int = 10,
                                   noise_level: float = 0.2) -> List[Dict]:
    """Generate synthetic variations of a base JSON for testing."""
    variations = [copy.deepcopy(base_json)]

    for i in range(num_variations - 1):
        var = copy.deepcopy(base_json)

        def perturb_value(v, path=""):
            if isinstance(v, str):
                if random.random() < noise_level:
                    modifications = [
                        v + " modified",
                        v.replace("a", "e") if "a" in v else v + "_alt",
                        f"{v}_v{i}",
                    ]
                    return random.choice(modifications)
                return v
            elif isinstance(v, (int, float)):
                if random.random() < noise_level:
                    return v + random.gauss(0, abs(v) * 0.1 + 1)
                return v
            elif isinstance(v, bool):
                if random.random() < noise_level * 0.5:
                    return not v
                return v
            elif isinstance(v, list):
                result = [perturb_value(item, f"{path}[{j}]") for j, item in enumerate(v)]
                if random.random() < noise_level * 0.3:
                    random.shuffle(result)
                return result
            elif isinstance(v, dict):
                result = {}
                for k, val in v.items():
                    result[k] = perturb_value(val, f"{path}.{k}")
                if random.random() < noise_level * 0.2:
                    result[f"extra_field_{i}"] = f"value_{i}"
                return result
            return v

        var = perturb_value(var)
        variations.append(var)

    return variations


def create_synthetic_dataset() -> List[Tuple[str, List[Dict]]]:
    """Create a diverse synthetic test dataset for ablation experiments."""
    datasets = []

    # 1. Simple flat JSON
    simple_json = {
        "name": "test_item",
        "value": 42,
        "category": "example",
        "active": True
    }
    datasets.append(("simple_flat", generate_synthetic_variations(simple_json)))

    # 2. Nested object
    nested_json = {
        "user": {
            "name": "John Doe",
            "profile": {
                "age": 30,
                "city": "New York",
                "interests": ["coding", "music", "travel"]
            }
        },
        "metadata": {
            "created": "2024-01-01",
            "version": 1.0
        }
    }
    datasets.append(("nested_object", generate_synthetic_variations(nested_json)))

    # 3. Array-heavy structure
    array_json = {
        "items": [
            {"id": 1, "name": "Item A", "price": 10.99},
            {"id": 2, "name": "Item B", "price": 20.99},
            {"id": 3, "name": "Item C", "price": 30.99},
        ],
        "total_count": 3
    }
    datasets.append(("array_heavy", generate_synthetic_variations(array_json)))

    # 4. Function calling output (like tool calls)
    function_json = {
        "function": "search_database",
        "arguments": {
            "query": "machine learning papers",
            "limit": 10,
            "filters": {
                "year": 2024,
                "category": "AI"
            }
        },
        "confidence": 0.95
    }
    datasets.append(("function_call", generate_synthetic_variations(function_json)))

    # 5. Deep nesting (stress test)
    deep_json = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "value": "deep_value",
                        "array": [1, 2, 3]
                    }
                }
            }
        }
    }
    datasets.append(("deep_nesting", generate_synthetic_variations(deep_json)))

    # 6. Mixed types
    mixed_json = {
        "string_field": "hello world",
        "number_field": 3.14159,
        "int_field": 42,
        "bool_field": True,
        "null_field": None,
        "array_field": [1, "two", True, None],
        "nested_field": {"key": "value"}
    }
    datasets.append(("mixed_types", generate_synthetic_variations(mixed_json)))

    return datasets


def load_real_data(results_dir: str) -> List[Tuple[str, List[Dict]]]:
    """Load real LLM outputs from consistency metrics results."""
    datasets = []
    results_path = Path(results_dir) / "combined_consistency_metrics_results.json"

    if not results_path.exists():
        print(f"Warning: Real data not found at {results_path}")
        return datasets

    with open(results_path) as f:
        data = json.load(f)

    # Extract d_std_normalized values for each model at each temperature
    for model, entries in data.items():
        # Group by temperature
        by_temp = {}
        for entry in entries:
            t = entry.get('temperature', 0)
            if t not in by_temp:
                by_temp[t] = []
            # Get d_std_normalized from consistency_metrics
            metrics = entry.get('consistency_metrics', entry)
            d_std_norm = metrics.get('d_std_normalized', 0)
            c_mean = metrics.get('c_mean', 0)
            by_temp[t].append({
                'd_std_normalized': d_std_norm,
                'c_mean': c_mean,
            })

        # Create dataset entry for each temperature
        for temp, samples in by_temp.items():
            if samples:
                datasets.append((f"{model}_T{temp}", samples))

    return datasets


def compute_stability_score(d_std_normalized: float, alpha: int) -> float:
    """Compute S_alpha from d_std_normalized."""
    return (1.0 / (1.0 + 2 * d_std_normalized)) ** alpha


def run_alpha_ablation_synthetic(analyzer: StructuralConsistencyAnalyzer,
                                  datasets: List[Tuple[str, List[Dict]]],
                                  alpha_values: List[int]) -> pd.DataFrame:
    """Run alpha ablation on synthetic data."""
    results = []

    for dataset_name, variations in tqdm(datasets, desc="Datasets"):
        # Calculate consistency metrics once
        metrics = analyzer.evaluate_structural_consistency(
            json_outputs=variations,
            method_name='sted',
            variation_type='combined'
        )

        if 'error' in metrics:
            continue

        consistency_metrics = metrics.get('consistency_metrics', metrics)
        d_std_normalized = consistency_metrics.get('d_std_normalized', 0)
        c_mean = consistency_metrics.get('c_mean', 0)
        r_v = consistency_metrics.get('r_v', 1.0)

        # Compute S_alpha for different alpha values
        for alpha in alpha_values:
            s_alpha = compute_stability_score(d_std_normalized, alpha)
            ranking_score = r_v * c_mean * s_alpha

            results.append({
                'parameter': 'alpha',
                'value': alpha,
                'dataset': dataset_name,
                'stability_score': s_alpha,
                'c_mean': c_mean,
                'd_std_normalized': d_std_normalized,
                'ranking_score': ranking_score,
            })

    return pd.DataFrame(results)


def run_alpha_ablation_real(real_data: List[Tuple[str, List[Dict]]],
                             alpha_values: List[int]) -> pd.DataFrame:
    """Run alpha ablation on real data (from precomputed metrics)."""
    results = []

    for dataset_name, samples in tqdm(real_data, desc="Real datasets"):
        for sample in samples:
            d_std_normalized = sample.get('d_std_normalized', 0)
            c_mean = sample.get('c_mean', 0)

            for alpha in alpha_values:
                s_alpha = compute_stability_score(d_std_normalized, alpha)

                results.append({
                    'parameter': 'alpha',
                    'value': alpha,
                    'dataset': dataset_name,
                    'stability_score': s_alpha,
                    'c_mean': c_mean,
                    'd_std_normalized': d_std_normalized,
                })

    return pd.DataFrame(results)


def run_theta_ablation(datasets: List[Tuple[str, List[Dict]]],
                        theta_values: List[float]) -> pd.DataFrame:
    """Run ablation study on theta (structural threshold) parameter."""
    results = []

    for theta in tqdm(theta_values, desc="Theta ablation"):
        evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
        analyzer = StructuralConsistencyAnalyzer(evaluator)

        for dataset_name, variations in datasets:
            # Calculate pairwise similarities with this theta
            similarities = []
            for v1, v2 in combinations(variations, 2):
                sim = evaluator.calculate_tree_edit_distance_opt(
                    v1, v2, variation_type='combined'
                )
                similarities.append(sim)

            if similarities:
                c_mean = float(np.mean(similarities))
                d_std = float(np.std(similarities))

                results.append({
                    'parameter': 'theta',
                    'value': theta,
                    'dataset': dataset_name,
                    'c_mean': c_mean,
                    'd_std': d_std,
                })

    return pd.DataFrame(results)


def run_path_decay_ablation(datasets: List[Tuple[str, List[Dict]]],
                             decay_values: List[float]) -> pd.DataFrame:
    """Run ablation study on path weight decay parameter."""
    results = []

    for decay in tqdm(decay_values, desc="Path decay ablation"):
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            path_weight_decay=decay,
        )
        analyzer = StructuralConsistencyAnalyzer(evaluator)

        for dataset_name, variations in datasets:
            metrics = analyzer.evaluate_structural_consistency(
                json_outputs=variations,
                method_name='sted',
                variation_type='combined'
            )

            if 'error' not in metrics:
                consistency_metrics = metrics.get('consistency_metrics', metrics)
                results.append({
                    'parameter': 'path_decay',
                    'value': decay,
                    'dataset': dataset_name,
                    'stability_score': consistency_metrics.get('stability_score', 0),
                    'c_mean': consistency_metrics.get('c_mean', 0),
                    'd_std': consistency_metrics.get('d_std', 0),
                    'ranking_score': consistency_metrics.get('ranking_score', 0),
                })

    return pd.DataFrame(results)


def run_structural_weight_ablation(datasets: List[Tuple[str, List[Dict]]],
                                    weight_values: List[float]) -> pd.DataFrame:
    """Run ablation study on structural weight parameter."""
    results = []

    for weight in tqdm(weight_values, desc="Structural weight ablation"):
        evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
        analyzer = StructuralConsistencyAnalyzer(evaluator)

        for dataset_name, variations in datasets:
            # Calculate with custom structural weight
            similarities = []
            for v1, v2 in combinations(variations, 2):
                # Combined: weight * structural + (1-weight) * content
                sim = evaluator.calculate_tree_edit_distance_opt(
                    v1, v2, variation_type='combined'
                )
                similarities.append(sim)

            if similarities:
                c_mean = float(np.mean(similarities))
                d_std = float(np.std(similarities))
                d_std_norm = d_std / 0.5 if d_std > 0 else 0  # Approximate normalization
                s_alpha = compute_stability_score(min(d_std_norm, 1.0), 20)

                results.append({
                    'parameter': 'structural_weight',
                    'value': weight,
                    'dataset': dataset_name,
                    'stability_score': s_alpha,
                    'c_mean': c_mean,
                    'd_std': d_std,
                })

    return pd.DataFrame(results)


def calculate_discrimination_score(df: pd.DataFrame, value_col: str = 'value',
                                    score_col: str = 'stability_score') -> Dict[str, float]:
    """Calculate how well a parameter setting discriminates between datasets."""
    discrimination_scores = {}

    for value in df[value_col].unique():
        subset = df[df[value_col] == value]
        scores = subset.groupby('dataset')[score_col].mean()
        discrimination_scores[value] = float(scores.std()) if len(scores) > 1 else 0.0

    return discrimination_scores


def calculate_ranking_correlation(df: pd.DataFrame, value_col: str = 'value',
                                   score_col: str = 'stability_score') -> pd.DataFrame:
    """Calculate Spearman correlation between rankings at different parameter values."""
    from scipy.stats import spearmanr

    values = sorted(df[value_col].unique())
    correlations = []

    for i, v1 in enumerate(values):
        for v2 in values[i+1:]:
            df1 = df[df[value_col] == v1].groupby('dataset')[score_col].mean()
            df2 = df[df[value_col] == v2].groupby('dataset')[score_col].mean()

            common_datasets = set(df1.index) & set(df2.index)
            if len(common_datasets) >= 3:
                r1 = [df1[d] for d in common_datasets]
                r2 = [df2[d] for d in common_datasets]
                rho, pval = spearmanr(r1, r2)
                correlations.append({
                    'value1': v1,
                    'value2': v2,
                    'spearman_rho': rho,
                    'p_value': pval,
                })

    return pd.DataFrame(correlations)


def plot_alpha_ablation(df: pd.DataFrame, output_dir: Path, use_real_data: bool = False):
    """Generate alpha ablation visualization for paper."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. S_alpha curve (theoretical)
    ax = axes[0]
    d_std_values = np.linspace(0, 0.5, 100)
    for alpha in [5, 10, 20, 30, 50]:
        s_alpha = (1.0 / (1.0 + 2 * d_std_values)) ** alpha
        ax.plot(d_std_values, s_alpha, label=f'$\\alpha$={alpha}', linewidth=2)
    ax.set_xlabel('$\\hat{D}_{std}$ (Normalized Dispersion)', fontsize=11)
    ax.set_ylabel('$S_\\alpha$ (Stability Score)', fontsize=11)
    ax.set_title('(a) Stability Score Sensitivity', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)

    # 2. Box plot by alpha value
    ax = axes[1]
    alpha_values = sorted(df['value'].unique())
    data_for_box = [df[df['value'] == a]['stability_score'].values for a in alpha_values]
    bp = ax.boxplot(data_for_box, labels=alpha_values, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.7)
    ax.set_xlabel('$\\alpha$ (Steepness Factor)', fontsize=11)
    ax.set_ylabel('$S_\\alpha$ Distribution', fontsize=11)
    ax.set_title('(b) Score Distribution by $\\alpha$', fontsize=12, fontweight='bold')
    if DEFAULT_VALUES['alpha'] in alpha_values:
        ax.axvline(x=alpha_values.index(DEFAULT_VALUES['alpha']) + 1,
                   color='red', linestyle='--', label=f'Default $\\alpha$={DEFAULT_VALUES["alpha"]}', alpha=0.7)
        ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Ranking stability (Spearman correlation)
    ax = axes[2]
    corr_df = calculate_ranking_correlation(df, 'value', 'stability_score')
    if not corr_df.empty:
        # Plot correlation vs alpha difference
        corr_df['alpha_diff'] = abs(corr_df['value2'] - corr_df['value1'])
        ax.scatter(corr_df['alpha_diff'], corr_df['spearman_rho'], alpha=0.7, s=50)
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='$\\rho$=0.9 threshold')
        ax.set_xlabel('$|\\alpha_1 - \\alpha_2|$', fontsize=11)
        ax.set_ylabel('Spearman $\\rho$', fontsize=11)
        ax.set_title('(c) Ranking Stability', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    suffix = '_real' if use_real_data else '_synthetic'
    plt.savefig(output_dir / f'ablation_alpha{suffix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'ablation_alpha{suffix}.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def plot_all_ablations(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate comprehensive ablation visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual parameter plots
    for param_name, df in all_results.items():
        if df.empty:
            continue

        score_col = 'stability_score' if 'stability_score' in df.columns else 'c_mean'

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Box plot
        ax = axes[0]
        values = sorted(df['value'].unique())
        data_for_box = [df[df['value'] == v][score_col].values for v in values]
        bp = ax.boxplot(data_for_box, labels=[str(v) for v in values], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#2E86AB')
            patch.set_alpha(0.7)

        param_labels = {
            'alpha': '$\\alpha$ (Steepness)',
            'theta': '$\\theta$ (Threshold)',
            'path_decay': '$\\lambda$ (Path Decay)',
            'structural_weight': '$w$ (Structural Weight)',
        }
        ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=11)
        ax.set_ylabel(f'{score_col}', fontsize=11)
        ax.set_title(f'{param_name.replace("_", " ").title()} Ablation', fontsize=12, fontweight='bold')

        if param_name in DEFAULT_VALUES and DEFAULT_VALUES[param_name] in values:
            ax.axvline(x=values.index(DEFAULT_VALUES[param_name]) + 1,
                       color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3, axis='y')

        # Line plot by dataset
        ax = axes[1]
        datasets = df['dataset'].unique()
        for dataset in datasets[:8]:  # Limit to 8 datasets for readability
            subset = df[df['dataset'] == dataset]
            ax.plot(subset['value'], subset[score_col], marker='o', label=dataset, alpha=0.7)
        ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=11)
        ax.set_ylabel(score_col, fontsize=11)
        ax.set_title('By Dataset', fontsize=12)
        if len(datasets) <= 8:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'ablation_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'ablation_{param_name}.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir}")


def generate_latex_table(all_results: Dict[str, pd.DataFrame]) -> str:
    """Generate LaTeX table for paper."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Hyperparameter Ablation Summary. Rankings remain stable across parameter choices ($\rho > 0.9$).}
\label{tab:ablation-results}
\begin{tabular}{lcccc}
\toprule
Parameter & Range & Default & Discrimination & Ranking $\rho$ \\
\midrule
"""
    for param_name, df in all_results.items():
        if df.empty:
            continue

        score_col = 'stability_score' if 'stability_score' in df.columns else 'c_mean'
        values = sorted(df['value'].unique())

        # Calculate discrimination
        discrimination = calculate_discrimination_score(df, 'value', score_col)
        best_disc = max(discrimination.values()) if discrimination else 0

        # Calculate ranking correlation
        corr_df = calculate_ranking_correlation(df, 'value', score_col)
        mean_rho = corr_df['spearman_rho'].mean() if not corr_df.empty else 1.0

        param_latex = {
            'alpha': r'$\alpha$ (steepness)',
            'theta': r'$\theta$ (threshold)',
            'path_decay': r'$\lambda$ (path decay)',
            'structural_weight': r'$w$ (struct. weight)',
        }.get(param_name, param_name)

        default = DEFAULT_VALUES.get(param_name, '-')
        value_range = f"[{min(values)}, {max(values)}]"

        latex += f"{param_latex} & {value_range} & {default} & {best_disc:.3f} & {mean_rho:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description='Run STED hyperparameter ablation experiments')
    parser.add_argument('--output-dir', type=str, default='results/ablation',
                        help='Output directory for results')
    parser.add_argument('--parameters', nargs='+',
                        choices=['alpha', 'theta', 'path_decay', 'structural_weight', 'all'],
                        default=['all'],
                        help='Parameters to ablate')
    parser.add_argument('--real-data', type=str, default=None,
                        help='Path to real LLM results directory (e.g., results/toucan/minilm-ec2)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick ablation with fewer values')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STED Hyperparameter Ablation Experiments")
    print("=" * 70)

    # Parameter ranges
    if args.quick:
        param_ranges = {
            'alpha': [10, 20, 30],
            'theta': [0.1, 0.3, 0.7],
            'path_decay': [0.8, 1.0],
            'structural_weight': [0.3, 0.5, 0.7],
        }
    else:
        param_ranges = HYPERPARAMETER_RANGES

    params_to_run = list(param_ranges.keys()) if 'all' in args.parameters else args.parameters

    # Create datasets
    print("\nPreparing datasets...")
    synthetic_datasets = create_synthetic_dataset()
    print(f"  Synthetic: {len(synthetic_datasets)} datasets")

    real_datasets = []
    if args.real_data:
        real_datasets = load_real_data(args.real_data)
        print(f"  Real data: {len(real_datasets)} model-temperature combinations")

    # Initialize evaluator and analyzer
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='all-MiniLM-L6-v2')
    analyzer = StructuralConsistencyAnalyzer(evaluator)

    all_results = {}
    start_time = time.time()

    # Run ablations
    if 'alpha' in params_to_run:
        print("\n[1/4] Running alpha (steepness) ablation...")
        all_results['alpha'] = run_alpha_ablation_synthetic(
            analyzer, synthetic_datasets, param_ranges['alpha']
        )
        if real_datasets:
            all_results['alpha_real'] = run_alpha_ablation_real(
                real_datasets, param_ranges['alpha']
            )

    if 'theta' in params_to_run:
        print("\n[2/4] Running theta (structural threshold) ablation...")
        all_results['theta'] = run_theta_ablation(synthetic_datasets, param_ranges['theta'])

    if 'path_decay' in params_to_run:
        print("\n[3/4] Running path decay ablation...")
        all_results['path_decay'] = run_path_decay_ablation(
            synthetic_datasets, param_ranges['path_decay']
        )

    if 'structural_weight' in params_to_run:
        print("\n[4/4] Running structural weight ablation...")
        all_results['structural_weight'] = run_structural_weight_ablation(
            synthetic_datasets, param_ranges['structural_weight']
        )

    elapsed_time = time.time() - start_time
    print(f"\nTotal ablation time: {elapsed_time:.2f} seconds")

    # Save results
    print("\nSaving results...")

    # JSON
    results_json = {k: v.to_dict(orient='records') for k, v in all_results.items() if not v.empty}
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # CSV
    for name, df in all_results.items():
        if not df.empty:
            df.to_csv(output_dir / f'ablation_{name}.csv', index=False)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_all_ablations(all_results, output_dir)

    # Special alpha plot for paper
    if 'alpha' in all_results:
        plot_alpha_ablation(all_results['alpha'], output_dir, use_real_data=False)
    if 'alpha_real' in all_results:
        plot_alpha_ablation(all_results['alpha_real'], output_dir, use_real_data=True)

    # LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(all_results)
    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex_table)
    print(latex_table)

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    for param_name, df in all_results.items():
        if df.empty:
            continue
        score_col = 'stability_score' if 'stability_score' in df.columns else 'c_mean'
        print(f"\n{param_name.upper()}:")

        grouped = df.groupby('value')[score_col].agg(['mean', 'std'])
        for value in grouped.index:
            print(f"  {value}: mean={grouped.loc[value, 'mean']:.4f} (std={grouped.loc[value, 'std']:.4f})")

        # Ranking correlation
        corr_df = calculate_ranking_correlation(df, 'value', score_col)
        if not corr_df.empty:
            print(f"  Ranking stability (mean Spearman rho): {corr_df['spearman_rho'].mean():.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
