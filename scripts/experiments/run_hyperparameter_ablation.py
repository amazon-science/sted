#!/usr/bin/env python3
"""
Hyperparameter Ablation Experiments for STED (ICML Submission)

This script runs comprehensive ablation studies on STED hyperparameters:
- α (alpha): Consistency score steepness factor
- θ (theta): Structural similarity threshold for content matching
- λ_path (path_weight_decay): Depth-based weight decay
- λ_size (size_penalty): Penalty for unmatched elements
- w (structural_weight): Weight balancing structural vs content costs

Reference: docs/sted_theory_icml.tex Table 1
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted import SemanticJsonTreeConsistencyEvaluator


# Hyperparameter ranges for ablation
HYPERPARAMETER_RANGES = {
    'alpha': [5, 10, 15, 20, 25, 30],  # Consistency score steepness
    'theta': [0.1, 0.2, 0.3, 0.5, 0.7],  # Structural threshold
    'path_decay': [0.7, 0.8, 0.9, 1.0],  # Path decay
    'structural_weight': [0.3, 0.5, 0.7],  # Structural vs content weight
}


def generate_synthetic_variations(base_json: Dict, num_variations: int = 10,
                                   noise_level: float = 0.2) -> List[Dict]:
    """Generate synthetic variations of a base JSON for testing."""
    import copy
    import random

    variations = [copy.deepcopy(base_json)]

    for i in range(num_variations - 1):
        var = copy.deepcopy(base_json)

        # Apply random perturbations
        def perturb_value(v, path=""):
            if isinstance(v, str):
                # Randomly modify strings
                if random.random() < noise_level:
                    modifications = [
                        v + " modified",
                        v.replace("a", "e") if "a" in v else v + "_alt",
                        f"{v}_v{i}",
                    ]
                    return random.choice(modifications)
                return v
            elif isinstance(v, (int, float)):
                # Add numeric noise
                if random.random() < noise_level:
                    return v + random.gauss(0, abs(v) * 0.1 + 1)
                return v
            elif isinstance(v, bool):
                if random.random() < noise_level * 0.5:
                    return not v
                return v
            elif isinstance(v, list):
                # Optionally reorder or modify list
                result = [perturb_value(item, f"{path}[{j}]") for j, item in enumerate(v)]
                if random.random() < noise_level * 0.3:
                    random.shuffle(result)
                return result
            elif isinstance(v, dict):
                result = {}
                for k, val in v.items():
                    result[k] = perturb_value(val, f"{path}.{k}")
                # Optionally add/remove keys
                if random.random() < noise_level * 0.2:
                    result[f"extra_field_{i}"] = f"value_{i}"
                return result
            return v

        var = perturb_value(var)
        variations.append(var)

    return variations


def create_test_dataset() -> List[Tuple[str, List[Dict]]]:
    """Create a diverse test dataset for ablation experiments."""
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

    # 4. Function calling output (like xlam/glaive)
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


def run_alpha_ablation(evaluator: SemanticJsonTreeConsistencyEvaluator,
                       datasets: List[Tuple[str, List[Dict]]],
                       alpha_values: List[int]) -> pd.DataFrame:
    """Run ablation study on alpha (steepness factor) parameter."""
    results = []

    for alpha in tqdm(alpha_values, desc="Alpha ablation"):
        for dataset_name, variations in datasets:
            metrics = evaluator.calculate_variation_consistency(
                variations,
                method='sted',
                variation_type='combined',
                apply_power_transform=True,
                steepness_factor=alpha
            )

            results.append({
                'parameter': 'alpha',
                'value': alpha,
                'dataset': dataset_name,
                'consistency_score': metrics['consistency_score'],
                'mean_distance': metrics.get('mean_distance', 0),
                'std_distance': metrics.get('std_distance', 0),
            })

    return pd.DataFrame(results)


def run_theta_ablation(datasets: List[Tuple[str, List[Dict]]],
                       theta_values: List[float]) -> pd.DataFrame:
    """Run ablation study on theta (structural threshold) parameter."""
    results = []

    for theta in tqdm(theta_values, desc="Theta ablation"):
        # Create custom evaluator - need to patch the method
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
        )

        # Store original method
        original_method = evaluator._calculate_content_similarity

        # Patch to use custom theta
        def patched_content_similarity(node1, node2, structural_sim_threshold=theta):
            return original_method(node1, node2, structural_sim_threshold=theta)

        evaluator._calculate_content_similarity = patched_content_similarity

        for dataset_name, variations in datasets:
            # Calculate pairwise similarities with this theta
            from itertools import combinations

            similarities = []
            for v1, v2 in combinations(variations, 2):
                sim = evaluator.calculate_tree_edit_distance_opt(v1, v2, variation_type='combined')
                similarities.append(sim)

            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)

            results.append({
                'parameter': 'theta',
                'value': theta,
                'dataset': dataset_name,
                'mean_similarity': mean_sim,
                'std_similarity': std_sim,
            })

    return pd.DataFrame(results)


def run_path_decay_ablation(datasets: List[Tuple[str, List[Dict]]],
                            decay_values: List[float]) -> pd.DataFrame:
    """Run ablation study on path weight decay (lambda_path) parameter."""
    results = []

    for decay in tqdm(decay_values, desc="Path decay ablation"):
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
            path_weight_decay=decay,
        )

        for dataset_name, variations in datasets:
            metrics = evaluator.calculate_variation_consistency(
                variations,
                method='sted',
                variation_type='combined',
                apply_power_transform=True,
                steepness_factor=20
            )

            results.append({
                'parameter': 'path_decay',
                'value': decay,
                'dataset': dataset_name,
                'consistency_score': metrics['consistency_score'],
                'mean_distance': metrics.get('mean_distance', 0),
                'std_distance': metrics.get('std_distance', 0),
            })

    return pd.DataFrame(results)


def run_structural_weight_ablation(datasets: List[Tuple[str, List[Dict]]],
                                    weight_values: List[float]) -> pd.DataFrame:
    """Run ablation study on structural weight (alpha in combined cost) parameter."""
    results = []

    for weight in tqdm(weight_values, desc="Structural weight ablation"):
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='all-MiniLM-L6-v2',
        )

        # Patch the update_cost method to use custom weight
        original_update_cost = evaluator.update_cost

        def patched_update_cost(node1, node2, structural_weight=weight):
            return original_update_cost(node1, node2, structural_weight=weight)

        evaluator.update_cost = patched_update_cost

        for dataset_name, variations in datasets:
            metrics = evaluator.calculate_variation_consistency(
                variations,
                method='sted',
                variation_type='combined',
                apply_power_transform=True,
                steepness_factor=20
            )

            results.append({
                'parameter': 'structural_weight',
                'value': weight,
                'dataset': dataset_name,
                'consistency_score': metrics['consistency_score'],
                'mean_distance': metrics.get('mean_distance', 0),
                'std_distance': metrics.get('std_distance', 0),
            })

    return pd.DataFrame(results)


def calculate_discrimination_score(df: pd.DataFrame, value_col: str = 'value',
                                    score_col: str = 'consistency_score') -> Dict[str, float]:
    """
    Calculate how well a parameter setting discriminates between datasets.
    Higher is better - indicates the parameter creates meaningful distinctions.
    """
    discrimination_scores = {}

    for value in df[value_col].unique():
        subset = df[df[value_col] == value]
        scores = subset.groupby('dataset')[score_col].mean()

        # Discrimination = variance across datasets (higher = more discriminating)
        discrimination_scores[value] = float(scores.std())

    return discrimination_scores


def plot_ablation_results(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate comprehensive ablation visualizations."""

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Alpha (steepness) ablation plot
    if 'alpha' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df = all_results['alpha']

        # Box plot by alpha value
        alpha_values = sorted(df['value'].unique())
        data_for_box = [df[df['value'] == a]['consistency_score'].values for a in alpha_values]
        axes[0].boxplot(data_for_box, labels=alpha_values)
        axes[0].set_xlabel('α (Steepness Factor)')
        axes[0].set_ylabel('Consistency Score')
        axes[0].set_title('Consistency Score Distribution by α')
        axes[0].axvline(x=alpha_values.index(20) + 1, color='r', linestyle='--',
                        label=f'Default α=20', alpha=0.7)
        axes[0].legend()

        # Line plot by dataset
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            axes[1].plot(subset['value'], subset['consistency_score'],
                        marker='o', label=dataset)
        axes[1].set_xlabel('α (Steepness Factor)')
        axes[1].set_ylabel('Consistency Score')
        axes[1].set_title('Consistency Score by Dataset')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_alpha.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Theta (structural threshold) ablation plot
    if 'theta' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df = all_results['theta']

        theta_values = sorted(df['value'].unique())
        data_for_box = [df[df['value'] == t]['mean_similarity'].values for t in theta_values]
        axes[0].boxplot(data_for_box, labels=[f'{t:.1f}' for t in theta_values])
        axes[0].set_xlabel('θ (Structural Threshold)')
        axes[0].set_ylabel('Mean Similarity')
        axes[0].set_title('Similarity Distribution by θ')
        if 0.3 in theta_values:
            axes[0].axvline(x=theta_values.index(0.3) + 1, color='r', linestyle='--',
                            label='Default θ=0.3', alpha=0.7)
            axes[0].legend()

        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            axes[1].plot(subset['value'], subset['mean_similarity'],
                        marker='o', label=dataset)
        axes[1].set_xlabel('θ (Structural Threshold)')
        axes[1].set_ylabel('Mean Similarity')
        axes[1].set_title('Similarity by Dataset')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_theta.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Path decay ablation plot
    if 'path_decay' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df = all_results['path_decay']

        decay_values = sorted(df['value'].unique())
        data_for_box = [df[df['value'] == d]['consistency_score'].values for d in decay_values]
        axes[0].boxplot(data_for_box, labels=[f'{d:.1f}' for d in decay_values])
        axes[0].set_xlabel('λ_path (Path Weight Decay)')
        axes[0].set_ylabel('Consistency Score')
        axes[0].set_title('Consistency by Path Decay')
        if 1.0 in decay_values:
            axes[0].axvline(x=decay_values.index(1.0) + 1, color='r', linestyle='--',
                            label='Default λ=1.0', alpha=0.7)
            axes[0].legend()

        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            axes[1].plot(subset['value'], subset['consistency_score'],
                        marker='o', label=dataset)
        axes[1].set_xlabel('λ_path (Path Weight Decay)')
        axes[1].set_ylabel('Consistency Score')
        axes[1].set_title('Consistency by Dataset')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_path_decay.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Structural weight ablation plot
    if 'structural_weight' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df = all_results['structural_weight']

        weight_values = sorted(df['value'].unique())
        data_for_box = [df[df['value'] == w]['consistency_score'].values for w in weight_values]
        axes[0].boxplot(data_for_box, labels=[f'{w:.1f}' for w in weight_values])
        axes[0].set_xlabel('w (Structural Weight)')
        axes[0].set_ylabel('Consistency Score')
        axes[0].set_title('Consistency by Structural Weight')
        if 0.5 in weight_values:
            axes[0].axvline(x=weight_values.index(0.5) + 1, color='r', linestyle='--',
                            label='Default w=0.5', alpha=0.7)
            axes[0].legend()

        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            axes[1].plot(subset['value'], subset['consistency_score'],
                        marker='o', label=dataset)
        axes[1].set_xlabel('w (Structural Weight)')
        axes[1].set_ylabel('Consistency Score')
        axes[1].set_title('Consistency by Dataset')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_structural_weight.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Summary heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    summary_data = []
    for param_name, df in all_results.items():
        score_col = 'consistency_score' if 'consistency_score' in df.columns else 'mean_similarity'
        discrimination = calculate_discrimination_score(df, 'value', score_col)
        for value, disc_score in discrimination.items():
            summary_data.append({
                'parameter': param_name,
                'value': str(value),
                'discrimination': disc_score
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        pivot_df = summary_df.pivot(index='parameter', columns='value', values='discrimination')

        im = ax.imshow(pivot_df.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(pivot_df.columns)))
        ax.set_xticklabels(pivot_df.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_df.index)))
        ax.set_yticklabels(pivot_df.index)
        ax.set_title('Hyperparameter Discrimination Scores\n(Higher = Better Dataset Differentiation)')
        plt.colorbar(im, ax=ax, label='Discrimination Score')

        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_summary_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir}")


def generate_latex_table(all_results: Dict[str, pd.DataFrame]) -> str:
    """Generate LaTeX table for ICML paper."""

    latex = """
\\begin{table}[h]
\\centering
\\caption{Hyperparameter Ablation Results}
\\label{tab:ablation-results}
\\begin{tabular}{lcccc}
\\toprule
Parameter & Range & Default & Optimal & Selection Criterion \\\\
\\midrule
"""

    for param_name, df in all_results.items():
        score_col = 'consistency_score' if 'consistency_score' in df.columns else 'mean_similarity'

        # Find optimal value (highest discrimination)
        discrimination = calculate_discrimination_score(df, 'value', score_col)
        optimal_value = max(discrimination, key=discrimination.get)

        # Get range
        values = sorted(df['value'].unique())
        value_range = f"[{min(values)}, {max(values)}]"

        # Default values
        defaults = {'alpha': 20, 'theta': 0.3, 'path_decay': 1.0, 'structural_weight': 0.5}
        default = defaults.get(param_name, '-')

        # Criterion
        criteria = {
            'alpha': 'Max discrimination in $\\hat{\\sigma} \\in [0, 0.1]$',
            'theta': 'Balance flexibility vs. coherence',
            'path_decay': 'Weight root vs. leaf importance',
            'structural_weight': 'Balance structure vs. content'
        }
        criterion = criteria.get(param_name, '-')

        # Format parameter name
        param_latex = {
            'alpha': '$\\alpha$ (consistency steepness)',
            'theta': '$\\theta$ (structural threshold)',
            'path_decay': '$\\lambda$ (path decay)',
            'structural_weight': '$w$ (structural weight)'
        }.get(param_name, param_name)

        latex += f"{param_latex} & {value_range} & {default} & {optimal_value} & {criterion} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description='Run STED hyperparameter ablation experiments')
    parser.add_argument('--output-dir', type=str, default='research/ablation_results',
                        help='Output directory for results')
    parser.add_argument('--parameters', nargs='+',
                        choices=['alpha', 'theta', 'path_decay', 'structural_weight', 'all'],
                        default=['all'],
                        help='Parameters to ablate')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick ablation with fewer values')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STED Hyperparameter Ablation Experiments")
    print("For ICML 2026 Submission")
    print("=" * 70)

    # Create test datasets
    print("\nGenerating test datasets...")
    datasets = create_test_dataset()
    print(f"Created {len(datasets)} test datasets:")
    for name, variations in datasets:
        print(f"  - {name}: {len(variations)} variations")

    # Initialize base evaluator
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='all-MiniLM-L6-v2',
    )

    # Run ablations
    all_results = {}
    params_to_run = HYPERPARAMETER_RANGES.keys() if 'all' in args.parameters else args.parameters

    if args.quick:
        # Reduce parameter ranges for quick testing
        quick_ranges = {
            'alpha': [10, 20, 30],
            'theta': [0.1, 0.3, 0.7],
            'path_decay': [0.9, 1.0],
            'structural_weight': [0.3, 0.5, 0.7],
        }
        param_ranges = {k: quick_ranges.get(k, v) for k, v in HYPERPARAMETER_RANGES.items()}
    else:
        param_ranges = HYPERPARAMETER_RANGES

    start_time = time.time()

    if 'alpha' in params_to_run:
        print("\n[1/4] Running alpha (steepness) ablation...")
        all_results['alpha'] = run_alpha_ablation(evaluator, datasets, param_ranges['alpha'])

    if 'theta' in params_to_run:
        print("\n[2/4] Running theta (structural threshold) ablation...")
        all_results['theta'] = run_theta_ablation(datasets, param_ranges['theta'])

    if 'path_decay' in params_to_run:
        print("\n[3/4] Running path decay ablation...")
        all_results['path_decay'] = run_path_decay_ablation(datasets, param_ranges['path_decay'])

    if 'structural_weight' in params_to_run:
        print("\n[4/4] Running structural weight ablation...")
        all_results['structural_weight'] = run_structural_weight_ablation(
            datasets, param_ranges['structural_weight']
        )

    elapsed_time = time.time() - start_time
    print(f"\nTotal ablation time: {elapsed_time:.2f} seconds")

    # Save results
    print("\nSaving results...")

    # Save as JSON
    results_json = {}
    for name, df in all_results.items():
        results_json[name] = df.to_dict(orient='records')

    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save as CSV
    for name, df in all_results.items():
        df.to_csv(output_dir / f'ablation_{name}.csv', index=False)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_ablation_results(all_results, output_dir)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(all_results)
    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex_table)
    print(latex_table)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    for param_name, df in all_results.items():
        score_col = 'consistency_score' if 'consistency_score' in df.columns else 'mean_similarity'
        print(f"\n{param_name.upper()}:")

        # Group by value and compute mean score
        grouped = df.groupby('value')[score_col].agg(['mean', 'std'])
        for value in grouped.index:
            mean_score = grouped.loc[value, 'mean']
            std_score = grouped.loc[value, 'std']
            print(f"  {value}: mean={mean_score:.4f} (std={std_score:.4f})")

        # Compute discrimination scores
        discrimination = calculate_discrimination_score(df, 'value', score_col)
        best_value = max(discrimination, key=discrimination.get)
        print(f"  Best discriminating value: {best_value} (score={discrimination[best_value]:.4f})")

    print(f"\nResults saved to: {output_dir}")
    print("Files generated:")
    print(f"  - ablation_results.json")
    print(f"  - ablation_*.csv")
    print(f"  - ablation_*.png")
    print(f"  - ablation_table.tex")


if __name__ == '__main__':
    main()
