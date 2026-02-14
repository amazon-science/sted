#!/usr/bin/env python3
"""
Embedding Model Ablation Study for STED

This script compares STED performance across different embedding models
to validate the claim that "choice of embedding model has minimal impact."

Embedding models tested:
1. Amazon Titan Text Embeddings v2 (1024-dim, commercial)
2. all-MiniLM-L6-v2 (384-dim, open-source, lightweight)
3. paraphrase-mpnet-base-v2 (768-dim, open-source, high quality)
4. all-mpnet-base-v2 (768-dim, open-source, general purpose)

Usage:
    python scripts/analysis/compare_embedding_models.py \
        --data-dir data/synthetic_variations \
        --output-dir results/embedding_ablation \
        --models all  # or specific model names
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str
    model_id: str
    dimension: int
    source: str  # 'bedrock' or 'sentence-transformers'
    description: str


# Define embedding models to compare
EMBEDDING_MODELS = {
    'titan-v2': EmbeddingModelConfig(
        name='Amazon Titan v2',
        model_id='amazon.titan-embed-text-v2:0',
        dimension=1024,
        source='bedrock',
        description='Commercial, AWS Bedrock'
    ),
    'minilm': EmbeddingModelConfig(
        name='all-MiniLM-L6-v2',
        model_id='all-MiniLM-L6-v2',
        dimension=384,
        source='sentence-transformers',
        description='Lightweight, fast'
    ),
    'mpnet': EmbeddingModelConfig(
        name='all-mpnet-base-v2',
        model_id='all-mpnet-base-v2',
        dimension=768,
        source='sentence-transformers',
        description='High quality, general'
    ),
    'paraphrase-mpnet': EmbeddingModelConfig(
        name='paraphrase-mpnet-base-v2',
        model_id='paraphrase-mpnet-base-v2',
        dimension=768,
        source='sentence-transformers',
        description='Optimized for paraphrase'
    ),
}


@dataclass
class VariationSample:
    """A single variation sample for evaluation."""
    original: Dict[str, Any]
    variant: Dict[str, Any]
    variation_type: str
    variation_ratio: float
    sample_id: str


@dataclass
class EvaluationResult:
    """Result from evaluating a single sample."""
    model_name: str
    variation_type: str
    variation_ratio: float
    similarity_score: float
    computation_time_ms: float
    sample_id: str


def load_variation_data(data_dir: str) -> Dict[str, List[VariationSample]]:
    """
    Load synthetic variation data from directory.

    Expected structure:
        data_dir/
            schema_variations/
                field_name_0.1.json, field_name_0.2.json, ...
                flat_structure.json
                nested_changes.json
            expression_variations/
                expression_0.1.json, expression_0.2.json, ...
            semantic_variations/
                semantic_0.1.json, semantic_0.2.json, ...

    Returns:
        Dictionary mapping variation_type -> list of VariationSamples
    """
    data_path = Path(data_dir)
    variations = defaultdict(list)

    # Try to load from expected structure
    variation_dirs = {
        'schema': data_path / 'schema_variations',
        'expression': data_path / 'expression_variations',
        'semantic': data_path / 'semantic_variations'
    }

    for var_type, var_dir in variation_dirs.items():
        if var_dir.exists():
            for json_file in var_dir.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    # Parse filename to extract ratio
                    filename = json_file.stem
                    ratio = 0.0
                    if '_' in filename:
                        try:
                            ratio = float(filename.split('_')[-1])
                        except ValueError:
                            pass

                    # Handle both list and dict formats
                    samples = data if isinstance(data, list) else [data]

                    for idx, sample in enumerate(samples):
                        if 'original' in sample and 'variant' in sample:
                            variations[var_type].append(VariationSample(
                                original=sample['original'],
                                variant=sample['variant'],
                                variation_type=var_type,
                                variation_ratio=ratio,
                                sample_id=f"{var_type}_{filename}_{idx}"
                            ))
                except Exception as e:
                    warnings.warn(f"Failed to load {json_file}: {e}")

    # If no structured data found, try loading from a single combined file
    if not any(variations.values()):
        combined_file = data_path / 'variations.json'
        if combined_file.exists():
            with open(combined_file, 'r') as f:
                data = json.load(f)
            for item in data:
                var_type = item.get('type', 'unknown')
                variations[var_type].append(VariationSample(
                    original=item['original'],
                    variant=item['variant'],
                    variation_type=var_type,
                    variation_ratio=item.get('ratio', 0.0),
                    sample_id=item.get('id', f"{var_type}_{len(variations[var_type])}")
                ))

    return dict(variations)


def generate_test_variations(num_samples: int = 50) -> Dict[str, List[VariationSample]]:
    """
    Generate synthetic test variations if no data file exists.

    Creates controlled variations for testing embedding model robustness.
    """
    print("Generating synthetic test variations...")

    variations = defaultdict(list)

    # Base templates for variation generation
    base_samples = [
        {"user_name": "John Doe", "email": "john@example.com", "age": 30},
        {"product_name": "Laptop", "price": 999.99, "category": "Electronics"},
        {"title": "Meeting Notes", "content": "Discussed project timeline", "date": "2024-01-15"},
        {"customer_id": "C123", "order_total": 150.00, "items": ["item1", "item2"]},
        {"name": "Test Report", "status": "completed", "score": 85},
    ]

    # Schema variations: field name changes
    field_name_mappings = [
        ("user_name", "userName"), ("user_name", "username"), ("user_name", "user"),
        ("email", "email_address"), ("email", "emailAddr"), ("email", "mail"),
        ("product_name", "productName"), ("product_name", "name"), ("product_name", "item_name"),
        ("customer_id", "customerId"), ("customer_id", "cust_id"), ("customer_id", "id"),
    ]

    for ratio_idx, ratio in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        for idx, base in enumerate(base_samples[:num_samples]):
            # Create variant with field name changes proportional to ratio
            variant = base.copy()
            keys_to_change = list(base.keys())[:int(len(base.keys()) * ratio) + 1]

            for old_key in keys_to_change:
                for old, new in field_name_mappings:
                    if old_key == old:
                        variant[new] = variant.pop(old_key)
                        break

            variations['schema'].append(VariationSample(
                original=base,
                variant=variant,
                variation_type='schema',
                variation_ratio=ratio,
                sample_id=f"schema_{ratio}_{idx}"
            ))

    # Expression variations: paraphrased values
    for ratio_idx, ratio in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        for idx, base in enumerate(base_samples[:num_samples]):
            variant = {}
            for key, value in base.items():
                if isinstance(value, str) and np.random.random() < ratio:
                    # Simple paraphrase simulation
                    variant[key] = value.replace(" ", "_").title() if " " in value else value
                else:
                    variant[key] = value

            variations['expression'].append(VariationSample(
                original=base,
                variant=variant,
                variation_type='expression',
                variation_ratio=ratio,
                sample_id=f"expression_{ratio}_{idx}"
            ))

    # Semantic variations: value changes
    for ratio_idx, ratio in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        for idx, base in enumerate(base_samples[:num_samples]):
            variant = {}
            for key, value in base.items():
                if isinstance(value, str) and np.random.random() < ratio:
                    variant[key] = f"different_{value}"
                elif isinstance(value, (int, float)) and np.random.random() < ratio:
                    variant[key] = value * 1.5
                else:
                    variant[key] = value

            variations['semantic'].append(VariationSample(
                original=base,
                variant=variant,
                variation_type='semantic',
                variation_ratio=ratio,
                sample_id=f"semantic_{ratio}_{idx}"
            ))

    return dict(variations)


def evaluate_with_model(
    model_config: EmbeddingModelConfig,
    samples: List[VariationSample],
    show_progress: bool = True
) -> List[EvaluationResult]:
    """
    Evaluate STED similarity for samples using a specific embedding model.

    Args:
        model_config: Configuration for the embedding model
        samples: List of variation samples to evaluate
        show_progress: Whether to show progress bar

    Returns:
        List of evaluation results
    """
    print(f"\nEvaluating with {model_config.name} ({model_config.model_id})...")

    # Initialize evaluator with specific model
    try:
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id=model_config.model_id,
            region_name='us-west-2'
        )
    except Exception as e:
        warnings.warn(f"Failed to initialize {model_config.name}: {e}")
        return []

    # Pre-compute embeddings for all samples
    all_jsons = []
    for sample in samples:
        all_jsons.append(sample.original)
        all_jsons.append(sample.variant)

    print(f"Pre-computing embeddings for {len(all_jsons)} JSON objects...")
    try:
        evaluator.precompute_embeddings(all_jsons, batch_size=64, show_progress=show_progress)
    except Exception as e:
        warnings.warn(f"Embedding precomputation failed: {e}")

    results = []
    iterator = tqdm(samples, desc=f"Evaluating {model_config.name}") if show_progress else samples

    for sample in iterator:
        try:
            start_time = time.perf_counter()

            similarity = evaluator.calculate_tree_edit_distance_opt(
                sample.original,
                sample.variant,
                variation_type='combined'
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            results.append(EvaluationResult(
                model_name=model_config.name,
                variation_type=sample.variation_type,
                variation_ratio=sample.variation_ratio,
                similarity_score=similarity,
                computation_time_ms=elapsed_ms,
                sample_id=sample.sample_id
            ))
        except Exception as e:
            warnings.warn(f"Evaluation failed for {sample.sample_id}: {e}")

    return results


def analyze_results(results: List[EvaluationResult]) -> pd.DataFrame:
    """
    Analyze evaluation results and compute summary statistics.

    Returns:
        DataFrame with aggregated statistics
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            'model': r.model_name,
            'variation_type': r.variation_type,
            'ratio': r.variation_ratio,
            'similarity': r.similarity_score,
            'time_ms': r.computation_time_ms
        }
        for r in results
    ])

    # Aggregate by model and variation type
    summary = df.groupby(['model', 'variation_type']).agg({
        'similarity': ['mean', 'std', 'min', 'max'],
        'time_ms': ['mean', 'std']
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    return summary.reset_index()


def compute_model_agreement(results: List[EvaluationResult]) -> pd.DataFrame:
    """
    Compute agreement between embedding models.

    Measures how similar the STED scores are across different embedding models
    for the same samples.
    """
    if not results:
        return pd.DataFrame()

    # Pivot to get model scores side by side
    df = pd.DataFrame([
        {
            'sample_id': r.sample_id,
            'model': r.model_name,
            'similarity': r.similarity_score
        }
        for r in results
    ])

    pivot = df.pivot(index='sample_id', columns='model', values='similarity')

    # Compute pairwise correlations
    correlations = pivot.corr()

    # Compute mean absolute difference between models
    models = pivot.columns.tolist()
    differences = {}

    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            diff = np.abs(pivot[m1] - pivot[m2]).mean()
            differences[f"{m1} vs {m2}"] = diff

    return correlations, differences


def generate_latex_table(summary: pd.DataFrame, output_path: str = None) -> str:
    """Generate LaTeX table for paper."""

    # Pivot for better formatting
    pivot = summary.pivot(
        index='model',
        columns='variation_type',
        values=['similarity_mean', 'similarity_std']
    )

    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{STED Similarity Scores Across Embedding Models}\n"
    latex += "\\label{tab:embedding-ablation}\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{lccc}\n\\toprule\n"
    latex += "\\textbf{Embedding Model} & \\textbf{Schema} & \\textbf{Expression} & \\textbf{Semantic} \\\\\n"
    latex += "\\midrule\n"

    for model in summary['model'].unique():
        model_data = summary[summary['model'] == model]
        row = f"{model}"
        for vtype in ['schema', 'expression', 'semantic']:
            vdata = model_data[model_data['variation_type'] == vtype]
            if not vdata.empty:
                mean = vdata['similarity_mean'].values[0]
                std = vdata['similarity_std'].values[0]
                row += f" & {mean:.3f}$\\pm${std:.3f}"
            else:
                row += " & --"
        latex += row + " \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


def main():
    parser = argparse.ArgumentParser(description='Embedding Model Ablation Study for STED')
    parser.add_argument('--data-dir', type=str, default='data/synthetic_variations',
                        help='Directory containing variation data')
    parser.add_argument('--output-dir', type=str, default='results/embedding_ablation',
                        help='Directory to save results')
    parser.add_argument('--models', type=str, nargs='+', default=['all'],
                        help='Models to evaluate (all, titan-v2, minilm, mpnet, paraphrase-mpnet)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples per variation ratio (if generating)')
    parser.add_argument('--skip-bedrock', action='store_true',
                        help='Skip Bedrock models (for testing without AWS access)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to evaluate
    if 'all' in args.models:
        models_to_eval = list(EMBEDDING_MODELS.keys())
    else:
        models_to_eval = args.models

    if args.skip_bedrock:
        models_to_eval = [m for m in models_to_eval if EMBEDDING_MODELS[m].source != 'bedrock']

    print(f"Models to evaluate: {models_to_eval}")

    # Load or generate variation data
    data_path = Path(args.data_dir)
    if data_path.exists():
        variations = load_variation_data(args.data_dir)
        print(f"Loaded variations: {', '.join(f'{k}: {len(v)}' for k, v in variations.items())}")
    else:
        print(f"Data directory {args.data_dir} not found, generating synthetic data...")
        variations = generate_test_variations(args.num_samples)

    if not any(variations.values()):
        print("No variation data found. Generating minimal test set...")
        variations = generate_test_variations(args.num_samples)

    # Flatten all samples
    all_samples = []
    for var_type, samples in variations.items():
        all_samples.extend(samples)

    print(f"\nTotal samples to evaluate: {len(all_samples)}")

    # Evaluate with each model
    all_results = []

    for model_key in models_to_eval:
        if model_key not in EMBEDDING_MODELS:
            warnings.warn(f"Unknown model: {model_key}")
            continue

        model_config = EMBEDDING_MODELS[model_key]

        try:
            results = evaluate_with_model(model_config, all_samples)
            all_results.extend(results)

            # Save intermediate results
            intermediate_df = pd.DataFrame([
                {
                    'model': r.model_name,
                    'variation_type': r.variation_type,
                    'ratio': r.variation_ratio,
                    'similarity': r.similarity_score,
                    'time_ms': r.computation_time_ms,
                    'sample_id': r.sample_id
                }
                for r in results
            ])
            intermediate_df.to_csv(output_dir / f'{model_key}_results.csv', index=False)

        except Exception as e:
            warnings.warn(f"Failed to evaluate {model_key}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No results collected. Check model configurations and data.")
        return

    # Analyze results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    summary = analyze_results(all_results)
    print("\nAggregated Statistics:")
    print(summary.to_string(index=False))

    # Compute model agreement
    correlations, differences = compute_model_agreement(all_results)

    print("\nModel Correlations:")
    print(correlations.round(3).to_string())

    print("\nMean Absolute Differences:")
    for pair, diff in sorted(differences.items(), key=lambda x: x[1]):
        print(f"  {pair}: {diff:.4f}")

    # Save results
    summary.to_csv(output_dir / 'summary.csv', index=False)
    correlations.to_csv(output_dir / 'correlations.csv')

    with open(output_dir / 'differences.json', 'w') as f:
        json.dump(differences, f, indent=2)

    # Generate LaTeX table
    latex_table = generate_latex_table(summary, output_dir / 'table.tex')
    print("\nLaTeX Table:")
    print(latex_table)

    # Final summary for paper
    print("\n" + "="*60)
    print("KEY FINDINGS FOR PAPER")
    print("="*60)

    # Check if claim holds: max difference should be < 3%
    max_diff = max(differences.values()) if differences else 0
    mean_diff = np.mean(list(differences.values())) if differences else 0

    print(f"Maximum pairwise difference: {max_diff:.4f} ({max_diff*100:.2f}%)")
    print(f"Mean pairwise difference: {mean_diff:.4f} ({mean_diff*100:.2f}%)")

    if max_diff < 0.03:
        print("\n[VALIDATED] Embedding model choice has minimal impact (<3% difference)")
    elif max_diff < 0.05:
        print("\n[ACCEPTABLE] Embedding models show small differences (<5%)")
    else:
        print("\n[WARNING] Embedding model choice shows significant impact (>5% difference)")
        print("Consider revising the paper's claim about embedding model impact.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
