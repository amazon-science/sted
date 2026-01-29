#!/usr/bin/env python3
"""
Embedding Model Ablation Study for STED.

Compare how different embedding models affect STED consistency scores.
Uses sampling for fast iteration.

Usage:
    python embedding_model_ablation.py --results-dir llm_gen_results/toucan --num-samples 100
"""
import json
import os
import argparse
import random
from collections import defaultdict
import numpy as np
from scipy import stats

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer
from tqdm import tqdm


def load_samples(results_dir: str, num_samples: int, seed: int = 42) -> list:
    """Load and sample results from all model directories."""
    random.seed(seed)

    all_samples = []

    # Find all result directories
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if not os.path.isdir(item_path):
            continue

        # Check for all_results.json
        results_file = os.path.join(item_path, 'all_results.json')
        if not os.path.exists(results_file):
            # Check nested structure
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    results_file = os.path.join(subitem_path, 'all_results.json')
                    if os.path.exists(results_file):
                        break
            else:
                continue

        with open(results_file, 'r') as f:
            data = json.load(f)

        model_name = data.get('metadata', {}).get('display_name', item)

        for idx, sample in enumerate(data['results']):
            all_samples.append({
                'model': model_name,
                'sample_idx': idx,
                'ground_truth': sample['ground_truth'],
                'responses': sample.get('responses') or sample.get('generated_runs', [])
            })

    # Sample
    if len(all_samples) > num_samples:
        samples = random.sample(all_samples, num_samples)
    else:
        samples = all_samples

    print(f"Loaded {len(all_samples)} total samples, using {len(samples)} for ablation")
    return samples


def calculate_sted_scores(samples: list, embedding_model: str, dimension: int = 512,
                          region: str = 'us-west-2', variation_type: str = 'combined') -> list:
    """Calculate STED scores for all samples using specified embedding model."""

    # Initialize evaluator
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id=embedding_model,
        region_name=region,
        embedding_dim=dimension
    )
    analyzer = StructuralConsistencyAnalyzer(evaluator)

    # Collect all JSON objects for batch embedding
    all_json_objects = []
    for sample in samples:
        gt = sample['ground_truth']
        if gt:
            all_json_objects.append(gt)
        for resp in sample['responses'][:10]:
            if resp:
                all_json_objects.append(resp)

    # Precompute embeddings
    print(f"  Precomputing embeddings for {len(all_json_objects)} JSON objects...")
    evaluator.precompute_embeddings(
        all_json_objects,
        batch_size=64,
        show_progress=True,
        max_workers=10,
        use_async=True,
        max_concurrent=25
    )

    # Calculate STED scores
    scores = []
    for sample in tqdm(samples, desc="  Calculating STED"):
        gt = sample['ground_truth']
        responses = sample['responses'][:10]
        valid_responses = [r for r in responses if r]

        if len(valid_responses) >= 1:
            total_runs = len(responses)
            valid_runs = len(valid_responses)
            validity_rate = valid_runs / total_runs if total_runs > 0 else 0.0

            report = analyzer.evaluate_structural_consistency(
                valid_responses, gt, method_name="sted",
                variation_type=variation_type, validity_rate=validity_rate
            )
            pairwise_similarities = report.get('raw_similarities', [])
            metrics = analyzer._calculate_consistency_metrics(pairwise_similarities, validity_rate)

            scores.append({
                'model': sample['model'],
                'sample_idx': sample['sample_idx'],
                'c_mean': metrics['c_mean'],
                'stability_score': metrics['stability_score'],
                'c_adj': metrics['c_adj']
            })
        else:
            scores.append({
                'model': sample['model'],
                'sample_idx': sample['sample_idx'],
                'c_mean': 0.0,
                'stability_score': 0.0,
                'c_adj': 0.0
            })

    return scores


def compare_embedding_models(results: dict, metric: str = 'c_mean'):
    """Compare STED scores across embedding models using correlation."""

    model_names = list(results.keys())
    n_models = len(model_names)

    print(f"\n{'='*60}")
    print(f"Correlation Analysis (metric: {metric})")
    print(f"{'='*60}")

    # Extract scores for each embedding model
    scores_by_model = {}
    for emb_model in model_names:
        scores_by_model[emb_model] = [s[metric] for s in results[emb_model]]

    # Compute pairwise correlations
    print(f"\n{'Spearman Correlation Matrix':^60}")
    print("-" * 60)

    # Header
    header = f"{'':20}"
    for name in model_names:
        short_name = name.split('.')[-1][:12]
        header += f"{short_name:>12}"
    print(header)

    correlations = {}
    for i, model1 in enumerate(model_names):
        row = f"{model_names[i].split('.')[-1][:18]:20}"
        for j, model2 in enumerate(model_names):
            if i <= j:
                corr, pval = stats.spearmanr(scores_by_model[model1], scores_by_model[model2])
                correlations[(model1, model2)] = corr
                row += f"{corr:12.4f}"
            else:
                row += f"{correlations[(model2, model1)]:12.4f}"
        print(row)

    # Pearson correlation
    print(f"\n{'Pearson Correlation Matrix':^60}")
    print("-" * 60)
    print(header)

    pearson_corrs = {}
    for i, model1 in enumerate(model_names):
        row = f"{model_names[i].split('.')[-1][:18]:20}"
        for j, model2 in enumerate(model_names):
            if i <= j:
                corr, pval = stats.pearsonr(scores_by_model[model1], scores_by_model[model2])
                pearson_corrs[(model1, model2)] = corr
                row += f"{corr:12.4f}"
            else:
                row += f"{pearson_corrs[(model2, model1)]:12.4f}"
        print(row)

    # Summary statistics
    print(f"\n{'Summary Statistics':^60}")
    print("-" * 60)
    print(f"{'Model':30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for model in model_names:
        scores = scores_by_model[model]
        print(f"{model.split('.')[-1][:28]:30} {np.mean(scores):10.4f} {np.std(scores):10.4f} {np.min(scores):10.4f} {np.max(scores):10.4f}")

    return correlations, pearson_corrs


def main():
    parser = argparse.ArgumentParser(description='Embedding Model Ablation for STED')
    parser.add_argument('--results-dir', default='llm_gen_results/toucan',
                        help='Directory containing LLM generation results')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to use for ablation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--variation-type', default='combined',
                        choices=['structural', 'content', 'combined'],
                        help='Variation type for STED calculation')
    parser.add_argument('--output', default=None, help='Output JSON file for results')

    # Embedding models to compare
    parser.add_argument('--embedding-models', nargs='+', default=[
        # Local sentence-transformers models
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'BAAI/bge-base-en-v1.5',
        'BAAI/bge-large-en-v1.5',
        'intfloat/e5-base-v2',
        'intfloat/e5-large-v2',
        # AWS Bedrock models
        'amazon.titan-embed-text-v2:0:256',
        'amazon.titan-embed-text-v2:0:512',
        'amazon.titan-embed-text-v2:0:1024',
        'cohere.embed-multilingual-v3',
        'cohere.embed-english-v3',
    ], help='Embedding models to compare (format: model_id or model_id:dimension)')

    args = parser.parse_args()

    # Load samples
    print(f"\n{'='*60}")
    print("Loading samples...")
    print(f"{'='*60}")
    samples = load_samples(args.results_dir, args.num_samples, args.seed)

    # Calculate STED with each embedding model
    results = {}

    for emb_spec in args.embedding_models:
        # Parse model spec (model_id or model_id:dimension)
        if ':' in emb_spec and emb_spec.count(':') >= 2:
            # Format: amazon.titan-embed-text-v2:0:512
            parts = emb_spec.rsplit(':', 1)
            model_id = parts[0]
            dimension = int(parts[1])
        elif emb_spec.startswith('amazon.titan'):
            model_id = emb_spec
            dimension = 512  # default for Titan
        elif emb_spec.startswith('cohere.embed-multilingual'):
            model_id = emb_spec
            dimension = 1024  # Cohere multilingual v3 is 1024-dim
        elif emb_spec.startswith('cohere.embed-english'):
            model_id = emb_spec
            dimension = 1024  # Cohere english v3 is 1024-dim
        elif emb_spec.startswith('cohere.embed-v4'):
            model_id = emb_spec
            dimension = 1024  # Cohere v4 is 1024-dim
        elif emb_spec in ['all-MiniLM-L6-v2']:
            model_id = emb_spec
            dimension = 384  # MiniLM is 384-dim
        elif emb_spec in ['all-mpnet-base-v2']:
            model_id = emb_spec
            dimension = 768  # mpnet is 768-dim
        elif emb_spec.startswith('BAAI/bge-base'):
            model_id = emb_spec
            dimension = 768  # bge-base is 768-dim
        elif emb_spec.startswith('BAAI/bge-large'):
            model_id = emb_spec
            dimension = 1024  # bge-large is 1024-dim
        elif emb_spec.startswith('intfloat/e5-base'):
            model_id = emb_spec
            dimension = 768  # e5-base is 768-dim
        elif emb_spec.startswith('intfloat/e5-large'):
            model_id = emb_spec
            dimension = 1024  # e5-large is 1024-dim
        else:
            model_id = emb_spec
            dimension = 384  # default for sentence-transformers

        print(f"\n{'='*60}")
        print(f"Processing: {model_id} (dim={dimension})")
        print(f"{'='*60}")

        scores = calculate_sted_scores(
            samples, model_id, dimension,
            args.region, args.variation_type
        )
        results[emb_spec] = scores

    # Compare results
    spearman_corrs, pearson_corrs = compare_embedding_models(results)

    # Save results if output specified
    if args.output:
        output_data = {
            'num_samples': len(samples),
            'embedding_models': args.embedding_models,
            'variation_type': args.variation_type,
            'results': results,
            'spearman_correlations': {f"{k[0]}__vs__{k[1]}": v for k, v in spearman_corrs.items()},
            'pearson_correlations': {f"{k[0]}__vs__{k[1]}": v for k, v in pearson_corrs.items()}
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print(f"\n{'='*60}")
    print("Ablation study complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
