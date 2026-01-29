#!/usr/bin/env python3
import json
import numpy as np
import argparse
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer
from tqdm import tqdm


def analyze_variation_consistency_progression(file_path, method='sted', variation_type='combined'):
    """Analyze variation consistency across different variation ratios"""

    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='amazon.titan-embed-text-v2:0')
    analyzer = StructuralConsistencyAnalyzer(evaluator)

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Results by variation ratio
    results_by_ratio = {}

    for sample in tqdm(data, desc="Processing samples"):
        base_sample = sample.get('base_sample', {})
        variants = sample.get('variants', [])

        if not variants:
            continue

        # Group by variation ratio
        for variant in variants:
            variation_ratio = round(variant.get('variation_ratio'), 1)
            variation = variant.get('variation', {})

            if variation_ratio not in results_by_ratio:
                results_by_ratio[variation_ratio] = []

            # Collect all variations including base sample for this ratio
            all_variations = [base_sample, variation]
            results_by_ratio[variation_ratio].append(all_variations)

    # Calculate consistency metrics for each ratio
    consistency_results = {}

    for ratio in sorted(results_by_ratio.keys()):
        ratio_metrics = []

        for variations in results_by_ratio[ratio]:
            # Use analyzer.evaluate_structural_consistency instead of local function
            report = analyzer.evaluate_structural_consistency(
                json_outputs=variations,
                method_name=method,
                variation_type=variation_type
            )

            # Extract metrics from report
            consistency_metrics = report.get('consistency_metrics', {})
            supporting_stats = report.get('supporting_stats', {})

            metrics = {
                'empty_ratio': consistency_metrics.get('empty_ratio', 0.0),
                'consistency_score': consistency_metrics.get('stability_score', float('inf')),
                'penalized_consistency': consistency_metrics.get('penalized_stability_score', float('inf')),
                'mean_distance': 1.0 - supporting_stats.get('mean_similarity', 0.0),
                'valid_count': report.get('num_outputs_analyzed', 0)
            }
            ratio_metrics.append(metrics)

        # Aggregate metrics
        consistency_results[ratio] = {
            'avg_empty_ratio': np.mean([m['empty_ratio'] for m in ratio_metrics]),
            'avg_consistency_score': np.mean([m['consistency_score'] for m in ratio_metrics if m['consistency_score'] != float('inf')]),
            'avg_penalized_consistency': np.mean([m['penalized_consistency'] for m in ratio_metrics if m['penalized_consistency'] != float('inf')]),
            'avg_mean_distance': np.mean([m['mean_distance'] for m in ratio_metrics if 'mean_distance' in m]),
            'samples_count': len(ratio_metrics)
        }

    return consistency_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze variation consistency using pairwise distances')
    parser.add_argument('file', help='Dataset file to analyze')
    parser.add_argument('--method', default='sted', choices=['ted', 'sted', 'bertscore', 'deepdiff', 'gnn'],
                        help='Similarity calculation method')
    parser.add_argument('--variation-type', default='combined', choices=['structural', 'content', 'combined'],
                        help='Type of variation to analyze')
    parser.add_argument('--output-dir', default='results', help='Directory to save output files')

    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f'variation_consistency_{args.method}_{args.variation_type}.json'
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"Analyzing variation consistency for: {args.file}")
    print(f"Method: {args.method}, Variation Type: {args.variation_type}")

    results = analyze_variation_consistency_progression(args.file, args.method, args.variation_type)

    # Save results
    output_data = {
        'file_path': args.file,
        'method': args.method,
        'variation_type': args.variation_type,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("Variation Consistency Analysis Summary")
    print("="*80)
    print(f"\n{'Ratio':<8} {'Empty%':<10} {'Consistency':<15} {'Penalized':<15} {'Mean Dist':<12}")
    print("-"*70)

    for ratio in sorted(results.keys()):
        r = results[ratio]
        print(f"{ratio:<8.1f} {r['avg_empty_ratio']*100:<10.2f} {r['avg_consistency_score']:<15.4f} "
              f"{r['avg_penalized_consistency']:<15.4f} {r['avg_mean_distance']:<12.4f}")
