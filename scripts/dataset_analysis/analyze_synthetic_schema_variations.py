#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from tqdm import tqdm

def analyze_schema_variations(dataset_file, output_dir='results'):
    """Analyze similarity scores for different schema variation types"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='amazon.titan-embed-text-v2:0')
    methods = ['ted', 'sted', 'bertscore', 'deepdiff', 'gnn']
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Separate samples by variation type
    variation_types = {}
    for sample in data:
        var_type = sample.get('variation_type', 'unknown')
        if var_type not in variation_types:
            variation_types[var_type] = []
        variation_types[var_type].append(sample)
    
    results = {}
    
    # Analyze field_name_change with variation ratios
    if 'field_name_change' in variation_types:
        print("Analyzing field_name_change variations...")
        field_results = {method: {round(i*0.1, 1): [] for i in range(1, 11)} for method in methods}
        
        for sample in tqdm(variation_types['field_name_change']):
            base_sample = sample.get('base_sample', {})
            variants = sample.get('variants', [])
            
            for variant in variants:
                variation_ratio = round(variant.get('variation_ratio', 0), 1)
                variation = variant.get('variation', {})
                
                for method in methods:
                    try:
                        similarity = evaluator.calculate_similarity_method[method](base_sample, variation)
                        field_results[method][variation_ratio].append(similarity)
                    except Exception as e:
                        print(f"Error with {method}: {e}")
        
        # Calculate averages
        field_avg = {method: [] for method in methods}
        ratios = [round(i*0.1, 1) for i in range(1, 11)]
        
        for method in methods:
            for ratio in ratios:
                scores = field_results[method][ratio]
                field_avg[method].append(np.mean(scores) if scores else 0)
        
        results['field_name_change'] = {'ratios': ratios, 'averages': field_avg}
    
    # Analyze flat_structure and nested_change (single comparison)
    for var_type in ['flat_structure', 'nested_change']:
        if var_type in variation_types:
            print(f"Analyzing {var_type} variations...")
            type_results = {method: [] for method in methods}
            
            for sample in tqdm(variation_types[var_type]):
                base_sample = sample.get('ground_truth', {})
                variation = sample.get('variation', {})
                
                for method in methods:
                    try:
                        similarity = evaluator.calculate_similarity_method[method](base_sample, variation)
                        type_results[method].append(similarity)
                    except Exception as e:
                        print(f"Error with {method}: {e}")
            
            # Calculate averages
            type_avg = {method: np.mean(scores) if scores else 0 for method, scores in type_results.items()}
            results[var_type] = {'average': type_avg}
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot field_name_change progression
    if 'field_name_change' in results:
        ax = axes[0]
        for i, method in enumerate(methods):
            ax.plot(results['field_name_change']['ratios'], 
                   results['field_name_change']['averages'][method],
                   marker='o', color=colors[i], label=method.upper(), linewidth=2)
        ax.set_xlabel('Variation Ratio')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Field Name Change Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Plot flat_structure comparison
    if 'flat_structure' in results:
        ax = axes[1]
        method_names = list(results['flat_structure']['average'].keys())
        scores = list(results['flat_structure']['average'].values())
        bars = ax.bar(method_names, scores, color=colors[:len(method_names)])
        ax.set_ylabel('Similarity Score')
        ax.set_title('Flat Structure Similarity')
        ax.set_ylim(0, 1)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom')
    
    # Plot nested_change comparison
    if 'nested_change' in results:
        ax = axes[2]
        method_names = list(results['nested_change']['average'].keys())
        scores = list(results['nested_change']['average'].values())
        bars = ax.bar(method_names, scores, color=colors[:len(method_names)])
        ax.set_ylabel('Similarity Score')
        ax.set_title('Nested Change Similarity')
        ax.set_ylim(0, 1)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/schema_variation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results
    print("\n" + "="*80)
    print("SCHEMA VARIATION ANALYSIS RESULTS")
    print("="*80)
    
    if 'field_name_change' in results:
        print("\nField Name Change Progression:")
        print(f"{'Ratio':<6} {'TED':<8} {'STED':<8} {'BERT':<8} {'DEEP':<8} {'GRAPH':<8}")
        print("-" * 50)
        for i, ratio in enumerate(results['field_name_change']['ratios']):
            row = f"{ratio:<6}"
            for method in methods:
                row += f" {results['field_name_change']['averages'][method][i]:<7.3f}"
            print(row)
    
    for var_type in ['flat_structure', 'nested_change']:
        if var_type in results:
            print(f"\n{var_type.replace('_', ' ').title()} Average Similarities:")
            for method in methods:
                score = results[var_type]['average'][method]
                print(f"  {method.upper()}: {score:.3f}")
    
    # Save results
    with open(f'{output_dir}/schema_variation_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFiles saved to {output_dir}/:")
    print(f"- schema_variation_analysis.png")
    print(f"- schema_variation_analysis_results.json")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze schema variation patterns')
    parser.add_argument('dataset_file', help='Schema variation dataset file to analyze')
    parser.add_argument('--output-dir', default='results', help='Directory to save output files')
    
    args = parser.parse_args()
    analyze_schema_variations(args.dataset_file, args.output_dir)
