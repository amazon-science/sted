#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from src.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from tqdm import tqdm

def get_variation_type_from_filename(filename):
    """Extract variation type from filename"""
    basename = os.path.basename(filename).lower()
    if 'expression' in basename:
        return 'expression'
    elif 'structure' in basename:
        return 'structure'
    elif 'semantic' in basename:
        return 'semantic'
    elif 'field' in basename:
        return 'field'
    else:
        return 'unknown'

def analyze_variation_progression(file_paths):
    """Analyze how similarity changes as variation number increases"""
    
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='amazon.titan-embed-text-v2:0')
    methods = ['ted', 'sted', 'bertscore', 'deepdiff', 'gnn']
    
    # Results structure: {file: {method: {variation_num: [similarities]}}}
    all_results = {}
    
    for file_path in file_paths:
        variation_type = get_variation_type_from_filename(file_path)
        print(f"Processing {variation_type} variation file: {file_path}")
        
        results = {method: {round(i/10, 1): [] for i in range(1, 11)} for method in methods}
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process each sample
        for sample in tqdm(data, desc=f"Processing {variation_type} samples"):
            base_sample = sample.get('base_sample', {})
            variants = sample.get('variants', [])
            
            if not base_sample or not variants:
                continue
                
            # Process each variation number
            for variant in variants:
                variation_ratio = round(variant.get('variation_ratio'), 1)
                variation = variant.get('variation', {})
                
                if not variation or not variation_ratio:
                    continue
                    
                # Calculate similarity for each method
                for method in methods:
                    try:
                        similarity = evaluator.calculate_similarity_method[method](base_sample, variation)
                        results[method][variation_ratio].append(similarity)
                    except Exception as e:
                        print(f"Error with {method} on variation {variation_ratio}: {e}")
        
        all_results[file_path] = {'type': variation_type, 'results': results}
    
    # Process results for each file
    for file_path, file_data in all_results.items():
        variation_type = file_data['type']
        results = file_data['results']
        
        # Calculate averages for each variation number
        avg_results = {method: [] for method in methods}
        variation_ratios = [round(i/10, 1) for i in range(1, 11)]
        
        for method in methods:
            for var_ratio in variation_ratios:
                scores = results[method][var_ratio]
                avg_score = np.mean(scores) if scores else 0
                avg_results[method].append(avg_score)
        
        # Save results
        output_filename = f'{variation_type}_variation_progression_results.json'
        output_data = {
            'variation_type': variation_type,
            'file_path': file_path,
            'variation_ratios': variation_ratios,
            'average_similarities': avg_results,
            'raw_results': results
        }
        
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, method in enumerate(methods):
            plt.plot(variation_ratios, avg_results[method], 
                    marker=markers[i], color=colors[i], label=method.upper(), 
                    linewidth=2, markersize=8)
        
        plt.xlabel('Variation Number', fontsize=12)
        plt.ylabel('Average Similarity Score', fontsize=12)
        plt.title(f'{variation_type.title()} Variation: Similarity Change as Variation Number Increases', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.xticks(variation_ratios)
        
        plt.tight_layout()
        plt.savefig(f'{variation_type}_variation_progression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print analysis for this variation type
        print(f"\n{'='*80}")
        print(f"{variation_type.title()} Variation Progression Analysis")
        print("=" * 80)
        
        print(f"\nAverage similarity by variation number:")
        print(f"{'Var#':<4} {'TED':<8} {'STED':<8} {'BERT':<8} {'DEEP':<8} {'GNN':<8}")
        print("-" * 50)
        
        for i, var_num in enumerate(variation_ratios):
            row = f"{var_num:<4}"
            for method in methods:
                row += f" {avg_results[method][i]:<7.3f}"
            print(row)
        
        # Calculate trends
        print(f"\nTrend Analysis for {variation_type}:")
        print("-" * 30)
        
        for method in methods:
            scores = avg_results[method]
            if len(scores) > 1:
                x = np.array(variation_ratios)
                y = np.array(scores)
                slope = np.polyfit(x, y, 1)[0]
                total_drop = scores[0] - scores[-1]
                
                print(f"{method.upper()}:")
                print(f"  Slope: {slope:.4f} per variation")
                print(f"  Total drop (1→10): {total_drop:.3f}")
                print(f"  Relative drop: {(total_drop/scores[0]*100):.1f}%" if scores[0] > 0 else "  Relative drop: N/A")
                print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze variation progression from dataset files')
    parser.add_argument('files', nargs='+', help='Dataset files to analyze')
    
    args = parser.parse_args()
    analyze_variation_progression(args.files)
