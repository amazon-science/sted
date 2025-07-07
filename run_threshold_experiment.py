#!/usr/bin/env python
"""
Semantic Threshold Sensitivity Experiment

This script analyzes the impact of different semantic thresholds on evaluation results.

Usage:
    python run_threshold_experiment.py --input-file ./generations/*/all_results.json --output-dir ./threshold_experiment
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

# Import semantic comparison functionality
try:
    from semantic_json_tree_consistency import (
        SemanticJsonTreeConsistencyEvaluator,
        evaluate_semantic_json_consistency
    )
    SEMANTIC_COMPARISON_AVAILABLE = True
except ImportError:
    print("Warning: semantic_json_tree_consistency module not available.")
    SEMANTIC_COMPARISON_AVAILABLE = False

def load_generations(input_file: str) -> Dict[str, Any]:
    """
    Load generated outputs from a file.
    
    Args:
        input_file: Path to the file containing generated outputs
        
    Returns:
        Dictionary with loaded data
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    return data

def evaluate_with_threshold(data: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
    """
    Evaluate generated outputs with a specific semantic threshold.
    
    Args:
        data: Dictionary with generated outputs
        threshold: Semantic threshold to use
        
    Returns:
        List of dictionaries with evaluation results for each sample
    """
    if not SEMANTIC_COMPARISON_AVAILABLE:
        raise ImportError("semantic_json_tree_consistency module is required for this experiment")
    
    results = []
    
    for sample in data.get('results', []):
        sample_id = sample.get('sample_id', 'unknown')
        ground_truth = sample.get('ground_truth', {})
        responses = sample.get('responses', [])
        
        print(f"Evaluating sample {sample_id} with threshold {threshold}...")
        
        # Create a list with ground truth and all generated outputs
        all_outputs = [ground_truth] + [output for output in responses if output]
        
        # Use the evaluate_semantic_json_consistency function with the specified threshold
        consistency_result = evaluate_semantic_json_consistency(
            outputs=all_outputs,
            array_order_matters=False,
            use_semantic_similarity=True,
            semantic_threshold=threshold
        )
        
        # Initialize the semantic evaluator with the specified threshold
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            array_order_matters=False,
            use_semantic_similarity=True,
            semantic_threshold=threshold,
            string_method='semantic'
        )
        
        # Calculate similarity scores between each generated output and ground truth
        similarity_scores = []
        for output in responses:
            if not output:  # Skip empty outputs
                continue
                
            # Calculate tree edit distance similarity
            similarity = evaluator.calculate_tree_edit_distance(output, ground_truth)[0]
            similarity_scores.append(similarity)
        
        # Calculate metrics
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            std_similarity = np.std(similarity_scores) if len(similarity_scores) > 1 else 0
            min_similarity = min(similarity_scores)
            max_similarity = max(similarity_scores)
        else:
            avg_similarity = std_similarity = min_similarity = max_similarity = 0
        
        # Calculate stability as inverse of standard deviation
        stability = 1 - (std_similarity if std_similarity < 1 else 1)
        
        # Extract consistency metrics
        consistency_metrics = consistency_result.get('consistency_metrics', {})
        
        results.append({
            'sample_id': sample_id,
            'threshold': threshold,
            'semantic_similarity': {
                'mean': float(avg_similarity),
                'std': float(std_similarity),
                'min': float(min_similarity),
                'max': float(max_similarity),
                'stability': float(stability)
            },
            'cross_run_consistency': consistency_metrics.get('mean_similarity', 0),
            'perfect_consistency': consistency_metrics.get('perfect_consistency', False),
            'num_responses': len(responses)
        })
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """
    Create visualizations of the threshold sensitivity analysis.
    
    Args:
        results: List of dictionaries with evaluation results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    for result in results:
        for sample in result['sample_results']:
            plot_data.append({
                'threshold': sample['threshold'],
                'semantic_similarity_mean': sample['semantic_similarity']['mean'],
                'semantic_similarity_std': sample['semantic_similarity']['std'],
                'stability': sample['semantic_similarity']['stability'],
                'cross_run_consistency': sample['cross_run_consistency'],
                'sample_id': sample['sample_id']
            })
    
    df = pd.DataFrame(plot_data)
    
    # Calculate averages across samples for each threshold
    avg_df = df.groupby('threshold').mean().reset_index()
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # 1. Line plot of semantic similarity vs threshold
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='threshold', y='semantic_similarity_mean', data=avg_df, marker='o')
    plt.title('Average Semantic Similarity vs Threshold')
    plt.xlabel('Semantic Threshold')
    plt.ylabel('Semantic Similarity (Accuracy)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_vs_similarity.png'), dpi=300)
    
    # 2. Line plot of stability vs threshold
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='threshold', y='stability', data=avg_df, marker='o')
    plt.title('Average Stability vs Threshold')
    plt.xlabel('Semantic Threshold')
    plt.ylabel('Stability')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_vs_stability.png'), dpi=300)
    
    # 3. Line plot of cross-run consistency vs threshold
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='threshold', y='cross_run_consistency', data=avg_df, marker='o')
    plt.title('Cross-Run Consistency vs Threshold')
    plt.xlabel('Semantic Threshold')
    plt.ylabel('Cross-Run Consistency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_vs_consistency.png'), dpi=300)
    
    # 4. Combined plot with multiple metrics
    plt.figure(figsize=(12, 8))
    
    plt.plot(avg_df['threshold'], avg_df['semantic_similarity_mean'], 'o-', label='Semantic Similarity')
    plt.plot(avg_df['threshold'], avg_df['stability'], 's-', label='Stability')
    plt.plot(avg_df['threshold'], avg_df['cross_run_consistency'], '^-', label='Cross-Run Consistency')
    
    plt.title('Impact of Semantic Threshold on Different Metrics')
    plt.xlabel('Semantic Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_combined_metrics.png'), dpi=300)
    
    # 5. Heatmap of metrics across thresholds
    plt.figure(figsize=(10, 6))
    heatmap_data = avg_df.set_index('threshold')[['semantic_similarity_mean', 'stability', 'cross_run_consistency']]
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Metrics Across Different Thresholds')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_metrics_heatmap.png'), dpi=300)
    
    # 6. Individual sample plots
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='threshold', y='semantic_similarity_mean', hue='sample_id', data=df, marker='o')
    plt.title('Semantic Similarity vs Threshold (By Sample)')
    plt.xlabel('Semantic Threshold')
    plt.ylabel('Semantic Similarity (Accuracy)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_vs_similarity_by_sample.png'), dpi=300)
    
    print(f"Visualizations saved to {output_dir}")

def run_experiment(args):
    """
    Run the semantic threshold sensitivity experiment.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load generated outputs
    data = load_generations(args.input_file)
    
    # Define thresholds to test
    if args.thresholds:
        thresholds = args.thresholds
    else:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    all_results = []
    
    # Evaluate with each threshold
    for threshold in thresholds:
        results = evaluate_with_threshold(data, threshold)
        all_results.append({
            'threshold': threshold,
            'sample_results': results
        })
    
    # Create visualizations
    create_visualizations(all_results, os.path.join(args.output_dir, "visualizations"))
    
    # Save results
    results_file = os.path.join(args.output_dir, "threshold_sensitivity_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'results': all_results,
            'parameters': {
                'thresholds': thresholds,
                'input_file': args.input_file
            }
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Print summary of findings
    print("\n=== Semantic Threshold Sensitivity Analysis ===")
    
    # Calculate average metrics across all samples for each threshold
    summary = []
    for result in all_results:
        threshold = result['threshold']
        
        # Calculate averages
        avg_similarity = np.mean([sample['semantic_similarity']['mean'] for sample in result['sample_results']])
        avg_stability = np.mean([sample['semantic_similarity']['stability'] for sample in result['sample_results']])
        avg_consistency = np.mean([sample['cross_run_consistency'] for sample in result['sample_results']])
        
        summary.append({
            'threshold': threshold,
            'avg_similarity': avg_similarity,
            'avg_stability': avg_stability,
            'avg_consistency': avg_consistency,
            'combined_score': avg_similarity * avg_stability  # Simple combined metric
        })
    
    # Sort by combined score
    summary.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print("\nThresholds ranked by combined score (similarity * stability):")
    for i, item in enumerate(summary):
        print(f"{i+1}. Threshold {item['threshold']}: {item['combined_score']:.4f} (similarity: {item['avg_similarity']:.4f}, stability: {item['avg_stability']:.4f})")
    
    # Recommend optimal threshold
    optimal_threshold = summary[0]['threshold']
    print(f"\nRecommended optimal threshold: {optimal_threshold}")

def main():
    parser = argparse.ArgumentParser(description="Run semantic threshold sensitivity experiment.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the JSON file containing the generated outputs.")
    parser.add_argument("--output-dir", type=str, default="./threshold_experiment", help="Directory to save experiment results.")
    parser.add_argument("--thresholds", type=float, nargs="+", help="List of thresholds to test. Default: 0.5, 0.6, 0.7, 0.8, 0.9")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()