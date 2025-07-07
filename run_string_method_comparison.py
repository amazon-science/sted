#!/usr/bin/env python
"""
String Method Comparison Experiment

This script compares different string comparison methods in the semantic tree evaluation.

Usage:
    python run_string_method_comparison.py --data-dir extracted_sharegpt_data --output-dir ./string_method_experiment
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import subprocess
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

# Define available string comparison methods
STRING_METHODS = ["levenshtein", "semantic", "exact", "jaccard"]

def run_generation(data_dir: str, output_dir: str, run_num: int, include_schema: bool) -> str:
    """
    Run LLM generation with specified parameters.
    
    Args:
        data_dir: Directory containing the data files
        output_dir: Directory to save generation results
        run_num: Number of runs to perform
        include_schema: Whether to include schema in the prompt
        
    Returns:
        Path to the generated results file
    """
    cmd = [
        "python", "llm_gen_simple.py",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--run-num", str(run_num)
    ]
    
    if include_schema:
        cmd.append("--include-schema")
    
    print(f"Running generation...")
    subprocess.run(cmd, check=True)
    
    # Find the most recent results directory
    result_dirs = list(Path(output_dir).glob("llm_gen_results_*"))
    
    if not result_dirs:
        raise FileNotFoundError("No results found")
    
    # Sort by creation time (most recent first)
    result_dir = sorted(result_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    results_file = result_dir / "all_results.json"
    
    return str(results_file)

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

def evaluate_with_string_method(data: Dict[str, Any], string_method: str) -> List[Dict[str, Any]]:
    """
    Evaluate generated outputs with a specific string comparison method.
    
    Args:
        data: Dictionary with generated outputs
        string_method: String comparison method to use
        
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
        
        print(f"Evaluating sample {sample_id} with string method {string_method}...")
        
        # Create a list with ground truth and all generated outputs
        all_outputs = [ground_truth] + [output for output in responses if output]
        
        # Initialize the semantic evaluator with the specified string method
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            array_order_matters=False,
            use_semantic_similarity=True,
            semantic_threshold=0.7,
            string_method=string_method
        )
        
        # Evaluate consistency
        consistency_result = evaluator.evaluate_structural_consistency(all_outputs)
        
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
            'string_method': string_method,
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
    Create visualizations of the string method comparison.
    
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
                'string_method': sample['string_method'],
                'semantic_similarity_mean': sample['semantic_similarity']['mean'],
                'semantic_similarity_std': sample['semantic_similarity']['std'],
                'stability': sample['semantic_similarity']['stability'],
                'cross_run_consistency': sample['cross_run_consistency'],
                'sample_id': sample['sample_id']
            })
    
    df = pd.DataFrame(plot_data)
    
    # Calculate averages across samples for each string method
    avg_df = df.groupby('string_method').mean().reset_index()
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # 1. Bar plot of semantic similarity by string method
    plt.figure(figsize=(10, 6))
    sns.barplot(x='string_method', y='semantic_similarity_mean', data=avg_df)
    plt.title('Average Semantic Similarity by String Method')
    plt.xlabel('String Method')
    plt.ylabel('Semantic Similarity (Accuracy)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'string_method_vs_similarity.png'), dpi=300)
    
    # 2. Bar plot of stability by string method
    plt.figure(figsize=(10, 6))
    sns.barplot(x='string_method', y='stability', data=avg_df)
    plt.title('Average Stability by String Method')
    plt.xlabel('String Method')
    plt.ylabel('Stability')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'string_method_vs_stability.png'), dpi=300)
    
    # 3. Bar plot of cross-run consistency by string method
    plt.figure(figsize=(10, 6))
    sns.barplot(x='string_method', y='cross_run_consistency', data=avg_df)
    plt.title('Cross-Run Consistency by String Method')
    plt.xlabel('String Method')
    plt.ylabel('Cross-Run Consistency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'string_method_vs_consistency.png'), dpi=300)
    
    # 4. Combined plot with multiple metrics
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(avg_df))
    width = 0.25
    
    plt.bar(x - width, avg_df['semantic_similarity_mean'], width, label='Semantic Similarity')
    plt.bar(x, avg_df['stability'], width, label='Stability')
    plt.bar(x + width, avg_df['cross_run_consistency'], width, label='Cross-Run Consistency')
    
    plt.title('Impact of String Method on Different Metrics')
    plt.xlabel('String Method')
    plt.ylabel('Metric Value')
    plt.xticks(x, avg_df['string_method'])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'string_method_combined_metrics.png'), dpi=300)
    
    # 5. Heatmap of metrics across string methods
    plt.figure(figsize=(10, 6))
    heatmap_data = avg_df.set_index('string_method')[['semantic_similarity_mean', 'stability', 'cross_run_consistency']]
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Metrics Across Different String Methods')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'string_method_metrics_heatmap.png'), dpi=300)
    
    # 6. Individual sample plots
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='string_method', y='semantic_similarity_mean', data=df)
    plt.title('Semantic Similarity by String Method (All Samples)')
    plt.xlabel('String Method')
    plt.ylabel('Semantic Similarity (Accuracy)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'string_method_vs_similarity_boxplot.png'), dpi=300)
    
    print(f"Visualizations saved to {output_dir}")

def run_experiment(args):
    """
    Run the string method comparison experiment.
    
    Args:
        args: Command-line arguments
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    generations_dir = os.path.join(args.output_dir, "generations")
    visualizations_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(generations_dir, exist_ok=True)
    
    # Run generation if input file is not provided
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = run_generation(
            data_dir=args.data_dir,
            output_dir=generations_dir,
            run_num=args.run_num,
            include_schema=args.include_schema
        )
    
    # Load generated outputs
    data = load_generations(input_file)
    
    # Define string methods to test
    if args.string_methods:
        string_methods = args.string_methods
    else:
        string_methods = STRING_METHODS
    
    all_results = []
    
    # Evaluate with each string method
    for method in string_methods:
        results = evaluate_with_string_method(data, method)
        all_results.append({
            'string_method': method,
            'sample_results': results
        })
    
    # Create visualizations
    create_visualizations(all_results, visualizations_dir)
    
    # Save results
    results_file = os.path.join(args.output_dir, "string_method_comparison_data.json")
    with open(results_file, 'w') as f:
        json.dump({
            'results': all_results,
            'parameters': {
                'string_methods': string_methods,
                'input_file': input_file
            }
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Print summary of findings
    print("\n=== String Method Comparison Analysis ===")
    
    # Calculate average metrics across all samples for each string method
    summary = []
    for result in all_results:
        method = result['string_method']
        
        # Calculate averages
        avg_similarity = np.mean([sample['semantic_similarity']['mean'] for sample in result['sample_results']])
        avg_stability = np.mean([sample['semantic_similarity']['stability'] for sample in result['sample_results']])
        avg_consistency = np.mean([sample['cross_run_consistency'] for sample in result['sample_results']])
        
        summary.append({
            'string_method': method,
            'avg_similarity': avg_similarity,
            'avg_stability': avg_stability,
            'avg_consistency': avg_consistency,
            'combined_score': avg_similarity * avg_stability  # Simple combined metric
        })
    
    # Sort by combined score
    summary.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print("\nString methods ranked by combined score (similarity * stability):")
    for i, item in enumerate(summary):
        print(f"{i+1}. {item['string_method']}: {item['combined_score']:.4f} (similarity: {item['avg_similarity']:.4f}, stability: {item['avg_stability']:.4f})")
    
    # Recommend optimal string method
    optimal_method = summary[0]['string_method']
    print(f"\nRecommended optimal string method: {optimal_method}")

def main():
    parser = argparse.ArgumentParser(description="Run string method comparison experiment.")
    parser.add_argument("--data-dir", type=str, help="Directory containing the data files.")
    parser.add_argument("--input-file", type=str, help="Path to the JSON file containing the generated outputs. If not provided, generation will be run.")
    parser.add_argument("--output-dir", type=str, default="./string_method_experiment", help="Directory to save experiment results.")
    parser.add_argument("--run-num", type=int, default=5, help="Number of runs to perform.")
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt.")
    parser.add_argument("--string-methods", type=str, nargs="+", choices=STRING_METHODS, help=f"List of string methods to test. Default: {STRING_METHODS}")
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_file and not args.data_dir:
        parser.error("Either --input-file or --data-dir must be provided")
    
    run_experiment(args)

if __name__ == "__main__":
    main()