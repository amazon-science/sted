#!/usr/bin/env python
"""
Hungarian Algorithm Effectiveness Experiment

This script evaluates the effectiveness of the Hungarian algorithm for comparing arrays
and long free text in the semantic tree evaluation approach.

Usage:
    python run_hungarian_experiment.py --data-dir extracted_sharegpt_data --output-dir ./hungarian_experiment
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import subprocess
import pandas as pd
from pathlib import Path
import copy
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from semantic_json_tree_consistency import (
    SemanticJsonTreeConsistencyEvaluator,
    evaluate_semantic_json_consistency
)

def run_generation(data_dir: str, output_dir: str, run_num: int, include_schema: bool, sample_limit: int = 10) -> str:
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
        "--run-num", str(run_num),
        "--sample-limit", str(sample_limit),
        "--include-schema"
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

def analyze_complex_fields(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyze the generated data to identify samples with array fields and long text fields.
    
    Args:
        data: Dictionary with generated outputs
        
    Returns:
        List of samples with complex fields and their statistics
    """
    field_stats = []
    
    # Define what constitutes a "long" text
    LONG_TEXT_THRESHOLD = 100  # characters
    
    for sample in data.get('results', []):
        sample_id = sample.get('sample_id', 'unknown')
        ground_truth = sample.get('ground_truth', {})
        
        # Find array fields and long text fields in ground truth
        array_fields = []
        long_text_fields = []
        
        def find_complex_fields(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    # Check for arrays
                    if isinstance(value, list) and len(value) > 1:
                        array_fields.append({
                            'path': new_path,
                            'length': len(value),
                            'contains_objects': any(isinstance(item, dict) for item in value)
                        })
                    
                    # Check for long text
                    elif isinstance(value, str) and len(value) > LONG_TEXT_THRESHOLD:
                        # Count paragraphs (separated by double newlines)
                        paragraphs = value.split('\n\n')
                        paragraphs = [p for p in paragraphs if p.strip()]
                        
                        # Count sentences (roughly)
                        sentences = value.replace('!', '.').replace('?', '.').split('.')
                        sentences = [s for s in sentences if s.strip()]
                        
                        long_text_fields.append({
                            'path': new_path,
                            'length': len(value),
                            'paragraph_count': len(paragraphs),
                            'sentence_count': len(sentences),
                            'has_structure': len(paragraphs) > 1 or len(sentences) > 3
                        })
                    
                    find_complex_fields(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    find_complex_fields(item, new_path)
        
        find_complex_fields(ground_truth)
        
        if array_fields or long_text_fields:
            field_stats.append({
                'sample_id': sample_id,
                'array_fields': array_fields,
                'long_text_fields': long_text_fields,
                'has_arrays': len(array_fields) > 0,
                'has_long_texts': len(long_text_fields) > 0,
                'array_count': len(array_fields),
                'long_text_count': len(long_text_fields),
                'max_array_length': max(field['length'] for field in array_fields) if array_fields else 0,
                'max_text_length': max(field['length'] for field in long_text_fields) if long_text_fields else 0,
                'has_object_arrays': any(field['contains_objects'] for field in array_fields),
                'has_structured_text': any(field['has_structure'] for field in long_text_fields)
            })
    
    return field_stats

def evaluate_real_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Evaluate real generated data with and without Hungarian algorithm,
    comparing both accuracy and stability metrics.
    
    Args:
        data: Dictionary with generated outputs
        
    Returns:
        List of evaluation results
    """
    
    results = []
    
    for sample in tqdm(data.get('results', []), desc="iterate evaluation for each sample"):
        sample_id = sample.get('sample_id', 'unknown')
        ground_truth = sample.get('ground_truth', {})
        responses = sample.get('responses', [])
        
        print(f"Evaluating sample {sample_id}...")
        
        # Skip if no responses
        if not responses:
            continue
        
        # Filter out empty responses
        valid_responses = [r for r in responses if r]
        if not valid_responses:
            continue
            
        # Create a list with ground truth and all generated outputs
        all_outputs_with_gt = [ground_truth] + valid_responses
        
        # Evaluate with Hungarian algorithm
        with_hungarian_result = evaluate_semantic_json_consistency(
            outputs=all_outputs_with_gt,
            array_order_matters=False,
            use_semantic_similarity=True,
            semantic_threshold=0.7,
            use_hungarian=True,  # Explicitly enable Hungarian algorithm
            long_string_method='hungarian'
        )
        
        # Evaluate without Hungarian algorithm
        without_hungarian_result = evaluate_semantic_json_consistency(
            outputs=all_outputs_with_gt,
            array_order_matters=False,  # Keep this the same for fair comparison
            use_semantic_similarity=True,
            semantic_threshold=0.7,
            use_hungarian=False,  # Explicitly disable Hungarian algorithm
            long_string_method='direct'  # Use direct comparison instead
        )
        
        # Initialize evaluators for individual comparisons
        evaluator_with_hungarian = SemanticJsonTreeConsistencyEvaluator(
            array_order_matters=False,
            use_semantic_similarity=True,
            semantic_threshold=0.7,
            string_method='semantic',
            use_hungarian=True,  # Explicitly enable Hungarian algorithm
            long_string_method='hungarian'
        )
        
        evaluator_without_hungarian = SemanticJsonTreeConsistencyEvaluator(
            array_order_matters=False,  # Keep this the same for fair comparison
            use_semantic_similarity=True,
            semantic_threshold=0.7,
            string_method='semantic',
            use_hungarian=False,  # Explicitly disable Hungarian algorithm
            long_string_method='direct'  # Use direct comparison instead
        )
        
        # Calculate individual similarity scores with Hungarian
        with_hungarian_scores = []
        for response in valid_responses:
            similarity, _ = evaluator_with_hungarian.calculate_tree_edit_distance(
                response, ground_truth
            )
            with_hungarian_scores.append(similarity)
        
        # Calculate individual similarity scores without Hungarian
        without_hungarian_scores = []
        for response in valid_responses:
            similarity, _ = evaluator_without_hungarian.calculate_tree_edit_distance(
                response, ground_truth
            )
            without_hungarian_scores.append(similarity)
        
        # Extract consistency metrics
        with_hungarian_consistency = with_hungarian_result.get('consistency_metrics', {})
        without_hungarian_consistency = without_hungarian_result.get('consistency_metrics', {})
        
        # Calculate accuracy metrics
        avg_with_hungarian = sum(with_hungarian_scores) / len(with_hungarian_scores) if with_hungarian_scores else 0
        avg_without_hungarian = sum(without_hungarian_scores) / len(without_hungarian_scores) if without_hungarian_scores else 0
        
        # Calculate stability metrics
        std_with_hungarian = np.std(with_hungarian_scores) if len(with_hungarian_scores) > 1 else 0
        std_without_hungarian = np.std(without_hungarian_scores) if len(without_hungarian_scores) > 1 else 0
        
        # Calculate stability as inverse of standard deviation
        stability_with_hungarian = 1 - (std_with_hungarian if std_with_hungarian < 1 else 1)
        stability_without_hungarian = 1 - (std_without_hungarian if std_without_hungarian < 1 else 1)
        
        # Calculate improvements
        accuracy_improvement = avg_with_hungarian - avg_without_hungarian
        stability_improvement = stability_with_hungarian - stability_without_hungarian
        
        # Calculate overall consistency improvement
        consistency_with_hungarian = with_hungarian_consistency.get('mean_similarity', 0)
        consistency_without_hungarian = without_hungarian_consistency.get('mean_similarity', 0)
        consistency_improvement = consistency_with_hungarian - consistency_without_hungarian
        
        results.append({
            'sample_id': sample_id,
            'accuracy': {
                'with_hungarian': float(avg_with_hungarian),
                'without_hungarian': float(avg_without_hungarian),
                'improvement': float(accuracy_improvement)
            },
            'stability': {
                'with_hungarian': float(stability_with_hungarian),
                'without_hungarian': float(stability_without_hungarian),
                'improvement': float(stability_improvement)
            },
            'consistency': {
                'with_hungarian': float(consistency_with_hungarian),
                'without_hungarian': float(consistency_without_hungarian),
                'improvement': float(consistency_improvement)
            },
            'std_deviation': {
                'with_hungarian': float(std_with_hungarian),
                'without_hungarian': float(std_without_hungarian),
                'difference': float(std_without_hungarian - std_with_hungarian)
            },
            'individual_scores': {
                'with_hungarian': [float(s) for s in with_hungarian_scores],
                'without_hungarian': [float(s) for s in without_hungarian_scores]
            }
        })
    
    return results

def create_visualizations(real_data_results: List[Dict[str, Any]], field_stats: List[Dict[str, Any]], output_dir: str):
    """
    Create visualizations of the Hungarian algorithm effectiveness for both accuracy and stability.
    
    Args:
        real_data_results: List of real data evaluation results
        field_stats: List of complex field statistics
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    # Convert nested dictionaries to flat structure for easier plotting
    flat_data = []
    for result in real_data_results:
        flat_data.append({
            'sample_id': result['sample_id'],
            'accuracy_with_hungarian': result['accuracy']['with_hungarian'],
            'accuracy_without_hungarian': result['accuracy']['without_hungarian'],
            'accuracy_improvement': result['accuracy']['improvement'],
            'stability_with_hungarian': result['stability']['with_hungarian'],
            'stability_without_hungarian': result['stability']['without_hungarian'],
            'stability_improvement': result['stability']['improvement'],
            'consistency_with_hungarian': result['consistency']['with_hungarian'],
            'consistency_without_hungarian': result['consistency']['without_hungarian'],
            'consistency_improvement': result['consistency']['improvement'],
            'std_with_hungarian': result['std_deviation']['with_hungarian'],
            'std_without_hungarian': result['std_deviation']['without_hungarian']
        })
    
    real_df = pd.DataFrame(flat_data)
    field_df = pd.DataFrame(field_stats) if field_stats else pd.DataFrame()
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    if not real_df.empty:
        # 1. Bar chart comparing accuracy with and without Hungarian
        plt.figure(figsize=(14, 8))
        accuracy_df_melted = pd.melt(
            real_df, 
            id_vars=['sample_id'], 
            value_vars=['accuracy_with_hungarian', 'accuracy_without_hungarian'],
            var_name='method', 
            value_name='accuracy'
        )
        
        sns.barplot(x='sample_id', y='accuracy', hue='method', data=accuracy_df_melted)
        plt.title('Accuracy With vs. Without Hungarian Algorithm')
        plt.xlabel('Sample ID')
        plt.ylabel('Accuracy (Similarity to Ground Truth)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
        
        # 2. Bar chart comparing stability with and without Hungarian
        plt.figure(figsize=(14, 8))
        stability_df_melted = pd.melt(
            real_df, 
            id_vars=['sample_id'], 
            value_vars=['stability_with_hungarian', 'stability_without_hungarian'],
            var_name='method', 
            value_name='stability'
        )
        
        sns.barplot(x='sample_id', y='stability', hue='method', data=stability_df_melted)
        plt.title('Stability With vs. Without Hungarian Algorithm')
        plt.xlabel('Sample ID')
        plt.ylabel('Stability (1 - Standard Deviation)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stability_comparison.png'), dpi=300)
        
        # 3. Bar chart comparing consistency with and without Hungarian
        plt.figure(figsize=(14, 8))
        consistency_df_melted = pd.melt(
            real_df, 
            id_vars=['sample_id'], 
            value_vars=['consistency_with_hungarian', 'consistency_without_hungarian'],
            var_name='method', 
            value_name='consistency'
        )
        
        sns.barplot(x='sample_id', y='consistency', hue='method', data=consistency_df_melted)
        plt.title('Cross-Run Consistency With vs. Without Hungarian Algorithm')
        plt.xlabel('Sample ID')
        plt.ylabel('Consistency Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'consistency_comparison.png'), dpi=300)
        
        # 4. Bar chart showing improvements for all metrics
        plt.figure(figsize=(14, 8))
        improvements_df_melted = pd.melt(
            real_df, 
            id_vars=['sample_id'], 
            value_vars=['accuracy_improvement', 'stability_improvement', 'consistency_improvement'],
            var_name='metric', 
            value_name='improvement'
        )
        
        sns.barplot(x='sample_id', y='improvement', hue='metric', data=improvements_df_melted)
        plt.title('Improvements from Hungarian Algorithm Across Metrics')
        plt.xlabel('Sample ID')
        plt.ylabel('Improvement')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_metrics_improvement.png'), dpi=300)
        
        # 5. Scatter plot of accuracy vs. stability
        plt.figure(figsize=(12, 8))
        
        # Plot points for with Hungarian
        sns.scatterplot(
            x='accuracy_with_hungarian', 
            y='stability_with_hungarian', 
            data=real_df, 
            label='With Hungarian',
            s=100,  # Size
            marker='o'
        )
        
        # Plot points for without Hungarian
        sns.scatterplot(
            x='accuracy_without_hungarian', 
            y='stability_without_hungarian', 
            data=real_df, 
            label='Without Hungarian',
            s=100,  # Size
            marker='x'
        )
        
        # Connect paired points with arrows
        for _, row in real_df.iterrows():
            plt.arrow(
                row['accuracy_without_hungarian'], 
                row['stability_without_hungarian'],
                row['accuracy_with_hungarian'] - row['accuracy_without_hungarian'], 
                row['stability_with_hungarian'] - row['stability_without_hungarian'],
                width=0.002, 
                head_width=0.01, 
                head_length=0.01, 
                fc='gray', 
                ec='gray',
                alpha=0.5
            )
        
        plt.title('Accuracy vs. Stability: With and Without Hungarian Algorithm')
        plt.xlabel('Accuracy (Similarity to Ground Truth)')
        plt.ylabel('Stability (1 - Standard Deviation)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_stability.png'), dpi=300)
        
        # 6. Box plots of improvements
        plt.figure(figsize=(10, 6))
        plt.boxplot(
            [real_df['accuracy_improvement'], real_df['stability_improvement'], real_df['consistency_improvement']],
            labels=['Accuracy', 'Stability', 'Consistency']
        )
        plt.title('Distribution of Improvements from Hungarian Algorithm')
        plt.ylabel('Improvement')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvements_distribution.png'), dpi=300)
        
        # 7. Correlation with field complexity
        if not field_df.empty:
            # Try to merge dataframes if possible
            try:
                merged_df = pd.merge(real_df, field_df, on='sample_id')
                
                # Scatter plot of array count vs accuracy improvement
                if 'array_count' in merged_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x='array_count', y='accuracy_improvement', data=merged_df)
                    plt.title('Array Count vs. Accuracy Improvement')
                    plt.xlabel('Number of Arrays in Sample')
                    plt.ylabel('Accuracy Improvement')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'array_count_vs_accuracy.png'), dpi=300)
                
                # Scatter plot of array count vs stability improvement
                if 'array_count' in merged_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x='array_count', y='stability_improvement', data=merged_df)
                    plt.title('Array Count vs. Stability Improvement')
                    plt.xlabel('Number of Arrays in Sample')
                    plt.ylabel('Stability Improvement')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'array_count_vs_stability.png'), dpi=300)
                
                # Scatter plot of long text count vs accuracy improvement
                if 'long_text_count' in merged_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x='long_text_count', y='accuracy_improvement', data=merged_df)
                    plt.title('Long Text Count vs. Accuracy Improvement')
                    plt.xlabel('Number of Long Texts in Sample')
                    plt.ylabel('Accuracy Improvement')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'text_count_vs_accuracy.png'), dpi=300)
                
                # Scatter plot of long text count vs stability improvement
                if 'long_text_count' in merged_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x='long_text_count', y='stability_improvement', data=merged_df)
                    plt.title('Long Text Count vs. Stability Improvement')
                    plt.xlabel('Number of Long Texts in Sample')
                    plt.ylabel('Stability Improvement')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'text_count_vs_stability.png'), dpi=300)
            except Exception as e:
                print(f"Error creating field correlation plots: {e}")
    
    print(f"Visualizations saved to {output_dir}")
    
def find_latest_generation(generations_dir):
    """
    Find the latest generation result in the given directory.

    Args:
        generations_dir: Directory containing generation results
    """
    generation_files = [f for f in os.listdir(generations_dir) if f.endswith('all_results.json')]
    if not generation_files:
        raise FileNotFoundError("No generation results found.")
    latest_file = max(generation_files, key=lambda f: os.path.getmtime(os.path.join(generations_dir, f)))
    return os.path.join(generations_dir, latest_file)    

def run_experiment(args):
    """
    Run the Hungarian algorithm effectiveness experiment.
    
    Args:
        args: Command-line arguments
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    generations_dir = os.path.join(args.output_dir, "generations")
    visualizations_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(generations_dir, exist_ok=True)
    
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
    
    # Analyze complex fields (arrays and long texts) in the data
    field_stats = analyze_complex_fields(data)
    
    # Evaluate real data
    real_data_results = evaluate_real_data(data)
    
    # Create visualizations
    create_visualizations(real_data_results, field_stats, visualizations_dir)
    
    # Save results
    results_file = os.path.join(args.output_dir, "hungarian_algorithm_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'real_data_results': real_data_results,
            'field_stats': field_stats,
            'parameters': {
                'input_file': input_file
            }
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Print summary of findings
    print("\n=== Hungarian Algorithm Effectiveness Analysis ===")
    
    # Calculate average improvements for real data
    if real_data_results:
        # Calculate accuracy improvements
        accuracy_improvements = [r['accuracy']['improvement'] for r in real_data_results]
        avg_accuracy_improvement = np.mean(accuracy_improvements)
        max_accuracy_improvement = max(accuracy_improvements)
        
        # Calculate stability improvements
        stability_improvements = [r['stability']['improvement'] for r in real_data_results]
        avg_stability_improvement = np.mean(stability_improvements)
        max_stability_improvement = max(stability_improvements)
        
        # Calculate consistency improvements
        consistency_improvements = [r['consistency']['improvement'] for r in real_data_results]
        avg_consistency_improvement = np.mean(consistency_improvements)
        max_consistency_improvement = max(consistency_improvements)
        
        print(f"\nHungarian Algorithm Impact Analysis:")
        
        # Accuracy metrics
        print(f"\nAccuracy Metrics (Similarity to Ground Truth):")
        print(f"Average improvement: {avg_accuracy_improvement:.4f}")
        print(f"Maximum improvement: {max_accuracy_improvement:.4f}")
        
        # Calculate percentage of samples with accuracy improvement
        improved_accuracy_samples = sum(1 for imp in accuracy_improvements if imp > 0)
        total_samples = len(real_data_results)
        accuracy_improvement_percentage = improved_accuracy_samples / total_samples * 100 if total_samples > 0 else 0
        print(f"Samples with accuracy improvement: {improved_accuracy_samples}/{total_samples} ({accuracy_improvement_percentage:.1f}%)")
        
        # Stability metrics
        print(f"\nStability Metrics (Consistency Across Runs):")
        print(f"Average improvement: {avg_stability_improvement:.4f}")
        print(f"Maximum improvement: {max_stability_improvement:.4f}")
        
        # Calculate percentage of samples with stability improvement
        improved_stability_samples = sum(1 for imp in stability_improvements if imp > 0)
        stability_improvement_percentage = improved_stability_samples / total_samples * 100 if total_samples > 0 else 0
        print(f"Samples with stability improvement: {improved_stability_samples}/{total_samples} ({stability_improvement_percentage:.1f}%)")
        
        # Consistency metrics
        print(f"\nCross-Run Consistency Metrics:")
        print(f"Average improvement: {avg_consistency_improvement:.4f}")
        print(f"Maximum improvement: {max_consistency_improvement:.4f}")
        
        # Calculate percentage of samples with consistency improvement
        improved_consistency_samples = sum(1 for imp in consistency_improvements if imp > 0)
        consistency_improvement_percentage = improved_consistency_samples / total_samples * 100 if total_samples > 0 else 0
        print(f"Samples with consistency improvement: {improved_consistency_samples}/{total_samples} ({consistency_improvement_percentage:.1f}%)")
        
        # Find samples with highest improvements
        if real_data_results:
            best_accuracy_sample = max(real_data_results, key=lambda x: x['accuracy']['improvement'])
            best_stability_sample = max(real_data_results, key=lambda x: x['stability']['improvement'])
            best_consistency_sample = max(real_data_results, key=lambda x: x['consistency']['improvement'])
            
            print(f"\nBest Samples:")
            print(f"Sample with highest accuracy improvement: {best_accuracy_sample['sample_id']} ({best_accuracy_sample['accuracy']['improvement']:.4f})")
            print(f"Sample with highest stability improvement: {best_stability_sample['sample_id']} ({best_stability_sample['stability']['improvement']:.4f})")
            print(f"Sample with highest consistency improvement: {best_consistency_sample['sample_id']} ({best_consistency_sample['consistency']['improvement']:.4f})")
            
            # Calculate correlation between accuracy and stability improvements
            correlation = np.corrcoef(accuracy_improvements, stability_improvements)[0, 1]
            print(f"\nCorrelation between accuracy and stability improvements: {correlation:.4f}")
            
            # Analyze correlation with field complexity
            if field_stats:
                field_samples = {stat['sample_id']: stat for stat in field_stats}
                array_improvements = []
                long_text_improvements = []
                both_improvements = []
                neither_improvements = []
                
                for result in real_data_results:
                    if result['sample_id'] in field_samples:
                        stat = field_samples[result['sample_id']]
                        has_arrays = stat['has_arrays']
                        has_long_texts = stat['has_long_texts']
                        
                        # Use accuracy improvement as the primary metric
                        accuracy_improvement = result['accuracy']['improvement']
                        
                        if has_arrays and has_long_texts:
                            both_improvements.append(accuracy_improvement)
                        elif has_arrays:
                            array_improvements.append(accuracy_improvement)
                        elif has_long_texts:
                            long_text_improvements.append(accuracy_improvement)
                        else:
                            neither_improvements.append(accuracy_improvement)
                    else:
                        neither_improvements.append(result['accuracy']['improvement'])
                
                print(f"\nField Type Analysis:")
                
                # Calculate and print average improvements by field type
                if array_improvements:
                    avg_array_improvement = sum(array_improvements) / len(array_improvements)
                    print(f"Average improvement for samples with arrays only: {avg_array_improvement:.4f}")
                
                if long_text_improvements:
                    avg_text_improvement = sum(long_text_improvements) / len(long_text_improvements)
                    print(f"Average improvement for samples with long texts only: {avg_text_improvement:.4f}")
                
                if both_improvements:
                    avg_both_improvement = sum(both_improvements) / len(both_improvements)
                    print(f"Average improvement for samples with both arrays and long texts: {avg_both_improvement:.4f}")
                
                if neither_improvements:
                    avg_neither_improvement = sum(neither_improvements) / len(neither_improvements)
                    print(f"Average improvement for samples with neither arrays nor long texts: {avg_neither_improvement:.4f}")
                
                # Calculate improvement ratios
                if array_improvements and neither_improvements:
                    array_ratio = avg_array_improvement / avg_neither_improvement if avg_neither_improvement > 0 else float('inf')
                    print(f"Improvement ratio (arrays vs. neither): {array_ratio:.2f}x")
                
                if long_text_improvements and neither_improvements:
                    text_ratio = avg_text_improvement / avg_neither_improvement if avg_neither_improvement > 0 else float('inf')
                    print(f"Improvement ratio (long texts vs. neither): {text_ratio:.2f}x")
                
                if both_improvements and neither_improvements:
                    both_ratio = avg_both_improvement / avg_neither_improvement if avg_neither_improvement > 0 else float('inf')
                    print(f"Improvement ratio (both vs. neither): {both_ratio:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Run Hungarian algorithm effectiveness experiment.")
    parser.add_argument("--data-dir", type=str, help="Directory containing the data files.")
    parser.add_argument("--input-file", type=str, help="Path to the JSON file containing the generated outputs. If not provided, generation will be run.")
    parser.add_argument("--output-dir", type=str, default="./hungarian_experiment", help="Directory to save experiment results.")
    parser.add_argument("--run-num", type=int, default=5, help="Number of runs to perform.")
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation")
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_file and not args.data_dir:
        parser.error("Either --input-file or --data-dir must be provided")
    
    run_experiment(args)

if __name__ == "__main__":
    main()