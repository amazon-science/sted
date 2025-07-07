#!/usr/bin/env python
"""
Temperature-Stability Correlation Experiment

This script runs a comprehensive experiment to analyze the relationship between
temperature settings and output stability in LLM generations.

Usage:
    python run_temperature_experiment.py --data-dir extracted_sharegpt_data --output-dir ./temperature_experiment
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Any
import subprocess
import pandas as pd
from pathlib import Path

def run_generation(data_dir: str, output_dir: str, temperature: float, run_num: int, include_schema: bool) -> str:
    """
    Run LLM generation with specified parameters.
    
    Args:
        data_dir: Directory containing the data files
        output_dir: Directory to save generation results
        temperature: Temperature setting for generation
        run_num: Number of runs to perform
        include_schema: Whether to include schema in the prompt
        
    Returns:
        Path to the generated results file
    """
    cmd = [
        "python", "llm_gen_simple.py",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--temperature", str(temperature),
        "--run-num", str(run_num)
    ]
    
    if include_schema:
        cmd.append("--include-schema")
    
    print(f"Running generation with temperature {temperature}...")
    subprocess.run(cmd, check=True)
    
    # Find the most recent results directory for this temperature
    temp_str = f"temp_{temperature:.2f}".replace('.', '_')
    result_dirs = list(Path(output_dir).glob(f"llm_gen_results_*{temp_str}*"))
    
    if not result_dirs:
        raise FileNotFoundError(f"No results found for temperature {temperature}")
    
    # Sort by creation time (most recent first)
    result_dir = sorted(result_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    results_file = result_dir / "all_results.json"
    
    return str(results_file)

def run_evaluation(input_file: str, output_dir: str) -> str:
    """
    Run evaluation on generated outputs.
    
    Args:
        input_file: Path to the generated results file
        output_dir: Directory to save evaluation results
        
    Returns:
        Path to the evaluation results file
    """
    cmd = [
        "python", "evaluate_generations.py",
        "--input-file", input_file,
        "--output-dir", output_dir,
        "--metrics", "all"
    ]
    
    print(f"Running evaluation on {input_file}...")
    subprocess.run(cmd, check=True)
    
    # Find the most recent evaluation results file
    eval_files = list(Path(output_dir).glob("evaluation_results_*.json"))
    
    if not eval_files:
        raise FileNotFoundError("No evaluation results found")
    
    # Sort by creation time (most recent first)
    eval_file = sorted(eval_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    return str(eval_file)

def extract_metrics(eval_file: str) -> Dict[str, Any]:
    """
    Extract metrics from evaluation results file.
    
    Args:
        eval_file: Path to the evaluation results file
        
    Returns:
        Dictionary with extracted metrics
    """
    with open(eval_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Initialize aggregated metrics
    metrics = {
        'semantic_similarity_mean': [],
        'semantic_similarity_std': [],
        'semantic_stability': [],
        'cross_run_consistency': [],
        'bleu_mean': [],
        'rouge_l_mean': [],
        'bert_score_mean': [],
        'jaccard_mean': []
    }
    
    # Extract metrics from each sample
    for sample in results:
        # Extract semantic metrics if available
        if 'semantic_tree_metrics' in sample and sample['semantic_tree_metrics']:
            if 'ground_truth_accuracy' in sample['semantic_tree_metrics']:
                gt_metrics = sample['semantic_tree_metrics']['ground_truth_accuracy']
                if 'semantic_similarity' in gt_metrics:
                    metrics['semantic_similarity_mean'].append(gt_metrics['semantic_similarity'].get('mean', 0))
                    metrics['semantic_similarity_std'].append(gt_metrics['semantic_similarity'].get('std', 0))
                    metrics['semantic_stability'].append(gt_metrics['semantic_similarity'].get('stability', 0))
            
            if 'cross_run_consistency' in sample['semantic_tree_metrics'] and 'consistency_metrics' in sample['semantic_tree_metrics']['cross_run_consistency']:
                consistency = sample['semantic_tree_metrics']['cross_run_consistency']['consistency_metrics']
                metrics['cross_run_consistency'].append(consistency.get('mean_similarity', 0))
        
        # Extract NLP metrics if available
        if 'nlp_metrics' in sample and sample['nlp_metrics']:
            if 'accuracy_metrics' in sample['nlp_metrics']:
                acc_metrics = sample['nlp_metrics']['accuracy_metrics']
                
                if 'bleu' in acc_metrics:
                    metrics['bleu_mean'].append(acc_metrics['bleu'].get('mean', 0))
                
                if 'rouge_l' in acc_metrics:
                    metrics['rouge_l_mean'].append(acc_metrics['rouge_l'].get('mean', 0))
                
                if 'bert_score' in acc_metrics:
                    metrics['bert_score_mean'].append(acc_metrics['bert_score'].get('mean', 0))
                
                if 'jaccard' in acc_metrics:
                    metrics['jaccard_mean'].append(acc_metrics['jaccard'].get('mean', 0))
    
    # Calculate averages
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = sum(values) / len(values)
        else:
            result[key] = 0
    
    return result

def analyze_temperature_stability_relationship(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the relationship between temperature and stability.
    
    Args:
        results: List of dictionaries with temperature and metrics
        
    Returns:
        Dictionary with analysis results
    """
    # Extract data for analysis
    temperatures = [r['temperature'] for r in results]
    semantic_stabilities = [r['semantic_stability'] for r in results]
    cross_run_consistencies = [r['cross_run_consistency'] for r in results]
    semantic_similarities = [r['semantic_similarity_mean'] for r in results]
    
    # Calculate correlations
    pearson_stability = stats.pearsonr(temperatures, semantic_stabilities)
    spearman_stability = stats.spearmanr(temperatures, semantic_stabilities)
    pearson_consistency = stats.pearsonr(temperatures, cross_run_consistencies)
    spearman_consistency = stats.spearmanr(temperatures, cross_run_consistencies)
    pearson_similarity = stats.pearsonr(temperatures, semantic_similarities)
    spearman_similarity = stats.spearmanr(temperatures, semantic_similarities)
    
    # Fit linear regression for stability
    slope, intercept, r_value, p_value, std_err = stats.linregress(temperatures, semantic_stabilities)
    
    # Fit polynomial regression for stability
    poly_degree = min(3, len(temperatures) - 1)  # Avoid overfitting
    poly_coeffs = np.polyfit(temperatures, semantic_stabilities, poly_degree)
    poly_r_squared = np.corrcoef(temperatures, np.polyval(poly_coeffs, temperatures))[0, 1] ** 2
    
    return {
        'correlations': {
            'pearson_stability': {
                'correlation': pearson_stability[0],
                'p_value': pearson_stability[1]
            },
            'spearman_stability': {
                'correlation': spearman_stability[0],
                'p_value': spearman_stability[1]
            },
            'pearson_consistency': {
                'correlation': pearson_consistency[0],
                'p_value': pearson_consistency[1]
            },
            'spearman_consistency': {
                'correlation': spearman_consistency[0],
                'p_value': spearman_consistency[1]
            },
            'pearson_similarity': {
                'correlation': pearson_similarity[0],
                'p_value': pearson_similarity[1]
            },
            'spearman_similarity': {
                'correlation': spearman_similarity[0],
                'p_value': spearman_similarity[1]
            }
        },
        'linear_regression': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'equation': f"stability = {slope:.4f} * temperature + {intercept:.4f}"
        },
        'polynomial_regression': {
            'coefficients': poly_coeffs.tolist(),
            'degree': poly_degree,
            'r_squared': poly_r_squared
        }
    }

def create_visualizations(results: List[Dict[str, Any]], output_dir: str, analysis: Dict[str, Any]):
    """
    Create visualizations of the temperature-stability relationship.
    
    Args:
        results: List of dictionaries with temperature and metrics
        output_dir: Directory to save visualizations
        analysis: Dictionary with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 1. Scatter plot with regression line for stability
    plt.subplot(2, 2, 1)
    sns.regplot(x='temperature', y='semantic_stability', data=df, scatter_kws={'alpha':0.7}, line_kws={'color':'red'})
    plt.title('Temperature vs. Semantic Stability')
    plt.xlabel('Temperature')
    plt.ylabel('Semantic Stability')
    
    # Add regression equation
    equation = analysis['linear_regression']['equation']
    r_squared = analysis['linear_regression']['r_squared']
    plt.annotate(f"{equation}\nR² = {r_squared:.4f}", 
                xy=(0.05, 0.05), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 2. Scatter plot with regression line for cross-run consistency
    plt.subplot(2, 2, 2)
    sns.regplot(x='temperature', y='cross_run_consistency', data=df, scatter_kws={'alpha':0.7}, line_kws={'color':'blue'})
    plt.title('Temperature vs. Cross-Run Consistency')
    plt.xlabel('Temperature')
    plt.ylabel('Cross-Run Consistency')
    
    # 3. Dual y-axis plot for stability and accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Semantic Stability', color=color)
    ax1.plot(df['temperature'], df['semantic_stability'], 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Semantic Similarity (Accuracy)', color=color)
    ax2.plot(df['temperature'], df['semantic_similarity_mean'], 's-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Temperature vs. Stability and Accuracy')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_stability_accuracy.png'), dpi=300)
    
    # 4. Heatmap of correlations between metrics
    plt.figure(figsize=(10, 8))
    correlation_columns = ['temperature', 'semantic_stability', 'cross_run_consistency', 
                          'semantic_similarity_mean', 'bleu_mean', 'rouge_l_mean', 
                          'bert_score_mean', 'jaccard_mean']
    correlation_df = df[correlation_columns].corr()
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlations.png'), dpi=300)
    
    # 5. Line plot comparing different metrics across temperatures
    plt.figure(figsize=(12, 6))
    metrics = ['semantic_stability', 'cross_run_consistency', 'semantic_similarity_mean', 
              'bleu_mean', 'rouge_l_mean', 'bert_score_mean', 'jaccard_mean']
    
    for metric in metrics:
        plt.plot(df['temperature'], df[metric], 'o-', label=metric)
    
    plt.xlabel('Temperature')
    plt.ylabel('Metric Value')
    plt.title('Comparison of Metrics Across Temperatures')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    
    # Save all figures
    plt.savefig(os.path.join(output_dir, 'temperature_stability.png'), dpi=300)
    
    print(f"Visualizations saved to {output_dir}")

def run_experiment(args):
    """
    Run the temperature-stability correlation experiment.
    
    Args:
        args: Command-line arguments
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    generations_dir = os.path.join(args.output_dir, "generations")
    evaluations_dir = os.path.join(args.output_dir, "evaluations")
    visualizations_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(generations_dir, exist_ok=True)
    os.makedirs(evaluations_dir, exist_ok=True)
    
    # Define temperatures to test
    if args.temperatures:
        temperatures = args.temperatures
    else:
        temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    # Run generation and evaluation for each temperature
    for temp in temperatures:
        # Run generation
        gen_results_file = run_generation(
            data_dir=args.data_dir,
            output_dir=generations_dir,
            temperature=temp,
            run_num=args.run_num,
            include_schema=args.include_schema
        )
        
        # Run evaluation
        eval_results_file = run_evaluation(
            input_file=gen_results_file,
            output_dir=evaluations_dir
        )
        
        # Extract metrics
        metrics = extract_metrics(eval_results_file)
        metrics['temperature'] = temp
        results.append(metrics)
    
    # Analyze the relationship between temperature and stability
    analysis = analyze_temperature_stability_relationship(results)
    
    # Create visualizations
    create_visualizations(results, visualizations_dir, analysis)
    
    # Save results and analysis
    results_file = os.path.join(args.output_dir, "temperature_stability_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'analysis': analysis,
            'parameters': {
                'temperatures': temperatures,
                'run_num': args.run_num,
                'include_schema': args.include_schema,
                'data_dir': args.data_dir
            }
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Print summary of findings
    print("\n=== Temperature-Stability Correlation Analysis ===")
    print(f"Pearson correlation: {analysis['correlations']['pearson_stability']['correlation']:.4f} (p-value: {analysis['correlations']['pearson_stability']['p_value']:.4f})")
    print(f"Spearman correlation: {analysis['correlations']['spearman_stability']['correlation']:.4f} (p-value: {analysis['correlations']['spearman_stability']['p_value']:.4f})")
    print(f"Linear regression: {analysis['linear_regression']['equation']}")
    print(f"R-squared (linear): {analysis['linear_regression']['r_squared']:.4f}")
    print(f"R-squared (polynomial): {analysis['polynomial_regression']['r_squared']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Run temperature-stability correlation experiment.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the data files.")
    parser.add_argument("--output-dir", type=str, default="./temperature_experiment", help="Directory to save experiment results.")
    parser.add_argument("--run-num", type=int, default=10, help="Number of runs per temperature.")
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt.")
    parser.add_argument("--temperatures", type=float, nargs="+", help="List of temperatures to test. Default: 0.0 to 1.0 in 0.1 increments.")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()