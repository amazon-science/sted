#!/usr/bin/env python
"""
Temperature-Stability Correlation Experiment

This script runs a comprehensive experiment to analyze the relationship between
temperature settings and standard deviation of mean similarity in LLM generations.
It focuses on how temperature affects the variability of similarity scores.

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
from tqdm import tqdm

from calculate_similarity_stats import (
    load_generation_results, 
    compare_with_multiple_generations
)

def run_generation(data_dir: str, output_dir: str, temperature: float, run_num: int, include_schema: bool, model_id: str, sample_limit: int=40) -> str:
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
        "python", "llm_gen.py",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--temperature", str(temperature),
        "--run-num", str(run_num),
        "--sample-limit", str(sample_limit),
        "--model-id", model_id
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

def calculate_similarity_metrics(generation_file: str, methods: List[str] = ["ted", "bertscore", "deepdiff"], output_dir: str=".", embedding_model="amazon.titan-embed-text-v2:0") -> Dict[str, Any]:
    """
    Calculate similarity metrics for generated outputs using our similarity statistics approach.
    
    Args:
        generation_file: Path to the generation results file
        methods: List of similarity methods to use
        
    Returns:
        Dictionary with similarity metrics and statistics
    """
    print(f"Calculating similarity metrics for {generation_file}...")
    
    # Load generation results
    ground_truth_list, generated_responses_list = load_generation_results(generation_file)
    
    # Calculate similarity statistics with multiple generations
    results = compare_with_multiple_generations(
        ground_truth_list, 
        generated_responses_list,
        methods=methods,
        model_id=embedding_model,
        output_dir=output_dir
    )
    
    return results

def extract_temperature_metrics(similarity_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metrics from similarity results focusing on temperature-stability correlation.
    
    Args:
        similarity_results: Results from calculate_similarity_metrics
        
    Returns:
        Dictionary with extracted metrics focused on standard deviation analysis
    """
    metrics = {}
    
    for method in similarity_results:
        method_results = similarity_results[method]
        
        # Extract overall statistics
        overall_stats = method_results['overall_stats']
        sample_level_stats = method_results['sample_level_stats']
        
        # Key metrics for temperature-stability analysis
        metrics[f'{method}_mean_similarity'] = overall_stats['mean']
        metrics[f'{method}_std_similarity'] = overall_stats['std']
        metrics[f'{method}_mean_of_means'] = sample_level_stats['mean_of_means']
        metrics[f'{method}_std_of_means'] = sample_level_stats['std_of_means']  # This is key for temperature correlation
        metrics[f'{method}_mean_of_stds'] = sample_level_stats['mean_of_stds']
        metrics[f'{method}_std_of_stds'] = sample_level_stats['std_of_stds']
        
        # Calculate coefficient of variation (normalized variability)
        if overall_stats['mean'] > 0:
            metrics[f'{method}_cv'] = overall_stats['std'] / overall_stats['mean']
        else:
            metrics[f'{method}_cv'] = 0
        
        # Calculate stability score (inverse of variability)
        metrics[f'{method}_stability'] = 1.0 / (1.0 + overall_stats['std'])
    
    return metrics

def analyze_temperature_std_correlation(results: List[Dict[str, Any]], methods: List[str] = ["ted", "bertscore", "deepdiff"]) -> Dict[str, Any]:
    """
    Analyze the correlation between temperature and standard deviation of mean similarity.
    
    Args:
        results: List of dictionaries with temperature and metrics
        methods: List of similarity methods used
        
    Returns:
        Dictionary with detailed correlation analysis results
    """
    temperatures = [r['temperature'] for r in results]
    analysis_results = {}
    
    for method in methods:
        method_analysis = {}
        
        # Extract key metrics for this method
        std_similarities = [r[f'{method}_std_similarity'] for r in results]
        std_of_means = [r[f'{method}_std_of_means'] for r in results]
        mean_of_stds = [r[f'{method}_mean_of_stds'] for r in results]
        cv_values = [r[f'{method}_cv'] for r in results]
        stability_scores = [r[f'{method}_stability'] for r in results]
        mean_similarities = [r[f'{method}_mean_similarity'] for r in results]
        
        # Calculate correlations for different variability metrics
        correlations = {}
        
        # Temperature vs Standard deviation of similarities
        if len(std_similarities) > 1:
            pearson_std = stats.pearsonr(temperatures, std_similarities)
            spearman_std = stats.spearmanr(temperatures, std_similarities)
            correlations['temp_vs_std_similarity'] = {
                'pearson': {'r': pearson_std[0], 'p': pearson_std[1]},
                'spearman': {'r': spearman_std[0], 'p': spearman_std[1]}
            }
        
        # Temperature vs Standard deviation of means (key metric)
        if len(std_of_means) > 1:
            pearson_std_means = stats.pearsonr(temperatures, std_of_means)
            spearman_std_means = stats.spearmanr(temperatures, std_of_means)
            correlations['temp_vs_std_of_means'] = {
                'pearson': {'r': pearson_std_means[0], 'p': pearson_std_means[1]},
                'spearman': {'r': spearman_std_means[0], 'p': spearman_std_means[1]}
            }
        
        # Temperature vs Mean of standard deviations
        if len(mean_of_stds) > 1:
            pearson_mean_stds = stats.pearsonr(temperatures, mean_of_stds)
            spearman_mean_stds = stats.spearmanr(temperatures, mean_of_stds)
            correlations['temp_vs_mean_of_stds'] = {
                'pearson': {'r': pearson_mean_stds[0], 'p': pearson_mean_stds[1]},
                'spearman': {'r': spearman_mean_stds[0], 'p': spearman_mean_stds[1]}
            }
        
        # Temperature vs Coefficient of Variation
        if len(cv_values) > 1:
            pearson_cv = stats.pearsonr(temperatures, cv_values)
            spearman_cv = stats.spearmanr(temperatures, cv_values)
            correlations['temp_vs_cv'] = {
                'pearson': {'r': pearson_cv[0], 'p': pearson_cv[1]},
                'spearman': {'r': spearman_cv[0], 'p': spearman_cv[1]}
            }
        
        # Temperature vs Stability scores
        if len(stability_scores) > 1:
            pearson_stability = stats.pearsonr(temperatures, stability_scores)
            spearman_stability = stats.spearmanr(temperatures, stability_scores)
            correlations['temp_vs_stability'] = {
                'pearson': {'r': pearson_stability[0], 'p': pearson_stability[1]},
                'spearman': {'r': spearman_stability[0], 'p': spearman_stability[1]}
            }
        
        # Temperature vs Mean similarity (accuracy)
        if len(mean_similarities) > 1:
            pearson_mean = stats.pearsonr(temperatures, mean_similarities)
            spearman_mean = stats.spearmanr(temperatures, mean_similarities)
            correlations['temp_vs_mean_similarity'] = {
                'pearson': {'r': pearson_mean[0], 'p': pearson_mean[1]},
                'spearman': {'r': spearman_mean[0], 'p': spearman_mean[1]}
            }
        
        # Linear regression for key metric (std of means)
        if len(std_of_means) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(temperatures, std_of_means)
            linear_regression = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err,
                'equation': f"std_of_means = {slope:.4f} * temperature + {intercept:.4f}"
            }
        else:
            linear_regression = {}
        
        # Polynomial regression for std of means
        if len(std_of_means) > 2:
            poly_degree = min(2, len(temperatures) - 1)
            poly_coeffs = np.polyfit(temperatures, std_of_means, poly_degree)
            poly_predictions = np.polyval(poly_coeffs, temperatures)
            poly_r_squared = np.corrcoef(std_of_means, poly_predictions)[0, 1] ** 2
            polynomial_regression = {
                'coefficients': poly_coeffs.tolist(),
                'degree': poly_degree,
                'r_squared': poly_r_squared
            }
        else:
            polynomial_regression = {}
        
        method_analysis = {
            'correlations': correlations,
            'linear_regression': linear_regression,
            'polynomial_regression': polynomial_regression,
            'summary_stats': {
                'std_similarity_range': [min(std_similarities), max(std_similarities)] if std_similarities else [0, 0],
                'std_of_means_range': [min(std_of_means), max(std_of_means)] if std_of_means else [0, 0],
                'cv_range': [min(cv_values), max(cv_values)] if cv_values else [0, 0],
                'stability_range': [min(stability_scores), max(stability_scores)] if stability_scores else [0, 0]
            }
        }
        
        analysis_results[method] = method_analysis
    
    return analysis_results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str, analysis: Dict[str, Any], methods: List[str] = ["ted", "bertscore", "deepdiff"]):
    """
    Create visualizations focusing on temperature vs standard deviation of mean similarity.
    
    Args:
        results: List of dictionaries with temperature and metrics
        output_dir: Directory to save visualizations
        analysis: Dictionary with analysis results
        methods: List of similarity methods used
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # Create a comprehensive figure with subplots for each method
    fig, axes = plt.subplots(len(methods), 3, figsize=(18, 6*len(methods)))
    if len(methods) == 1:
        axes = axes.reshape(1, -1)
    
    for i, method in enumerate(methods):
        # 1. Temperature vs Standard Deviation of Means (key plot)
        ax1 = axes[i, 0]
        std_of_means_col = f'{method}_std_of_means'
        if std_of_means_col in df.columns:
            sns.regplot(x='temperature', y=std_of_means_col, data=df, 
                       scatter_kws={'alpha':0.7}, line_kws={'color':'red'}, ax=ax1)
            ax1.set_title(f'{method.upper()}: Temperature vs Std of Means')
            ax1.set_xlabel('Temperature')
            ax1.set_ylabel('Standard Deviation of Means')
            
            # Add regression equation if available
            if method in analysis and 'linear_regression' in analysis[method]:
                lr = analysis[method]['linear_regression']
                if 'equation' in lr:
                    ax1.annotate(f"{lr['equation']}\nR² = {lr['r_squared']:.4f}", 
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                verticalalignment='top')
        
        # 2. Temperature vs Overall Standard Deviation
        ax2 = axes[i, 1]
        std_similarity_col = f'{method}_std_similarity'
        if std_similarity_col in df.columns:
            sns.regplot(x='temperature', y=std_similarity_col, data=df, 
                       scatter_kws={'alpha':0.7}, line_kws={'color':'blue'}, ax=ax2)
            ax2.set_title(f'{method.upper()}: Temperature vs Overall Std')
            ax2.set_xlabel('Temperature')
            ax2.set_ylabel('Overall Standard Deviation')
        
        # 3. Temperature vs Coefficient of Variation
        ax3 = axes[i, 2]
        cv_col = f'{method}_cv'
        if cv_col in df.columns:
            sns.regplot(x='temperature', y=cv_col, data=df, 
                       scatter_kws={'alpha':0.7}, line_kws={'color':'green'}, ax=ax3)
            ax3.set_title(f'{method.upper()}: Temperature vs CV')
            ax3.set_xlabel('Temperature')
            ax3.set_ylabel('Coefficient of Variation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_std_correlation_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary plot comparing all methods for std of means
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, method in enumerate(methods):
        std_of_means_col = f'{method}_std_of_means'
        if std_of_means_col in df.columns:
            plt.plot(df['temperature'], df[std_of_means_col], 'o-', 
                    color=colors[i % len(colors)], label=f'{method.upper()}', linewidth=2, markersize=6)
    
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Standard Deviation of Means', fontsize=12)
    plt.title('Temperature vs Standard Deviation of Mean Similarity\n(Comparison Across Methods)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_std_means_comparison.png'), dpi=300)
    plt.close()
    
    # Create correlation heatmap for each method
    for method in methods:
        method_cols = [col for col in df.columns if col.startswith(method) or col == 'temperature']
        if len(method_cols) > 2:
            plt.figure(figsize=(10, 8))
            correlation_df = df[method_cols].corr()
            sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                       square=True, linewidths=0.5)
            plt.title(f'{method.upper()}: Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{method}_correlation_matrix.png'), dpi=300)
            plt.close()
    
    # Create a dual-axis plot showing mean similarity vs std of means
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        mean_col = f'{method}_mean_similarity'
        std_col = f'{method}_std_of_means'
        
        if mean_col in df.columns and std_col in df.columns:
            color = colors[i % len(colors)]
            
            # Plot mean similarity on left axis
            ax1.plot(df['temperature'], df[mean_col], 'o-', color=color, 
                    label=f'{method.upper()} Mean', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Mean Similarity (Accuracy)', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create second y-axis for std of means
    ax2 = ax1.twinx()
    
    for i, method in enumerate(methods):
        std_col = f'{method}_std_of_means'
        
        if std_col in df.columns:
            color = colors[i % len(colors)]
            
            # Plot std of means on right axis with dashed line
            ax2.plot(df['temperature'], df[std_col], 's--', color=color, 
                    label=f'{method.upper()} Std', linewidth=2, markersize=6, alpha=0.7)
    
    ax2.set_ylabel('Standard Deviation of Means (Variability)', color='gray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    plt.title('Temperature vs Accuracy and Variability\n(Dual-Axis Comparison)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_accuracy_vs_variability.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def run_experiment(args):
    """
    Run the temperature-standard deviation correlation experiment.
    
    Args:
        args: Command-line arguments
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    generations_dir = os.path.join(args.output_dir, "generations")
    visualizations_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(generations_dir, exist_ok=True)
    
    # Define temperatures to test
    if args.temperatures:
        temperatures = args.temperatures
    else:
        temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Define similarity methods to use
    methods = ["ted", "bertscore", "deepdiff"]
    
    results = []
    
    # Run generation and evaluation for each temperature
    for temp in tqdm(temperatures, desc="Processing temperatures"):
        print(f"\n=== Processing Temperature {temp} ===")
        
        # Check if we already have generation results for this temperature and this model
        temp_str = f"temp_{temp:.2f}".replace('.', '_')
        existing_results = list(Path(generations_dir).glob(f"llm_gen_results_{args.model_id}_*{temp_str}*"))
        
        gen_results_file = None
                
        if existing_results:
            result_dir = sorted(existing_results, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            print(f"Found existing results for temperature {temp}. Using {result_dir}")
            gen_results_file = str(result_dir / "all_results.json")
        
        if gen_results_file is None or not os.path.exists(gen_results_file) or args.force_regenerate:
            # Run generation
            gen_results_file = run_generation(
                data_dir=args.data_dir,
                output_dir=generations_dir,
                temperature=temp,
                run_num=args.run_num,
                include_schema=args.include_schema,
                model_id=args.model_id
            )
            
        
        if args.exe_evaluation:
            # get the directory of gen_results_file
            result_dir = os.path.dirname(gen_results_file)
            
            # Calculate similarity metrics
            similarity_results = calculate_similarity_metrics(gen_results_file, methods, result_dir)
            
            # Extract temperature-focused metrics
            metrics = extract_temperature_metrics(similarity_results)
            metrics['temperature'] = temp
            results.append(metrics)
            
            print(f"Temperature {temp} completed. Key metrics:")
            for method in methods:
                std_of_means = metrics.get(f'{method}_std_of_means', 0)
                mean_similarity = metrics.get(f'{method}_mean_similarity', 0)
                print(f"  {method.upper()}: Mean={mean_similarity:.4f}, Std of Means={std_of_means:.4f}")
    
    if args.exe_evaluation:
        # Analyze the relationship between temperature and standard deviation
        analysis = analyze_temperature_std_correlation(results, methods)
        
        # Create visualizations
        create_visualizations(results, visualizations_dir, analysis, methods)
        
        # Save results and analysis
        results_file = os.path.join(args.output_dir, "temperature_std_correlation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'results': results,
                'analysis': analysis,
                'parameters': {
                    'temperatures': temperatures,
                    'methods': methods,
                    'run_num': args.run_num,
                    'include_schema': args.include_schema,
                    'data_dir': args.data_dir
                }
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Run temperature vs standard deviation correlation experiment.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the data files.")
    parser.add_argument("--output-dir", type=str, default="./temperature_experiment", help="Directory to save experiment results.")
    parser.add_argument("--run-num", type=int, default=10, help="Number of runs per temperature.")
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt.")
    parser.add_argument("--temperatures", type=float, nargs="+", help="List of temperatures to test. Default: 0.0 to 1.0 in 0.1 increments.")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regeneration even if results already exist.")
    parser.add_argument("--model-id", type=str, default="us.anthropic.claude-3-5-sonnet-20241022-v2:0", help="Model id")
    parser.add_argument("--embedding-model-id", type=str, default="amazon.titan-embed-text-v2:0", help="Embedding model id")
    parser.add_argument("--exe-evaluation", type=str, default=True, help="Execute evaluation")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()