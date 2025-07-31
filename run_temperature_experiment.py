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

from datetime import datetime

from calculate_similarity_stats import (
    load_generation_results, 
    compare_with_multiple_generations
)

def run_generation(data_dir: str, output_dir: str, temperature: float, run_num: int, include_schema: bool, model_id: str, sample_limit: int=40, max_tokens: int=8000) -> str:
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
        "--model-id", model_id,
        "--max-tokens", str(max_tokens)
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
        ground_truth_list[:10], 
        generated_responses_list[:10],
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
        metrics[f'{method}_mean_pairwise_similarity'] = overall_stats['mean']  # Average similarity between output pairs
        metrics[f'{method}_std_pairwise_similarity'] = overall_stats['std']  # Average std of pairwise similarities per sample
        metrics[f'{method}_mean_of_means'] = sample_level_stats['mean_of_means']
        metrics[f'{method}_std_of_means'] = sample_level_stats['std_of_means']  # This is key for temperature correlation
        metrics[f'{method}_mean_of_stds'] = sample_level_stats['mean_of_stds']
        metrics[f'{method}_std_of_stds'] = sample_level_stats['std_of_stds']
        
        # New consistency metrics from overall_stats
        metrics[f'{method}_consistency_coefficient'] = overall_stats.get('consistency_coefficient', 0.0)
        metrics[f'{method}_stability_score'] = overall_stats.get('stability_score', 0.0)
        metrics[f'{method}_std_normalized_cv'] = overall_stats.get('std_normalized_cv', 0.0)
        metrics[f'{method}_std_normalized_relative'] = overall_stats.get('std_normalized_relative', 0.0)
        
        # Essential distribution metrics
        metrics[f'{method}_iqr'] = overall_stats.get('iqr', 0.0)
        metrics[f'{method}_range'] = overall_stats.get('range', 0.0)
        
        # Legacy metrics for backward compatibility
        if overall_stats['mean'] > 0:
            metrics[f'{method}_cv'] = overall_stats['std'] / overall_stats['mean']
        else:
            metrics[f'{method}_cv'] = 0
        
        # Legacy stability score (keep for backward compatibility)
        metrics[f'{method}_stability'] = 1.0 / (1.0 + overall_stats['std'])
    
    return metrics

def detect_outliers(data, method='iqr', threshold=1.5):
    """
    Detect outliers in data using IQR or Z-score method.
    
    Args:
        data: Array of data points
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    data = np.array(data)
    
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold
    else:
        return np.zeros(len(data), dtype=bool)

def analyze_temperature_std_correlation(results: List[Dict[str, Any]], methods: List[str] = ["ted", "bertscore", "deepdiff"], remove_outliers: bool = True) -> Dict[str, Any]:
    """
    Analyze the correlation between temperature and standard deviation of mean similarity.
    
    Args:
        results: List of dictionaries with temperature and metrics
        methods: List of similarity methods used
        remove_outliers: Whether to remove outliers before correlation analysis
        
    Returns:
        Dictionary with detailed correlation analysis results
    """
    temperatures = [r['temperature'] for r in results]
    analysis_results = {}
    
    for method in methods:
        method_analysis = {}
        
        # Extract key metrics for this method
        std_pairwise_similarities = [r.get(f'{method}_std_pairwise_similarity', r.get(f'{method}_std_similarity', 0.0)) for r in results]
        std_of_means = [r[f'{method}_std_of_means'] for r in results]
        mean_of_stds = [r[f'{method}_mean_of_stds'] for r in results]
        cv_values = [r[f'{method}_cv'] for r in results]
        stability_scores = [r[f'{method}_stability'] for r in results]
        mean_pairwise_similarities = [r.get(f'{method}_mean_pairwise_similarity', r.get(f'{method}_mean_similarity', 0.0)) for r in results]
        
        # New consistency metrics
        consistency_coefficients = [r.get(f'{method}_consistency_coefficient', 0.0) for r in results]
        stability_scores_new = [r.get(f'{method}_stability_score', 0.0) for r in results]
        std_normalized_cv = [r.get(f'{method}_std_normalized_cv', 0.0) for r in results]
        std_normalized_relative = [r.get(f'{method}_std_normalized_relative', 0.0) for r in results]
        
        # Outlier detection and removal if requested
        if remove_outliers:
            # Detect outliers in key metrics
            cc_outliers = detect_outliers(consistency_coefficients, method='iqr', threshold=1.5)
            std_means_outliers = detect_outliers(std_of_means, method='iqr', threshold=1.5)
            
            # Combine outlier masks (outlier if flagged by any key metric)
            combined_outliers = cc_outliers | std_means_outliers
            outlier_count = np.sum(combined_outliers)
            
            if outlier_count > 0:
                print(f"\n⚠️  {method.upper()}: Detected {outlier_count} outliers, removing from correlation analysis")
                
                # Create clean datasets without outliers
                clean_mask = ~combined_outliers
                temperatures_clean = [t for i, t in enumerate(temperatures) if clean_mask[i]]
                std_pairwise_similarities_clean = [s for i, s in enumerate(std_pairwise_similarities) if clean_mask[i]]
                std_of_means_clean = [s for i, s in enumerate(std_of_means) if clean_mask[i]]
                mean_of_stds_clean = [s for i, s in enumerate(mean_of_stds) if clean_mask[i]]
                cv_values_clean = [s for i, s in enumerate(cv_values) if clean_mask[i]]
                stability_scores_clean = [s for i, s in enumerate(stability_scores) if clean_mask[i]]
                mean_pairwise_similarities_clean = [s for i, s in enumerate(mean_pairwise_similarities) if clean_mask[i]]
                consistency_coefficients_clean = [s for i, s in enumerate(consistency_coefficients) if clean_mask[i]]
                stability_scores_new_clean = [s for i, s in enumerate(stability_scores_new) if clean_mask[i]]
                std_normalized_cv_clean = [s for i, s in enumerate(std_normalized_cv) if clean_mask[i]]
                std_normalized_relative_clean = [s for i, s in enumerate(std_normalized_relative) if clean_mask[i]]
            else:
                # No outliers detected, use original data
                temperatures_clean = temperatures
                std_pairwise_similarities_clean = std_pairwise_similarities
                std_of_means_clean = std_of_means
                mean_of_stds_clean = mean_of_stds
                cv_values_clean = cv_values
                stability_scores_clean = stability_scores
                mean_pairwise_similarities_clean = mean_pairwise_similarities
                consistency_coefficients_clean = consistency_coefficients
                stability_scores_new_clean = stability_scores_new
                std_normalized_cv_clean = std_normalized_cv
                std_normalized_relative_clean = std_normalized_relative
        else:
            # Use original data without outlier removal
            temperatures_clean = temperatures
            std_pairwise_similarities_clean = std_pairwise_similarities
            std_of_means_clean = std_of_means
            mean_of_stds_clean = mean_of_stds
            cv_values_clean = cv_values
            stability_scores_clean = stability_scores
            mean_pairwise_similarities_clean = mean_pairwise_similarities
            consistency_coefficients_clean = consistency_coefficients
            stability_scores_new_clean = stability_scores_new
            std_normalized_cv_clean = std_normalized_cv
            std_normalized_relative_clean = std_normalized_relative
            outlier_count = 0
        
        # Calculate correlations for different variability metrics using clean data
        correlations = {}
        
        # Temperature vs Standard deviation of pairwise similarities
        if len(std_pairwise_similarities_clean) > 1:
            pearson_std = stats.pearsonr(temperatures_clean, std_pairwise_similarities_clean)
            spearman_std = stats.spearmanr(temperatures_clean, std_pairwise_similarities_clean)
            correlations['temp_vs_std_pairwise_similarity'] = {
                'pearson': {'r': pearson_std[0], 'p': pearson_std[1]},
                'spearman': {'r': spearman_std[0], 'p': spearman_std[1]}
            }
        
        # Temperature vs Standard deviation of means (key metric)
        if len(std_of_means_clean) > 1:
            pearson_std_means = stats.pearsonr(temperatures_clean, std_of_means_clean)
            spearman_std_means = stats.spearmanr(temperatures_clean, std_of_means_clean)
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
        
        # Temperature vs Mean pairwise similarity (consistency)
        if len(mean_pairwise_similarities) > 1:
            pearson_mean = stats.pearsonr(temperatures, mean_pairwise_similarities)
            spearman_mean = stats.spearmanr(temperatures, mean_pairwise_similarities)
            correlations['temp_vs_mean_pairwise_similarity'] = {
                'pearson': {'r': pearson_mean[0], 'p': pearson_mean[1]},
                'spearman': {'r': spearman_mean[0], 'p': spearman_mean[1]}
            }
        
        # Temperature vs Consistency Coefficient (key new metric)
        if len(consistency_coefficients) > 1:
            pearson_cc = stats.pearsonr(temperatures, consistency_coefficients)
            spearman_cc = stats.spearmanr(temperatures, consistency_coefficients)
            correlations['temp_vs_consistency_coefficient'] = {
                'pearson': {'r': pearson_cc[0], 'p': pearson_cc[1]},
                'spearman': {'r': spearman_cc[0], 'p': spearman_cc[1]}
            }
        
        # Temperature vs New Stability Score
        if len(stability_scores_new) > 1:
            pearson_stability_new = stats.pearsonr(temperatures, stability_scores_new)
            spearman_stability_new = stats.spearmanr(temperatures, stability_scores_new)
            correlations['temp_vs_stability_score'] = {
                'pearson': {'r': pearson_stability_new[0], 'p': pearson_stability_new[1]},
                'spearman': {'r': spearman_stability_new[0], 'p': spearman_stability_new[1]}
            }
        
        # Temperature vs Normalized CV
        if len(std_normalized_cv) > 1:
            pearson_norm_cv = stats.pearsonr(temperatures, std_normalized_cv)
            spearman_norm_cv = stats.spearmanr(temperatures, std_normalized_cv)
            correlations['temp_vs_std_normalized_cv'] = {
                'pearson': {'r': pearson_norm_cv[0], 'p': pearson_norm_cv[1]},
                'spearman': {'r': spearman_norm_cv[0], 'p': spearman_norm_cv[1]}
            }
        
        # Temperature vs Normalized Relative Std
        if len(std_normalized_relative) > 1:
            pearson_norm_rel = stats.pearsonr(temperatures, std_normalized_relative)
            spearman_norm_rel = stats.spearmanr(temperatures, std_normalized_relative)
            correlations['temp_vs_std_normalized_relative'] = {
                'pearson': {'r': pearson_norm_rel[0], 'p': pearson_norm_rel[1]},
                'spearman': {'r': spearman_norm_rel[0], 'p': spearman_norm_rel[1]}
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
    
    # Create a comprehensive figure with subplots for each method (4 key metrics)
    fig, axes = plt.subplots(len(methods), 4, figsize=(20, 5*len(methods)))
    if len(methods) == 1:
        axes = axes.reshape(1, -1)
    
    for i, method in enumerate(methods):
        # 1. Temperature vs Consistency Coefficient (PRIMARY METRIC)
        ax1 = axes[i, 0]
        cc_col = f'{method}_consistency_coefficient'
        if cc_col in df.columns:
            sns.regplot(x='temperature', y=cc_col, data=df, 
                       scatter_kws={'alpha':0.7, 's':60}, line_kws={'color':'red', 'linewidth':2}, ax=ax1)
            ax1.set_title(f'{method.upper()}: Temperature vs Consistency Coefficient\n(PRIMARY METRIC)', fontweight='bold')
            ax1.set_xlabel('Temperature')
            ax1.set_ylabel('Consistency Coefficient')
            
            # Add correlation info if available
            if method in analysis and 'correlations' in analysis[method]:
                corr_data = analysis[method]['correlations'].get('temp_vs_consistency_coefficient', {})
                if 'pearson' in corr_data:
                    r = corr_data['pearson']['r']
                    p = corr_data['pearson']['p']
                    ax1.annotate(f"Pearson r = {r:.3f}\np = {p:.4f}", 
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", alpha=0.8),
                                verticalalignment='top', fontweight='bold')
        
        # 2. Temperature vs Standard Deviation of Means
        ax2 = axes[i, 1]
        std_of_means_col = f'{method}_std_of_means'
        if std_of_means_col in df.columns:
            sns.regplot(x='temperature', y=std_of_means_col, data=df, 
                       scatter_kws={'alpha':0.7}, line_kws={'color':'blue'}, ax=ax2)
            ax2.set_title(f'{method.upper()}: Temperature vs Std of Means')
            ax2.set_xlabel('Temperature')
            ax2.set_ylabel('Standard Deviation of Means')
            
            # Add correlation info
            if method in analysis and 'correlations' in analysis[method]:
                corr_data = analysis[method]['correlations'].get('temp_vs_std_of_means', {})
                if 'pearson' in corr_data:
                    r = corr_data['pearson']['r']
                    p = corr_data['pearson']['p']
                    ax2.annotate(f"r = {r:.3f}, p = {p:.4f}", 
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
                                verticalalignment='top')
        
        # 3. Temperature vs Stability Score
        ax3 = axes[i, 2]
        stability_col = f'{method}_stability_score'
        if stability_col in df.columns:
            sns.regplot(x='temperature', y=stability_col, data=df, 
                       scatter_kws={'alpha':0.7}, line_kws={'color':'green'}, ax=ax3)
            ax3.set_title(f'{method.upper()}: Temperature vs Stability Score')
            ax3.set_xlabel('Temperature')
            ax3.set_ylabel('Stability Score')
            
            # Add correlation info
            if method in analysis and 'correlations' in analysis[method]:
                corr_data = analysis[method]['correlations'].get('temp_vs_stability_score', {})
                if 'pearson' in corr_data:
                    r = corr_data['pearson']['r']
                    p = corr_data['pearson']['p']
                    ax3.annotate(f"r = {r:.3f}, p = {p:.4f}", 
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
                                verticalalignment='top')
        
        # 4. Temperature vs Normalized Relative Std
        ax4 = axes[i, 3]
        norm_std_col = f'{method}_std_normalized_relative'
        if norm_std_col in df.columns:
            sns.regplot(x='temperature', y=norm_std_col, data=df, 
                       scatter_kws={'alpha':0.7}, line_kws={'color':'purple'}, ax=ax4)
            ax4.set_title(f'{method.upper()}: Temperature vs Normalized Std')
            ax4.set_xlabel('Temperature')
            ax4.set_ylabel('Normalized Relative Std')
            
            # Add correlation info
            if method in analysis and 'correlations' in analysis[method]:
                corr_data = analysis[method]['correlations'].get('temp_vs_std_normalized_relative', {})
                if 'pearson' in corr_data:
                    r = corr_data['pearson']['r']
                    p = corr_data['pearson']['p']
                    ax4.annotate(f"r = {r:.3f}, p = {p:.4f}", 
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.8),
                                verticalalignment='top')

    
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
    
    # Create quantile analysis for consistency evaluation
    create_quantile_analysis(results, output_dir, methods)
    
    # Create a focused plot for Consistency Coefficient across all methods
    plt.figure(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        cc_col = f'{method}_consistency_coefficient'
        if cc_col in df.columns:
            plt.plot(df['temperature'], df[cc_col], 'o-', 
                    color=colors[i % len(colors)], label=f'{method.upper()}', 
                    linewidth=3, markersize=8)
    
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Consistency Coefficient', fontsize=12)
    plt.title('Temperature vs Consistency Coefficient (PRIMARY METRIC)\nComparison Across Methods', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_consistency_coefficient_comparison.png'), dpi=300)
    plt.close()
    
    # Create a dual-axis plot showing pairwise similarity vs std of means
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        mean_col = f'{method}_mean_pairwise_similarity'
        std_col = f'{method}_std_of_means'
        
        if mean_col in df.columns and std_col in df.columns:
            color = colors[i % len(colors)]
            
            # Plot mean pairwise similarity on left axis
            ax1.plot(df['temperature'], df[mean_col], 'o-', color=color, 
                    label=f'{method.upper()} Pairwise Sim', linewidth=2, markersize=6)
    
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

def create_quantile_analysis(results: List[Dict[str, Any]], output_dir: str, methods: List[str] = ["ted", "bertscore", "deepdiff"]):
    """
    Create quantile plots for consistency analysis instead of correlation matrices.
    
    Args:
        results: List of dictionaries with temperature and metrics
        output_dir: Directory to save visualizations
        methods: List of similarity methods used
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    temperatures = sorted(df['temperature'].unique())
    
    for method in methods:
        # Create a comprehensive quantile analysis figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{method.upper()}: Quantile Analysis for Consistency Evaluation', fontsize=16)
        
        # 1. Box plots of normalized std across temperatures
        ax1 = axes[0, 0]
        std_col = f'{method}_std_of_means'
        if std_col in df.columns:
            # Prepare data for box plot
            box_data = []
            temp_labels = []
            for temp in temperatures:
                temp_data = df[df['temperature'] == temp][std_col].values
                if len(temp_data) > 0:
                    box_data.append(temp_data)
                    temp_labels.append(f'{temp:.1f}')
            
            if box_data:
                bp = ax1.boxplot(box_data, labels=temp_labels, patch_artist=True)
                # Color boxes with a gradient
                colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax1.set_title('Distribution of Std of Means Across Temperatures')
                ax1.set_xlabel('Temperature')
                ax1.set_ylabel('Standard Deviation of Means')
                ax1.grid(True, alpha=0.3)
        
        # 2. Quantile evolution plot
        ax2 = axes[0, 1]
        if std_col in df.columns:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            quantile_data = {q: [] for q in quantiles}
            
            for temp in temperatures:
                temp_data = df[df['temperature'] == temp][std_col].values
                if len(temp_data) > 0:
                    for q in quantiles:
                        quantile_data[q].append(np.quantile(temp_data, q))
                else:
                    for q in quantiles:
                        quantile_data[q].append(np.nan)
            
            # Plot quantile lines
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i, q in enumerate(quantiles):
                ax2.plot(temperatures, quantile_data[q], 'o-', 
                        color=colors[i], label=f'{int(q*100)}th percentile', 
                        linewidth=2, markersize=4)
            
            ax2.set_title('Quantile Evolution Across Temperatures')
            ax2.set_xlabel('Temperature')
            ax2.set_ylabel('Standard Deviation of Means')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot comparing low vs high temperature
        ax3 = axes[1, 0]
        if std_col in df.columns and len(temperatures) >= 2:
            low_temp = temperatures[0]
            high_temp = temperatures[-1]
            
            low_data = df[df['temperature'] == low_temp][std_col].values
            high_data = df[df['temperature'] == high_temp][std_col].values
            
            if len(low_data) > 0 and len(high_data) > 0:
                # Create Q-Q plot
                min_len = min(len(low_data), len(high_data))
                if min_len > 1:
                    low_sorted = np.sort(low_data)[:min_len]
                    high_sorted = np.sort(high_data)[:min_len]
                    
                    ax3.scatter(low_sorted, high_sorted, alpha=0.7, s=50)
                    
                    # Add diagonal line for reference
                    min_val = min(np.min(low_sorted), np.min(high_sorted))
                    max_val = max(np.max(low_sorted), np.max(high_sorted))
                    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                    
                    ax3.set_title(f'Q-Q Plot: Temp {low_temp:.1f} vs Temp {high_temp:.1f}')
                    ax3.set_xlabel(f'Temperature {low_temp:.1f} Quantiles')
                    ax3.set_ylabel(f'Temperature {high_temp:.1f} Quantiles')
                    ax3.grid(True, alpha=0.3)
        
        # 4. Consistency score evolution
        ax4 = axes[1, 1]
        if std_col in df.columns:
            # Calculate consistency score (inverse of normalized std)
            consistency_scores = []
            for temp in temperatures:
                temp_data = df[df['temperature'] == temp][std_col].values
                if len(temp_data) > 0:
                    # Consistency = 1 / (1 + mean_std)
                    mean_std = np.mean(temp_data)
                    consistency = 1.0 / (1.0 + mean_std)
                    consistency_scores.append(consistency)
                else:
                    consistency_scores.append(np.nan)
            
            ax4.plot(temperatures, consistency_scores, 'o-', 
                    color='purple', linewidth=3, markersize=8)
            ax4.set_title('Consistency Score vs Temperature')
            ax4.set_xlabel('Temperature')
            ax4.set_ylabel('Consistency Score (Higher = More Consistent)')
            ax4.grid(True, alpha=0.3)
            
            # Add trend line
            if len(temperatures) > 1:
                z = np.polyfit(temperatures, consistency_scores, 1)
                p = np.poly1d(z)
                ax4.plot(temperatures, p(temperatures), "r--", alpha=0.8, 
                        label=f'Trend: slope={z[0]:.4f}')
                ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method}_quantile_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a summary quantile comparison across all methods
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Method Quantile Analysis Summary', fontsize=16)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # 1. Median consistency across methods
    ax1 = axes[0, 0]
    for i, method in enumerate(methods):
        std_col = f'{method}_std_of_means'
        if std_col in df.columns:
            medians = []
            for temp in temperatures:
                temp_data = df[df['temperature'] == temp][std_col].values
                if len(temp_data) > 0:
                    medians.append(np.median(temp_data))
                else:
                    medians.append(np.nan)
            
            ax1.plot(temperatures, medians, 'o-', 
                    color=colors[i % len(colors)], label=f'{method.upper()}', 
                    linewidth=2, markersize=6)
    
    ax1.set_title('Median Std of Means Across Methods')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Median Standard Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Interquartile range across methods
    ax2 = axes[0, 1]
    for i, method in enumerate(methods):
        std_col = f'{method}_std_of_means'
        if std_col in df.columns:
            iqrs = []
            for temp in temperatures:
                temp_data = df[df['temperature'] == temp][std_col].values
                if len(temp_data) > 0:
                    q75 = np.percentile(temp_data, 75)
                    q25 = np.percentile(temp_data, 25)
                    iqrs.append(q75 - q25)
                else:
                    iqrs.append(np.nan)
            
            ax2.plot(temperatures, iqrs, 'o-', 
                    color=colors[i % len(colors)], label=f'{method.upper()}', 
                    linewidth=2, markersize=6)
    
    ax2.set_title('Interquartile Range Across Methods')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('IQR of Standard Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Consistency ranking
    ax3 = axes[1, 0]
    consistency_rankings = {}
    for temp in temperatures:
        temp_rankings = []
        for method in methods:
            std_col = f'{method}_std_of_means'
            if std_col in df.columns:
                temp_data = df[df['temperature'] == temp][std_col].values
                if len(temp_data) > 0:
                    mean_std = np.mean(temp_data)
                    consistency = 1.0 / (1.0 + mean_std)
                    temp_rankings.append((method, consistency))
        
        # Sort by consistency (higher is better)
        temp_rankings.sort(key=lambda x: x[1], reverse=True)
        consistency_rankings[temp] = temp_rankings
    
    # Plot ranking evolution
    for i, method in enumerate(methods):
        rankings = []
        for temp in temperatures:
            if temp in consistency_rankings:
                for rank, (m, _) in enumerate(consistency_rankings[temp]):
                    if m == method:
                        rankings.append(rank + 1)  # 1-based ranking
                        break
                else:
                    rankings.append(len(methods))  # Worst rank if not found
            else:
                rankings.append(len(methods))
        
        ax3.plot(temperatures, rankings, 'o-', 
                color=colors[i % len(colors)], label=f'{method.upper()}', 
                linewidth=2, markersize=6)
    
    ax3.set_title('Consistency Ranking Across Methods')
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('Rank (1 = Most Consistent)')
    ax3.set_ylim(0.5, len(methods) + 0.5)
    ax3.invert_yaxis()  # Better rank at top
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Temperature sensitivity analysis
    ax4 = axes[1, 1]
    for i, method in enumerate(methods):
        std_col = f'{method}_std_of_means'
        if std_col in df.columns:
            temp_sensitivity = []
            baseline_std = None
            
            for temp in temperatures:
                temp_data = df[df['temperature'] == temp][std_col].values
                if len(temp_data) > 0:
                    mean_std = np.mean(temp_data)
                    if baseline_std is None:
                        baseline_std = mean_std
                        temp_sensitivity.append(0.0)  # Baseline
                    else:
                        # Relative change from baseline
                        sensitivity = (mean_std - baseline_std) / baseline_std if baseline_std > 0 else 0
                        temp_sensitivity.append(sensitivity)
                else:
                    temp_sensitivity.append(np.nan)
            
            ax4.plot(temperatures, temp_sensitivity, 'o-', 
                    color=colors[i % len(colors)], label=f'{method.upper()}', 
                    linewidth=2, markersize=6)
    
    ax4.set_title('Temperature Sensitivity (Relative to Baseline)')
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Relative Change in Std')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_method_quantile_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Quantile analysis plots saved to {output_dir}")

def interpret_correlation_strength(r: float, p_value: float) -> str:
    """
    Interpret correlation coefficient with practical meaning.
    
    Args:
        r: Pearson correlation coefficient
        p_value: Statistical significance p-value
        
    Returns:
        String interpretation of the correlation strength
    """
    if p_value > 0.05:
        return "Not statistically significant"
    
    abs_r = abs(r)
    if abs_r > 0.8:
        return "Very strong relationship"
    elif abs_r > 0.5:
        return "Strong relationship" 
    elif abs_r > 0.3:
        return "Moderate relationship"
    else:
        return "Weak relationship"

def print_correlation_analysis(analysis: Dict[str, Any], methods: List[str]):
    """
    Print detailed correlation analysis with interpretations.
    
    Args:
        analysis: Analysis results from analyze_temperature_std_correlation
        methods: List of similarity methods used
    """
    print("\n" + "="*80)
    print("TEMPERATURE-CONSISTENCY CORRELATION ANALYSIS")
    print("="*80)
    
    print("\nExpected: Higher Temperature → Higher Std → Less Consistency")
    print("Strong positive correlation (r > 0.7) confirms expected behavior\n")
    
    for method in methods:
        if method not in analysis:
            continue
            
        method_analysis = analysis[method]
        correlations = method_analysis.get('correlations', {})
        
        print(f"\n{method.upper()} METHOD:")
        print("-" * 40)
        
        # Temperature vs Std of Means (primary metric)
        if 'temp_vs_std_of_means' in correlations:
            corr_data = correlations['temp_vs_std_of_means']
            pearson_r = corr_data['pearson']['r']
            pearson_p = corr_data['pearson']['p']
            spearman_r = corr_data['spearman']['r']
            spearman_p = corr_data['spearman']['p']
            
            pearson_interp = interpret_correlation_strength(pearson_r, pearson_p)
            spearman_interp = interpret_correlation_strength(spearman_r, spearman_p)
            
            print(f"📊 Temperature vs Std of Means (Primary Consistency Metric):")
            print(f"   Pearson:  r = {pearson_r:+.3f}, p = {pearson_p:.4f} → {pearson_interp}")
            print(f"   Spearman: r = {spearman_r:+.3f}, p = {spearman_p:.4f} → {spearman_interp}")
            
            # Interpretation
            if pearson_p < 0.05:
                if pearson_r > 0.7:
                    print(f"   ✅ EXCELLENT: Strong positive correlation confirms expected behavior")
                    print(f"      → Temperature effectively controls {method} consistency")
                    print(f"      → Recommend temp < 0.4 for consistent outputs")
                elif pearson_r > 0.5:
                    print(f"   ✅ GOOD: Clear temperature-consistency relationship")
                    print(f"      → Temperature has meaningful effect on {method} consistency")
                elif pearson_r > 0.3:
                    print(f"   ⚠️  MODERATE: Some temperature effect, but other factors matter")
                elif pearson_r > 0:
                    print(f"   ⚠️  WEAK: Minimal temperature effect on consistency")
                else:
                    print(f"   ❌ UNEXPECTED: Negative correlation - investigate further")
            else:
                print(f"   ❌ NOT SIGNIFICANT: Relationship could be due to chance")
        
        # Temperature vs Overall Std
        if 'temp_vs_std_similarity' in correlations:
            corr_data = correlations['temp_vs_std_similarity']
            pearson_r = corr_data['pearson']['r']
            pearson_p = corr_data['pearson']['p']
            
            print(f"\n📈 Temperature vs Overall Std:")
            print(f"   Pearson: r = {pearson_r:+.3f}, p = {pearson_p:.4f} → {interpret_correlation_strength(pearson_r, pearson_p)}")
        
        # Temperature vs Coefficient of Variation
        if 'temp_vs_cv' in correlations:
            corr_data = correlations['temp_vs_cv']
            pearson_r = corr_data['pearson']['r']
            pearson_p = corr_data['pearson']['p']
            
            print(f"\n📊 Temperature vs Coefficient of Variation:")
            print(f"   Pearson: r = {pearson_r:+.3f}, p = {pearson_p:.4f} → {interpret_correlation_strength(pearson_r, pearson_p)}")
        
        # Temperature vs Consistency Coefficient (KEY NEW METRIC)
        if 'temp_vs_consistency_coefficient' in correlations:
            corr_data = correlations['temp_vs_consistency_coefficient']
            pearson_r = corr_data['pearson']['r']
            pearson_p = corr_data['pearson']['p']
            spearman_r = corr_data['spearman']['r']
            spearman_p = corr_data['spearman']['p']
            
            print(f"\n🎯 Temperature vs Consistency Coefficient (KEY METRIC):")
            print(f"   Pearson:  r = {pearson_r:+.3f}, p = {pearson_p:.4f} → {interpret_correlation_strength(pearson_r, pearson_p)}")
            print(f"   Spearman: r = {spearman_r:+.3f}, p = {spearman_p:.4f} → {interpret_correlation_strength(spearman_r, spearman_p)}")
            
            # Special interpretation for Consistency Coefficient
            if pearson_p < 0.05:
                if pearson_r < -0.7:
                    print(f"   ✅ EXCELLENT: Strong negative correlation confirms CC captures temperature effects")
                    print(f"      → Consistency Coefficient is highly sensitive to temperature changes")
                elif pearson_r < -0.5:
                    print(f"   ✅ GOOD: Clear negative correlation shows CC effectiveness")
                elif pearson_r < -0.3:
                    print(f"   ⚠️  MODERATE: Some temperature sensitivity in CC")
                else:
                    print(f"   ⚠️  WEAK: Limited temperature sensitivity in CC")
        
        # Temperature vs New Stability Score
        if 'temp_vs_stability_score' in correlations:
            corr_data = correlations['temp_vs_stability_score']
            pearson_r = corr_data['pearson']['r']
            pearson_p = corr_data['pearson']['p']
            
            print(f"\n📈 Temperature vs Stability Score:")
            print(f"   Pearson: r = {pearson_r:+.3f}, p = {pearson_p:.4f} → {interpret_correlation_strength(pearson_r, pearson_p)}")
        
        # Temperature vs Normalized CV
        if 'temp_vs_std_normalized_cv' in correlations:
            corr_data = correlations['temp_vs_std_normalized_cv']
            pearson_r = corr_data['pearson']['r']
            pearson_p = corr_data['pearson']['p']
            
            print(f"\n📊 Temperature vs Normalized CV:")
            print(f"   Pearson: r = {pearson_r:+.3f}, p = {pearson_p:.4f} → {interpret_correlation_strength(pearson_r, pearson_p)}")
        
        # Temperature vs Normalized Relative Std
        if 'temp_vs_std_normalized_relative' in correlations:
            corr_data = correlations['temp_vs_std_normalized_relative']
            pearson_r = corr_data['pearson']['r']
            pearson_p = corr_data['pearson']['p']
            
            print(f"\n📊 Temperature vs Normalized Relative Std:")
            print(f"   Pearson: r = {pearson_r:+.3f}, p = {pearson_p:.4f} → {interpret_correlation_strength(pearson_r, pearson_p)}")
        

        
        # Linear regression summary
        if 'linear_regression' in method_analysis:
            lr = method_analysis['linear_regression']
            if 'slope' in lr and 'r_squared' in lr:
                print(f"\n📈 Linear Regression:")
                print(f"   Equation: {lr.get('equation', 'N/A')}")
                print(f"   R² = {lr['r_squared']:.4f} ({lr['r_squared']*100:.1f}% variance explained)")
                print(f"   Slope = {lr['slope']:+.4f} (std change per temperature unit)")
    
    print("\n" + "="*80)
    print("SUMMARY RECOMMENDATIONS:")
    print("="*80)
    
    # Overall recommendations
    strong_methods = []
    moderate_methods = []
    weak_methods = []
    
    for method in methods:
        if method in analysis and 'correlations' in analysis[method]:
            corr_data = analysis[method]['correlations'].get('temp_vs_std_of_means', {})
            if 'pearson' in corr_data:
                r = corr_data['pearson']['r']
                p = corr_data['pearson']['p']
                
                if p < 0.05:
                    if r > 0.7:
                        strong_methods.append(method)
                    elif r > 0.3:
                        moderate_methods.append(method)
                    else:
                        weak_methods.append(method)
    
    if strong_methods:
        print(f"✅ STRONG temperature-consistency relationship: {', '.join(strong_methods).upper()}")
        print(f"   → These methods are highly sensitive to temperature changes")
        print(f"   → Use temp ≤ 0.3 for consistent outputs")
    
    if moderate_methods:
        print(f"⚠️  MODERATE temperature-consistency relationship: {', '.join(moderate_methods).upper()}")
        print(f"   → These methods show some temperature sensitivity")
        print(f"   → Consider temp ≤ 0.5 for reasonably consistent outputs")
    
    if weak_methods:
        print(f"❌ WEAK temperature-consistency relationship: {', '.join(weak_methods).upper()}")
        print(f"   → These methods may not be ideal for consistency evaluation")
        print(f"   → Consider other factors affecting consistency")
    
    print("\n💡 Use quantile plots to visualize these relationships and identify optimal temperature ranges!")

def run_experiment(args):
    """
    Run the temperature-standard deviation correlation experiment.
    
    Args:
        args: Command-line arguments
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # get the string of current time 
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generations_dir = os.path.join(args.output_dir, f"generations-{current_time}")
    visualizations_dir = os.path.join(args.output_dir, f"visualizations-{current_time}")
    os.makedirs(generations_dir, exist_ok=True)
    
    # Define temperatures to test
    if args.temperatures:
        temperatures = args.temperatures
    else:
        #temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        temperatures = np.arange(0.0, 1.0, 0.05)
    
    # Define similarity methods to use
    methods = ["ted", "bertscore", "deepdiff"]
    
    results = []
    
    # Run generation and evaluation for each temperature
    for temp in tqdm(temperatures, desc="Processing temperatures"):
        print(f"\n=== Processing Temperature {temp} ===")
        
        # Check if we already have generation results for this temperature and this model
        temp_str = f"temp_{temp:.3f}".replace('.', '_')
        existing_results = list(Path(generations_dir).glob(f"llm_gen_results_*{temp_str}*"))
        
        gen_results_file = None
                
        if existing_results:
            result_dir = sorted(existing_results, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            print(f"Found existing results for temperature {temp}. Using {result_dir}")
            gen_results_file = str(result_dir / "all_results.json")
            print(f"gen_results_file: {gen_results_file}")
            
        
        if gen_results_file is None or not os.path.exists(gen_results_file) or args.force_regenerate:
            # Run generation
            gen_results_file = run_generation(
                data_dir=args.data_dir,
                output_dir=generations_dir,
                temperature=temp,
                run_num=args.run_num,
                include_schema=args.include_schema,
                model_id=args.model_id,
                sample_limit=args.sample_limit,
                max_tokens=args.max_tokens
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
        
        # Print detailed correlation analysis
        print_correlation_analysis(analysis, methods)
        
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
    parser.add_argument("--exe-evaluation", action="store_true", help="Execute evaluation")
    parser.add_argument("--sample-limit", type=int, default=0, help="Limit the number of samples to process.")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Limit the number of max_tokens of LLM generation.")
    parser.add_argument("--remove-outliers", action="store_true", help="Remove outliers before correlation analysis")
    parser.add_argument("--outlier-method", type=str, default="iqr", choices=["iqr", "zscore"], help="Method for outlier detection")
    parser.add_argument("--outlier-threshold", type=float, default=1.5, help="Threshold for outlier detection")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()
