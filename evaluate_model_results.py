#!/usr/bin/env python
"""
Multi-Model Results Evaluation Script

This script evaluates existing model results from different models (e.g., Claude variants)
and creates comprehensive comparisons focusing on the three primary consistency metrics:
1. Consistency Coefficient - Combines accuracy and stability
2. Normalized CV - Scale-independent variability measure  
3. Stability Score - Intuitive 0-1 stability measure

Usage:
    python evaluate_model_results.py --results-dir ./temperature_experiment --output-dir ./model_comparison
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Import functions from existing modules
from calculate_similarity_stats import (
    load_generation_results, 
    compare_with_multiple_generations,
    save_comparison_summary
)


def extract_temperature_metrics(similarity_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract essential metrics focusing on the three primary metrics.
    
    Args:
        similarity_results: Results from calculate_similarity_metrics
        
    Returns:
        Dictionary with essential metrics only
    """
    metrics = {}
    
    for method in similarity_results:
        method_results = similarity_results[method]
        overall_stats = method_results['overall_stats']
        sample_level_stats = method_results['sample_level_stats']
        
        # === PRIMARY METRICS ONLY ===
        metrics[f'{method}_consistency_coefficient'] = overall_stats.get('consistency_coefficient', 0.0)
        metrics[f'{method}_std_normalized_cv'] = overall_stats.get('std_normalized_cv', 0.0)
        metrics[f'{method}_stability_score'] = overall_stats.get('stability_score', 0.0)
        
        # === MINIMAL SUPPORTING METRICS ===
        metrics[f'{method}_mean_pairwise_similarity'] = overall_stats['mean']
        metrics[f'{method}_mean_of_stds'] = sample_level_stats['mean_of_stds']
    
    return metrics


def calculate_similarity_metrics(generation_file: str, methods: List[str] = ["ted", "bertscore", "deepdiff"], 
                                output_dir: str = ".", embedding_model: str = "amazon.titan-embed-text-v2:0") -> Dict[str, Any]:
    """
    Calculate similarity metrics for generated outputs.
    
    Args:
        generation_file: Path to the generation results file
        methods: List of similarity methods to use
        output_dir: Directory to save intermediate results
        embedding_model: Embedding model ID
        
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


def detect_outliers(data, method='iqr', threshold=1.5):
    """
    Detect outliers in data using IQR or Z-score method.
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


def analyze_temperature_correlation(results: List[Dict[str, Any]], methods: List[str] = ["ted", "bertscore", "deepdiff"], 
                                  remove_outliers: bool = True) -> Dict[str, Any]:
    """
    Analyze correlation between temperature and primary consistency metrics.
    
    Args:
        results: List of dictionaries with temperature and metrics
        methods: List of similarity methods used
        remove_outliers: Whether to remove outliers before correlation analysis
        
    Returns:
        Dictionary with correlation analysis results
    """
    temperatures = [r['temperature'] for r in results]
    analysis_results = {}
    
    for method in methods:
        # Extract primary metrics
        consistency_coefficients = [r.get(f'{method}_consistency_coefficient', 0.0) for r in results]
        std_normalized_cv = [r.get(f'{method}_std_normalized_cv', 0.0) for r in results]
        stability_scores = [r.get(f'{method}_stability_score', 0.0) for r in results]
        mean_of_stds = [r.get(f'{method}_mean_of_stds', 0.0) for r in results]
        
        # Outlier detection and removal if requested
        if remove_outliers:
            cc_outliers = detect_outliers(consistency_coefficients, method='iqr', threshold=1.5)
            cv_outliers = detect_outliers(std_normalized_cv, method='iqr', threshold=1.5)
            stability_outliers = detect_outliers(stability_scores, method='iqr', threshold=1.5)
            
            combined_outliers = cc_outliers | cv_outliers | stability_outliers
            outlier_count = np.sum(combined_outliers)
            
            if outlier_count > 0:
                print(f"⚠️  {method.upper()}: Detected {outlier_count} outliers, removing from correlation analysis")
                
                clean_mask = ~combined_outliers
                temperatures_clean = [t for i, t in enumerate(temperatures) if clean_mask[i]]
                consistency_coefficients_clean = [s for i, s in enumerate(consistency_coefficients) if clean_mask[i]]
                std_normalized_cv_clean = [s for i, s in enumerate(std_normalized_cv) if clean_mask[i]]
                stability_scores_clean = [s for i, s in enumerate(stability_scores) if clean_mask[i]]
                mean_of_stds_clean = [s for i, s in enumerate(mean_of_stds) if clean_mask[i]]
            else:
                temperatures_clean = temperatures
                consistency_coefficients_clean = consistency_coefficients
                std_normalized_cv_clean = std_normalized_cv
                stability_scores_clean = stability_scores
                mean_of_stds_clean = mean_of_stds
        else:
            temperatures_clean = temperatures
            consistency_coefficients_clean = consistency_coefficients
            std_normalized_cv_clean = std_normalized_cv
            stability_scores_clean = stability_scores
            mean_of_stds_clean = mean_of_stds
        
        # Calculate correlations for PRIMARY METRICS ONLY
        correlations = {}
        
        # 1. Temperature vs Consistency Coefficient
        if len(consistency_coefficients_clean) > 1:
            pearson_cc = stats.pearsonr(temperatures_clean, consistency_coefficients_clean)
            correlations['temp_vs_consistency_coefficient'] = {
                'pearson': {'r': pearson_cc[0], 'p': pearson_cc[1]}
            }
        
        # 2. Temperature vs Normalized CV
        if len(std_normalized_cv_clean) > 1:
            pearson_norm_cv = stats.pearsonr(temperatures_clean, std_normalized_cv_clean)
            correlations['temp_vs_std_normalized_cv'] = {
                'pearson': {'r': pearson_norm_cv[0], 'p': pearson_norm_cv[1]}
            }
        
        # 3. Temperature vs Stability Score
        if len(stability_scores_clean) > 1:
            pearson_stability = stats.pearsonr(temperatures_clean, stability_scores_clean)
            correlations['temp_vs_stability_score'] = {
                'pearson': {'r': pearson_stability[0], 'p': pearson_stability[1]}
            }
        
        # Keep mean_of_stds for backward compatibility
        if len(mean_of_stds_clean) > 1:
            pearson_mean_stds = stats.pearsonr(temperatures_clean, mean_of_stds_clean)
            correlations['temp_vs_mean_of_stds'] = {
                'pearson': {'r': pearson_mean_stds[0], 'p': pearson_mean_stds[1]}
            }
        
        analysis_results[method] = {
            'correlations': correlations,
            'outliers_removed': outlier_count if remove_outliers else 0
        }
    
    return analysis_results


def discover_model_results(results_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover model result folders and their temperature result files.
    
    Args:
        results_dir: Path to directory containing model result folders
        
    Returns:
        Dictionary mapping model names to lists of result file paths
    """
    model_results = {}
    
    # Find all model result folders
    model_folders = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('generations-')]
    
    for model_folder in model_folders:
        model_name = model_folder.name.replace('generations-', '')
        
        # Find all temperature result files in this model folder
        result_files = list(model_folder.glob("**/all_results.json"))
        
        if result_files:
            model_results[model_name] = sorted(result_files)
            print(f"📁 Found {len(result_files)} result files for model: {model_name}")
        else:
            print(f"⚠️  No result files found for model: {model_name}")
    
    return model_results


def process_model_results(model_results: Dict[str, List[Path]], methods: List[str], 
                         embedding_model: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all model results and extract metrics.
    
    Args:
        model_results: Dictionary mapping model names to result file paths
        methods: List of similarity methods to use
        embedding_model: Embedding model ID
        
    Returns:
        Dictionary mapping model names to lists of temperature results
    """
    all_model_data = {}
    
    for model_name, result_files in model_results.items():
        print(f"\n{'='*80}")
        print(f"🤖 PROCESSING MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        model_data = []
        
        for result_file in tqdm(result_files, desc=f"Processing {model_name}"):
            # Extract temperature from path
            temp_match = re.search(r'temp_(\d+)_(\d+)', str(result_file))
            if temp_match:
                temp = float(f"{temp_match.group(1)}.{temp_match.group(2)}")
            else:
                print(f"⚠️  Could not extract temperature from {result_file}")
                continue
            
            if temp < 0.9:
                continue
            
            try:
                # Calculate similarity metrics
                similarity_results = calculate_similarity_metrics(
                    str(result_file), methods, str(result_file.parent), embedding_model
                )
                
                # Save comparison summary for this temperature
                temp_result_dir = result_file.parent / f"temp_{temp:.2f}_analysis"
                save_comparison_summary(similarity_results, temp_result_dir)
                
                # Extract temperature-focused metrics
                metrics = extract_temperature_metrics(similarity_results)
                metrics['temperature'] = temp
                metrics['model'] = model_name
                model_data.append(metrics)
                
            except Exception as e:
                print(f"❌ Error processing {result_file}: {e}")
                continue
        
        if model_data:
            all_model_data[model_name] = model_data
            print(f"✅ Model {model_name} completed: {len(model_data)} temperatures processed")
        else:
            print(f"❌ No valid results for model {model_name}")
    
    return all_model_data


def create_model_comparison_plots(df: pd.DataFrame, output_dir: str, methods: List[str]):
    """
    Create visualizations comparing different models across primary metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    sns.set(style="whitegrid")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create comparison plots for each primary metric
    metrics_info = [
        ('consistency_coefficient', 'Consistency Coefficient', 'Higher = Better'),
        ('std_normalized_cv', 'Normalized CV', 'Lower = More Stable'),
        ('stability_score', 'Stability Score', 'Higher = More Stable')
    ]
    
    for metric_suffix, metric_name, interpretation in metrics_info:
        fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]
        
        fig.suptitle(f'Model Comparison: {metric_name} vs Temperature\n({interpretation})', 
                    fontsize=16, fontweight='bold')
        
        for method_idx, method in enumerate(methods):
            ax = axes[method_idx]
            metric_col = f'{method}_{metric_suffix}'
            
            if metric_col in df.columns:
                # Plot each model with different colors
                for model_idx, model in enumerate(df['model'].unique()):
                    model_data = df[df['model'] == model]
                    label = model.replace('claude-', '').replace('3-', '3 ')
                    ax.plot(model_data['temperature'], model_data[metric_col], 
                           'o-', color=colors[model_idx % len(colors)], 
                           label=label, linewidth=2, markersize=6, alpha=0.8)
                
                ax.set_title(f'{method.upper()}: {metric_name}', fontweight='bold')
                ax.set_xlabel('Temperature')
                ax.set_ylabel(metric_name)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'model_comparison_{metric_suffix}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create comprehensive overview plot
    create_comprehensive_overview(df, output_dir, methods)


def create_comprehensive_overview(df: pd.DataFrame, output_dir: str, methods: List[str]):
    """
    Create a comprehensive overview plot showing all models and metrics.
    """
    fig, axes = plt.subplots(len(methods), 3, figsize=(18, 6*len(methods)))
    if len(methods) == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    metrics = [
        ('consistency_coefficient', 'Consistency Coefficient'),
        ('std_normalized_cv', 'Normalized CV'),
        ('stability_score', 'Stability Score')
    ]
    
    for method_idx, method in enumerate(methods):
        for metric_idx, (metric_suffix, metric_name) in enumerate(metrics):
            ax = axes[method_idx, metric_idx]
            metric_col = f'{method}_{metric_suffix}'
            
            if metric_col in df.columns:
                for model_idx, model in enumerate(df['model'].unique()):
                    model_data = df[df['model'] == model]
                    label = model.replace('claude-', '').replace('3-', '3 ')
                    ax.plot(model_data['temperature'], model_data[metric_col], 
                           'o-', color=colors[model_idx % len(colors)], 
                           label=label, linewidth=2, markersize=4, alpha=0.8)
                
                ax.set_title(f'{method.upper()}: {metric_name}', fontweight='bold')
                ax.set_xlabel('Temperature')
                ax.set_ylabel(metric_name)
                if method_idx == 0 and metric_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Model Comparison: All Methods and Primary Metrics', 
                fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_model_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def save_model_comparison_summary(model_correlations: Dict, df: pd.DataFrame, output_dir: str, methods: List[str]):
    """
    Save comprehensive model comparison summary to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text summary
    summary_file = os.path.join(output_dir, "model_comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("🤖 MULTI-MODEL COMPARISON SUMMARY\n")
        f.write("="*120 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Compared: {', '.join(df['model'].unique())}\n")
        f.write(f"Methods Used: {', '.join(methods).upper()}\n")
        f.write(f"Temperature Range: {df['temperature'].min():.2f} - {df['temperature'].max():.2f}\n")
        f.write(f"Total Data Points: {len(df)} temperature-model combinations\n\n")
        
        f.write("🎯 PRIMARY METRICS FOCUS:\n")
        f.write("• Consistency Coefficient: Combines accuracy and stability (Higher = Better)\n")
        f.write("• Normalized CV: Scale-independent variability (Lower = More Stable)\n")
        f.write("• Stability Score: Intuitive 0-1 stability measure (Higher = More Stable)\n\n")
        
        # Primary metrics comparison for each method
        for method in methods:
            f.write(f"\n{'='*80}\n")
            f.write(f"📊 {method.upper()} METHOD - TEMPERATURE CORRELATIONS\n")
            f.write(f"{'='*80}\n")
            
            f.write(f"{'Model':<25} {'Consistency Coeff':<20} {'Normalized CV':<20} {'Stability Score':<20}\n")
            f.write(f"{'':25} {'(r, p-value)':<20} {'(r, p-value)':<20} {'(r, p-value)':<20}\n")
            f.write("-" * 90 + "\n")
            
            for model in sorted(df['model'].unique()):
                if model in model_correlations:
                    correlations = model_correlations[model].get(method, {}).get('correlations', {})
                    
                    cc_corr = correlations.get('temp_vs_consistency_coefficient', {}).get('pearson', {'r': 0.0, 'p': 1.0})
                    cv_corr = correlations.get('temp_vs_std_normalized_cv', {}).get('pearson', {'r': 0.0, 'p': 1.0})
                    stability_corr = correlations.get('temp_vs_stability_score', {}).get('pearson', {'r': 0.0, 'p': 1.0})
                    
                    model_display = model.replace('claude-', '').replace('3-', '3 ')
                    f.write(f"{model_display:<25} ({cc_corr['r']:+.3f}, {cc_corr['p']:.3f}){'':<8} "
                           f"({cv_corr['r']:+.3f}, {cv_corr['p']:.3f}){'':<8} "
                           f"({stability_corr['r']:+.3f}, {stability_corr['p']:.3f})\n")
        
        # Best performing models summary
        f.write(f"\n{'='*80}\n")
        f.write(f"🏆 BEST PERFORMING MODELS BY METRIC\n")
        f.write(f"{'='*80}\n")
        
        for method in methods:
            f.write(f"\n{method.upper()} Method:\n")
            
            # Find best model for each primary metric
            best_models = find_best_models(model_correlations, method, df['model'].unique())
            
            for metric_name, (model, r_value) in best_models.items():
                if model:
                    model_display = model.replace('claude-', '').replace('3-', '3 ')
                    f.write(f"  • {metric_name}: {model_display} (r = {r_value:.3f})\n")
        
        # Recommendations
        f.write(f"\n{'='*80}\n")
        f.write(f"💡 RECOMMENDATIONS\n")
        f.write(f"{'='*80}\n")
        
        # Find overall best model
        overall_best = find_overall_best_model(model_correlations, methods, df['model'].unique())
        if overall_best:
            f.write(f"🥇 Overall Best Model: {overall_best.replace('claude-', '').replace('3-', '3 ')}\n")
            f.write(f"   → Most consistent temperature-stability correlations across metrics\n\n")
        
        f.write("📈 Temperature Recommendations:\n")
        f.write("• For maximum consistency: Use temperature ≤ 0.3\n")
        f.write("• For balanced creativity/consistency: Use temperature 0.5-0.7\n")
        f.write("• Monitor Consistency Coefficient as primary metric\n\n")
        
        f.write("="*120 + "\n")
    
    # Save JSON summary
    json_file = os.path.join(output_dir, "model_comparison_summary.json")
    json_data = {
        'experiment_type': 'multi_model_comparison',
        'timestamp': datetime.now().isoformat(),
        'models_compared': list(df['model'].unique()),
        'methods_used': methods,
        'temperature_range': {'min': float(df['temperature'].min()), 'max': float(df['temperature'].max())},
        'total_data_points': len(df),
        'correlations_by_model': {}
    }
    
    for model in df['model'].unique():
        if model in model_correlations:
            json_data['correlations_by_model'][model] = {}
            for method in methods:
                if method in model_correlations[model]:
                    correlations = model_correlations[model][method].get('correlations', {})
                    json_data['correlations_by_model'][model][method] = {
                        'consistency_coefficient': correlations.get('temp_vs_consistency_coefficient', {}),
                        'normalized_cv': correlations.get('temp_vs_std_normalized_cv', {}),
                        'stability_score': correlations.get('temp_vs_stability_score', {})
                    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"📄 Model comparison summary saved to: {summary_file}")
    print(f"📊 Model comparison JSON saved to: {json_file}")


def find_best_models(model_correlations: Dict, method: str, models: List[str]) -> Dict[str, tuple]:
    """
    Find the best performing model for each primary metric.
    """
    best_models = {
        'Consistency Coefficient': (None, float('inf')),
        'Normalized CV': (None, float('-inf')),
        'Stability Score': (None, float('inf'))
    }
    
    for model in models:
        if model in model_correlations and method in model_correlations[model]:
            correlations = model_correlations[model][method].get('correlations', {})
            
            # For Consistency Coefficient, we want strongest negative correlation
            cc_r = correlations.get('temp_vs_consistency_coefficient', {}).get('pearson', {'r': 0.0})['r']
            if cc_r < best_models['Consistency Coefficient'][1]:
                best_models['Consistency Coefficient'] = (model, cc_r)
            
            # For Normalized CV, we want strongest positive correlation
            cv_r = correlations.get('temp_vs_std_normalized_cv', {}).get('pearson', {'r': 0.0})['r']
            if cv_r > best_models['Normalized CV'][1]:
                best_models['Normalized CV'] = (model, cv_r)
            
            # For Stability Score, we want strongest negative correlation
            stability_r = correlations.get('temp_vs_stability_score', {}).get('pearson', {'r': 0.0})['r']
            if stability_r < best_models['Stability Score'][1]:
                best_models['Stability Score'] = (model, stability_r)
    
    return best_models


def find_overall_best_model(model_correlations: Dict, methods: List[str], models: List[str]) -> str:
    """
    Find the overall best performing model across all metrics and methods.
    """
    model_scores = {}
    
    for model in models:
        total_score = 0
        valid_correlations = 0
        
        for method in methods:
            if model in model_correlations and method in model_correlations[model]:
                correlations = model_correlations[model][method].get('correlations', {})
                
                # Score based on correlation strength (absolute value)
                cc_r = abs(correlations.get('temp_vs_consistency_coefficient', {}).get('pearson', {'r': 0.0})['r'])
                cv_r = abs(correlations.get('temp_vs_std_normalized_cv', {}).get('pearson', {'r': 0.0})['r'])
                stability_r = abs(correlations.get('temp_vs_stability_score', {}).get('pearson', {'r': 0.0})['r'])
                
                total_score += cc_r + cv_r + stability_r
                valid_correlations += 3
        
        if valid_correlations > 0:
            model_scores[model] = total_score / valid_correlations
    
    if model_scores:
        return max(model_scores, key=model_scores.get)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing model results and create comprehensive comparisons.")
    parser.add_argument("--results-dir", type=str, required=True, 
                       help="Directory containing model result folders (e.g., generations-claude-3-5-sonnet-v2)")
    parser.add_argument("--output-dir", type=str, default="./model_comparison_results", 
                       help="Directory to save comparison results")
    parser.add_argument("--methods", nargs='+', default=["ted", "bertscore", "deepdiff"], 
                       help="Similarity methods to use")
    parser.add_argument("--embedding-model", type=str, default="amazon.titan-embed-text-v2:0", 
                       help="Embedding model ID")
    parser.add_argument("--remove-outliers", action="store_true", 
                       help="Remove outliers before correlation analysis")
    
    args = parser.parse_args()
    
    # Validate input directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1
    
    print(f"🚀 Starting multi-model evaluation...")
    print(f"📁 Results directory: {results_dir}")
    print(f"📊 Methods: {', '.join(args.methods)}")
    print(f"🤖 Embedding model: {args.embedding_model}")
    
    # Discover model results
    model_results = discover_model_results(results_dir)
    if not model_results:
        print("❌ No model result folders found")
        return 1
    
    # Process all model results
    all_model_data = process_model_results(model_results, args.methods, args.embedding_model)
    if not all_model_data:
        print("❌ No valid results found for any model")
        return 1
    
    # Create output directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"model_comparison_{current_time}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all results into DataFrame
    combined_data = []
    for model_name, model_data in all_model_data.items():
        combined_data.extend(model_data)
    
    df = pd.DataFrame(combined_data)
    
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   • Models processed: {len(all_model_data)}")
    print(f"   • Total data points: {len(df)}")
    print(f"   • Temperature range: {df['temperature'].min():.2f} - {df['temperature'].max():.2f}")
    
    # Analyze correlations for each model
    print(f"\n🔍 ANALYZING CORRELATIONS...")
    model_correlations = {}
    for model_name, model_data in all_model_data.items():
        print(f"   Analyzing {model_name}...")
        analysis = analyze_temperature_correlation(model_data, args.methods, args.remove_outliers)
        model_correlations[model_name] = analysis
    
    # Create visualizations
    print(f"\n📈 CREATING VISUALIZATIONS...")
    create_model_comparison_plots(df, output_dir, args.methods)
    
    # Save comprehensive summary
    print(f"\n💾 SAVING RESULTS...")
    save_model_comparison_summary(model_correlations, df, output_dir, args.methods)
    
    print(f"\n🎉 Multi-model evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n📋 Key files generated:")
    print(f"   • model_comparison_summary.txt - Main analysis report")
    print(f"   • model_comparison_summary.json - Programmatic access")
    print(f"   • model_comparison_*.png - Visualization plots")
    print(f"   • comprehensive_model_comparison.png - Overview plot")
    
    return 0


if __name__ == "__main__":
    exit(main())