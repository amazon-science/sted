#!/usr/bin/env python
"""
Model Comparison Experiment

This script runs a comprehensive experiment to compare different models using
tree-based semantic evaluation with stability consideration.

Usage:
    python run_model_comparison.py --data-dir extracted_sharegpt_data --output-dir ./model_comparison
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

# Define available models
AVAILABLE_MODELS = {
    "claude-3-5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-opus": "us.anthropic.claude-3-opus-20240229-v1:0",
    "claude-3-haiku": "us.anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-sonnet": "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-2": "us.anthropic.claude-2:1",
    "llama-3-70b": "meta.llama3-70b-instruct-v1:0",
    "llama-3-8b": "meta.llama3-8b-instruct-v1:0",
    "mistral-7b": "mistral.mistral-7b-instruct-v0:2"
}

def run_generation(data_dir: str, output_dir: str, model_id: str, temperature: float, run_num: int, include_schema: bool) -> str:
    """
    Run LLM generation with specified parameters.
    
    Args:
        data_dir: Directory containing the data files
        output_dir: Directory to save generation results
        model_id: Model ID to use
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
        "--model-id", model_id,
        "--temperature", str(temperature),
        "--run-num", str(run_num)
    ]
    
    if include_schema:
        cmd.append("--include-schema")
    
    print(f"Running generation with model {model_id}...")
    subprocess.run(cmd, check=True)
    
    # Find the most recent results directory for this model
    model_name = model_id.split('/')[-1].split(':')[0]
    result_dirs = list(Path(output_dir).glob(f"llm_gen_results_{model_name}*"))
    
    if not result_dirs:
        raise FileNotFoundError(f"No results found for model {model_id}")
    
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

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """
    Create visualizations comparing different models.
    
    Args:
        results: List of dictionaries with model and metrics
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # 1. Bar chart comparing semantic similarity across models
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='model_name', y='semantic_similarity_mean', data=df)
    plt.title('Semantic Similarity by Model')
    plt.xlabel('Model')
    plt.ylabel('Semantic Similarity (Accuracy)')
    plt.xticks(rotation=45, ha='right')
    
    # Add error bars
    for i, row in enumerate(df.itertuples()):
        ax.errorbar(i, row.semantic_similarity_mean, yerr=row.semantic_similarity_std, fmt='none', color='black', capsize=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_semantic_similarity.png'), dpi=300)
    
    # 2. Bar chart comparing stability across models
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_name', y='semantic_stability', data=df)
    plt.title('Semantic Stability by Model')
    plt.xlabel('Model')
    plt.ylabel('Semantic Stability')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_semantic_stability.png'), dpi=300)
    
    # 3. Scatter plot of accuracy vs. stability
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x='semantic_similarity_mean', y='semantic_stability', 
                             hue='model_name', size='cross_run_consistency',
                             sizes=(100, 400), data=df)
    
    # Add labels to points
    for i, row in enumerate(df.itertuples()):
        plt.annotate(row.model_name, 
                    (row.semantic_similarity_mean, row.semantic_stability),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Model Comparison: Accuracy vs. Stability')
    plt.xlabel('Semantic Similarity (Accuracy)')
    plt.ylabel('Semantic Stability')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_accuracy_vs_stability.png'), dpi=300)
    
    # 4. Heatmap of all metrics across models
    plt.figure(figsize=(12, 8))
    metrics_columns = ['semantic_similarity_mean', 'semantic_stability', 'cross_run_consistency', 
                      'bleu_mean', 'rouge_l_mean', 'bert_score_mean', 'jaccard_mean']
    
    # Normalize metrics for fair comparison
    normalized_df = df.copy()
    for col in metrics_columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    # Create heatmap
    heatmap_data = normalized_df.set_index('model_name')[metrics_columns]
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Normalized Metrics Across Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_metrics_heatmap.png'), dpi=300)
    
    # 5. Radar chart for top models
    if len(df) >= 3:
        top_models = df.nlargest(3, 'semantic_similarity_mean')
        
        # Prepare data for radar chart
        categories = metrics_columns
        N = len(categories)
        
        # Create angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one line per model and fill area
        for i, row in enumerate(top_models.itertuples()):
            values = [getattr(row, col) for col in metrics_columns]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row.model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Top 3 Models Comparison', size=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_models_radar.png'), dpi=300)
    
    print(f"Visualizations saved to {output_dir}")

def run_experiment(args):
    """
    Run the model comparison experiment.
    
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
    
    # Define models to test
    if args.models:
        models = args.models
    else:
        # Use a subset of available models by default
        models = ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku", "llama-3-70b"]
    
    # Validate models
    for model in models:
        if model not in AVAILABLE_MODELS:
            print(f"Warning: Model '{model}' not in available models list. Available models: {list(AVAILABLE_MODELS.keys())}")
            if args.strict:
                raise ValueError(f"Invalid model: {model}")
    
    results = []
    
    # Run generation and evaluation for each model
    for model_name in models:
        if model_name not in AVAILABLE_MODELS:
            print(f"Skipping unknown model: {model_name}")
            continue
            
        model_id = AVAILABLE_MODELS[model_name]
        
        try:
            # Run generation
            gen_results_file = run_generation(
                data_dir=args.data_dir,
                output_dir=generations_dir,
                model_id=model_id,
                temperature=args.temperature,
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
            metrics['model_name'] = model_name
            metrics['model_id'] = model_id
            results.append(metrics)
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            if args.strict:
                raise
    
    # Create visualizations
    create_visualizations(results, visualizations_dir)
    
    # Save results
    results_file = os.path.join(args.output_dir, "model_comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'parameters': {
                'models': models,
                'temperature': args.temperature,
                'run_num': args.run_num,
                'include_schema': args.include_schema,
                'data_dir': args.data_dir
            }
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Print summary of findings
    print("\n=== Model Comparison Summary ===")
    
    # Sort models by semantic similarity (accuracy)
    accuracy_ranking = sorted(results, key=lambda x: x['semantic_similarity_mean'], reverse=True)
    print("\nModels ranked by accuracy (semantic similarity):")
    for i, model in enumerate(accuracy_ranking):
        print(f"{i+1}. {model['model_name']}: {model['semantic_similarity_mean']:.4f}")
    
    # Sort models by stability
    stability_ranking = sorted(results, key=lambda x: x['semantic_stability'], reverse=True)
    print("\nModels ranked by stability:")
    for i, model in enumerate(stability_ranking):
        print(f"{i+1}. {model['model_name']}: {model['semantic_stability']:.4f}")
    
    # Calculate combined score (accuracy * stability)
    for model in results:
        model['combined_score'] = model['semantic_similarity_mean'] * model['semantic_stability']
    
    combined_ranking = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    print("\nModels ranked by combined score (accuracy * stability):")
    for i, model in enumerate(combined_ranking):
        print(f"{i+1}. {model['model_name']}: {model['combined_score']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Run model comparison experiment.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the data files.")
    parser.add_argument("--output-dir", type=str, default="./model_comparison", help="Directory to save experiment results.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Fixed temperature to use for all models.")
    parser.add_argument("--run-num", type=int, default=10, help="Number of runs per model.")
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt.")
    parser.add_argument("--models", type=str, nargs="+", help="List of models to test. Default: claude-3-5-sonnet, claude-3-opus, claude-3-haiku, llama-3-70b")
    parser.add_argument("--strict", action="store_true", help="Fail if any model encounters an error.")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()