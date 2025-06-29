#!/usr/bin/env python3
"""
Model Comparison for Semantic Tree Consistency

This script runs the LLM generation with different models at a fixed temperature
and evaluates the consistency using the Semantic Tree Consistency framework.
"""

import os
import json
import subprocess
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from semantic_json_tree_consistency import evaluate_semantic_json_consistency

def run_generation_with_models(data_dir, output_dir, models, temperature=0.7, run_num=10, sample_limit=5):
    """Run LLM generation with different models at a fixed temperature."""
    results = {}
    
    for model_id in models:
        print(f"\n{'='*80}")
        print(f"RUNNING GENERATION WITH MODEL {model_id}")
        print(f"{'='*80}")
        
        # Extract model name for display
        model_name = model_id.split('/')[-1].split(':')[0]
        
        # Run the llm_gen.py script with the specified model
        cmd = [
            "python", "llm_gen.py",
            "--data-dir", data_dir,
            "--output-dir", output_dir,
            "--model-id", model_id,
            "--temperature", str(temperature),
            "--run-num", str(run_num),
            "--sample-limit", str(sample_limit)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Find the most recent output directory for this model
            model_dirs = []
            for dir_name in os.listdir(output_dir):
                if dir_name.startswith(f"llm_gen_results_{model_name}_") and f"temp_{str(temperature).replace('.', '_')}" in dir_name:
                    dir_path = os.path.join(output_dir, dir_name)
                    if os.path.isdir(dir_path):
                        model_dirs.append((dir_path, os.path.getmtime(dir_path)))
            
            if not model_dirs:
                print(f"Warning: No output directory found for model {model_id}")
                continue
                
            # Get the most recent directory
            most_recent_dir = sorted(model_dirs, key=lambda x: x[1], reverse=True)[0][0]
            results[model_id] = most_recent_dir
            print(f"Results for model {model_id} saved in {most_recent_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running generation with model {model_id}: {e}")
    
    return results

def evaluate_consistency(result_dirs):
    """Evaluate consistency for each model."""
    consistency_results = {}
    empty_response_rates = {}
    
    for model_id, result_dir in result_dirs.items():
        all_results_path = os.path.join(result_dir, "all_results.json")
        
        if not os.path.exists(all_results_path):
            print(f"Warning: Results file not found for model {model_id}: {all_results_path}")
            continue
            
        with open(all_results_path, 'r') as f:
            data = json.load(f)
        
        # Extract model name for display
        model_name = model_id.split('/')[-1].split(':')[0]
        
        # Evaluate consistency for each sample
        sample_consistencies = []
        sample_consistencies_standard = []
        
        for sample in data["results"]:
            responses = sample["responses"]
            
            # Handle empty or invalid responses
            if not responses:
                print(f"Warning: Empty responses for sample {sample['sample_id']}")
                continue
                
            # Check for empty dictionaries (failed generations)
            empty_count = sum(1 for r in responses if isinstance(r, dict) and not r)
            
            # If all responses are empty, assign a very low consistency score
            if empty_count == len(responses):
                print(f"Warning: All responses are empty for sample {sample['sample_id']}")
                sample_consistencies.append(0.0)  # Penalize with lowest possible score
                sample_consistencies_standard.append(0.0)  # Penalize with lowest possible score
                
                # Store penalty metrics
                sample["consistency_analysis"] = {
                    "with_semantic": {
                        "mean_similarity": 0.0,
                        "std_deviation": 0.0,
                        "consistency_coefficient": 0.0,
                        "min_similarity": 0.0,
                        "max_similarity": 0.0,
                        "similarity_range": 0.0,
                        "empty_responses": empty_count,
                        "total_responses": len(responses)
                    },
                    "without_semantic": {
                        "mean_similarity": 0.0,
                        "std_deviation": 0.0,
                        "consistency_coefficient": 0.0,
                        "min_similarity": 0.0,
                        "max_similarity": 0.0,
                        "similarity_range": 0.0,
                        "empty_responses": empty_count,
                        "total_responses": len(responses)
                    }
                }
                continue
                
            # If some responses are empty but not all, filter them out but track the count
            if empty_count > 0:
                print(f"Warning: {empty_count}/{len(responses)} empty responses for sample {sample['sample_id']}")
                responses = [r for r in responses if isinstance(r, dict) and r]
                
            # Skip if we don't have at least 2 valid responses after filtering
            if len(responses) < 2:
                print(f"Warning: Not enough valid responses for sample {sample['sample_id']}")
                continue
                
            # Evaluate with semantic tree consistency
            semantic_result = evaluate_semantic_json_consistency(
                responses,
                use_semantic_similarity=True,
                semantic_threshold=0.7
            )
            
            # Evaluate with standard tree consistency (semantic features disabled)
            standard_result = evaluate_semantic_json_consistency(
                responses,
                use_semantic_similarity=False
            )
            
            # Extract basic consistency metrics
            sample_consistencies.append(semantic_result["consistency_metrics"]["mean_similarity"])
            sample_consistencies_standard.append(standard_result["consistency_metrics"]["mean_similarity"])
            
            # Store detailed metrics for this sample
            sample["consistency_analysis"] = {
                "with_semantic": {
                    "mean_similarity": semantic_result["consistency_metrics"]["mean_similarity"],
                    "std_deviation": semantic_result["consistency_metrics"]["std_deviation"],
                    "consistency_coefficient": semantic_result["consistency_metrics"]["consistency_coefficient"],
                    "min_similarity": semantic_result["consistency_metrics"]["min_similarity"],
                    "max_similarity": semantic_result["consistency_metrics"]["max_similarity"],
                    "similarity_range": semantic_result["consistency_metrics"]["similarity_range"],
                    "statistical_metrics": semantic_result["statistical_metrics"]
                },
                "without_semantic": {
                    "mean_similarity": standard_result["consistency_metrics"]["mean_similarity"],
                    "std_deviation": standard_result["consistency_metrics"]["std_deviation"],
                    "consistency_coefficient": standard_result["consistency_metrics"]["consistency_coefficient"],
                    "min_similarity": standard_result["consistency_metrics"]["min_similarity"],
                    "max_similarity": standard_result["consistency_metrics"]["max_similarity"],
                    "similarity_range": standard_result["consistency_metrics"]["similarity_range"],
                    "statistical_metrics": standard_result["statistical_metrics"]
                }
            }
        
        # Calculate empty response rate
        total_responses = 0
        empty_responses = 0
        
        for sample in data["results"]:
            responses = sample["responses"]
            total_responses += len(responses)
            empty_responses += sum(1 for r in responses if isinstance(r, dict) and not r)
        
        empty_rate = empty_responses / total_responses if total_responses > 0 else 0
        empty_response_rates[model_name] = empty_rate
        
        # Calculate average consistency across samples
        if sample_consistencies:
            avg_consistency = np.mean(sample_consistencies)
            std_consistency = np.std(sample_consistencies)
            avg_consistency_standard = np.mean(sample_consistencies_standard)
            std_consistency_standard = np.std(sample_consistencies_standard)
            
            min_consistency = np.min(sample_consistencies) if sample_consistencies else 0
            max_consistency = np.max(sample_consistencies) if sample_consistencies else 0
            
            min_consistency_standard = np.min(sample_consistencies_standard) if sample_consistencies_standard else 0
            max_consistency_standard = np.max(sample_consistencies_standard) if sample_consistencies_standard else 0
            
            consistency_results[model_name] = {
                "model_id": model_id,
                "with_semantic": {
                    "mean": avg_consistency,
                    "std": std_consistency,
                    "samples": sample_consistencies,
                    "min": min_consistency,
                    "max": max_consistency
                },
                "without_semantic": {
                    "mean": avg_consistency_standard,
                    "std": std_consistency_standard,
                    "samples": sample_consistencies_standard,
                    "min": min_consistency_standard,
                    "max": max_consistency_standard
                },
                "sample_count": len(sample_consistencies),
                "empty_response_rate": empty_rate,
                "empty_responses": empty_responses,
                "total_responses": total_responses
            }
            
            print(f"Model {model_name}: With Semantic = {avg_consistency:.4f} ± {std_consistency:.4f}, "
                  f"Without Semantic = {avg_consistency_standard:.4f} ± {std_consistency_standard:.4f}, "
                  f"Empty Rate = {empty_rate:.2%}")
        else:
            print(f"Warning: No valid samples for model {model_name}")
    
    return consistency_results, empty_response_rates

def plot_results(consistency_results, empty_response_rates, output_dir, temperature):
    """Plot consistency across different models."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Sort models by semantic consistency score
    models = sorted(consistency_results.keys(), 
                   key=lambda m: consistency_results[m]["with_semantic"]["mean"],
                   reverse=True)
    
    # Create main plot directory
    plots_dir = os.path.join(output_dir, f"model_comparison_temp_{str(temperature).replace('.', '_')}_{timestamp}")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Mean Consistency by Model
    plt.figure(figsize=(12, 8))
    
    semantic_means = [consistency_results[m]["with_semantic"]["mean"] for m in models]
    semantic_stds = [consistency_results[m]["with_semantic"]["std"] for m in models]
    standard_means = [consistency_results[m]["without_semantic"]["mean"] for m in models]
    standard_stds = [consistency_results[m]["without_semantic"]["std"] for m in models]
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, semantic_means, width, label='With Semantic Similarity',
            yerr=semantic_stds, capsize=5, color='royalblue', alpha=0.8)
    plt.bar(x + width/2, standard_means, width, label='Without Semantic Similarity',
            yerr=standard_stds, capsize=5, color='lightcoral', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Consistency Score')
    plt.title(f'Consistency Comparison Across Models (Temperature = {temperature})')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot
    mean_plot_path = os.path.join(plots_dir, "mean_consistency_by_model.png")
    plt.savefig(mean_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mean consistency plot saved to {mean_plot_path}")
    
    # Plot 2: Consistency Coefficient by Model
    plt.figure(figsize=(12, 8))
    
    semantic_coef = [consistency_results[m]["with_semantic"]["mean"] * 
                    (1 - min(consistency_results[m]["with_semantic"]["std"] / 
                             max(consistency_results[m]["with_semantic"]["mean"], 0.001), 1)) 
                    for m in models]
    
    standard_coef = [consistency_results[m]["without_semantic"]["mean"] * 
                     (1 - min(consistency_results[m]["without_semantic"]["std"] / 
                              max(consistency_results[m]["without_semantic"]["mean"], 0.001), 1)) 
                     for m in models]
    
    # Create bars
    plt.bar(x - width/2, semantic_coef, width, label='With Semantic Similarity',
            color='royalblue', alpha=0.8)
    plt.bar(x + width/2, standard_coef, width, label='Without Semantic Similarity',
            color='lightcoral', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Consistency Coefficient')
    plt.title(f'Consistency Coefficient by Model (Temperature = {temperature})')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot
    coef_plot_path = os.path.join(plots_dir, "consistency_coefficient_by_model.png")
    plt.savefig(coef_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Consistency coefficient plot saved to {coef_plot_path}")
    
    # Plot 3: Improvement from Semantic Similarity
    plt.figure(figsize=(12, 8))
    
    improvements = [consistency_results[m]["with_semantic"]["mean"] - 
                   consistency_results[m]["without_semantic"]["mean"] 
                   for m in models]
    
    improvement_pct = [(consistency_results[m]["with_semantic"]["mean"] - 
                      consistency_results[m]["without_semantic"]["mean"]) / 
                      max(consistency_results[m]["without_semantic"]["mean"], 0.001) * 100
                      for m in models]
    
    # Sort models by improvement
    models_by_improvement = sorted(zip(models, improvements, improvement_pct), 
                                 key=lambda x: x[1], reverse=True)
    models_imp = [m[0] for m in models_by_improvement]
    improvements_sorted = [m[1] for m in models_by_improvement]
    improvement_pct_sorted = [m[2] for m in models_by_improvement]
    
    # Create bars
    plt.figure(figsize=(12, 8))
    bars = plt.bar(np.arange(len(models_imp)), improvements_sorted, color='mediumseagreen', alpha=0.8)
    
    # Add percentage labels on top of bars
    for i, (bar, pct) in enumerate(zip(bars, improvement_pct_sorted)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Absolute Improvement')
    plt.title(f'Improvement from Semantic Similarity by Model (Temperature = {temperature})')
    plt.xticks(np.arange(len(models_imp)), models_imp, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot
    improvement_plot_path = os.path.join(plots_dir, "semantic_improvement_by_model.png")
    plt.savefig(improvement_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Improvement plot saved to {improvement_plot_path}")
    
    # Plot 4: Standard Deviation Comparison
    plt.figure(figsize=(12, 8))
    
    semantic_std = [consistency_results[m]["with_semantic"]["std"] for m in models]
    standard_std = [consistency_results[m]["without_semantic"]["std"] for m in models]
    
    # Create bars
    plt.bar(x - width/2, semantic_std, width, label='With Semantic Similarity',
            color='royalblue', alpha=0.8)
    plt.bar(x + width/2, standard_std, width, label='Without Semantic Similarity',
            color='lightcoral', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Standard Deviation')
    plt.title(f'Consistency Variation by Model (Temperature = {temperature})')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot
    std_plot_path = os.path.join(plots_dir, "consistency_std_by_model.png")
    plt.savefig(std_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Standard deviation plot saved to {std_plot_path}")
    
    # Plot 5: Empty Response Rate by Model
    plt.figure(figsize=(12, 8))
    
    empty_rates = [empty_response_rates[m] for m in models]
    
    # Create bars
    bars = plt.bar(np.arange(len(models)), empty_rates, color='firebrick', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Empty Response Rate')
    plt.title(f'Empty Response Rate by Model (Temperature = {temperature})')
    plt.xticks(np.arange(len(models)), models, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{empty_rates[i]:.1%}", ha='center', va='bottom', fontsize=9)
    
    # Save the plot
    empty_plot_path = os.path.join(plots_dir, "empty_response_rate_by_model.png")
    plt.savefig(empty_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Empty response rate plot saved to {empty_plot_path}")
    
    # Plot 6: Consistency vs Empty Response Rate
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter([empty_response_rates[m] for m in models], 
               [consistency_results[m]["with_semantic"]["mean"] for m in models],
               s=80, alpha=0.7, c='royalblue', label='With Semantic')
    
    plt.scatter([empty_response_rates[m] for m in models], 
               [consistency_results[m]["without_semantic"]["mean"] for m in models],
               s=80, alpha=0.7, c='lightcoral', label='Without Semantic')
    
    # Add model labels to points
    for i, m in enumerate(models):
        plt.annotate(m, 
                    (empty_response_rates[m], consistency_results[m]["with_semantic"]["mean"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Empty Response Rate')
    plt.ylabel('Consistency Score')
    plt.title(f'Consistency vs Empty Response Rate (Temperature = {temperature})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the plot
    corr_plot_path = os.path.join(plots_dir, "consistency_vs_empty_rate.png")
    plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation plot saved to {corr_plot_path}")
    
    # Save all data
    data_path = os.path.join(plots_dir, "model_comparison_data.json")
    with open(data_path, 'w') as f:
        json.dump({
            "temperature": temperature,
            "models": models,
            "results": consistency_results,
            "empty_response_rates": empty_response_rates,
            "improvements": {
                m: {
                    "absolute": consistency_results[m]["with_semantic"]["mean"] - 
                               consistency_results[m]["without_semantic"]["mean"],
                    "percentage": (consistency_results[m]["with_semantic"]["mean"] - 
                                  consistency_results[m]["without_semantic"]["mean"]) / 
                                  max(consistency_results[m]["without_semantic"]["mean"], 0.001) * 100
                } for m in models
            }
        }, f, indent=2)
    print(f"Data saved to {data_path}")
    
    return plots_dir, data_path

def main():
    parser = argparse.ArgumentParser(description="Run model comparison for consistency evaluation")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the data files")
    parser.add_argument("--output-dir", type=str, default="./model_comparison", help="Directory to save results")
    parser.add_argument("--temperature", type=float, default=0.7, help="Fixed temperature to use for all models")
    parser.add_argument("--run-num", type=int, default=10, help="Number of runs per model")
    parser.add_argument("--sample-limit", type=int, default=5, help="Number of samples to process")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define models to test
    models = [
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.nova-premier-v1:0"
    ]
    
    print(f"Running model comparison with {len(models)} models:")
    for model in models:
        print(f"  - {model}")
    print(f"Fixed temperature: {args.temperature}")
    print(f"Runs per model: {args.run_num}")
    print(f"Samples per run: {args.sample_limit}")
    
    # Run generation with different models
    result_dirs = run_generation_with_models(
        args.data_dir,
        args.output_dir,
        models,
        args.temperature,
        args.run_num,
        args.sample_limit
    )
    
    # Evaluate consistency
    consistency_results, empty_response_rates = evaluate_consistency(result_dirs)
    
    # Plot results
    plots_dir, data_path = plot_results(consistency_results, empty_response_rates, args.output_dir, args.temperature)
    
    print("\nExperiment completed!")
    print(f"Plots directory: {plots_dir}")
    print(f"Data: {data_path}")

if __name__ == "__main__":
    main()