#!/usr/bin/env python3
"""
Temperature Experiment for Semantic Tree Consistency

This script runs the LLM generation with different temperature settings
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

def run_generation_with_temperature(data_dir, output_dir, model_id, temperatures, run_num=10, sample_limit=5):
    """Run LLM generation with different temperature settings."""
    results = {}
    
    for temp in temperatures:
        print(f"\n{'='*80}")
        print(f"RUNNING GENERATION WITH TEMPERATURE {temp}")
        print(f"{'='*80}")
        
        # Format temperature for command line
        temp_str = str(temp)
        
        # Run the llm_gen.py script with the specified temperature
        cmd = [
            "python", "llm_gen.py",
            "--data-dir", data_dir,
            "--output-dir", output_dir,
            "--model-id", model_id,
            "--temperature", temp_str,
            "--run-num", str(run_num),
            "--sample-limit", str(sample_limit)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Find the most recent output directory for this temperature
            temp_dirs = []
            for dir_name in os.listdir(output_dir):
                if dir_name.startswith(f"llm_gen_results_") and f"temp_{temp_str.replace('.', '_')}" in dir_name:
                    dir_path = os.path.join(output_dir, dir_name)
                    if os.path.isdir(dir_path):
                        temp_dirs.append((dir_path, os.path.getmtime(dir_path)))
            
            if not temp_dirs:
                print(f"Warning: No output directory found for temperature {temp}")
                continue
                
            # Get the most recent directory
            most_recent_dir = sorted(temp_dirs, key=lambda x: x[1], reverse=True)[0][0]
            results[temp] = most_recent_dir
            print(f"Results for temperature {temp} saved in {most_recent_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running generation with temperature {temp}: {e}")
    
    return results

def evaluate_consistency(result_dirs):
    """Evaluate consistency for each temperature setting."""
    consistency_results = {}
    empty_response_rates = {}
    empty_response_rates = {}
    
    for temp, result_dir in result_dirs.items():
        all_results_path = os.path.join(result_dir, "all_results.json")
        
        if not os.path.exists(all_results_path):
            print(f"Warning: Results file not found for temperature {temp}: {all_results_path}")
            continue
            
        with open(all_results_path, 'r') as f:
            data = json.load(f)
        
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
        empty_response_rates[temp] = empty_rate
        
        # Calculate average consistency across samples
        if sample_consistencies:
            avg_consistency = np.mean(sample_consistencies)
            std_consistency = np.std(sample_consistencies)
            avg_consistency_standard = np.mean(sample_consistencies_standard)
            std_consistency_standard = np.std(sample_consistencies_standard)
            
            min_consistency = np.min(sample_consistencies)
            max_consistency = np.max(sample_consistencies)
            
            min_consistency_standard = np.min(sample_consistencies_standard)
            max_consistency_standard = np.max(sample_consistencies_standard)
            
            consistency_results[temp] = {
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
            
            print(f"Temperature {temp}: With Semantic = {avg_consistency:.4f} ± {std_consistency:.4f}, "
                  f"Without Semantic = {avg_consistency_standard:.4f} ± {std_consistency_standard:.4f}, "
                  f"Empty Rate = {empty_rate:.2%}")
        else:
            print(f"Warning: No valid samples for temperature {temp}")
    
    return consistency_results, empty_response_rates

def plot_results(consistency_results, empty_response_rates, output_dir):
    """Plot consistency vs temperature."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    temperatures = sorted(consistency_results.keys())
    
    # Create main plot directory
    plots_dir = os.path.join(output_dir, f"plots_{timestamp}")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Mean Similarity
    semantic_means = [consistency_results[t]["with_semantic"]["mean"] for t in temperatures]
    semantic_stds = [consistency_results[t]["with_semantic"]["std"] for t in temperatures]
    standard_means = [consistency_results[t]["without_semantic"]["mean"] for t in temperatures]
    standard_stds = [consistency_results[t]["without_semantic"]["std"] for t in temperatures]
    
    plt.figure(figsize=(10, 6))
    
    # Plot semantic consistency
    plt.errorbar(temperatures, semantic_means, yerr=semantic_stds, 
                 marker='o', linestyle='-', label='With Semantic Similarity')
    
    # Plot standard consistency
    plt.errorbar(temperatures, standard_means, yerr=standard_stds, 
                 marker='s', linestyle='--', label='Without Semantic Similarity')
    
    plt.xlabel('Temperature')
    plt.ylabel('Consistency Score')
    plt.title('Mean Consistency vs Temperature')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add correlation coefficient
    semantic_corr = np.corrcoef(temperatures, semantic_means)[0, 1]
    standard_corr = np.corrcoef(temperatures, standard_means)[0, 1]
    plt.annotate(f"With Semantic: r = {semantic_corr:.4f}\nWithout Semantic: r = {standard_corr:.4f}",
                xy=(0.05, 0.05), xycoords='axes fraction')
    
    # Save the plot
    mean_plot_path = os.path.join(plots_dir, "mean_consistency.png")
    plt.savefig(mean_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mean consistency plot saved to {mean_plot_path}")
    
    # Plot 2: Consistency Coefficient
    plt.figure(figsize=(10, 6))
    
    semantic_coef = [consistency_results[t]["with_semantic"]["mean"] * 
                    (1 - min(consistency_results[t]["with_semantic"]["std"] / 
                             max(consistency_results[t]["with_semantic"]["mean"], 0.001), 1)) 
                    for t in temperatures]
    
    standard_coef = [consistency_results[t]["without_semantic"]["mean"] * 
                     (1 - min(consistency_results[t]["without_semantic"]["std"] / 
                              max(consistency_results[t]["without_semantic"]["mean"], 0.001), 1)) 
                     for t in temperatures]
    
    plt.plot(temperatures, semantic_coef, marker='o', linestyle='-', label='With Semantic Similarity')
    plt.plot(temperatures, standard_coef, marker='s', linestyle='--', label='Without Semantic Similarity')
    
    plt.xlabel('Temperature')
    plt.ylabel('Consistency Coefficient')
    plt.title('Consistency Coefficient vs Temperature')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    coef_plot_path = os.path.join(plots_dir, "consistency_coefficient.png")
    plt.savefig(coef_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Consistency coefficient plot saved to {coef_plot_path}")
    
    # Plot 3: Standard Deviation
    plt.figure(figsize=(10, 6))
    
    semantic_std = [consistency_results[t]["with_semantic"]["std"] for t in temperatures]
    standard_std = [consistency_results[t]["without_semantic"]["std"] for t in temperatures]
    
    plt.plot(temperatures, semantic_std, marker='o', linestyle='-', label='With Semantic Similarity')
    plt.plot(temperatures, standard_std, marker='s', linestyle='--', label='Without Semantic Similarity')
    
    plt.xlabel('Temperature')
    plt.ylabel('Standard Deviation')
    plt.title('Consistency Variation vs Temperature')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    std_plot_path = os.path.join(plots_dir, "consistency_std.png")
    plt.savefig(std_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Standard deviation plot saved to {std_plot_path}")
    
    # Plot 4: Min-Max Range
    plt.figure(figsize=(10, 6))
    
    semantic_range = [consistency_results[t]["with_semantic"]["max"] - 
                      consistency_results[t]["with_semantic"]["min"] 
                      for t in temperatures]
    
    standard_range = [consistency_results[t]["without_semantic"]["max"] - 
                       consistency_results[t]["without_semantic"]["min"] 
                       for t in temperatures]
    
    plt.plot(temperatures, semantic_range, marker='o', linestyle='-', label='With Semantic Similarity')
    plt.plot(temperatures, standard_range, marker='s', linestyle='--', label='Without Semantic Similarity')
    
    plt.xlabel('Temperature')
    plt.ylabel('Min-Max Range')
    plt.title('Consistency Range vs Temperature')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    range_plot_path = os.path.join(plots_dir, "consistency_range.png")
    plt.savefig(range_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Range plot saved to {range_plot_path}")
    
    # Plot 5: Empty Response Rate
    plt.figure(figsize=(10, 6))
    
    empty_rates = [empty_response_rates[t] for t in temperatures]
    
    plt.plot(temperatures, empty_rates, marker='o', linestyle='-', color='firebrick')
    plt.fill_between(temperatures, [0] * len(temperatures), empty_rates, alpha=0.2, color='firebrick')
    
    plt.xlabel('Temperature')
    plt.ylabel('Empty Response Rate')
    plt.title('Empty Response Rate vs Temperature')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add correlation coefficient
    empty_corr = np.corrcoef(temperatures, empty_rates)[0, 1] if len(temperatures) > 1 else 0
    plt.annotate(f"Correlation: r = {empty_corr:.4f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 va='top', ha='left')
    
    empty_plot_path = os.path.join(plots_dir, "empty_response_rate.png")
    plt.savefig(empty_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Empty response rate plot saved to {empty_plot_path}")
    
    # Save all data
    data_path = os.path.join(plots_dir, "consistency_data.json")
    with open(data_path, 'w') as f:
        json.dump({
            "temperatures": temperatures,
            "results": consistency_results,
            "empty_response_rates": empty_response_rates,
            "correlations": {
                "with_semantic": float(semantic_corr),
                "without_semantic": float(standard_corr),
                "empty_rate": float(empty_corr)
            }
        }, f, indent=2)
    print(f"Data saved to {data_path}")
    
    return plots_dir, data_path

def main():
    parser = argparse.ArgumentParser(description="Run temperature experiment for consistency evaluation")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the data files")
    parser.add_argument("--output-dir", type=str, default="./temperature_experiment", help="Directory to save results")
    parser.add_argument("--model-id", type=str, default="us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
                        help="Model ID to use")
    parser.add_argument("--run-num", type=int, default=10, help="Number of runs per temperature")
    parser.add_argument("--sample-limit", type=int, default=5, help="Number of samples to process")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define temperature range to test
    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    print(f"Running temperature experiment with {len(temperatures)} temperature settings:")
    print(f"Temperatures: {temperatures}")
    print(f"Model: {args.model_id}")
    print(f"Runs per temperature: {args.run_num}")
    print(f"Samples per run: {args.sample_limit}")
    
    # Run generation with different temperatures
    result_dirs = run_generation_with_temperature(
        args.data_dir,
        args.output_dir,
        args.model_id,
        temperatures,
        args.run_num,
        args.sample_limit
    )
    
    # Evaluate consistency
    consistency_results, empty_response_rates = evaluate_consistency(result_dirs)
    
    # Plot results
    plot_path, data_path = plot_results(consistency_results, empty_response_rates, args.output_dir)
    
    print("\nExperiment completed!")
    print(f"Plot: {plot_path}")
    print(f"Data: {data_path}")

if __name__ == "__main__":
    main()