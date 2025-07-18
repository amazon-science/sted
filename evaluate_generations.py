#!/usr/bin/env python
"""
Evaluate LLM generations using various metrics.

This script evaluates previously generated LLM outputs using different metrics:
1. Semantic tree-based metrics (accuracy and cross-run consistency)
2. NLP-based metrics (BLEU, ROUGE, BERTScore)

Usage:
    python evaluate_generations.py --input-file path/to/generations.json --output-dir ./evaluation_results
"""

import argparse
import json
import os
import time
import statistics
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from collections import defaultdict

# Import NLP evaluation metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score
from semantic_json_tree_consistency import (
    SemanticJsonTreeConsistencyEvaluator,
    evaluate_semantic_json_consistency,
    parse_json_outputs
)

nltk.download('punkt', quiet=True)

# Function to recursively convert objects to JSON-serializable types
def make_json_serializable(obj):
    """Recursively convert all values to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.numel() > 1 else float(obj.item())
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif obj is np.True_:
        return True
    elif obj is np.False_:
        return False
    elif hasattr(obj, 'item'):
        try:
            return obj.item()
        except:
            pass
    elif hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except:
            pass
    # Handle any other NumPy types we might have missed
    elif type(obj).__module__ == 'numpy':
        try:
            return obj.item() if hasattr(obj, 'item') else obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        except:
            return str(obj)
    else:
        return obj

def calculate_semantic_metrics(generated_outputs: List[Dict], ground_truth: Dict) -> Dict[str, Any]:
    """
    Calculate semantic comparison metrics between generated outputs and ground truth.
    This function evaluates both:
    1. Cross-run consistency: How consistent the outputs are across multiple runs
    2. Ground truth accuracy: How similar the outputs are to the ground truth
    
    Args:
        generated_outputs: List of generated output dictionaries from multiple runs
        ground_truth: Ground truth dictionary to compare against
        
    Returns:
        Dictionary with semantic comparison metrics including:
        - cross_run_consistency: Metrics about consistency across runs
        - ground_truth_accuracy: Metrics about similarity to ground truth
    """
    try:
        # Create a list with ground truth and all generated outputs
        all_outputs = [output for output in generated_outputs if output]
        
        # Use the existing evaluate_semantic_json_consistency function
        consistency_result = evaluate_semantic_json_consistency(
            outputs=all_outputs,
            array_order_matters=False,
            use_semantic_similarity=True,
            semantic_threshold=0.7
        )
        
        # Convert the entire consistency_result to JSON-serializable types
        print("Converting consistency_result to JSON-serializable types...")
        consistency_result = make_json_serializable(consistency_result)
        print("Conversion complete.")
        
        # Verify that perfect_consistency is a Python bool
        if 'consistency_metrics' in consistency_result and 'perfect_consistency' in consistency_result['consistency_metrics']:
            pc_value = consistency_result['consistency_metrics']['perfect_consistency']
            print(f"perfect_consistency type: {type(pc_value)}, value: {pc_value}")
            # Force it to be a Python bool
            consistency_result['consistency_metrics']['perfect_consistency'] = bool(pc_value)
        
        # Extract metrics from the consistency result
        consistency_metrics = consistency_result.get('consistency_metrics', {})
        
        # Calculate ground truth to generated outputs similarity
        # Initialize the semantic evaluator
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            array_order_matters=False,
            use_semantic_similarity=True,
            semantic_threshold=0.7,
            string_method='semantic'
        )
        
        # Calculate similarity scores between each generated output and ground truth
        similarity_scores = []
        for output in generated_outputs:
            if not output:  # Skip empty outputs
                continue
                
            # Calculate tree edit distance similarity
            similarity = evaluator.calculate_tree_edit_distance(output, ground_truth)[0]
            similarity_scores.append(similarity)
        
        # Calculate metrics
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            std_similarity = statistics.stdev(similarity_scores) if len(similarity_scores) > 1 else 0
            min_similarity = min(similarity_scores)
            max_similarity = max(similarity_scores)
        else:
            avg_similarity = std_similarity = min_similarity = max_similarity = 0
        
        # Note: In semantic comparison, a single similarity score is more meaningful than
        # traditional precision/recall which are designed for exact matching scenarios
        
        # Calculate stability as inverse of standard deviation
        # This measures how stable/consistent the outputs are across multiple runs
        stability = 1 - (std_similarity if std_similarity < 1 else 1)
        
        return {
            "cross_run_consistency": consistency_result,  # Metrics about consistency across multiple runs
            "ground_truth_accuracy": {
                "semantic_similarity": {
                    "mean": float(avg_similarity),  # Average similarity to ground truth
                    "std": float(std_similarity),   # Standard deviation of similarity (lower = more consistent)
                    "min": float(min_similarity),   # Minimum similarity to ground truth
                    "max": float(max_similarity),   # Maximum similarity to ground truth
                    "stability": float(stability)   # Measure of stability across runs (1 - std)
                },
                "individual_scores": [float(score) for score in similarity_scores]  # Individual similarity scores
            }
        }
    except Exception as e:
        print(f"Error calculating semantic metrics: {e}")
        return {
            "error": str(e),
            "cross_run_consistency": {},
            "ground_truth_accuracy": {
                "semantic_similarity": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "stability": 0.0},
                "individual_scores": []
            }
        }

def calculate_nlp_metrics(generated_outputs: List[Dict], ground_truth: Dict) -> Dict[str, Any]:
    """
    Calculate NLP-based evaluation metrics between generated outputs and ground truth.
    Uses standard NLP metrics like BLEU, ROUGE, and BERTScore to evaluate both
    accuracy (similarity to ground truth) and cross-run consistency.
    
    Args:
        generated_outputs: List of generated output dictionaries from multiple runs
        ground_truth: Ground truth dictionary to compare against
        
    Returns:
        Dictionary with NLP metrics including:
        - accuracy_metrics: Metrics about similarity to ground truth
        - consistency_metrics: Metrics about consistency across runs
        - individual_scores: Raw scores for each metric and run
    """
    try:
        # Convert dictionaries to strings for text-based metrics
        gt_str = json.dumps(ground_truth, sort_keys=True, indent=2)
        
        # Prepare for NLTK metrics
        # Tokenize ground truth
        gt_tokens = nltk.word_tokenize(gt_str.lower())
        # Initialize Rouge
        rouge = Rouge()
        
        # Calculate metrics for each output
        bleu_scores = []
        rouge_scores_1 = []
        rouge_scores_2 = []
        rouge_scores_l = []
        bert_scores = []
        jaccard_scores = []
        
        # Also keep track of key-based similarity
        gt_keys = set(_extract_all_keys(ground_truth))
        
        for output in generated_outputs:
            if not output:  # Skip empty outputs
                continue
                
            # Convert output to string
            output_str = json.dumps(output, sort_keys=True, indent=2)
            
            # Calculate BLEU score
            output_tokens = nltk.word_tokenize(output_str.lower())
            try:
                # Use smoothing to avoid zero scores due to n-gram mismatches
                smoothie = SmoothingFunction().method1
                bleu = sentence_bleu([gt_tokens], output_tokens, smoothing_function=smoothie)
                bleu_scores.append(bleu)
            except Exception as e:
                print(f"Error calculating BLEU score: {e}")
            
            # Calculate ROUGE scores
            try:
                rouge_scores = rouge.get_scores(output_str, gt_str)[0]
                rouge_scores_1.append(rouge_scores['rouge-1']['f'])
                rouge_scores_2.append(rouge_scores['rouge-2']['f'])
                rouge_scores_l.append(rouge_scores['rouge-l']['f'])
            except Exception as e:
                print(f"Error calculating ROUGE scores: {e}")
        
            # Calculate BERTScore
            try:
                P, R, F1 = bert_score([output_str], [gt_str], lang="en")
                # Convert PyTorch tensor to Python float
                if hasattr(F1, 'item'):
                    bert_scores.append(float(F1.item()))
                else:
                    bert_scores.append(float(F1))
            except Exception as e:
                print(f"Error calculating BERTScore: {e}")
            
            # Calculate Jaccard similarity on keys as a fallback
            output_keys = set(_extract_all_keys(output))
            common_keys = gt_keys.intersection(output_keys)
            union_keys = gt_keys.union(output_keys)
            jaccard = len(common_keys) / len(union_keys) if union_keys else 0
            jaccard_scores.append(jaccard)
        
        # Calculate statistics for each metric
        metrics = {}
        
        # Helper function to calculate statistics
        def calc_stats(scores, name):
            if not scores:
                return {name: {"mean": 0, "std": 0, "min": 0, "max": 0}}
            # Convert any non-native types to Python native types
            scores = [float(score) for score in scores]
            return {name: {
                "mean": float(sum(scores) / len(scores)),
                "std": float(statistics.stdev(scores) if len(scores) > 1 else 0),
                "min": float(min(scores)),
                "max": float(max(scores))
            }}
        
        # Add available metrics
        if bleu_scores:
            metrics.update(calc_stats(bleu_scores, "bleu"))
        if rouge_scores_1:
            metrics.update(calc_stats(rouge_scores_1, "rouge_1"))
            metrics.update(calc_stats(rouge_scores_2, "rouge_2"))
            metrics.update(calc_stats(rouge_scores_l, "rouge_l"))
        if bert_scores:
            metrics.update(calc_stats(bert_scores, "bert_score"))
        
        # Always include Jaccard similarity as a fallback
        metrics.update(calc_stats(jaccard_scores, "jaccard"))
        
        # Calculate overall consistency as average of standard deviations
        std_values = [m["std"] for m in metrics.values()]
        if std_values:
            avg_std = sum(std_values) / len(std_values)
            consistency = 1 - (avg_std if avg_std < 1 else 1)
        else:
            consistency = 0
        
        # Add individual scores
        individual_scores = {}
        if bleu_scores:
            individual_scores["bleu"] = bleu_scores
        if rouge_scores_1:
            individual_scores["rouge_1"] = rouge_scores_1
            individual_scores["rouge_2"] = rouge_scores_2
            individual_scores["rouge_l"] = rouge_scores_l
        if bert_scores:
            individual_scores["bert_score"] = bert_scores
        individual_scores["jaccard"] = jaccard_scores
        
        return {
            "accuracy_metrics": metrics,  # Metrics about similarity to ground truth
            "cross_run_stability": consistency,  # Overall stability across runs (1 - avg_std)
            "individual_scores": individual_scores,  # Raw scores for each metric and run
            "available_metrics": list(metrics.keys())  # List of available metrics
        }
    except Exception as e:
        print(f"Error calculating basic metrics: {e}")
        return {
            "error": str(e),
            "accuracy_metrics": {"jaccard": {"mean": 0, "std": 0, "min": 0, "max": 0}},
            "cross_run_stability": 0,
            "individual_scores": {"jaccard": []},
            "available_metrics": ["jaccard"]
        }

def _extract_all_keys(obj: Dict, prefix: str = "") -> List[str]:
    """
    Extract all keys from a nested dictionary, including nested paths.
    
    Args:
        obj: Dictionary to extract keys from
        prefix: Prefix for nested keys
        
    Returns:
        List of all keys, including nested paths
    """
    keys = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            
            if isinstance(value, (dict, list)):
                keys.extend(_extract_all_keys(value, full_key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            full_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            
            if isinstance(item, (dict, list)):
                keys.extend(_extract_all_keys(item, full_key))
    
    return keys

def evaluate_generations(input_file: str, output_dir: str, metrics_to_calculate: List[str]) -> Dict[str, Any]:
    """
    Evaluate previously generated LLM outputs using various metrics.
    
    Args:
        input_file: Path to the JSON file containing the generated outputs
        output_dir: Directory to save the evaluation results
        metrics_to_calculate: List of metrics to calculate ('semantic', 'nlp', or 'all')
        
    Returns:
        Dictionary with evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the generated outputs
    print(f"Loading generated outputs from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    metadata = data.get('metadata', {})
    results = data.get('results', [])
    
    print(f"Loaded {len(results)} samples with metadata: {metadata}")
    
    # Determine which metrics to calculate
    calculate_semantic = 'semantic' in metrics_to_calculate or 'all' in metrics_to_calculate
    calculate_nlp = 'nlp' in metrics_to_calculate or 'all' in metrics_to_calculate
    
    # Initialize aggregated metrics for overall statistics
    overall_metrics = {
        "semantic": {
            "ground_truth_similarity": [],
            "cross_run_consistency": []
        },
        "nlp": {
            "bleu": [],
            "rouge_1": [],
            "rouge_2": [],
            "rouge_l": [],
            "bert_score": [],
            "jaccard": [],
            "cross_run_stability": []
        }
    }
    
    # Process each sample
    evaluation_results = []
    
    for sample_idx, sample in enumerate(results):
        sample_id = sample.get('sample_id', f"sample_{sample_idx}")
        print(f"\n{'='*60}")
        print(f"EVALUATING SAMPLE {sample_idx + 1}/{len(results)}: {sample_id}")
        print(f"{'='*60}")
        
        # Extract ground truth and responses
        ground_truth = sample.get('ground_truth', {})
        responses = sample.get('responses', [])
        
        if not responses:
            print(f"Warning: No responses found for sample {sample_id}")
            continue
        
        print(f"Found {len(responses)} responses for sample {sample_id}")
        
        # Initialize metrics dictionaries
        semantic_metrics = {}
        nlp_metrics = {}
        
        # Calculate semantic tree-based metrics if requested
        if calculate_semantic:
            print("\nCalculating semantic tree-based metrics (accuracy and cross-run consistency)...")
            semantic_metrics = calculate_semantic_metrics(responses, ground_truth)
        
        # Calculate NLP-based metrics if requested
        if calculate_nlp:
            print("\nCalculating NLP-based metrics (BLEU, ROUGE, BERTScore)...")
            nlp_metrics = calculate_nlp_metrics(responses, ground_truth)
        
        # Print summary of metrics
        if semantic_metrics and calculate_semantic:
            print("\n=== Semantic Tree-Based Metrics ===")
            if 'ground_truth_accuracy' in semantic_metrics:
                gt_metrics = semantic_metrics['ground_truth_accuracy']
                print("Ground Truth Accuracy:")
                print(f"  Semantic similarity: {gt_metrics['semantic_similarity']['mean']:.4f} ± {gt_metrics['semantic_similarity']['std']:.4f}")
                print(f"  Min similarity: {gt_metrics['semantic_similarity']['min']:.4f}, Max similarity: {gt_metrics['semantic_similarity']['max']:.4f}")
                print(f"  Stability across runs: {gt_metrics['semantic_similarity']['stability']:.4f} (higher is better)")
                
                # Print cross-run consistency metrics if available
                if 'cross_run_consistency' in semantic_metrics and 'consistency_metrics' in semantic_metrics['cross_run_consistency']:
                    consistency_metrics = semantic_metrics['cross_run_consistency']['consistency_metrics']
                    print("\nCross-Run Consistency:")
                    print(f"  Overall consistency score: {consistency_metrics.get('mean_similarity', 0):.4f}")
                    print(f"  Perfect consistency: {consistency_metrics.get('perfect_consistency', False)}")
        
        if nlp_metrics and calculate_nlp:
            print("\n=== NLP-Based Metrics ===")
            metrics = nlp_metrics['accuracy_metrics']
            print(f"Available metrics: {nlp_metrics.get('available_metrics', [])}")
            print("\nGround Truth Accuracy:")
            
            # Print BLEU score
            if 'bleu' in metrics:
                bleu = metrics['bleu']
                print(f"  BLEU score: {bleu['mean']:.4f} ± {bleu['std']:.4f} (higher is better)")
            
            # Print ROUGE scores
            if 'rouge_l' in metrics:
                rouge_l = metrics['rouge_l']
                print(f"  ROUGE-L F1: {rouge_l['mean']:.4f} ± {rouge_l['std']:.4f} (higher is better)")
            
            # Print BERTScore
            if 'bert_score' in metrics:
                bert = metrics['bert_score']
                print(f"  BERTScore: {bert['mean']:.4f} ± {bert['std']:.4f} (higher is better)")
            
            # Print Jaccard similarity
            if 'jaccard' in metrics:
                jaccard = metrics['jaccard']
                print(f"  Jaccard similarity: {jaccard['mean']:.4f} ± {jaccard['std']:.4f} (higher is better)")
            
            print("\nCross-Run Stability:")
            print(f"  Overall stability: {nlp_metrics.get('cross_run_stability', 0):.4f} (higher is better)")
            print("  (Stability measures how consistent the outputs are across multiple runs)")
            print("  (A value of 1.0 means perfect consistency across all runs)")
        
        # Create evaluation result for this sample
        evaluation_result = {
            "sample_id": sample_id,
            "semantic_tree_metrics": semantic_metrics if calculate_semantic else {},
            "nlp_metrics": nlp_metrics if calculate_nlp else {},
        }
        
        # Collect metrics for overall statistics
        if calculate_semantic and semantic_metrics:
            if 'ground_truth_accuracy' in semantic_metrics and 'semantic_similarity' in semantic_metrics['ground_truth_accuracy']:
                overall_metrics['semantic']['ground_truth_similarity'].append(
                    semantic_metrics['ground_truth_accuracy']['semantic_similarity']['mean']
                )
            
            if 'cross_run_consistency' in semantic_metrics and 'consistency_metrics' in semantic_metrics['cross_run_consistency']:
                overall_metrics['semantic']['cross_run_consistency'].append(
                    semantic_metrics['cross_run_consistency']['consistency_metrics'].get('mean_similarity', 0)
                )
        
        if calculate_nlp and nlp_metrics:
            if 'cross_run_stability' in nlp_metrics:
                overall_metrics['nlp']['cross_run_stability'].append(nlp_metrics['cross_run_stability'])
            
            if 'accuracy_metrics' in nlp_metrics:
                for metric in ['bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'bert_score', 'jaccard']:
                    if metric in nlp_metrics['accuracy_metrics']:
                        overall_metrics['nlp'][metric].append(nlp_metrics['accuracy_metrics'][metric]['mean'])
        
        evaluation_results.append(evaluation_result)
    
    # Calculate overall metrics across all samples
    overall_summary = {}
    
    # Helper function to calculate statistics for a list of values
    def calc_overall_stats(values, name):
        if not values:
            return {name: {"mean": 0, "std": 0, "min": 0, "max": 0}}
        return {name: {
            "mean": float(sum(values) / len(values)),
            "std": float(statistics.stdev(values) if len(values) > 1 else 0),
            "min": float(min(values)),
            "max": float(max(values))
        }}
    
    # Calculate semantic metrics summary if available
    if calculate_semantic:
        semantic_summary = {}
        if overall_metrics['semantic']['ground_truth_similarity']:
            semantic_summary.update(calc_overall_stats(
                overall_metrics['semantic']['ground_truth_similarity'], 
                "ground_truth_similarity"
            ))
        if overall_metrics['semantic']['cross_run_consistency']:
            semantic_summary.update(calc_overall_stats(
                overall_metrics['semantic']['cross_run_consistency'], 
                "cross_run_consistency"
            ))
        overall_summary["semantic"] = semantic_summary
    
    # Calculate NLP metrics summary if available
    if calculate_nlp:
        nlp_summary = {}
        for metric in ['bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'bert_score', 'jaccard']:
            if overall_metrics['nlp'][metric]:
                nlp_summary.update(calc_overall_stats(overall_metrics['nlp'][metric], metric))
        if overall_metrics['nlp']['cross_run_stability']:
            nlp_summary.update(calc_overall_stats(
                overall_metrics['nlp']['cross_run_stability'], 
                "cross_run_stability"
            ))
        overall_summary["nlp"] = nlp_summary
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL METRICS SUMMARY ACROSS ALL SAMPLES")
    print("=" * 60)
    
    if "semantic" in overall_summary:
        print("\n=== Semantic Tree-Based Metrics (Average Across All Samples) ===")
        if "ground_truth_similarity" in overall_summary["semantic"]:
            gt_sim = overall_summary["semantic"]["ground_truth_similarity"]
            print(f"Ground Truth Similarity: {gt_sim['mean']:.4f} ± {gt_sim['std']:.4f}")
            print(f"Min: {gt_sim['min']:.4f}, Max: {gt_sim['max']:.4f}")
        
        if "cross_run_consistency" in overall_summary["semantic"]:
            consistency = overall_summary["semantic"]["cross_run_consistency"]
            print(f"Cross-Run Consistency: {consistency['mean']:.4f} ± {consistency['std']:.4f}")
            print(f"Min: {consistency['min']:.4f}, Max: {consistency['max']:.4f}")
    
    if "nlp" in overall_summary:
        print("\n=== NLP-Based Metrics (Average Across All Samples) ===")
        for metric in ["bleu", "rouge_1", "rouge_2", "rouge_l", "bert_score", "jaccard"]:
            if metric in overall_summary["nlp"]:
                m = overall_summary["nlp"][metric]
                print(f"{metric.upper()}: {m['mean']:.4f} ± {m['std']:.4f} (Min: {m['min']:.4f}, Max: {m['max']:.4f})")
        
        if "cross_run_stability" in overall_summary["nlp"]:
            stability = overall_summary["nlp"]["cross_run_stability"]
            print(f"Cross-Run Stability: {stability['mean']:.4f} ± {stability['std']:.4f}")
            print(f"Min: {stability['min']:.4f}, Max: {stability['max']:.4f}")
    
    # Create the final evaluation results
    final_results = {
        "metadata": {
            "original_metadata": metadata,
            "evaluation_timestamp": time.strftime('%Y%m%d_%H%M%S'),
            "metrics_calculated": metrics_to_calculate
        },
        "overall_metrics": overall_summary,
        "results": evaluation_results
    }
    
    # Convert all objects to JSON-serializable types
    serializable_results = make_json_serializable(final_results)
    
    # Save the evaluation results
    output_file = os.path.join(output_dir, f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nEvaluation results saved to {output_file}")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM generations using various metrics.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the JSON file containing the generated outputs.")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Directory to save the evaluation results.")
    parser.add_argument("--metrics", type=str, default="all", choices=["semantic", "nlp", "all"], help="Which metrics to calculate.")
    args = parser.parse_args()
    
    # Convert metrics to a list
    if args.metrics == "all":
        metrics_to_calculate = ["semantic", "nlp"]
    else:
        metrics_to_calculate = [args.metrics]
    
    # Evaluate the generations
    evaluate_generations(args.input_file, args.output_dir, metrics_to_calculate)

if __name__ == "__main__":
    main()