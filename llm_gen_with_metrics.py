import boto3
from bedrock_utils import build_message, inference_with_converse_api
import argparse
import json
import ast
import concurrent.futures
import time
import os
import statistics
from collections import Counter
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple
import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import semantic comparison functionality
from semantic_json_tree_consistency import (
    SemanticJsonTreeConsistencyEvaluator,
    evaluate_semantic_json_consistency,
    parse_json_outputs
)

# Import NLP evaluation metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

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
    elif isinstance(obj, (bool, np.bool_, np.bool)):
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

# Custom JSON encoder to handle non-serializable objects
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() > 1 else float(obj.item())
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, bool):
            return bool(obj)  # Explicitly convert numpy.bool_ to Python bool
        if isinstance(obj, np.bool_):
            return bool(obj)  # Handle numpy boolean type
        try:
            # Try to convert to a basic type
            if hasattr(obj, 'item'):
                return obj.item()  # Handle objects with .item() method
            elif hasattr(obj, 'tolist'):
                return obj.tolist()  # Handle objects with .tolist() method
        except:
            pass
        return super(NpEncoder, self).default(obj)


def read_sharegpt(data_dir="data"):
    """
    Reads the dataset from the specified directory.
    """
    import os
    import json
    data_path = os.path.join(data_dir, f"all_conversations.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} does not exist.")

    dataset_dict = []
    with open(data_path, 'r') as file:
        data = json.load(file)
        """
        for item in data:
            if 'conversations' in item:
                conversations_new = []
                for conversation in item['conversations']:
                    if 'value' in conversation and isinstance(conversation['value'], str) and conversation['from'] == 'gpt':
                        print(conversation['value'])
                        conversation['value'] = json.loads(conversation['value'])
                        conversations_new.append(conversation)
                item['conversations'] = conversations_new
            dataset_dict.append(item)
        """
    
    return data

def _single_inference(client, model_id, messages, system_prompts=None, max_tokens=8000, temperature=0.1, top_p=0.9, top_k=200, task_id=None):
    """
    Helper function to run a single inference request using threads.
    """
    try:
        # Create a new client for each thread to avoid potential thread safety issues
        thread_client = boto3.client('bedrock-runtime', region_name='us-west-2')
        
        # Print task ID if provided (useful for debugging)
        if task_id is not None:
            print(f"Starting task {task_id}")
        
        # Attempt to run inference
        response = inference_with_converse_api(
            thread_client,
            model_id=model_id,
            messages=messages,
            system_prompts=system_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        if not response or not isinstance(response, list) or len(response) == 0:
            print(f"Warning: Empty or invalid response received for task {task_id}")
            return {}
            
        print(f"Response received for task {task_id}: {len(response)} items")
        response_text = response[0].get('text', '{}')
        
        # Parse the response with safer error handling
        try:
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]  # Remove backticks and 'json'
            if isinstance(response_text, str):
                parsed_response = ast.literal_eval(response_text)  # Convert string to dict if necessary
            elif isinstance(response_text, bytes):
                response_text = response_text.decode('utf-8')  # Decode bytes to string
                parsed_response = ast.literal_eval(response_text)  # Convert string to dict if necessary
            elif isinstance(response_text, dict):
                parsed_response = response_text
            else:
                print(f"Unexpected response type: {type(response_text)}")
                return {}
                
            if task_id is not None:
                print(f"Task {task_id} completed successfully")
            return parsed_response
        except (SyntaxError, ValueError) as parse_error:
            print(f"Error parsing response for task {task_id}: {parse_error}")
            print(f"Raw response text: {response_text[:100]}...")
            return {}
    except Exception as e:
        print(f"Error during inference for task {task_id}: {e}")
        return {}  # Return an empty dict on error

def create_json_schema(ground_truth_dict, max_depth=10):
    """
    Creates a JSON schema from the ground truth dictionary.
    
    Args:
        ground_truth_dict: The ground truth dictionary to create schema from
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        A dictionary representing the JSON schema
    """
    if max_depth <= 0:
        return {"type": "object"}
    
    if not isinstance(ground_truth_dict, dict):
        return {"type": type(ground_truth_dict).__name__}
    
    properties = {}
    required = []
    
    for key, value in ground_truth_dict.items():
        required.append(key)
        
        if isinstance(value, dict):
            properties[key] = create_json_schema(value, max_depth - 1)
        elif isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                # For lists of dictionaries, extract schema from the first item
                properties[key] = {
                    "type": "array",
                    "items": create_json_schema(value[0], max_depth - 1)
                }
            else:
                # For other lists, determine the type of items
                item_type = "string"  # Default type
                if value:
                    if all(isinstance(item, int) for item in value):
                        item_type = "integer"
                    elif all(isinstance(item, float) for item in value):
                        item_type = "number"
                    elif all(isinstance(item, bool) for item in value):
                        item_type = "boolean"
                
                properties[key] = {
                    "type": "array",
                    "items": {"type": item_type}
                }
        elif isinstance(value, str):
            properties[key] = {"type": "string"}
        elif isinstance(value, int):
            properties[key] = {"type": "integer"}
        elif isinstance(value, float):
            properties[key] = {"type": "number"}
        elif isinstance(value, bool):
            properties[key] = {"type": "boolean"}
        else:
            properties[key] = {"type": "string"}
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }

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
        all_outputs = [ground_truth] + [output for output in generated_outputs if output]
        
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
    except Exception as e:
        print(f"Error calculating basic metrics: {e}")
        return {
            "error": str(e),
            "basic_similarity": {"mean": 0, "std": 0, "consistency": 0},
            "precision": {"mean": 0, "std": 0},
            "recall": {"mean": 0, "std": 0},
            "f1_score": {"mean": 0, "std": 0},
            "individual_scores": {"precision": [], "recall": [], "f1": []}
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

def extract_schema(response_dict, max_depth=10):
    """
    Recursively extracts the schema (keys and their types) from a nested dictionary.
    
    Args:
        response_dict: The dictionary to extract schema from
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        A dictionary representing the schema with keys and their types
    """
    if max_depth <= 0:
        return "MAX_DEPTH_REACHED"
    
    if not isinstance(response_dict, dict):
        return type(response_dict).__name__
    
    schema = {}
    for key, value in response_dict.items():
        if isinstance(value, dict):
            schema[key] = extract_schema(value, max_depth - 1)
        elif isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                # For lists of dictionaries, extract schema from the first item
                schema[key] = [extract_schema(value[0], max_depth - 1)]
            else:
                # For other lists, just note the type and length
                item_types = set(type(item).__name__ for item in value) if value else {"empty"}
                schema[key] = f"list[{','.join(item_types)}]({len(value)})"
        else:
            schema[key] = type(value).__name__
    
    return schema

def create_prompt_with_schema(user_prompt, ground_truth_dict):
    """
    Creates a prompt that includes the JSON schema derived from the ground truth.
    
    Args:
        user_prompt: The original user prompt
        ground_truth_dict: The ground truth dictionary
        include_ground_truth: Whether to include the actual ground truth values in the prompt
        
    Returns:
        A modified prompt with the schema information
    """
    # Create JSON schema from ground truth
    schema = create_json_schema(ground_truth_dict)
    
    # Format the schema as a string
    schema_str = json.dumps(schema, indent=2)
    
    # Create the modified prompt
    modified_prompt = f"{user_prompt}\n\n"
    modified_prompt += "Please ensure your response follows this JSON schema:\n"
    modified_prompt += f"```json\n{schema_str}\n```\n\n"
    
    modified_prompt += "Your response should be valid JSON that conforms to the schema above."
    
    return modified_prompt

def run_inference(client, model_id, messages, system_prompts=None, max_tokens=8000, temperature=0.1, top_p=0.9, top_k=200, run_num=5, max_workers=None):
    """
    Runs inference in parallel using ThreadPoolExecutor.
    
    Args:
        client: Bedrock client (not used directly, each thread creates its own)
        model_id: Model ID to use
        messages: List of messages to send
        system_prompts: Optional system prompts
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling (higher = more random outputs)
        top_p: Top-p for sampling
        top_k: Top-k for sampling
        run_num: Number of inference runs to perform
        max_workers: Maximum number of parallel workers (None = auto-determine based on CPU count)
    
    Returns:
        List of response dictionaries
    """
    responses = []
    start_time = time.time()
    
    # Determine optimal number of workers if not specified
    if max_workers is None:
        # Use CPU count or a reasonable default for I/O bound tasks
        max_workers = min(4, (os.cpu_count() or 4) * 4)  # 4x CPU count is good for I/O bound tasks
        print(f"Auto-determined max_workers: {max_workers}")
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all inference tasks
        future_to_task = {}
        for i in range(run_num):
            future = executor.submit(
                _single_inference,
                client,  # Not used directly, each thread creates its own client
                model_id,
                messages,
                system_prompts,
                max_tokens,
                temperature,
                top_p,
                top_k,
                i  # Pass task ID for better logging
            )
            future_to_task[future] = i
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                response = future.result()
                responses.append(response)
                print(f"Task {task_id} result collected")
            except Exception as e:
                print(f"Task {task_id} failed with exception: {e}")
                responses.append({})  # Append an empty dict on error
    
    elapsed_time = time.time() - start_time
    print(f"Completed {run_num} requests in {elapsed_time:.2f} seconds")
    print(f"Average time per request: {elapsed_time/run_num:.2f} seconds")
    
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using an LLM with parallel inference and optional JSON schema guidance.")
    parser.add_argument("--model-id", type=str, default="us.anthropic.claude-3-5-sonnet-20241022-v2:0", help="The ID of the Bedrock model to use.")
    parser.add_argument("--data-dir", type=str, required=True, help="The directory containing the data files.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers for inference. Default is auto-determined based on CPU count.")
    parser.add_argument("--run-num", type=int, default=5, help="Number of inference runs to perform.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling (0.1-2.0). Higher values produce more random outputs.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--top-k", type=int, default=200, help="Top-k sampling parameter.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save evaluation results.")
    parser.add_argument("--schema-only", action="store_true", help="Only evaluate schema consistency without saving full responses.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit the number of samples to process.")
    parser.add_argument("--skip-semantic-metrics", action="store_true", help="Skip calculation of semantic tree-based metrics and use NLP metrics instead.")
    parser.add_argument("--calculate-all-metrics", action="store_true", help="Calculate both semantic tree-based and NLP metrics (more comprehensive but slower).")
    
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt.")
    parser.add_argument("--include-ground-truth", action="store_true", help="Include ground truth values in the prompt (requires --include-schema).")
    args = parser.parse_args()
    
    # Create a client with appropriate configuration
    from botocore.config import Config
    boto_config = Config(
        retries={
            'max_attempts': 10,
            'mode': 'adaptive'
        },
        max_pool_connections=50  # Increase connection pool size
    )
    client = boto3.client('bedrock-runtime', region_name='us-west-2', config=boto_config)
    model_id = args.model_id
    
    include_schema = True

    # Configure logging based on verbose flag
    if not args.verbose:
        # Reduce boto3 logging noise
        import logging
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

    print(f"Starting parallel inference with {args.run_num} runs using ThreadPoolExecutor")
    print(f"Model: {model_id}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}, Top-k: {args.top_k}")
    
    # List dataset
    dataset_dict = read_sharegpt(args.data_dir)
    
    results = []
    all_sample_results = []  # Store results for all samples
    
    # Create timestamped output directory for this run
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    temp_str = f"temp_{args.temperature:.2f}".replace('.', '_')
    model_name = model_id.split('/')[-1].split(':')[0]
    run_output_dir = os.path.join(args.output_dir, f"llm_gen_results_{model_name}_{temp_str}_{run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_output_dir}")
    
    # Determine how many samples to process
    sample_limit = args.sample_limit if args.sample_limit is not None else 2
    samples_to_process = dataset_dict[:sample_limit]
    
    for sample_idx, item in enumerate(samples_to_process):
        sample_id = f"sample_{sample_idx:03d}"
        print(f"\n{'='*60}")
        print(f"PROCESSING SAMPLE {sample_idx + 1}/{len(samples_to_process)}: {sample_id}")
        print(f"{'='*60}")
        
        system_prompt = item['conversations'][0]['value']
        user_prompt = item['conversations'][1]['value']
        gt_value = item['conversations'][2]['value']
        
        try:
            gt_dict = json.loads(gt_value)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse ground truth JSON for {sample_id}: {e}")
            gt_dict = {"error": "Invalid JSON", "raw_value": gt_value}
        
        print(f'Ground truth keys: {list(gt_dict.keys()) if isinstance(gt_dict, dict) else "Not a dict"}')
        
        # Determine whether to use the original prompt or a modified one with schema
        if include_schema:
            # Create a modified prompt with the JSON schema
            modified_user_prompt = create_prompt_with_schema(
                user_prompt=user_prompt,
                ground_truth_dict=gt_dict
            )
            
            message = build_message(texts=[modified_user_prompt])
            
            print(f"\nRunning inference on prompt with schema")
            print(f"Original prompt: {user_prompt[:100]}...")
            print(f"System prompt: {system_prompt[:100]}...")
            print(f"Schema added to prompt for better comparison with ground truth")
            
            # Save the modified prompt for reference
            prompt_output_path = os.path.join(run_output_dir, f"{sample_id}_modified_prompt.txt")
            with open(prompt_output_path, 'w') as f:
                f.write(modified_user_prompt)
            print(f"Saved modified prompt to {prompt_output_path}")
        else:
            # Use original prompt without schema
            modified_user_prompt = user_prompt
            message = build_message(texts=[user_prompt])
            print(f"\nRunning inference on original prompt: {user_prompt[:100]}...")
            print(f"System prompt: {system_prompt[:100]}...")
        
        
        responses = run_inference(
            client=client,
            model_id=model_id,
            messages=[message],
            system_prompts=system_prompt,
            max_tokens=8000,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            run_num=args.run_num,
            max_workers=args.max_workers
        )
        
        print(responses)
        
        schema_list = [extract_schema(response) for response in responses]
        print(f"schema_list: {schema_list}")
        
        # Calculate metrics between responses and ground truth
        semantic_metrics = {}
        nlp_metrics = {}
        
        # Determine which metrics to calculate based on command-line arguments
        calculate_semantic = not args.skip_semantic_metrics or args.calculate_all_metrics
        calculate_nlp = args.skip_semantic_metrics or args.calculate_all_metrics
        
        # Calculate semantic tree-based metrics if requested
        if calculate_semantic:
            print("\nCalculating semantic tree-based metrics (accuracy and cross-run consistency)...")
            semantic_metrics = calculate_semantic_metrics(responses, gt_dict)
        else:
            print("\nSkipping semantic tree-based metrics (--skip-semantic-metrics flag used).")
            
        # Calculate NLP-based metrics if requested
        if calculate_nlp:
            print("\nCalculating NLP-based metrics (BLEU, ROUGE, BERTScore)...")
            nlp_metrics = calculate_nlp_metrics(responses, gt_dict)
        
        # Print summary of metrics if available
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
            else:
                print(f"Semantic similarity to ground truth: {semantic_metrics.get('semantic_similarity', {}).get('mean', 0):.4f} ± {semantic_metrics.get('semantic_similarity', {}).get('std', 0):.4f}")
                print(f"Stability across runs: {semantic_metrics.get('semantic_similarity', {}).get('stability', 0):.4f}")
        
        if nlp_metrics and calculate_nlp:
            print("\n=== NLP-Based Metrics ===")
            
            # Calculate NLP metrics
            nlp_metrics = calculate_nlp_metrics(responses, gt_dict)
            
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
            
            
        
        # Calculate NLP metrics if semantic metrics are skipped
        nlp_metrics = {}
        if args.skip_semantic_metrics:
            nlp_metrics = calculate_nlp_metrics(responses, gt_dict)
        
        # Save the responses and metadata
        sample_results = {
            "sample_id": sample_id,
            "original_prompt": user_prompt,
            "modified_prompt": modified_user_prompt,
            "system_prompt": system_prompt,
            "ground_truth": gt_dict,
            "ground_truth_schema": create_json_schema(gt_dict),
            "responses": responses,
            "schemas": schema_list,
            "semantic_tree_metrics": semantic_metrics if calculate_semantic else {},
            "nlp_metrics": nlp_metrics if calculate_nlp else {},
            "metadata": {
                "model_id": model_id,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "run_num": args.run_num
            }
        }
        
        all_sample_results.append(sample_results)
        
        # Save individual sample results
        # Debug function to find non-serializable objects
        def find_non_serializable(obj, path=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    try:
                        json.dumps(v)
                    except TypeError:
                        print(f"Non-serializable at {new_path}: {type(v)}, value: {v}")
                        find_non_serializable(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    try:
                        json.dumps(item)
                    except TypeError:
                        print(f"Non-serializable at {new_path}: {type(item)}, value: {item}")
                        find_non_serializable(item, new_path)
        
        # Try to find any non-serializable objects before saving
        try:
            json.dumps(sample_results)
        except TypeError:
            print("Found non-serializable objects in sample_results:")
            find_non_serializable(sample_results)
        
        # Debug function to find non-serializable objects
        def find_non_serializable(obj, path=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    try:
                        json.dumps(v)
                    except TypeError:
                        print(f"Non-serializable at {new_path}: {type(v)}, value: {v}")
                        find_non_serializable(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    try:
                        json.dumps(item)
                    except TypeError:
                        print(f"Non-serializable at {new_path}: {type(item)}, value: {item}")
                        find_non_serializable(item, new_path)
        
        # Convert all objects to JSON-serializable types
        serializable_results = make_json_serializable(sample_results)
        
        # Try to find any remaining non-serializable objects
        try:
            json.dumps(serializable_results)
        except TypeError:
            print("Found non-serializable objects in sample_results:")
            find_non_serializable(serializable_results)
        
        sample_output_path = os.path.join(run_output_dir, f"{sample_id}.json")
        with open(sample_output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Saved results for {sample_id} to {sample_output_path}")
    
    # Save all results in a single file
    all_results_path = os.path.join(run_output_dir, "all_results.json")
    
    # Create the final results object
    final_results = {
        "metadata": {
            "model_id": model_id,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "run_num": args.run_num,
            "timestamp": run_timestamp
        },
        "results": all_sample_results
    }
    
    # Convert all objects to JSON-serializable types
    serializable_results = make_json_serializable(final_results)
    
    # Try to find any remaining non-serializable objects
    try:
        json.dumps(serializable_results)
    except TypeError as e:
        print(f"Found non-serializable objects in final_results: {e}")
        # Use the find_non_serializable function defined earlier
        find_non_serializable(serializable_results)
    
    with open(all_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nAll results saved to {all_results_path}")