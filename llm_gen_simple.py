#!/usr/bin/env python
"""
Generate text using an LLM with parallel inference.

This script generates text using an LLM with parallel inference and saves the results.
It focuses solely on generation, without evaluation metrics.

Usage:
    python llm_gen_simple.py --data-dir extracted_sharegpt_data --output-dir ./generations
"""

import boto3
from bedrock_utils import build_message, inference_with_converse_api
import argparse
import json
import ast
import concurrent.futures
import time
import os
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple
import sys

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def read_sharegpt(data_dir="data"):
    """
    Reads the dataset from the specified directory.
    """
    import os
    import json
    data_path = os.path.join(data_dir, f"all_conversations.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} does not exist.")

    with open(data_path, 'r') as file:
        data = json.load(file)
    
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
        ground_truth_dict: The ground truth dictionary used to derive the schema
        
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
        max_workers = min(32, (os.cpu_count() or 4) * 4)  # 4x CPU count is good for I/O bound tasks
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
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, (bool, np.bool_, np.bool)):
        return bool(obj)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using an LLM with parallel inference.")
    parser.add_argument("--model-id", type=str, default="us.anthropic.claude-3-5-sonnet-20241022-v2:0", help="The ID of the Bedrock model to use.")
    parser.add_argument("--data-dir", type=str, required=True, help="The directory containing the data files.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers for inference. Default is auto-determined based on CPU count.")
    parser.add_argument("--run-num", type=int, default=5, help="Number of inference runs to perform.")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling (0.1-2.0). Higher values produce more random outputs.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--top-k", type=int, default=200, help="Top-k sampling parameter.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--output-dir", type=str, default="./generations", help="Directory to save generation results.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit the number of samples to process.")
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt to guide the output structure.")
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
        if not args.include_schema:
            # Use original prompt without schema
            modified_user_prompt = user_prompt
            message = build_message(texts=[user_prompt])
            print(f"\nRunning inference on original prompt: {user_prompt[:100]}...")
            print(f"System prompt: {system_prompt[:100]}...")
        else:
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
        
        print(f"Received {len(responses)} responses")
        
        # Extract schema for each response
        schema_list = [extract_schema(response) for response in responses]
        print(f"Extracted schemas for {len(schema_list)} responses")
        
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
            "metadata": {
                "model_id": model_id,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "run_num": args.run_num
            }
        }
        
        # Convert all objects to JSON-serializable types
        serializable_results = make_json_serializable(sample_results)
        
        all_sample_results.append(serializable_results)
        
        # Save individual sample results
        sample_output_path = os.path.join(run_output_dir, f"{sample_id}.json")
        with open(sample_output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Saved results for {sample_id} to {sample_output_path}")
    
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
    
    # Save all results in a single file
    all_results_path = os.path.join(run_output_dir, "all_results.json")
    with open(all_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nAll results saved to {all_results_path}")