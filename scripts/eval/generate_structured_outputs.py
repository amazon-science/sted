#!/usr/bin/env python
"""
Generate text using an LLM with parallel inference.

This script generates text using an LLM with parallel inference and saves the results.
It focuses solely on generation, without evaluation metrics.

Usage:
    python generate_structured_outputs.py --data-dir sharegpt_data --output-dir ./generations
"""

import boto3
from dotenv import load_dotenv
from sted.bedrock_utils import build_message, inference_with_converse_api
import argparse
import json
import concurrent.futures
import time
import os
import numpy as np
from typing import List, Dict, Any
import sys
from tqdm import tqdm
import re
import openai

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "not-set"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

provider_mapping = {
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": "bedrock",
    "us.anthropic.claude-sonnet-4-20250514-v1:0": "bedrock",
    "us.qwen.qwen3-235b-a22b-2507-v1:0": "bedrock",
    "us.deepseek.v3-v1:0": "bedrock",
    "openai/gpt-5": "openai",
    "openai/gpt-4o": "openai",
    "google/gemini-2.5-pro": "openai",
    "x-ai/grok-4": "openai"
}

def estimate_tokens(text):
    """
    Estimate token count using character-based approximation.
    Uses conservative 3:1 ratio (3 chars = 1 token) to avoid underestimation.
    """
    return len(text) // 3

def truncate_text_by_chars(text, max_tokens):
    """
    Truncate text to fit within estimated max_tokens limit.
    Uses conservative character-based estimation.
    """
    # Use conservative 3:1 ratio for truncation
    max_chars = max_tokens * 3
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [TRUNCATED]"

def extract_json_from_string(input_string):
    """
    Extract JSON data from a string with a prefix.
    Handles both object {} and array [] formats without modification.
    """
    # Find the first { or [
    match = re.search(r'[{[]', input_string)
    if not match:
        raise ValueError("No JSON data found in the string")
    
    start_pos = match.start()
    start_char = match.group()
    
    # Determine the matching closing character
    end_char = '}' if start_char == '{' else ']'
    
    # Find the matching closing bracket/brace
    count = 0
    end_pos = -1
    
    for i in range(start_pos, len(input_string)):
        if input_string[i] == start_char:
            count += 1
        elif input_string[i] == end_char:
            count -= 1
            if count == 0:
                end_pos = i
                break
    
    if end_pos == -1:
        raise ValueError("No matching closing bracket/brace found")
    
    # Extract the JSON string
    json_string = input_string[start_pos:end_pos + 1]
    
    # Parse and return the JSON
    return json.loads(json_string)

def read_sharegpt(dataset_dir="data"):
    """
    Reads the dataset from the specified directory.
    """    
    dataset_list = os.listdir(dataset_dir)
    print(f"dataset_list: {dataset_list}")
    
    json_data = []
    for dataset_name in dataset_list:
        print(f"Processing dataset: {dataset_name}")
        data_dir = os.path.join(dataset_dir, dataset_name)
        if not os.path.exists(data_dir) or '.DS_Store' == dataset_name:
            print(f"Dataset {dataset_name} does not exist. Skipping.")
            continue
        
        data_path = os.path.join(data_dir, f"all_conversations.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file {data_path} does not exist.")

        with open(data_path, 'r') as file:
            # convert json to list
            json_data.extend(json.load(file))
    
    return json_data

def _single_inference(model_id, user_prompt, system_prompts=None, max_tokens=8000, temperature=0.1, top_p=0.9, top_k=200, task_id=None):
    """
    Helper function to run a single inference request using threads.
    """
    try:
        # Create a new client for each thread to avoid potential thread safety issues
        # Uses AWS credentials from environment variables or ~/.aws/credentials
        thread_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Print task ID if provided (useful for debugging)
        if task_id is not None:
            print(f"Starting task {task_id}")
        
        provider = provider_mapping.get(model_id, "bedrock")
        print(f"provider: {provider} for model_id: {model_id}")
        if provider == "bedrock":
            message = build_message(texts=[user_prompt])
            # Attempt to run inference
            response = inference_with_converse_api(
                thread_client,
                model_id=model_id,
                messages=[message],
                system_prompts=system_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            if not response or not isinstance(response, list) or len(response) == 0:
                print(f"Warning: Empty or invalid response received for task {task_id}")
                return {}
        
            print(f"Response received for task {task_id}: {len(response)} items")
            response_text = response[0].get('text', '{}')
            print(f"original response: {response_text}")
        else:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": f"{system_prompts}"},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print(f"Response received for task {task_id}: {response}")
            response_text = response.choices[0].message.content
            print(f"Response received for task {task_id}: {response_text}")
        # remove space or new lines in ends of response_text
        response_text = response_text.strip()
        
        # Parse the response with safer error handling
        try:
            parsed_response = extract_json_from_string(response_text)
            return parsed_response
        except (SyntaxError, ValueError) as parse_error:
            print(f"Error parsing response for task {task_id}: {parse_error}")
            print(f"Raw response text: {response_text}...")
            return {}
    except Exception as e:
        print(f"Error during inference for task {task_id}: {e}")
        return {}  # Return an empty dict on error

def create_json_schema(ground_truth_dict, max_depth=10):
    """
    Creates a JSON schema from the ground truth dictionary or list of dictionaries.
    
    Args:
        ground_truth_dict: The ground truth dictionary or list of dictionaries to create schema from
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        A dictionary representing the JSON schema
    """
    if max_depth <= 0:
        return {"type": "object"}
    
    # Handle list of dictionaries
    if isinstance(ground_truth_dict, list):
        if ground_truth_dict and all(isinstance(item, dict) for item in ground_truth_dict):
            # Create schema from the first item in the list
            item_schema = create_json_schema(ground_truth_dict[0], max_depth - 1)
            return {
                "type": "array",
                "items": item_schema
            }
        else:
            # For other lists, determine the type of items
            item_type = "string"  # Default type
            if ground_truth_dict:
                if all(isinstance(item, int) for item in ground_truth_dict):
                    item_type = "integer"
                elif all(isinstance(item, float) for item in ground_truth_dict):
                    item_type = "number"
                elif all(isinstance(item, bool) for item in ground_truth_dict):
                    item_type = "boolean"
            
            return {
                "type": "array",
                "items": {"type": item_type}
            }
    
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

def run_inference(model_id, user_prompt, system_prompts=None, max_tokens=8000, temperature=0.1, top_p=0.9, top_k=200, run_num=5, max_workers=None):
    """
    Runs inference in parallel using ThreadPoolExecutor.
    
    Args:
        model_id: Model ID to use
        user_prompt: The user prompt to send to the model
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
                model_id,
                user_prompt,
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
    
    print(f"final responses: {responses}")
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
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif hasattr(obj, 'item'):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    elif hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except (ValueError, TypeError):
            pass
    # Handle any other NumPy types we might have missed
    elif type(obj).__module__ == 'numpy':
        try:
            return obj.item() if hasattr(obj, 'item') else obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        except (ValueError, TypeError):
            return str(obj)
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
    parser.add_argument("--sample-limit", type=int, default=-1, help="Limit the number of samples to process.")
    parser.add_argument("--include-schema", action="store_true", help="Include JSON schema in the prompt to guide the output structure.")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--max-context-tokens", type=int, default=32767, help="Maximum context tokens to use (default: 32767 for 32k models).")
    parser.add_argument("--skip-long-samples", action="store_true", help="Skip samples that exceed token limit instead of truncating.")
    args = parser.parse_args()
    
    
    args.include_schema = True
    
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
    
    print(f"Processing {len(dataset_dict)} samples, args.sample_limit: {args.sample_limit}")
    # Determine how many samples to process
    if args.sample_limit > 0:
        samples_to_process = dataset_dict[:args.sample_limit]
    else:
        samples_to_process = dataset_dict
    
    print(f"Processing {type(samples_to_process)} samples")
    for sample_idx, item in tqdm(enumerate(samples_to_process)):
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
            print(f"gt_value: {gt_value}")
            gt_dict = {"error": "Invalid JSON", "raw_value": gt_value}
        
        # Check estimated token count before processing
        total_estimated_tokens = estimate_tokens(system_prompt + user_prompt + gt_value)
        print(f"Estimated total tokens: {total_estimated_tokens}")
        
        # Use command line argument for max context length
        MAX_CONTEXT_TOKENS = args.max_context_tokens
        
        if total_estimated_tokens > MAX_CONTEXT_TOKENS:
            print(f"WARNING: Sample {sample_id} exceeds estimated token limit ({total_estimated_tokens} > {MAX_CONTEXT_TOKENS})")
            
            if args.skip_long_samples:
                print(f"Skipping sample {sample_id} due to --skip-long-samples flag")
                continue
            
            # Try to truncate the user prompt while keeping system prompt and ground truth
            system_estimated_tokens = estimate_tokens(system_prompt)
            gt_estimated_tokens = estimate_tokens(gt_value)
            available_tokens = MAX_CONTEXT_TOKENS - system_estimated_tokens - gt_estimated_tokens - 1000  # Buffer for schema
            
            if available_tokens < 1000:
                print(f"ERROR: Sample {sample_id} cannot be processed - system prompt and ground truth too long")
                print(f"System estimated tokens: {system_estimated_tokens}, GT estimated tokens: {gt_estimated_tokens}")
                continue
            
            print(f"Truncating user prompt from {estimate_tokens(user_prompt)} to ~{available_tokens} estimated tokens")
            user_prompt = truncate_text_by_chars(user_prompt, available_tokens)
            print(f"After truncation: {estimate_tokens(user_prompt)} estimated tokens")
        
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
            
            # Check if the modified prompt with schema exceeds limits
            modified_total_estimated_tokens = estimate_tokens(system_prompt + modified_user_prompt)
            if modified_total_estimated_tokens > MAX_CONTEXT_TOKENS:
                print(f"WARNING: Modified prompt with schema exceeds estimated limit ({modified_total_estimated_tokens} tokens)")
                # Try to truncate the user part of the modified prompt
                schema_part = modified_user_prompt[len(user_prompt):]
                schema_estimated_tokens = estimate_tokens(schema_part)
                available_for_user = MAX_CONTEXT_TOKENS - estimate_tokens(system_prompt) - schema_estimated_tokens - 1000
                
                if available_for_user < 500:
                    print(f"ERROR: Cannot fit schema - falling back to original prompt")
                    modified_user_prompt = user_prompt
                else:
                    truncated_user = truncate_text_by_chars(user_prompt, available_for_user)
                    modified_user_prompt = create_prompt_with_schema(
                        user_prompt=truncated_user,
                        ground_truth_dict=gt_dict
                    )
            
            print(f"\nRunning inference on prompt with schema")
            print(f"Original prompt: {user_prompt[:100]}...")
            print(f"System prompt: {system_prompt[:100]}...")
            print(f"Schema added to prompt for better comparison with ground truth")
            
            # Save the modified prompt for reference
            prompt_output_path = os.path.join(run_output_dir, f"{sample_id}_modified_prompt.txt")
            with open(prompt_output_path, 'w') as f:
                f.write(modified_user_prompt)
            print(f"Saved modified prompt to {prompt_output_path}")
        
        # Final estimated token count check
        final_estimated_tokens = estimate_tokens(system_prompt + modified_user_prompt)
        print(f"Final estimated tokens: {final_estimated_tokens}")
        
        if final_estimated_tokens > MAX_CONTEXT_TOKENS:
            print(f"ERROR: Sample {sample_id} still exceeds estimated token limit after truncation. Skipping.")
            continue
        
        responses = run_inference(
            model_id=model_id,
            user_prompt=modified_user_prompt,
            system_prompts=system_prompt,
            max_tokens=args.max_tokens,
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
