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
from typing import List, Dict, Any, Optional, Union, Callable, Set


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

def run_inference(client, model_id, messages, system_prompts=None, max_tokens=8000, temperature=0.1, top_p=0.9, top_k=200, run_num=5, max_workers=None):
    """
    Runs inference in parallel using ThreadPoolExecutor.
    
    Args:
        client: Bedrock client (not used directly, each thread creates its own)
        model_id: Model ID to use
        messages: List of messages to send
        system_prompts: Optional system prompts
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using an LLM with parallel inference.")
    parser.add_argument("--model-id", type=str, default="us.anthropic.claude-3-5-sonnet-20241022-v2:0", help="The ID of the Bedrock model to use.")
    parser.add_argument("--data-dir", type=str, required=True, help="The directory containing the data files.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers for inference. Default is auto-determined based on CPU count.")
    parser.add_argument("--run-num", type=int, default=5, help="Number of inference runs to perform.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save evaluation results.")
    parser.add_argument("--schema-only", action="store_true", help="Only evaluate schema consistency without saving full responses.")
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
    
    # List dataset
    dataset_dict = read_sharegpt(args.data_dir)
    
    results = []
    all_sample_results = []  # Store results for all samples
    
    # Create timestamped output directory for this run
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_output_dir = os.path.join(args.output_dir, f"llm_gen_results_{run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_output_dir}")
    
    for sample_idx, item in enumerate(dataset_dict[:2]):
        sample_id = f"sample_{sample_idx:03d}"
        print(f"\n{'='*60}")
        print(f"PROCESSING SAMPLE {sample_idx + 1}/{min(len(dataset_dict), 2)}: {sample_id}")
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
        
        message = build_message(texts=[user_prompt])
        
        print(f"\nRunning inference on prompt: {user_prompt[:100]}...")
        print(f"System prompt: {system_prompt[:100]}...")
        
        responses = run_inference(
            client=client,
            model_id=model_id,
            messages=[message],
            system_prompts=system_prompt,
            max_tokens=8000,
            temperature=0.1,
            top_p=0.9,
            top_k=200,
            run_num=args.run_num,
            max_workers=args.max_workers
        )
        
        print(responses)
        
        schema_list = [extract_schema(response) for response in responses]
        print(f"schema_list: {schema_list}")