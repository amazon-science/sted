#!/usr/bin/env python
"""
Generate text using an LLM with parallel inference.

This script generates text using an LLM with parallel inference and saves the results.
It focuses solely on generation, without evaluation metrics.

Usage:
    python generate_structured_outputs.py --data-dir sharegpt_data --output-dir ./generations
"""

import boto3
from botocore.config import Config
from dotenv import load_dotenv
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
import logging
from datetime import datetime

# Load environment variables from .env file (override=True to override existing env vars)
load_dotenv(override=True)

# Setup logging
def setup_logging(output_dir=None, log_level=logging.INFO):
    """Setup logging configuration with both file and console handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if output_dir provided)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all debug info
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger

# Initialize logger (will be reconfigured with output_dir in main)
logger = logging.getLogger(__name__)

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "not-set"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# Import directly from submodules to avoid __init__.py which requires bert_score
from sted.model_config import get_provider, get_display_name
from sted.bedrock_utils import build_message, inference_with_converse_api

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
    # Strip markdown code block wrappers (e.g., ```json ... ```)
    input_string = re.sub(r'^```(?:json)?\s*\n?', '', input_string.strip())
    input_string = re.sub(r'\n?```\s*$', '', input_string)

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

    Note: Uses sorted() to ensure deterministic sample ordering across runs.
    """
    # Filter out macOS metadata files (._*) and .DS_Store
    dataset_list = sorted([d for d in os.listdir(dataset_dir) if not d.startswith('._') and d != '.DS_Store'])
    print(f"dataset_list: {dataset_list}")

    json_data = []
    for dataset_name in dataset_list:
        print(f"Processing dataset: {dataset_name}")
        data_dir = os.path.join(dataset_dir, dataset_name)
        if not os.path.exists(data_dir):
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
    import time
    start_time = time.time()

    logger.debug(f"[Task {task_id}] Starting inference with model={model_id}, temp={temperature}, max_tokens={max_tokens}")
    logger.debug(f"[Task {task_id}] User prompt length: {len(user_prompt)} chars")

    try:
        # Create a new client for each thread to avoid potential thread safety issues
        # Uses AWS credentials from environment variables or ~/.aws/credentials
        aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
        aws_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')

        logger.debug(f"[Task {task_id}] AWS Region: {aws_region}, Key ID present: {bool(aws_key_id)}, Secret present: {bool(aws_secret)}")

        # Configure boto3 with increased timeout for slower models like Claude-Opus-4
        boto_config = Config(
            retries={'max_attempts': 5, 'mode': 'adaptive'},
            read_timeout=300,  # 5 minutes timeout for large model responses
            connect_timeout=30
        )
        thread_client = boto3.client(
            'bedrock-runtime',
            region_name=aws_region,
            aws_access_key_id=aws_key_id,
            aws_secret_access_key=aws_secret,
            config=boto_config
        )

        provider = get_provider(model_id)
        logger.info(f"[Task {task_id}] Using provider: {provider}")

        if provider == "bedrock":
            message = build_message(texts=[user_prompt])
            logger.debug(f"[Task {task_id}] Calling Bedrock Converse API...")

            response = inference_with_converse_api(
                thread_client,
                model_id=model_id,
                messages=[message],
                system_prompts=system_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            print(f"response: {response}")

            if not response or not isinstance(response, list) or len(response) == 0:
                logger.warning(f"[Task {task_id}] Empty or invalid response from Bedrock")
                return {}

            response_text = response[0].get('text', '{}')

            # Handle models that return reasoningContent + text (e.g., Minimax-M2)
            # Response format: [{'reasoningContent': {...}}, {'text': '[actual JSON]'}]
            if response_text == '{}' and len(response) > 1:
                for item in response:
                    if isinstance(item, dict) and 'text' in item:
                        response_text = item['text']
                        break
            logger.debug(f"[Task {task_id}] Bedrock response received, length: {len(response_text)} chars")
        else:
            logger.debug(f"[Task {task_id}] Calling OpenAI-compatible API...")
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
            response_text = response.choices[0].message.content
            logger.debug(f"[Task {task_id}] OpenAI response received, length: {len(response_text)} chars")

        response_text = response_text.strip()
        elapsed_time = time.time() - start_time
        logger.info(f"[Task {task_id}] API call completed in {elapsed_time:.2f}s, response length: {len(response_text)} chars")

        try:
            parsed_response = extract_json_from_string(response_text)
            logger.info(f"[Task {task_id}] JSON extraction successful")
            return parsed_response
        except (SyntaxError, ValueError) as e:
            # Log the failed extraction for debugging
            logger.error(f"[Task {task_id}] JSON extraction failed: {e}")
            logger.error(f"[Task {task_id}] Response length: {len(response_text)} chars")
            logger.debug(f"[Task {task_id}] Response preview (first 500 chars): {response_text[:500]}")
            logger.debug(f"[Task {task_id}] Response preview (last 200 chars): {response_text[-200:] if len(response_text) > 200 else response_text}")
            return {"_extraction_error": str(e), "_response_preview": response_text[:1000]}
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"[Task {task_id}] API Error after {elapsed_time:.2f}s: {e}")
        import traceback
        logger.debug(f"[Task {task_id}] Full traceback: {traceback.format_exc()}")
        return {"_api_error": str(e)}

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
        max_workers = min(32, (os.cpu_count() or 4) * 4)

    logger.info(f"Starting {run_num} parallel inference runs with {max_workers} workers")
    logger.debug(f"Model: {model_id}, Temperature: {temperature}, Max tokens: {max_tokens}")

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _single_inference, model_id, user_prompt, system_prompts,
                max_tokens, temperature, top_p, top_k, i
            ): i for i in range(run_num)
        }

        for future in concurrent.futures.as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                responses.append(future.result())
            except Exception as e:
                logger.error(f"[Task {task_id}] Failed: {e}")
                responses.append({})

    elapsed_time = time.time() - start_time
    valid_count = sum(1 for r in responses if r)
    logger.info(f"Completed {run_num} runs in {elapsed_time:.1f}s ({valid_count}/{run_num} valid)")
    print(f"  Completed {run_num} runs in {elapsed_time:.1f}s ({valid_count}/{run_num} valid)")
    
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
    parser.add_argument("--run-num", type=int, default=10, help="Number of inference runs to perform.")
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
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # Reconfigure the module-level logger with output directory
    setup_logging(args.output_dir, log_level)

    if not args.verbose:
        # Reduce boto3 logging noise
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

    print(f"Starting parallel inference with {args.run_num} runs using ThreadPoolExecutor")
    print(f"Model: {model_id}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}, Top-k: {args.top_k}")
    
    # List dataset
    dataset_dict = read_sharegpt(args.data_dir)
    
    all_sample_results = []  # Store results for all samples
    
    # Create timestamped output directory for this run
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    temp_str = f"temp_{args.temperature:.2f}".replace('.', '_')
    model_name = get_display_name(model_id)
    run_output_dir = os.path.join(args.output_dir, f"llm_gen_results_{model_name}_{temp_str}_{run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Print run configuration
    print(f"\n{'='*60}")
    print(f"LLM Structured Output Generation")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Temperature: {args.temperature}")
    print(f"Runs per sample: {args.run_num}")
    print(f"Output directory: {run_output_dir}")
    print(f"{'='*60}\n")
    
    # Determine how many samples to process
    if args.sample_limit > 0:
        samples_to_process = dataset_dict[:args.sample_limit]
    else:
        samples_to_process = dataset_dict
    
    total_samples = len(samples_to_process)
    print(f"Processing {total_samples} samples...\n")
    
    for sample_idx, item in tqdm(enumerate(samples_to_process), total=total_samples, desc="Generating outputs"):
        sample_id = f"sample_{sample_idx:03d}"
        logger.info(f"Processing {sample_id} ({sample_idx + 1}/{total_samples})")

        system_prompt = item['conversations'][0]['value']
        user_prompt = item['conversations'][1]['value']
        gt_value = item['conversations'][2]['value']

        logger.debug(f"[{sample_id}] System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}")

        try:
            gt_dict = json.loads(gt_value)
            logger.debug(f"[{sample_id}] Ground truth parsed successfully, keys: {list(gt_dict.keys()) if isinstance(gt_dict, dict) else 'list'}")
        except json.JSONDecodeError as e:
            logger.warning(f"[{sample_id}] Could not parse ground truth JSON: {e}")
            print(f"\n[{sample_id}] Warning: Could not parse ground truth JSON: {e}")
            gt_dict = {"error": "Invalid JSON", "raw_value": gt_value}
        
        # Check estimated token count before processing
        total_estimated_tokens = estimate_tokens(system_prompt + user_prompt + gt_value)
        logger.debug(f"[{sample_id}] Estimated total tokens: {total_estimated_tokens}")
        print(f"Estimated total tokens: {total_estimated_tokens}")

        # Use command line argument for max context length
        MAX_CONTEXT_TOKENS = args.max_context_tokens

        if total_estimated_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"[{sample_id}] Exceeds estimated token limit ({total_estimated_tokens} > {MAX_CONTEXT_TOKENS})")
            print(f"WARNING: Sample {sample_id} exceeds estimated token limit ({total_estimated_tokens} > {MAX_CONTEXT_TOKENS})")

            if args.skip_long_samples:
                logger.info(f"[{sample_id}] Skipping due to --skip-long-samples flag")
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
            
            user_prompt = truncate_text_by_chars(user_prompt, available_tokens)
        
        # Determine whether to use the original prompt or a modified one with schema
        if not args.include_schema:
            modified_user_prompt = user_prompt
        else:
            # Create a modified prompt with the JSON schema
            modified_user_prompt = create_prompt_with_schema(
                user_prompt=user_prompt,
                ground_truth_dict=gt_dict
            )
            
            # Check if the modified prompt with schema exceeds limits
            modified_total_estimated_tokens = estimate_tokens(system_prompt + modified_user_prompt)
            if modified_total_estimated_tokens > MAX_CONTEXT_TOKENS:
                schema_part = modified_user_prompt[len(user_prompt):]
                schema_estimated_tokens = estimate_tokens(schema_part)
                available_for_user = MAX_CONTEXT_TOKENS - estimate_tokens(system_prompt) - schema_estimated_tokens - 1000
                
                if available_for_user < 500:
                    modified_user_prompt = user_prompt
                else:
                    truncated_user = truncate_text_by_chars(user_prompt, available_for_user)
                    modified_user_prompt = create_prompt_with_schema(
                        user_prompt=truncated_user,
                        ground_truth_dict=gt_dict
                    )
            
            # Save the modified prompt for reference
            prompt_output_path = os.path.join(run_output_dir, f"{sample_id}_modified_prompt.txt")
            with open(prompt_output_path, 'w') as f:
                f.write(modified_user_prompt)
        
        # Final estimated token count check
        final_estimated_tokens = estimate_tokens(system_prompt + modified_user_prompt)
        if final_estimated_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"[{sample_id}] Skipped: exceeds token limit ({final_estimated_tokens} > {MAX_CONTEXT_TOKENS})")
            print(f"\n[{sample_id}] Skipped: exceeds token limit")
            continue

        logger.info(f"[{sample_id}] Starting inference with final token count: {final_estimated_tokens}")
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
        
        # Extract schema for each response
        schema_list = [extract_schema(response) for response in responses]
        
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

        # Handle both dict and list responses - lists are valid JSON responses
        valid_responses = sum(1 for r in responses if r and (isinstance(r, list) or (isinstance(r, dict) and not r.get('_api_error') and not r.get('_extraction_error'))))
        logger.info(f"[{sample_id}] Completed - {valid_responses}/{len(responses)} valid responses, saved to {sample_output_path}")
    
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

    logger.info(f"All results saved to: {all_results_path}")
    logger.info(f"Generation complete! Processed {len(all_sample_results)} samples")

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Processed: {len(all_sample_results)} samples")
    print(f"Results saved to: {run_output_dir}")
    print(f"{'='*60}")
