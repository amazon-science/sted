#!/usr/bin/env python3
"""
Generate Tool Calls for STED Consistency Evaluation

This script generates tool call outputs from the xLAM dataset using models
configured in sted/model_config.py. It captures the LLM's tool calling
decisions without executing the tools.

Supports:
- AWS Bedrock models (Claude, Llama, Nova, etc.)
- OpenAI-compatible models (GPT, Gemini via OpenRouter)

Usage:
    python generate_tool_calls.py --model us.anthropic.claude-3-5-sonnet-20241022-v2:0 --num-runs 5 --num-samples 100
    python generate_tool_calls.py --model openai/gpt-4.1-mini --temperature 0.7 --num-runs 10
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables (skip AWS vars - use default credential chain)
load_dotenv()
# Remove any placeholder AWS credentials from .env to use CLI credentials
for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']:
    if os.environ.get(key, '').startswith('your-'):
        del os.environ[key]

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted.model_config import MODEL_REGISTRY, get_provider, get_display_name
from sted.bedrock_utils import inference_with_converse_api


def get_bedrock_client():
    """Get AWS Bedrock client with retry configuration for parallel execution."""
    boto_config = Config(
        retries={'max_attempts': 20, 'mode': 'adaptive'},
        max_pool_connections=100,  # Higher for parallel workers
        connect_timeout=30,
        read_timeout=120
    )
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
        config=boto_config
    )


def get_openai_client():
    """Get OpenAI-compatible client."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def xlam_tool_to_bedrock_tool(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert xLAM tool definition to Bedrock Converse API tool schema format.

    Bedrock format:
    {
        "toolSpec": {
            "name": "function_name",
            "description": "...",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
    }
    """
    name = tool_def.get("name", "unknown_function")
    description = tool_def.get("description", "No description provided")
    params = tool_def.get("parameters", {})

    # Convert xLAM parameter format to JSON Schema
    properties = {}
    required = []

    for param_name, param_info in params.items():
        if isinstance(param_info, dict):
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            param_default = param_info.get("default")

            # Map xLAM types to JSON Schema types
            type_map = {
                "str": "string",
                "string": "string",
                "int": "integer",
                "integer": "integer",
                "float": "number",
                "number": "number",
                "bool": "boolean",
                "boolean": "boolean",
                "list": "array",
                "array": "array",
                "dict": "object",
                "object": "object",
            }
            json_type = type_map.get(param_type.lower(), "string")

            properties[param_name] = {
                "type": json_type,
                "description": param_desc,
            }

            # If no default, consider it required
            if param_default is None:
                required.append(param_name)
        else:
            # Simple format: just the type
            properties[param_name] = {"type": "string"}
            required.append(param_name)

    return {
        "toolSpec": {
            "name": name,
            "description": description,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    }


def xlam_tool_to_openai_tool(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert xLAM tool definition to OpenAI tool schema format.

    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "function_name",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
    }
    """
    name = tool_def.get("name", "unknown_function")
    description = tool_def.get("description", "No description provided")
    params = tool_def.get("parameters", {})

    # Convert xLAM parameter format to JSON Schema
    properties = {}
    required = []

    for param_name, param_info in params.items():
        if isinstance(param_info, dict):
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            param_default = param_info.get("default")

            # Map xLAM types to JSON Schema types
            type_map = {
                "str": "string",
                "string": "string",
                "int": "integer",
                "integer": "integer",
                "float": "number",
                "number": "number",
                "bool": "boolean",
                "boolean": "boolean",
                "list": "array",
                "array": "array",
                "dict": "object",
                "object": "object",
            }
            json_type = type_map.get(param_type.lower(), "string")

            properties[param_name] = {
                "type": json_type,
                "description": param_desc,
            }

            # If no default, consider it required
            if param_default is None:
                required.append(param_name)
        else:
            # Simple format: just the type
            properties[param_name] = {"type": "string"}
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    }


def has_long_tool_names(tools: List[Dict], max_length: int = 64) -> bool:
    """
    Check if any tool name exceeds the specified length limit.

    AWS Bedrock enforces a 64-character limit on tool names: ^[a-zA-Z0-9_-]{1,64}$
    This applies to both Converse API and invoke_model (Anthropic Messages format).

    Args:
        tools: List of tool definitions (xLAM format with 'name' field)
        max_length: Maximum allowed tool name length (default: 64 for Bedrock)

    Returns:
        True if any tool name exceeds the limit, False otherwise
    """
    for tool in tools:
        name = tool.get("name", "")
        if len(name) > max_length:
            return True
    return False


def get_long_tool_names(tools: List[Dict], max_length: int = 64) -> List[str]:
    """
    Get list of tool names that exceed the specified length limit.

    Args:
        tools: List of tool definitions (xLAM format with 'name' field)
        max_length: Maximum allowed tool name length (default: 64 for Bedrock)

    Returns:
        List of tool names that exceed the limit
    """
    long_names = []
    for tool in tools:
        name = tool.get("name", "")
        if len(name) > max_length:
            long_names.append(name)
    return long_names


def load_xlam_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """Load xLAM dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    if num_samples and num_samples > 0:
        data = data[:num_samples]

    return data


def convert_openai_tool_to_xlam(openai_tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAI-style tool definition to xLAM format.

    OpenAI format:
    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    xLAM format:
    {"name": "...", "description": "...", "parameters": {...}}
    """
    if "function" in openai_tool:
        func = openai_tool["function"]
        params = func.get("parameters", {})

        # Convert JSON Schema parameters to xLAM format
        xlam_params = {}
        properties = params.get("properties", {})
        required = params.get("required", [])

        for param_name, param_info in properties.items():
            xlam_params[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_name not in required:
                xlam_params[param_name]["default"] = None

        return {
            "name": func.get("name", "unknown"),
            "description": func.get("description", ""),
            "parameters": xlam_params
        }
    else:
        # Already in xLAM-like format
        return openai_tool


def load_toucan_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    Load Toucan dataset (extracted tool_calls format from download_toucan_data.py).

    Args:
        dataset_path: Path to the Toucan dataset JSON file
        num_samples: Number of samples to load (-1 or None for all samples)

    Converts to xLAM-compatible format for processing.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    if num_samples and num_samples > 0:
        data = data[:num_samples]

    converted = []
    for item in data:
        # Parse tools from string if needed
        tools_raw = item.get("tools", "[]")
        if isinstance(tools_raw, str):
            try:
                tools = json.loads(tools_raw)
            except json.JSONDecodeError:
                tools = []
        else:
            tools = tools_raw

        # Convert OpenAI-style tools to xLAM format
        xlam_tools = [convert_openai_tool_to_xlam(t) for t in tools]

        # Get ground truth tool calls
        ground_truth = item.get("tool_calls", [])

        converted.append({
            "id": item.get("id", item.get("uuid", "")),
            "query": item.get("question", ""),
            "tools": xlam_tools,
            "answers": ground_truth,
        })

    return converted


def load_toucan_from_huggingface(num_samples: int = -1) -> List[Dict]:
    """
    Load Toucan dataset directly from Hugging Face and convert to xLAM format.

    This extracts tools from the 'tools' field and converts them for use.

    Args:
        num_samples: Number of samples to load (-1 for all samples)
    """
    import ast
    import re
    from datasets import load_dataset

    ds = load_dataset("beyoru/Toucan-1.5M-structured-Qwen", split="train", streaming=True)

    converted = []
    for i, sample in enumerate(ds):
        if num_samples > 0 and i >= num_samples:
            break

        # Parse tools
        tools_raw = sample.get("tools", "[]")
        try:
            tools = json.loads(tools_raw)
        except json.JSONDecodeError:
            continue

        # Convert to xLAM format
        xlam_tools = [convert_openai_tool_to_xlam(t) for t in tools]

        # Extract tool calls from first assistant message
        messages = sample.get("messages", [])
        tool_calls = []
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Extract <tool_call>...</tool_call>
                pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        tc = ast.literal_eval(match.strip())
                        if "arguments" in tc and isinstance(tc["arguments"], str):
                            tc["arguments"] = json.loads(tc["arguments"])
                        tool_calls.append(tc)
                    except (ValueError, SyntaxError, json.JSONDecodeError):
                        pass
                break  # Only first assistant message

        if not xlam_tools:
            continue

        converted.append({
            "id": sample.get("uuid", f"toucan_{i}"),
            "query": sample.get("question", ""),
            "tools": xlam_tools,
            "answers": tool_calls,
        })

    return converted


def extract_tool_calls_bedrock(response_content: List[Dict]) -> List[Dict[str, Any]]:
    """Extract tool calls from Bedrock Converse API response."""
    tool_calls = []

    if not response_content:
        return tool_calls

    for content in response_content:
        if "toolUse" in content:
            tool_calls.append({
                "name": content["toolUse"]["name"],
                "arguments": content["toolUse"]["input"]
            })

    return tool_calls


def extract_tool_calls_openai(response) -> List[Dict[str, Any]]:
    """Extract tool calls from OpenAI response."""
    tool_calls = []

    if not response.choices:
        return tool_calls

    message = response.choices[0].message
    if message.tool_calls:
        for tc in message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "name": tc.function.name,
                "arguments": args
            })

    return tool_calls


def _single_bedrock_inference(
    client,
    model_id: str,
    messages: List[Dict],
    bedrock_tools: List[Dict],
    temperature: float,
    max_tokens: int,
    run_idx: int,
) -> List[Dict]:
    """Single inference call for parallel execution."""
    try:
        response_content = inference_with_converse_api(
            bedrock_client=client,
            model_id=model_id,
            messages=messages,
            tools=bedrock_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            return_content=True
        )
        return extract_tool_calls_bedrock(response_content)
    except Exception as e:
        print(f"  Run {run_idx + 1} failed: {e}")
        return []


def generate_tool_calls_bedrock(
    client,
    model_id: str,
    query: str,
    tools: List[Dict],
    num_runs: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_workers: int = None,
) -> List[List[Dict]]:
    """Generate tool calls using Bedrock Converse API with parallel execution."""
    bedrock_tools = [xlam_tool_to_bedrock_tool(t) for t in tools]
    messages = [{"role": "user", "content": [{"text": query}]}]

    # Determine optimal number of workers if not specified
    if max_workers is None:
        max_workers = min(num_runs, 10)  # Cap at 10 to avoid rate limits

    # Use ThreadPoolExecutor for parallel processing
    all_runs = [None] * num_runs
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _single_bedrock_inference,
                client, model_id, messages, bedrock_tools,
                temperature, max_tokens, i
            ): i for i in range(num_runs)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_runs[idx] = future.result()
            except Exception as e:
                print(f"  Run {idx + 1} failed: {e}")
                all_runs[idx] = []

    return all_runs


def generate_tool_calls_bedrock_multiturn(
    client,
    model_id: str,
    query: str,
    tools: List[Dict],
    conversation_turns: List[Dict],
    num_runs: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_turns: int = 10,
) -> List[List[Dict]]:
    """
    Generate tool calls using Bedrock Converse API with multi-turn support.

    Uses original tool responses from the dataset to continue the conversation
    until the model stops calling tools or max_turns is reached.

    Args:
        client: Bedrock client
        model_id: Model ID
        query: Initial user query
        tools: Tool definitions
        conversation_turns: Pre-extracted turns with tool_calls and tool_responses
        num_runs: Number of independent runs
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        max_turns: Maximum conversation turns to prevent infinite loops

    Returns:
        List of runs, where each run contains all tool calls made across turns
    """
    bedrock_tools = [xlam_tool_to_bedrock_tool(t) for t in tools]

    # Build a mapping from tool call (name + args) to response
    # This allows us to look up the appropriate response for any tool call
    tool_response_map = {}
    for turn in conversation_turns:
        turn_tool_calls = turn.get('tool_calls', [])
        turn_responses = turn.get('tool_responses', [])
        # Map each tool call to its response (by index)
        for i, tc in enumerate(turn_tool_calls):
            if i < len(turn_responses):
                # Create a key from tool name + serialized args
                key = (tc.get('name', ''), json.dumps(tc.get('arguments', {}), sort_keys=True))
                tool_response_map[key] = turn_responses[i]

    all_runs = []

    for run_idx in range(num_runs):
        run_tool_calls = []
        messages = [{"role": "user", "content": [{"text": query}]}]

        for turn_idx in range(max_turns):
            try:
                response_content = inference_with_converse_api(
                    bedrock_client=client,
                    model_id=model_id,
                    messages=messages,
                    tools=bedrock_tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    return_content=True
                )

                # Extract tool calls from this turn
                turn_calls = extract_tool_calls_bedrock(response_content)

                if not turn_calls:
                    # No tool calls - model finished
                    break

                run_tool_calls.extend(turn_calls)

                # Build assistant message with tool use
                assistant_content = []
                for content in response_content:
                    if "toolUse" in content:
                        assistant_content.append(content)
                    elif "text" in content:
                        assistant_content.append(content)

                messages.append({"role": "assistant", "content": assistant_content})

                # Build tool results for each tool call
                tool_results = []
                for tc in turn_calls:
                    key = (tc.get('name', ''), json.dumps(tc.get('arguments', {}), sort_keys=True))

                    # Look up the response
                    if key in tool_response_map:
                        response_content_str = tool_response_map[key]
                    else:
                        # Fallback: try to find by tool name only
                        fallback_response = None
                        for (name, _), resp in tool_response_map.items():
                            if name == tc.get('name', ''):
                                fallback_response = resp
                                break
                        if fallback_response:
                            response_content_str = fallback_response
                        else:
                            # No matching response found - use placeholder
                            response_content_str = {"status": "success", "result": "Tool executed successfully"}

                    # Format for Bedrock toolResult
                    if isinstance(response_content_str, dict):
                        response_str = json.dumps(response_content_str)
                    else:
                        response_str = str(response_content_str)

                    # Find the toolUseId from the response
                    tool_use_id = None
                    for content in assistant_content:
                        if "toolUse" in content and content["toolUse"]["name"] == tc.get('name', ''):
                            tool_use_id = content["toolUse"].get("toolUseId", f"tool_{turn_idx}")
                            break

                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id or f"tool_{turn_idx}",
                            "content": [{"text": response_str}]
                        }
                    })

                # Add tool results as user message
                messages.append({"role": "user", "content": tool_results})

            except Exception as e:
                print(f"  Run {run_idx + 1}, Turn {turn_idx + 1} failed: {e}")
                break

        all_runs.append(run_tool_calls)

    return all_runs


def _single_openai_inference(
    client,
    model_id: str,
    query: str,
    openai_tools: List[Dict],
    temperature: float,
    max_tokens: int,
    run_idx: int,
) -> List[Dict]:
    """Single OpenAI inference call for parallel execution."""
    try:
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=openai_tools,
            messages=[{"role": "user", "content": query}]
        )
        return extract_tool_calls_openai(response)
    except Exception as e:
        print(f"  Run {run_idx + 1} failed: {e}")
        return []


def generate_tool_calls_openai(
    client,
    model_id: str,
    query: str,
    tools: List[Dict],
    num_runs: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_workers: int = None,
) -> List[List[Dict]]:
    """Generate tool calls using OpenAI-compatible API with parallel execution."""
    openai_tools = [xlam_tool_to_openai_tool(t) for t in tools]

    # Determine optimal number of workers if not specified
    if max_workers is None:
        max_workers = min(num_runs, 10)  # Cap at 10 to avoid rate limits

    # Use ThreadPoolExecutor for parallel processing
    all_runs = [None] * num_runs
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _single_openai_inference,
                client, model_id, query, openai_tools,
                temperature, max_tokens, i
            ): i for i in range(num_runs)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_runs[idx] = future.result()
            except Exception as e:
                print(f"  Run {idx + 1} failed: {e}")
                all_runs[idx] = []

    return all_runs


def main():
    parser = argparse.ArgumentParser(
        description="Generate tool calls for STED evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        help="Model ID from model_config.py"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (higher = more variation)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs per sample for consistency evaluation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all samples)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="research/datasets/Salesforce_xlam-function-calling-60k.json",
        help="Path to dataset file"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["xlam", "toucan", "toucan-hf"],
        default="xlam",
        help="Dataset type: xlam (default), toucan (local file), toucan-hf (from HuggingFace)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/experiments/tool_calling_consistency",
        help="Output directory for results"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset (for resuming)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for response"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Enable multi-turn tool calling using original dataset responses"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns for multi-turn mode (default: 10)"
    )
    parser.add_argument(
        "--filter-long-tool-names",
        action="store_true",
        help="Filter out samples with tool names > 64 characters (required for AWS Bedrock)"
    )
    parser.add_argument(
        "--max-tool-name-length",
        type=int,
        default=64,
        help="Maximum tool name length when filtering (default: 64 for Bedrock)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel workers for inference (default: 10)"
    )

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("\nAvailable models:")
        print("=" * 70)
        for model_id, (provider, display_name) in MODEL_REGISTRY.items():
            print(f"  [{provider}] {display_name}")
            print(f"          {model_id}")
        print("=" * 70)
        return

    # Get provider and display name
    provider = get_provider(args.model)
    display_name = get_display_name(args.model)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_str = f"temp_{args.temperature:.2f}".replace(".", "_")
    model_short = display_name.replace(" ", "-")
    run_dir = Path(args.output_dir) / f"run_{model_short}_{temp_str}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Tool Call Generation for STED Consistency Evaluation")
    print("=" * 70)
    print(f"Model: {display_name} ({args.model})")
    print(f"Provider: {provider}")
    print(f"Temperature: {args.temperature}")
    print(f"Runs per sample: {args.num_runs}")
    print(f"Samples: {args.num_samples}")
    print(f"Multi-turn: {args.multi_turn}")
    if args.multi_turn:
        print(f"Max turns: {args.max_turns}")
    print(f"Filter long tool names: {args.filter_long_tool_names}")
    if args.filter_long_tool_names:
        print(f"Max tool name length: {args.max_tool_name_length}")
    print(f"Max workers (parallel): {args.max_workers}")
    print(f"Output: {run_dir}")
    print("=" * 70)

    # Load dataset based on type
    print(f"\nLoading {args.dataset_type} dataset...")
    # Calculate effective sample limit (-1 means all samples)
    effective_limit = None if args.num_samples <= 0 else args.num_samples + args.start_idx

    if args.dataset_type == "xlam":
        dataset = load_xlam_dataset(args.dataset_path, effective_limit)
        dataset = dataset[args.start_idx:]
    elif args.dataset_type == "toucan":
        dataset = load_toucan_dataset(args.dataset_path, effective_limit)
        dataset = dataset[args.start_idx:]
    elif args.dataset_type == "toucan-hf":
        print("  Streaming from HuggingFace (this may take a moment)...")
        dataset = load_toucan_from_huggingface(effective_limit if effective_limit else -1)
        dataset = dataset[args.start_idx:]
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    print(f"Loaded {len(dataset)} samples")

    # Initialize client based on provider
    print(f"\nInitializing {provider} client...")
    if provider == "bedrock":
        client = get_bedrock_client()
        generate_fn = generate_tool_calls_bedrock
    else:
        client = get_openai_client()
        generate_fn = generate_tool_calls_openai

    # Save run metadata
    metadata = {
        "model": args.model,
        "display_name": display_name,
        "provider": provider,
        "temperature": args.temperature,
        "num_runs": args.num_runs,
        "num_samples": len(dataset),
        "dataset_path": args.dataset_path,
        "timestamp": timestamp,
        "start_idx": args.start_idx,
        "max_tokens": args.max_tokens,
        "multi_turn": args.multi_turn,
        "max_turns": args.max_turns if args.multi_turn else None,
        "filter_long_tool_names": args.filter_long_tool_names,
        "max_tool_name_length": args.max_tool_name_length if args.filter_long_tool_names else None,
        "max_workers": args.max_workers,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Process samples
    all_results = []
    skipped_long_names = []  # Track samples skipped due to long tool names

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        sample_id = sample.get("id", idx)
        query = sample.get("query", "")

        # Parse tools (may be string or list)
        tools_raw = sample.get("tools", "[]")
        if isinstance(tools_raw, str):
            try:
                tools = json.loads(tools_raw)
            except json.JSONDecodeError:
                print(f"\n  Sample {sample_id}: Failed to parse tools")
                tools = []
        else:
            tools = tools_raw

        # Parse ground truth answers
        answers_raw = sample.get("answers", "[]")
        if isinstance(answers_raw, str):
            try:
                ground_truth = json.loads(answers_raw)
            except json.JSONDecodeError:
                ground_truth = []
        else:
            ground_truth = answers_raw

        # Skip if no tools defined
        if not tools:
            continue

        # Filter out samples with long tool names (Bedrock has 64-char limit)
        if args.filter_long_tool_names:
            if has_long_tool_names(tools, args.max_tool_name_length):
                long_names = get_long_tool_names(tools, args.max_tool_name_length)
                skipped_long_names.append({
                    "sample_id": sample_id,
                    "long_tool_names": long_names,
                    "max_length": max(len(n) for n in long_names)
                })
                continue

        # Get conversation turns for multi-turn mode
        conversation_turns = sample.get("conversation_turns", [])

        # Generate tool calls
        try:
            if args.multi_turn and provider == "bedrock" and conversation_turns:
                # Use multi-turn generation with original responses
                runs = generate_tool_calls_bedrock_multiturn(
                    client=client,
                    model_id=args.model,
                    query=query,
                    tools=tools,
                    conversation_turns=conversation_turns,
                    num_runs=args.num_runs,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_turns=args.max_turns,
                )
            else:
                # Single-turn generation (with parallel workers)
                runs = generate_fn(
                    client=client,
                    model_id=args.model,
                    query=query,
                    tools=tools,
                    num_runs=args.num_runs,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_workers=args.max_workers,
                )
        except Exception as e:
            print(f"\n  Sample {sample_id} failed: {e}")
            runs = [[] for _ in range(args.num_runs)]

        # Store result
        result = {
            "sample_id": sample_id,
            "query": query,
            "tools": tools,
            "ground_truth": ground_truth,
            "generated_runs": runs,
            "num_valid_runs": sum(1 for r in runs if r),
        }
        all_results.append(result)

        # Save intermediate results every 10 samples
        if (idx + 1) % 10 == 0:
            with open(run_dir / "intermediate_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Save final results
    final_output = {
        "metadata": metadata,
        "results": all_results,
        "summary": {
            "total_samples": len(all_results),
            "samples_with_all_valid": sum(
                1 for r in all_results if r["num_valid_runs"] == args.num_runs
            ),
            "avg_valid_runs": sum(r["num_valid_runs"] for r in all_results) / len(all_results) if all_results else 0,
        }
    }

    # Add filtering statistics if enabled
    if args.filter_long_tool_names:
        final_output["filtering"] = {
            "enabled": True,
            "max_tool_name_length": args.max_tool_name_length,
            "samples_filtered": len(skipped_long_names),
            "filtered_samples": skipped_long_names
        }

    with open(run_dir / "all_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"Total samples processed: {len(all_results)}")
    if args.filter_long_tool_names:
        print(f"Samples filtered (long tool names): {len(skipped_long_names)}")
    print(f"Samples with all valid runs: {final_output['summary']['samples_with_all_valid']}")
    print(f"Average valid runs per sample: {final_output['summary']['avg_valid_runs']:.2f}")
    print(f"Results saved to: {run_dir}")
    print("=" * 70)

    return run_dir


if __name__ == "__main__":
    main()
