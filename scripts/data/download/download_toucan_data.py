#!/usr/bin/env python3
"""
Download and extract tool_call data from Toucan-1.5M-structured-Qwen dataset.

This script extracts only the <tool_call> parts from the first assistant response,
with validation to ensure samples are compatible with Bedrock API.

Validation includes:
- Tool names must match pattern [a-zA-Z0-9_-]+ (Bedrock requirement)
- Tool calls must be parseable from <tool_call>...</tool_call> format
- Diverse subset coverage for comprehensive evaluation

Usage:
    python scripts/data/download_toucan_data.py --output-dir toucan_data --num-samples 1000
"""

import json
import os
import re
import argparse
from datasets import load_dataset

# Bedrock tool name pattern requirement
VALID_TOOL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


def extract_tool_calls(content: str) -> list:
    """
    Extract <tool_call> JSON objects from assistant response content.

    The Toucan dataset uses Python-style dicts with single quotes:
    {'name': 'tool-name', 'arguments': '{"param": "value"}'}

    Args:
        content: The assistant message content

    Returns:
        List of parsed JSON tool call objects with expanded arguments
    """
    import ast

    tool_calls = []

    # Pattern to match <tool_call>...</tool_call> blocks
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        try:
            # First try ast.literal_eval for Python-style dict with single quotes
            tool_call = ast.literal_eval(match.strip())

            # Parse the arguments string into actual JSON
            if 'arguments' in tool_call and isinstance(tool_call['arguments'], str):
                try:
                    tool_call['arguments'] = json.loads(tool_call['arguments'])
                except json.JSONDecodeError:
                    # Keep as string if can't parse
                    pass

            tool_calls.append(tool_call)
        except (ValueError, SyntaxError):
            # Try JSON parsing as fallback
            try:
                cleaned = match.strip()
                # Convert single quotes to double quotes for JSON
                cleaned = cleaned.replace("'", '"')
                tool_call = json.loads(cleaned)

                # Parse nested arguments
                if 'arguments' in tool_call and isinstance(tool_call['arguments'], str):
                    try:
                        tool_call['arguments'] = json.loads(tool_call['arguments'])
                    except json.JSONDecodeError:
                        pass

                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse tool_call: {match[:100]}...")
                continue

    return tool_calls


def extract_tool_responses(content: str) -> list:
    """
    Extract <tool_response> blocks from user message content.

    Args:
        content: The user message content containing tool responses

    Returns:
        List of tool response strings (may be JSON or plain text)
    """
    responses = []

    # Pattern to match <tool_response>...</tool_response> blocks
    pattern = r'<tool_response>\s*(.*?)\s*</tool_response>'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        content_str = match.strip()
        # Try to parse as JSON, otherwise keep as string
        try:
            parsed = json.loads(content_str)
            responses.append(parsed)
        except json.JSONDecodeError:
            # Keep as raw string (some responses are plain text)
            responses.append(content_str)

    return responses


def extract_conversation_turns(messages: list) -> list:
    """
    Extract all conversation turns with tool calls and responses.

    Returns a list of turns, where each turn contains:
    - tool_calls: List of tool calls made by assistant
    - tool_responses: List of tool responses from the next user message

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        List of turn dicts with tool_calls and tool_responses
    """
    turns = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            tool_calls = extract_tool_calls(content)

            if tool_calls:
                turn = {
                    'tool_calls': tool_calls,
                    'tool_responses': []
                }

                # Look for tool responses in the next user message
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get('role') == 'user':
                        next_content = next_msg.get('content', '')
                        if '<tool_response>' in next_content:
                            turn['tool_responses'] = extract_tool_responses(next_content)

                turns.append(turn)

        i += 1

    return turns


def get_first_assistant_response(messages: list) -> str:
    """
    Get the content of the first assistant message.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Content string of first assistant message, or empty string if not found
    """
    for msg in messages:
        if msg.get('role') == 'assistant':
            return msg.get('content', '')
    return ''


def validate_tool_names(tools: list) -> bool:
    """
    Validate that all tool names match Bedrock's required pattern [a-zA-Z0-9_-]+.

    Args:
        tools: List of tool definitions (OpenAI format with 'function' key)

    Returns:
        True if all tool names are valid, False otherwise
    """
    for tool in tools:
        # Handle OpenAI format: {"type": "function", "function": {"name": "..."}}
        if isinstance(tool, dict):
            if 'function' in tool:
                name = tool['function'].get('name', '')
            else:
                name = tool.get('name', '')

            if not name or not VALID_TOOL_NAME_PATTERN.match(name):
                return False
    return True


def get_tool_names_from_tools(tools: list) -> list:
    """Extract tool names from tool definitions."""
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            if 'function' in tool:
                names.append(tool['function'].get('name', ''))
            else:
                names.append(tool.get('name', ''))
    return names


def download_toucan_tool_calls(output_dir: str, num_samples: int = 100,
                                diverse_subsets: bool = True,
                                allowed_subsets: list = None):
    """
    Download Toucan dataset and extract tool_call data with validation.

    Args:
        output_dir: Directory to save extracted data
        num_samples: Total number of samples to extract
        diverse_subsets: If True, sample evenly across subset_name categories
        allowed_subsets: List of subset names to include (None = all subsets)

    Validation:
        - Tool names must match [a-zA-Z0-9_-]+ (Bedrock requirement)
        - Tool calls must be parseable
        - Tool definitions must exist
    """
    print("Loading Toucan-1.5M-structured-Qwen dataset (streaming)...")
    ds = load_dataset("beyoru/Toucan-1.5M-structured-Qwen", split="train", streaming=True)

    os.makedirs(output_dir, exist_ok=True)

    extracted_samples = []
    subset_counts = {}

    # Determine number of subsets for even distribution
    if allowed_subsets:
        num_subsets = len(allowed_subsets)
        print(f"Filtering to subsets: {allowed_subsets}")
    else:
        num_subsets = 3  # Default: single-turn-original, single-turn-diversify, multi-turn

    # For diverse sampling, distribute evenly across allowed subsets
    samples_per_subset = num_samples // num_subsets if diverse_subsets else num_samples

    # Track skip reasons for debugging
    skip_reasons = {
        'no_assistant': 0,
        'no_tool_calls': 0,
        'invalid_tool_names': 0,
        'no_tools': 0,
        'subset_full': 0,
        'excluded_subset': 0,
    }
    scanned_count = 0

    print(f"Extracting {num_samples} valid samples...")
    print("Validation: tool names must match [a-zA-Z0-9_-]+")

    for i, sample in enumerate(ds):
        scanned_count += 1

        # Get first assistant response
        messages = sample.get('messages', [])
        assistant_content = get_first_assistant_response(messages)

        if not assistant_content:
            skip_reasons['no_assistant'] += 1
            continue

        # Extract tool calls
        tool_calls = extract_tool_calls(assistant_content)

        if not tool_calls:
            skip_reasons['no_tool_calls'] += 1
            continue

        # Get tool definitions from the sample
        tools_raw = sample.get('tools', '[]')
        try:
            tools = json.loads(tools_raw)
        except json.JSONDecodeError:
            tools = []

        if not tools:
            skip_reasons['no_tools'] += 1
            continue

        # Validate tool names match Bedrock pattern
        if not validate_tool_names(tools):
            skip_reasons['invalid_tool_names'] += 1
            continue

        subset_name = sample.get('subset_name', 'unknown')

        # Filter by allowed subsets if specified
        if allowed_subsets and subset_name not in allowed_subsets:
            skip_reasons['excluded_subset'] += 1
            continue

        # Track diversity
        if diverse_subsets:
            if subset_name not in subset_counts:
                subset_counts[subset_name] = 0
            if subset_counts[subset_name] >= samples_per_subset:
                # Check if we have enough total
                if len(extracted_samples) >= num_samples:
                    break
                skip_reasons['subset_full'] += 1
                continue
            subset_counts[subset_name] += 1

        # Create extracted sample (single-turn only - no conversation_turns)
        extracted = {
            'id': sample.get('uuid', f'sample_{len(extracted_samples)}'),
            'subset_name': subset_name,
            'question': sample.get('question', ''),
            'target_tools': sample.get('target_tools', ''),
            'tools': tools,  # Include tool definitions for LLM calling
            'tool_calls': tool_calls,  # Ground truth from dataset (first turn)
            'num_tool_calls': len(tool_calls),
        }

        extracted_samples.append(extracted)

        if len(extracted_samples) >= num_samples:
            break

        if len(extracted_samples) % 100 == 0:
            print(f"  Extracted {len(extracted_samples)}/{num_samples} samples (scanned {scanned_count})...")

    print(f"\nExtracted {len(extracted_samples)} samples total")

    # Print subset distribution
    print("\nSubset distribution:")
    final_subset_counts = {}
    for sample in extracted_samples:
        subset = sample['subset_name']
        final_subset_counts[subset] = final_subset_counts.get(subset, 0) + 1
    for subset, count in sorted(final_subset_counts.items()):
        print(f"  {subset}: {count}")

    # Save all samples
    all_samples_file = os.path.join(output_dir, "toucan_tool_calls.json")
    with open(all_samples_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved all samples to: {all_samples_file}")

    # Save individual samples
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    for i, sample in enumerate(extracted_samples):
        sample_file = os.path.join(samples_dir, f"sample_{i+1:04d}.json")
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Saved individual samples to: {samples_dir}/")

    # Compute and save statistics
    stats = compute_statistics(extracted_samples)
    stats_file = os.path.join(output_dir, "toucan_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved statistics to: {stats_file}")

    # Print statistics
    print("\n=== TOOL CALL STATISTICS ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total tool calls: {stats['total_tool_calls']}")
    print(f"Avg tool calls per sample: {stats['avg_tool_calls_per_sample']:.2f}")
    print(f"Avg depth: {stats['avg_depth']:.2f}")
    print(f"Max depth: {stats['max_depth']}")
    print(f"Avg fields per tool call: {stats['avg_fields']:.2f}")
    print(f"Unique keys: {stats['unique_keys_count']}")

    return extracted_samples


def analyze_json_depth(obj, depth=0):
    """Calculate maximum depth of JSON object."""
    if isinstance(obj, dict):
        if not obj:
            return depth
        return max(analyze_json_depth(v, depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return depth
        return max(analyze_json_depth(item, depth + 1) for item in obj)
    return depth


def count_fields(obj):
    """Count total fields in JSON object recursively."""
    if isinstance(obj, dict):
        return len(obj) + sum(count_fields(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_fields(item) for item in obj)
    return 0


def collect_keys(obj, keys=None):
    """Collect all unique keys from JSON object."""
    if keys is None:
        keys = set()
    if isinstance(obj, dict):
        keys.update(obj.keys())
        for v in obj.values():
            collect_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            collect_keys(item, keys)
    return keys


def compute_statistics(samples: list) -> dict:
    """Compute statistics for extracted tool calls."""
    depths = []
    field_counts = []
    all_keys = set()
    tool_call_counts = []

    for sample in samples:
        tool_calls = sample.get('tool_calls', [])
        tool_call_counts.append(len(tool_calls))

        for tc in tool_calls:
            depth = analyze_json_depth(tc)
            fields = count_fields(tc)
            keys = collect_keys(tc)

            depths.append(depth)
            field_counts.append(fields)
            all_keys.update(keys)

    return {
        'total_samples': len(samples),
        'total_tool_calls': sum(tool_call_counts),
        'avg_tool_calls_per_sample': sum(tool_call_counts) / len(samples) if samples else 0,
        'avg_depth': sum(depths) / len(depths) if depths else 0,
        'max_depth': max(depths) if depths else 0,
        'min_depth': min(depths) if depths else 0,
        'avg_fields': sum(field_counts) / len(field_counts) if field_counts else 0,
        'max_fields': max(field_counts) if field_counts else 0,
        'unique_keys_count': len(all_keys),
        'sample_keys': list(all_keys)[:50]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract tool_call data from Toucan dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="toucan_data",
        help="Directory to save extracted data (default: toucan_data)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to extract (default: 100)"
    )
    parser.add_argument(
        "--no-diverse",
        action="store_true",
        help="Disable diverse subset sampling (take first N samples)"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=None,
        help="Subset names to include (default: all). Options: single-turn-original, single-turn-diversify, multi-turn"
    )

    args = parser.parse_args()

    download_toucan_tool_calls(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        diverse_subsets=not args.no_diverse,
        allowed_subsets=args.subsets
    )

    print("\nDownload complete!")
