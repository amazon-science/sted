#!/usr/bin/env python
"""
Generate 3 dataset categories for STED human-aligned consistency evaluation experiments.
Each category contains 10 datasets with variation ratios from 0.1 to 1.0.
Categories: 1. Schema Variation, 2. Expression Variation, 3. Semantic Variation
"""

import json
import random
import copy
from typing import List, Dict, Any, Tuple
import os
from itertools import permutations
import boto3
import time
from sted.bedrock_utils import build_message, inference_with_converse_api, get_json
from tqdm import tqdm
import json
from datetime import datetime
import copy
import ast
import re

user_prompt = """The original JSON data is as follows:
{original_data}
"""

system_prompt_semantic_by_field = """You are tasked with creating a variation of the following list of strings by changing their meanings.
"""

system_prompt_expression_by_field = """You are tasked with creating a variation of the following list of strings by changing their expression but keep semantic similarity.
"""

system_prompt_field_variants_by_field = """You are tasked with creating variants of the following list of field names in dict data by changing their expression with same meaning.
"""

user_prompt_fields = """The original list is as follows:
{string_list}

Use print_field_variants to print the result.
"""

print_field_semantic_variants_tool = [
    {
        "toolSpec": {
            "name": "print_field_variants",
            "description": "Print the list of variants with different meanings for the original string list",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "string_list": {
                            "type": "array",
                            "description": "The list of variants with different meanings",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["string_list"]
                }
            }
        }
    }
]

print_field_expression_variants_tool = [
    {
        "toolSpec": {
            "name": "print_field_variants",
            "description": "Print the list of variants with different expression but keeping the same meaning for the original string list",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "string_list": {
                            "type": "array",
                            "description": "The list of variants with different expression but same meaning",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["string_list"]
                }
            }
        }
    }
]



class ExperimentDatasetGenerator:
    """Generate 4 dataset categories with 10 datasets each at different variation ratios (0.1-1.0) for STED evaluation experiments."""

    def __init__(self, base_dataset_dir, model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", region_name="us-west-2", data_source="sharegpt"):
        """
        Initialize dataset generator.

        Args:
            base_dataset_dir: Directory containing base data
            model_id: Bedrock model ID for LLM-based variations
            region_name: AWS region
            data_source: "sharegpt" or "toucan" - determines base data format
        """
        self.data_source = data_source

        # Load base templates based on data source
        if data_source == "toucan":
            self.base_templates = self._load_toucan_samples(base_dataset_dir)
        else:
            self.base_templates = self._load_sharegpt_samples(base_dataset_dir)

        # Initialize Bedrock client for Claude 3.5 Sonnet with Converse API
        self.bedrock_region = os.getenv('BEDROCK_REGION', region_name)

        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.bedrock_region)
            self.claude_model_id = model_id

            # Test the connection with a simple converse call
            test_response = self.bedrock_client.converse(
                modelId=self.claude_model_id,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
                inferenceConfig={"maxTokens": 10, "temperature": 0.1}
            )
            print(f"✓ Bedrock Converse API initialized and tested for {self.claude_model_id}")
        except Exception as e:
            raise RuntimeError(f"Could not initialize Bedrock client: {e}. Please check AWS credentials and Bedrock access.")

    def _load_toucan_samples(self, base_dataset_dir) -> List[Dict]:
        """Load tool call samples from Toucan dataset."""
        samples = []

        # Look for Toucan JSON files
        toucan_files = []
        for root, dirs, files in os.walk(base_dataset_dir):
            for f in files:
                if f.endswith('.json') and 'toucan' in f.lower():
                    toucan_files.append(os.path.join(root, f))

        # Also check for toucan_data directory
        toucan_data_path = os.path.join(os.path.dirname(base_dataset_dir), "toucan_data")
        if os.path.exists(toucan_data_path):
            for f in os.listdir(toucan_data_path):
                if f.endswith('.json'):
                    toucan_files.append(os.path.join(toucan_data_path, f))

        for filepath in toucan_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        if 'tool_calls' in item and item['tool_calls']:
                            # Convert tool calls to a structured format for variations
                            tool_call_json = {
                                "tool_calls": item['tool_calls'],
                                "_meta": {
                                    "id": item.get('id'),
                                    "question": item.get('question', '')[:200],  # Truncate for readability
                                    "target_tools": item.get('target_tools', ''),
                                }
                            }
                            samples.append(tool_call_json)

            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                continue

        if len(samples) < 5:
            raise RuntimeError(f"Could not load sufficient Toucan samples from {base_dataset_dir}. Found {len(samples)} samples.")

        print(f"Loaded {len(samples)} tool call samples from Toucan data")
        return samples
        
    def _load_sharegpt_samples(self, base_dataset_dir) -> List[Dict]:
        """Load all available complex JSON samples from ShareGPT data."""
        samples = []
        
        # list all datasets under base_dataset_dir
        sharegpt_dirs = [d for d in os.listdir(base_dataset_dir) if os.path.isdir(os.path.join(base_dataset_dir, d)) and d.startswith("sharegpt")]
                
        for directory in sharegpt_dirs:
            try:
                dir_path = os.path.join(base_dataset_dir, directory)
                if not os.path.exists(dir_path):
                    print(f"Directory not found: {dir_path}")
                    continue
                    
                # Get all conversation files in the directory
                conversation_files = [f for f in os.listdir(dir_path) if f.startswith("conversation_") and f.endswith(".json")]
                conversation_files.sort()  # Sort to ensure consistent loading order
                
                print(f"Found {len(conversation_files)} conversation files in {dir_path}")
                
                # Load all available conversations
                for filename in conversation_files:
                    file_path = os.path.join(dir_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            conversation = json.load(f)
                            
                        # Extract the GPT response (which contains the structured JSON)
                        for conv in conversation.get("conversations", []):
                            if conv.get("from") == "gpt":
                                try:
                                    # Try to parse the GPT response as JSON
                                    gpt_json = json.loads(conv["value"])
                                    samples.append(gpt_json)
                                    break
                                except json.JSONDecodeError:
                                    # If it's not valid JSON, skip this sample
                                    continue
                    except Exception as e:
                        print(f"Warning: Could not load {filename}: {e}")
                        continue
                                    
            except Exception as e:
                print(f"Warning: Could not access directory {directory}: {e}")
                continue
        
        # Ensure we have samples loaded
        if len(samples) < 5:
            raise RuntimeError("Could not load sufficient ShareGPT samples. Please check the sharegpt_data directory.")
        
        print(f"Loaded {len(samples)} complex samples from ShareGPT data")
        return samples
    
    def generate_schema_variation_dataset(self, num_samples: int = 80, variation_ratio=1.0, output_dir="synthetic_dataset") -> List[Dict]:
        """
        Generate Dataset Category 2: Schema Variation Dataset V2
        Same semantic content in different structural formats using predefined variation types.
        """
        variation_types = ["flat_structure", "field_name_change", "nested_change"]
        
        all_samples = []
        for variation_type in variation_types:
            for idx in tqdm(range(min(num_samples, len(self.base_templates))), desc=f"Generating {variation_type} variations"):
                sample = self.base_templates[idx]
                
                if isinstance(sample, list):
                    sample = {"root": sample}
                
                total_fields = self._get_all_fields(sample)
                num_fields = len(total_fields)
                
                if variation_type == "field_name_change":
                    field_names = [field.split('.')[-1] for field in total_fields]
                    field_variants = []
                    
                    # Generate field variants using LLM
                    for attempt in range(3):
                        try:
                            if idx > 0:
                                time.sleep(1)  # Rate limiting
                            message = build_message(texts=[user_prompt_fields.format(string_list=field_names)])
                            response = inference_with_converse_api(
                                self.bedrock_client, self.claude_model_id, [message],
                                system_prompts=system_prompt_field_variants_by_field, max_tokens=8000,
                                temperature=0.1, tools=print_field_expression_variants_tool
                            )
                            result = get_json(response, "print_field_variants")
                            if result and 'string_list' in result:
                                field_variants = result['string_list']
                                if isinstance(field_variants, str):
                                    field_variants = ast.literal_eval(field_variants)
                                break
                        except Exception as e:
                            print(f"Error generating field variants (attempt {attempt+1}): {e}")
                            if attempt == 2:  # Last attempt
                                # Fallback to simple modifications
                                field_variants = [f"{name}_modified" for name in field_names]
                    
                    if not field_variants:
                        field_variants = [f"{name}_modified" for name in field_names]
                    
                    synthetic_sample = {
                        "base_sample": sample,
                        "sample_id": f"schema_variation_{idx:03d}",
                        "variation_type": variation_type,
                        "variants": []
                    }
                    
                    # Incrementally build variants by adding one field change at a time
                    current_variant = copy.deepcopy(sample)
                    
                    for i in range(10):
                        current_variation_ratio = round(0.1 + i * 0.1, 1)
                        num_fields_to_change = int(num_fields * current_variation_ratio)
                        
                        # Apply field changes up to the current level
                        for j in range(min(num_fields_to_change, len(field_variants))):
                            if j < len(field_names):
                                field_mapping = {total_fields[j]: field_variants[j]}
                                current_variant = self._rename_fields_recursive(current_variant, field_mapping)
                        
                        # Track changes for this variation level
                        changed_fields = field_names[:min(num_fields_to_change, len(field_variants))]
                        changed_variants = field_variants[:min(num_fields_to_change, len(field_variants))]
                        
                        synthetic_sample["variants"].append({
                            "variation_ratio": current_variation_ratio,
                            "variation": copy.deepcopy(current_variant),
                            "field_path": changed_fields,
                            "original_value": changed_fields,
                            "variant": changed_variants
                        })
                    
                    all_samples.append(synthetic_sample)
                
                elif variation_type == "flat_structure":
                    variant = self._flatten_structure(sample)
                    all_samples.append({
                        "sample_id": f"schema_flat_{idx:03d}",
                        "variation_type": variation_type,
                        "ground_truth": sample,
                        "variation": variant
                    })
                    
                elif variation_type == "nested_change":
                    variant = self._add_nesting(sample, variation_ratio)
                    all_samples.append({
                        "sample_id": f"schema_nested_{idx:03d}",
                        "variation_type": variation_type,
                        "ground_truth": sample,
                        "variation": variant
                    })
        
        # Save samples
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"schema_variation_dataset_{current_time_str}.json"), "w") as f:
            json.dump(all_samples, f, indent=4)
        
        return all_samples
    
    def _parse_python_list_string(self, variants: str) -> list:
        """Parse a Python list string with mixed quotes into a list."""
        # Replace newlines with space
        variants = variants.replace('\n', ' ')
        # Normalize curly quotes
        variants = re.sub(r'[""]', '"', variants)
        variants = re.sub(r"['']", "'", variants)
        # Unescape single quotes (valid in Python, invalid in JSON)
        variants = variants.replace("\\'", "'")
        # Convert single-quoted strings to double-quoted: ', ' or '] or ['
        variants = re.sub(r"', '", '", "', variants)
        variants = re.sub(r"'\]", '"]', variants)
        variants = re.sub(r"\['", '["', variants)
        variants = re.sub(r"', \"", '", "', variants)
        variants = re.sub(r"\", '", '", "', variants)
        return json.loads(variants)
        
    def _flatten_structure(self, sample: Dict) -> Dict:
        """Flatten nested structure based on ratio"""
        result = {}
        self._flatten_recursive(sample, result, "")
        return result
    
    def _flatten_recursive(self, obj, result: Dict, prefix: str):
        """Recursively flatten nested objects"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, (dict, list)) and len(str(v)) > 50:
                    self._flatten_recursive(v, result, new_key)
                else:
                    result[new_key] = v
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._flatten_recursive(item, result, f"{prefix}_{i}")
    
    def _add_nesting(self, sample: Dict, ratio: float = 1.0) -> Dict:
        """Add additional nesting levels based on ratio"""
        if ratio < 0.4:
            return sample
        elif ratio < 0.7:
            return {"wrapper": sample}
        else:
            return {"data": {"content": sample}}
        
    def _rename_fields_recursive(self, obj, field_renew_map: dict, prefix: str = ""):
        """Recursively rename fields in nested structures"""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                new_key = field_renew_map.get(f"{prefix}{k}", k)
                if isinstance(v, dict):
                    result[new_key] = self._rename_fields_recursive(v, field_renew_map, f"{prefix}{k}.")
                elif isinstance(v, list):
                    result[new_key] = [self._rename_fields_recursive(item, field_renew_map, f"{prefix}{k}.") if isinstance(item, dict) else item for item in v]
                else:
                    result[new_key] = v
            return result
        elif isinstance(obj, list):
            return [self._rename_fields_recursive(item, field_renew_map, f"{prefix}{k}[{idx}].") if isinstance(item, dict) else item for idx, item in enumerate(obj)]
        return obj
        
    def generate_expression_variation_dataset(self, num_samples: int = 50, max_variation_ratio=1.0, output_dir="synthetic_dataset") -> List[Dict]:
        """
        Generate Dataset Category 3: Expression Variation Dataset
        Synonymous terms and equivalent expressions.
        Expected: STED stable, BERTScore stable, DeepDiff unstable.
        """
        self.generate_semantic_variation_dataset(num_samples, max_variation_ratio=max_variation_ratio, semantic_variation=False)
    
    def _get_all_fields(self, sample: Dict, prefix: str = "") -> List[str]:
        """Get all fields in the sample, including nested ones."""
        fields = []
        
        if isinstance(sample, list):
            for i, item in enumerate(sample):
                if isinstance(item, dict):
                    fields.extend(self._get_all_fields(item, f"{prefix}[{i}]."))
                elif isinstance(item, list):
                    for j, subitem in enumerate(item):
                        if isinstance(subitem, dict):
                            fields.extend(self._get_all_fields(subitem, f"{prefix}[{i}][{j}]."))
                        else:
                            fields.append(f"{prefix}[{i}][{j}]")
                else:
                    fields.append(f"{prefix}[{i}]")
            return fields
        elif isinstance(sample, dict):
            for key, value in sample.items():
                if isinstance(value, dict):
                    fields.extend(self._get_all_fields(value, f"{prefix}{key}."))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            fields.extend(self._get_all_fields(item, f"{prefix}{key}[{i}]."))
                        else:
                            fields.append(f"{prefix}{key}[{i}]")
                else:
                    fields.append(f"{prefix}{key}")
            return fields
        
        return fields
    
    def _get_all_str_fields(self, sample: Dict, prefix: str = "") -> List[str]:
        """Get all string fields in the sample, including nested ones."""
        fields = []
        if isinstance(sample, str):
            return [prefix.strip('.')]
        elif isinstance(sample, list):
            for i, item in enumerate(sample):
                if isinstance(item, dict) or isinstance(item, list):
                    fields.extend(self._get_all_str_fields(item, f"{prefix}[{i}]."))
                elif isinstance(item, str):
                    fields.append(f"{prefix}[{i}]")
        elif isinstance(sample, dict):
            for key, value in sample.items():
                if isinstance(value, dict):
                    fields.extend(self._get_all_str_fields(value, f"{prefix}{key}."))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            fields.extend(self._get_all_str_fields(item, f"{prefix}{key}[{i}]."))
                        elif isinstance(item, str):
                            fields.append(f"{prefix}{key}[{i}]")
                elif isinstance(value, str):
                    fields.append(f"{prefix}{key}")
            return fields
        
        return fields
    
    def get_value_from_field(self, json_data, field_path):
        """Get value from nested JSON using dot notation and array indices."""
        if not field_path:
            return json_data
            
        parts = field_path.split('.')
        current = json_data
        
        for part in parts:
            if '[' in part and ']' in part:
                # Handle array access like "items[0]"
                field_name = part.split('[')[0]
                index_str = part.split('[')[1].split(']')[0]
                try:
                    index = int(index_str)
                    if isinstance(current, dict) and field_name in current:
                        current = current[field_name]
                        if isinstance(current, list) and 0 <= index < len(current):
                            current = current[index]
                        else:
                            return None
                    else:
                        return None
                except (ValueError, IndexError):
                    return None
            else:
                # Regular field access
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        
        return current
    
    def replace_field_value(self, json_data, field_path, new_value):
        """Replace value in nested JSON using dot notation and array indices."""
        if not field_path:
            return new_value
            
        parts = field_path.split('.')
        current = json_data
        
        # Navigate to the parent of the target field
        for i, part in enumerate(parts[:-1]):
            if '[' in part and ']' in part:
                # Handle array access
                field_name = part.split('[')[0]
                index_str = part.split('[')[1].split(']')[0]
                try:
                    index = int(index_str)
                    if isinstance(current, dict) and field_name in current:
                        current = current[field_name]
                        if isinstance(current, list) and 0 <= index < len(current):
                            current = current[index]
                        else:
                            print(f"failed to replace field value (field: {field_path}, new_value: {new_value}), invalid array index, json data: {json_data}")
                            return json_data
                    else:
                        print(f"failed to replace field value (field: {field_path}, new_value: {new_value}), field not found, json data: {json_data}")
                        return json_data
                except (ValueError, IndexError):
                    print(f"failed to replace field value (field: {field_path}, new_value: {new_value}), invalid index, json data: {json_data}")
                    return json_data
            else:
                # Regular field access
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    print(f"failed to replace field value (field: {field_path}, new_value: {new_value}), field not found, json data: {json_data}")
                    return json_data
        
        # Set the final value
        final_part = parts[-1]
        if '[' in final_part and ']' in final_part:
            # Handle array access for final part
            field_name = final_part.split('[')[0]
            index_str = final_part.split('[')[1].split(']')[0]
            try:
                index = int(index_str)
                if isinstance(current, dict) and field_name in current:
                    if isinstance(current[field_name], list) and 0 <= index < len(current[field_name]):
                        current[field_name][index] = new_value
                    else:
                        print(f"failed to replace field value (field: {field_path}, new_value: {new_value}), invalid final array index, json data: {json_data}")
                        return json_data
                else:
                    print(f"failed to replace field value (field: {field_path}, new_value: {new_value}), final field not found, json data: {json_data}")
                    return json_data
            except (ValueError, IndexError):
                print(f"failed to replace field value (field: {field_path}, new_value: {new_value}), invalid final index, json data: {json_data}")
                return json_data
        else:
            # Regular field assignment
            if isinstance(current, dict):
                current[final_part] = new_value
            else:
                print(f"failed to replace field value (field: {field_path}, new_value: {new_value}), cannot assign to non-dict, json data: {json_data}")
                return json_data
        
        return json_data
    
    def generate_semantic_variation_dataset(self, num_samples: int = 80, max_variation_ratio=1.0, semantic_variation=True, output_dir="synthetic_dataset") -> List[Dict]:
        samples = []
        system_prompt_by_field = system_prompt_semantic_by_field if semantic_variation else system_prompt_expression_by_field
        print_tool = print_field_semantic_variants_tool if semantic_variation else print_field_expression_variants_tool

        for idx, sample in tqdm(enumerate(self.base_templates[:num_samples]), desc="walk through all samples"):
            sample = {"root": sample} if not isinstance(sample, dict) else sample
            all_str_fields = self._get_all_str_fields(sample)
            num_str_fields = len(all_str_fields)
            
            if num_str_fields == 0:
                continue
            
            max_variation = int(num_str_fields * max_variation_ratio)
            target_values = [self.get_value_from_field(sample, all_str_fields[i]) for i in range(max_variation)]

            if semantic_variation:
                variants = ["this is a variant"] * len(target_values)
            else:
                # Generate variants with LLM
                for attempt in range(3):
                    try:
                        if idx > 0:
                            time.sleep(2)
                        message = build_message(texts=[user_prompt_fields.format(string_list=target_values)])
                        response = inference_with_converse_api(
                            self.bedrock_client, self.claude_model_id, [message],
                            system_prompts=system_prompt_by_field, max_tokens=8000,
                            temperature=0.1, tools=print_tool
                        )
                        variants = get_json(response, "print_field_variants")['string_list']
                        if isinstance(variants, str):
                            print(f"variants: {variants}")
                            variants = self._parse_python_list_string(variants)
                        break
                    except Exception as e:
                        import traceback
                        traceback.print_exc()

            data_type = "sematic" if semantic_variation else "expression"
            sythetic_sample = {
                "base_sample": sample,
                "sample_id": f"{data_type}_variation_{idx:03d}",
                "variants": []
            }
            
            # Incrementally build variants by adding one field change at a time
            current_variant = copy.deepcopy(sample)
            
            for i in range(10):
                variation_ratio = round(0.1 + i/10, 1)
                num_fields_to_change = int(num_str_fields * variation_ratio)
                
                # Add one more field change to the current variant
                if i > 0:
                    prev_num_fields = int(num_str_fields * (0.1 + (i-1)/10))
                    for j in range(prev_num_fields, min(num_fields_to_change, len(variants))):
                        current_variant = self.replace_field_value(current_variant, all_str_fields[j], variants[j])
                else:
                    # First iteration - change fields from 0 to num_fields_to_change
                    for j in range(min(num_fields_to_change, len(variants))):
                        current_variant = self.replace_field_value(current_variant, all_str_fields[j], variants[j])
                
                # Track changes for this variation level
                changed_fields = all_str_fields[:min(num_fields_to_change, len(variants))]
                changed_originals = target_values[:min(num_fields_to_change, len(variants))]
                changed_variants = variants[:min(num_fields_to_change, len(variants))]
                
                # Validate that all intended fields were actually changed
                for j, (orig, var) in enumerate(zip(changed_originals, changed_variants)):
                    if orig == var:
                        # Force a simple change if AI didn't change it
                        changed_variants[j] = f"{var} (modified)"
                        current_variant = self.replace_field_value(current_variant, changed_fields[j], changed_variants[j])
                
                sythetic_sample["variants"].append({
                    "variation_ratio": variation_ratio,
                    "variation": copy.deepcopy(current_variant),
                    "field_path": changed_fields,
                    "original_value": changed_originals,
                    "variant": changed_variants
                })
                
            samples.append(sythetic_sample)
        
        # Save samples
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{data_type}_variation_dataset_{current_time_str}.json"), "w") as f:
            json.dump(samples, f, indent=4)
        
        return samples

    def generate_tool_call_variation_dataset(self, num_samples: int = 80, output_dir="synthetic_dataset") -> List[Dict]:
        """
        Generate Dataset for Tool Call Variations (Toucan-specific).
        Creates variations on tool call parameters for human validation.

        Variation types:
        1. Parameter value variations (semantic equivalents)
        2. Tool call ordering variations
        3. Parameter format variations (e.g., date formats, number formats)
        """
        if self.data_source != "toucan":
            print("Warning: Tool call variations work best with Toucan data source")

        all_samples = []
        variation_types = ["param_value", "param_format", "tool_order"]

        for variation_type in variation_types:
            for idx in tqdm(range(min(num_samples, len(self.base_templates))), desc=f"Generating {variation_type} variations"):
                sample = self.base_templates[idx]
                tool_calls = sample.get("tool_calls", [])

                if not tool_calls:
                    continue

                synthetic_sample = {
                    "base_sample": sample,
                    "sample_id": f"tool_call_{variation_type}_{idx:03d}",
                    "variation_type": variation_type,
                    "variants": []
                }

                if variation_type == "param_value":
                    # Generate parameter value variations using LLM
                    all_param_values = []
                    param_paths = []

                    for tc_idx, tc in enumerate(tool_calls):
                        args = tc.get("arguments", {})
                        for param_name, param_value in args.items():
                            if isinstance(param_value, str) and len(param_value) > 2:
                                all_param_values.append(param_value)
                                param_paths.append((tc_idx, param_name))

                    if not all_param_values:
                        continue

                    # Generate variants using LLM
                    variants = []
                    for attempt in range(3):
                        try:
                            if idx > 0:
                                time.sleep(1)
                            message = build_message(texts=[user_prompt_fields.format(string_list=all_param_values)])
                            response = inference_with_converse_api(
                                self.bedrock_client, self.claude_model_id, [message],
                                system_prompts=system_prompt_expression_by_field, max_tokens=8000,
                                temperature=0.1, tools=print_field_expression_variants_tool
                            )
                            result = get_json(response, "print_field_variants")
                            if result and 'string_list' in result:
                                variants = result['string_list']
                                if isinstance(variants, str):
                                    variants = self._parse_python_list_string(variants)
                                break
                        except Exception as e:
                            print(f"Error generating param variants (attempt {attempt+1}): {e}")
                            if attempt == 2:
                                variants = [f"{v}_modified" for v in all_param_values]

                    if not variants:
                        variants = [f"{v}_modified" for v in all_param_values]

                    # Create incremental variations
                    num_params = len(param_paths)
                    current_variant = copy.deepcopy(sample)

                    for i in range(10):
                        variation_ratio = round(0.1 + i * 0.1, 1)
                        num_to_change = max(1, int(num_params * variation_ratio))

                        for j in range(min(num_to_change, len(variants))):
                            tc_idx, param_name = param_paths[j]
                            if tc_idx < len(current_variant["tool_calls"]):
                                current_variant["tool_calls"][tc_idx]["arguments"][param_name] = variants[j]

                        synthetic_sample["variants"].append({
                            "variation_ratio": variation_ratio,
                            "variation": copy.deepcopy(current_variant),
                            "num_params_changed": min(num_to_change, len(variants)),
                        })

                elif variation_type == "tool_order":
                    # Reorder tool calls (for multi-tool scenarios)
                    if len(tool_calls) < 2:
                        continue

                    # Create variations by shuffling tool call order
                    for i in range(10):
                        variation_ratio = round(0.1 + i * 0.1, 1)
                        variant = copy.deepcopy(sample)

                        # Progressively shuffle more tool calls
                        n_to_shuffle = max(2, int(len(tool_calls) * variation_ratio))
                        shuffled_calls = variant["tool_calls"][:n_to_shuffle]
                        random.shuffle(shuffled_calls)
                        variant["tool_calls"] = shuffled_calls + variant["tool_calls"][n_to_shuffle:]

                        synthetic_sample["variants"].append({
                            "variation_ratio": variation_ratio,
                            "variation": variant,
                            "tools_shuffled": n_to_shuffle,
                        })

                elif variation_type == "param_format":
                    # Change parameter formats (numbers, dates, etc.)
                    current_variant = copy.deepcopy(sample)

                    for i in range(10):
                        variation_ratio = round(0.1 + i * 0.1, 1)

                        for tc in current_variant["tool_calls"]:
                            args = tc.get("arguments", {})
                            for param_name, param_value in list(args.items()):
                                # Format variations for different types
                                if isinstance(param_value, (int, float)):
                                    # Number format variations
                                    if random.random() < variation_ratio:
                                        if isinstance(param_value, int):
                                            args[param_name] = str(param_value)  # int -> string
                                        else:
                                            args[param_name] = round(param_value, 2)  # precision change

                        synthetic_sample["variants"].append({
                            "variation_ratio": variation_ratio,
                            "variation": copy.deepcopy(current_variant),
                        })

                if synthetic_sample["variants"]:
                    all_samples.append(synthetic_sample)

        # Save samples
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"tool_call_variation_dataset_{current_time_str}.json")
        with open(output_path, "w") as f:
            json.dump(all_samples, f, indent=4)

        print(f"Saved {len(all_samples)} tool call variation samples to {output_path}")
        return all_samples


def main():
    """Main function to generate all experimental datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate STED experiment datasets with variable ratios")
    parser.add_argument("--output-dir", default="synthetic_dataset",
                       help="Output directory for datasets")
    parser.add_argument("--num-samples", type=int, default=80,
                       help="Number of samples per dataset")
    parser.add_argument("--base-dataset-dir", default="sharegpt_data",
                       help="Base dataset directory")
    parser.add_argument("--data-source", choices=["sharegpt", "toucan"], default="sharegpt",
                       help="Data source: 'sharegpt' for structured output, 'toucan' for tool calls")
    parser.add_argument("--toucan-only", action="store_true",
                       help="Generate only tool call variations (requires --data-source toucan)")


    args = parser.parse_args()

    generator = ExperimentDatasetGenerator(
        base_dataset_dir=args.base_dataset_dir,
        data_source=args.data_source
    )

    if args.data_source == "toucan" or args.toucan_only:
        # Generate tool call variations for Toucan
        generator.generate_tool_call_variation_dataset(num_samples=args.num_samples, output_dir=args.output_dir)
    else:
        # Generate standard variations for ShareGPT
        generator.generate_schema_variation_dataset(num_samples=args.num_samples, output_dir=args.output_dir)
        generator.generate_semantic_variation_dataset(num_samples=args.num_samples, output_dir=args.output_dir)
        generator.generate_expression_variation_dataset(num_samples=args.num_samples, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
