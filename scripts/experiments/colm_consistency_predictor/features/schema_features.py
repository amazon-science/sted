#!/usr/bin/env python3
"""
Schema Features Extraction (6 features)

Extracts complexity features from tool/function schemas.
These features capture the structural complexity of the expected output.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class SchemaFeatures:
    """Container for 6 schema complexity features."""
    num_tools: int = 0
    max_params: int = 0
    total_params: int = 0
    max_nesting_depth: int = 0
    has_array_params: bool = False
    has_object_params: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_tools': self.num_tools,
            'max_params': self.max_params,
            'total_params': self.total_params,
            'max_nesting_depth': self.max_nesting_depth,
            'has_array_params': 1 if self.has_array_params else 0,
            'has_object_params': 1 if self.has_object_params else 0,
        }


def compute_nesting_depth(schema: Dict, current_depth: int = 0) -> int:
    """Recursively compute maximum nesting depth of a schema."""
    if not isinstance(schema, dict):
        return current_depth

    max_depth = current_depth

    # Check for nested properties
    if 'properties' in schema:
        for prop_name, prop_schema in schema['properties'].items():
            depth = compute_nesting_depth(prop_schema, current_depth + 1)
            max_depth = max(max_depth, depth)

    # Check for array items
    if 'items' in schema:
        depth = compute_nesting_depth(schema['items'], current_depth + 1)
        max_depth = max(max_depth, depth)

    # Check for anyOf/oneOf/allOf
    for key in ['anyOf', 'oneOf', 'allOf']:
        if key in schema:
            for sub_schema in schema[key]:
                depth = compute_nesting_depth(sub_schema, current_depth + 1)
                max_depth = max(max_depth, depth)

    return max_depth


def count_parameters(schema: Dict) -> int:
    """Count number of parameters in a schema."""
    if not isinstance(schema, dict):
        return 0

    count = 0

    if 'properties' in schema:
        count += len(schema['properties'])
        for prop_schema in schema['properties'].values():
            # Count nested properties
            count += count_parameters(prop_schema)

    if 'items' in schema:
        count += count_parameters(schema['items'])

    return count


def has_type(schema: Dict, type_name: str) -> bool:
    """Check if schema contains a specific type anywhere."""
    if not isinstance(schema, dict):
        return False

    if schema.get('type') == type_name:
        return True

    if 'properties' in schema:
        for prop_schema in schema['properties'].values():
            if has_type(prop_schema, type_name):
                return True

    if 'items' in schema:
        if has_type(schema['items'], type_name):
            return True

    for key in ['anyOf', 'oneOf', 'allOf']:
        if key in schema:
            for sub_schema in schema[key]:
                if has_type(sub_schema, type_name):
                    return True

    return False


def extract_tool_schema_features(tool: Dict) -> Dict[str, Any]:
    """Extract features from a single tool definition."""
    features = {
        'params': 0,
        'nesting_depth': 0,
        'has_array': False,
        'has_object': False,
    }

    # Get parameters schema
    params_schema = tool.get('parameters', {})
    if isinstance(params_schema, dict):
        features['params'] = count_parameters(params_schema)
        features['nesting_depth'] = compute_nesting_depth(params_schema)
        features['has_array'] = has_type(params_schema, 'array')
        features['has_object'] = has_type(params_schema, 'object')

        # Also check for direct properties
        if 'properties' in params_schema:
            features['params'] = max(features['params'], len(params_schema['properties']))

    return features


def extract_schema_features(tools: List[Dict]) -> SchemaFeatures:
    """
    Extract schema features from a list of tool definitions.

    Args:
        tools: List of tool definitions, each with 'name', 'description', 'parameters'

    Returns:
        SchemaFeatures dataclass
    """
    features = SchemaFeatures()

    if not tools:
        return features

    features.num_tools = len(tools)

    max_params = 0
    total_params = 0
    max_depth = 0
    has_array = False
    has_object = False

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_features = extract_tool_schema_features(tool)

        max_params = max(max_params, tool_features['params'])
        total_params += tool_features['params']
        max_depth = max(max_depth, tool_features['nesting_depth'])
        has_array = has_array or tool_features['has_array']
        has_object = has_object or tool_features['has_object']

    features.max_params = max_params
    features.total_params = total_params
    features.max_nesting_depth = max_depth
    features.has_array_params = has_array
    features.has_object_params = has_object

    return features


# Test
if __name__ == '__main__':
    # Test with sample tool definitions
    test_tools = [
        {
            "name": "search_files",
            "description": "Search for files",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "extension": {"type": "string"},
                            "size_min": {"type": "integer"}
                        }
                    }
                }
            }
        },
        {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                }
            }
        },
        {
            "name": "get_status",
            "description": "Get status",
            "parameters": {}
        }
    ]

    print("Schema Feature Extraction Test")
    print("=" * 60)

    features = extract_schema_features(test_tools)
    feat_dict = features.to_dict()

    print("\nExtracted features:")
    for k, v in feat_dict.items():
        print(f"  {k}: {v}")

    # Test with individual tools
    print("\n\nPer-tool analysis:")
    for tool in test_tools:
        tool_feat = extract_tool_schema_features(tool)
        print(f"\n  {tool['name']}:")
        for k, v in tool_feat.items():
            print(f"    {k}: {v}")
