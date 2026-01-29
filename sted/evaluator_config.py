"""
Configuration constants and default values for the semantic JSON tree consistency evaluator.
"""

from typing import Dict, Tuple


def get_default_type_change_costs() -> Dict[Tuple[str, str], float]:
    """Define default costs for type changes."""
    costs = {}
    types = ["object", "array", "string", "number", "boolean", "null"]

    # Default cost is 1.0
    for t1 in types:
        for t2 in types:
            costs[(t1, t2)] = 1.0

    # Same type has zero cost
    for t in types:
        costs[(t, t)] = 0.0

    # Lower costs for some type conversions
    costs[("string", "number")] = costs[("number", "string")] = 0.5
    costs[("boolean", "string")] = costs[("string", "boolean")] = 0.7
    costs[("number", "boolean")] = costs[("boolean", "number")] = 0.7
    costs[("null", "string")] = costs[("string", "null")] = 0.5
    costs[("null", "number")] = costs[("number", "null")] = 0.5

    # Higher costs for structure changes
    costs[("object", "array")] = costs[("array", "object")] = 1
    costs[("object", "string")] = costs[("string", "object")] = 1
    costs[("object", "number")] = costs[("number", "object")] = 1
    costs[("array", "string")] = costs[("string", "array")] = 1
    costs[("array", "number")] = costs[("number", "array")] = 1

    return costs


def get_default_weights() -> Dict[str, float]:
    """Get default weights for different similarity components."""
    return {
        'type': 0.1,
        'value': 0.8,
        'key': 0.1
    }


def get_default_config() -> Dict:
    """Get default configuration for the evaluator."""
    return {
        'path_weight_decay': 0.9,
        'type_change_cost': get_default_type_change_costs(),
        'required_fields': set(),
        'model_id': 'all-MiniLM-L6-v2',
        'chunk_size': 300,
        'chunk_overlap': 50,
        'weights': get_default_weights(),
        'batch_size_bertscore': 2000,
        'key_sim_threshold': 0.8
    }
