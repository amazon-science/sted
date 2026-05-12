"""Default type-change costs for STED.

Extracted from semantic_json_tree_consistency.py during the v0.2.0 refactor.
The original function is re-exported there for backward compatibility.
"""
from __future__ import annotations

from typing import Dict, Tuple


def _get_default_type_change_costs() -> Dict[Tuple[str, str], float]:
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
    costs[("string", "number")] = costs[("number", "string")] = 0.1
    costs[("boolean", "string")] = costs[("string", "boolean")] = 0.1
    costs[("number", "boolean")] = costs[("boolean", "number")] = 0.1
    costs[("null", "string")] = costs[("string", "null")] = 0.1
    costs[("null", "number")] = costs[("number", "null")] = 0.1

    # Higher costs for structure changes
    costs[("object", "array")] = costs[("array", "object")] = 0.5
    costs[("object", "string")] = costs[("string", "object")] = 0.5
    costs[("object", "number")] = costs[("number", "object")] = 0.5
    costs[("array", "string")] = costs[("string", "array")] = 0.5
    costs[("array", "number")] = costs[("number", "array")] = 1

    return costs
