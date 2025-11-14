#!/usr/bin/env python3
"""
MCP Server for STED Consistency Evaluation

This server implements the Model Context Protocol (MCP) using FastMCP
to provide STED-based consistency evaluation tools for agentic systems.
"""

import sys
import os
from typing import List, Dict, Any, Literal

# Add parent directory to path to import sted
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp.server.fastmcp import FastMCP
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Initialize FastMCP server
mcp = FastMCP("STED Evaluator")

# Initialize STED evaluator (shared across all tool calls)
evaluator = SemanticJsonTreeConsistencyEvaluator(
    model_id='amazon.titan-embed-text-v2:0'
)


@mcp.tool()
def evaluate_consistency(
    json1: Dict[str, Any],
    json2: Dict[str, Any],
    variation_type: Literal["structural", "content", "combined"] = "combined"
) -> Dict[str, Any]:
    """
    Evaluate consistency between two JSON structures using STED.
    
    Args:
        json1: First JSON structure to compare
        json2: Second JSON structure to compare
        variation_type: Type of consistency evaluation - "structural" focuses on 
                       structure, "content" on semantics, "combined" on both
    
    Returns:
        Dictionary containing similarity score and variation type used
    """
    similarity = evaluator.calculate_tree_edit_distance_opt(
        json1, json2, variation_type=variation_type
    )
    
    return {
        "similarity": similarity,
        "variation_type": variation_type
    }


@mcp.tool()
def evaluate_batch_consistency(
    json_list: List[Dict[str, Any]],
    variation_type: Literal["structural", "content", "combined"] = "combined"
) -> Dict[str, Any]:
    """
    Evaluate consistency across multiple JSON structures using pairwise comparisons.
    
    Args:
        json_list: List of JSON structures to compare (minimum 2 required)
        variation_type: Type of consistency evaluation
    
    Returns:
        Dictionary with average consistency, comparison statistics, and min/max scores
    
    Raises:
        ValueError: If fewer than 2 JSON structures provided
    """
    if len(json_list) < 2:
        raise ValueError("Need at least 2 JSON structures for batch evaluation")
    
    # Calculate pairwise similarities
    scores = []
    for i in range(len(json_list)):
        for j in range(i + 1, len(json_list)):
            similarity = evaluator.calculate_tree_edit_distance_opt(
                json_list[i], json_list[j], variation_type=variation_type
            )
            scores.append(similarity)
    
    return {
        "average_consistency": sum(scores) / len(scores),
        "num_comparisons": len(scores),
        "min_similarity": min(scores),
        "max_similarity": max(scores),
        "variation_type": variation_type
    }


@mcp.tool()
def evaluate_tool_calls(
    tool_calls: List[Dict[str, Any]],
    variation_type: Literal["structural", "content", "combined"] = "combined"
) -> Dict[str, Any]:
    """
    Evaluate consistency of agent tool calls, checking both tool selection and parameters.
    
    Args:
        tool_calls: List of tool call objects (minimum 2 required). Each should have
                   'tool' or 'name' field and 'parameters' or 'params' field
        variation_type: Type of consistency evaluation
    
    Returns:
        Dictionary with tool selection consistency, unique tools used, and parameter
        consistency metrics
    
    Raises:
        ValueError: If fewer than 2 tool calls provided
    """
    if len(tool_calls) < 2:
        raise ValueError("Need at least 2 tool calls for evaluation")
    
    # Check tool selection consistency
    tool_names = [tc.get("tool") or tc.get("name") for tc in tool_calls]
    tool_consistent = len(set(tool_names)) == 1
    
    # Calculate parameter consistency
    param_scores = []
    for i in range(len(tool_calls)):
        for j in range(i + 1, len(tool_calls)):
            params1 = tool_calls[i].get("parameters") or tool_calls[i].get("params", {})
            params2 = tool_calls[j].get("parameters") or tool_calls[j].get("params", {})
            
            similarity = evaluator.calculate_tree_edit_distance_opt(
                params1, params2, variation_type=variation_type
            )
            param_scores.append(similarity)
    
    avg_param_consistency = sum(param_scores) / len(param_scores) if param_scores else 0
    
    return {
        "tool_selection_consistent": tool_consistent,
        "unique_tools": list(set(tool_names)),
        "parameter_consistency": avg_param_consistency,
        "num_comparisons": len(param_scores),
        "variation_type": variation_type
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
