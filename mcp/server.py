#!/usr/bin/env python3
"""MCP Server for STED Consistency Evaluation"""

import json
import sys
from typing import Any

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


class STEDMCPServer:
    def __init__(self):
        self.evaluator = SemanticJsonTreeConsistencyEvaluator(
            model_id='amazon.titan-embed-text-v2:0'
        )
    
    def handle_request(self, request: dict) -> dict:
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return self._list_tools()
        elif method == "tools/call":
            return self._call_tool(params)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _list_tools(self) -> dict:
        return {
            "tools": [
                {
                    "name": "evaluate_consistency",
                    "description": "Evaluate consistency between two JSON structures using STED",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "json1": {"type": "object", "description": "First JSON structure"},
                            "json2": {"type": "object", "description": "Second JSON structure"},
                            "variation_type": {
                                "type": "string",
                                "enum": ["structural", "content", "combined"],
                                "default": "combined",
                                "description": "Type of consistency to evaluate"
                            }
                        },
                        "required": ["json1", "json2"]
                    }
                },
                {
                    "name": "evaluate_batch_consistency",
                    "description": "Evaluate consistency across multiple JSON structures",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "json_list": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "List of JSON structures to compare"
                            },
                            "variation_type": {
                                "type": "string",
                                "enum": ["structural", "content", "combined"],
                                "default": "combined"
                            }
                        },
                        "required": ["json_list"]
                    }
                },
                {
                    "name": "evaluate_tool_calls",
                    "description": "Evaluate consistency of agent tool calls",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tool_calls": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "List of tool call objects"
                            },
                            "variation_type": {
                                "type": "string",
                                "enum": ["structural", "content", "combined"],
                                "default": "combined"
                            }
                        },
                        "required": ["tool_calls"]
                    }
                }
            ]
        }
    
    def _call_tool(self, params: dict) -> dict:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "evaluate_consistency":
                return self._evaluate_consistency(arguments)
            elif tool_name == "evaluate_batch_consistency":
                return self._evaluate_batch_consistency(arguments)
            elif tool_name == "evaluate_tool_calls":
                return self._evaluate_tool_calls(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _evaluate_consistency(self, args: dict) -> dict:
        json1 = args["json1"]
        json2 = args["json2"]
        variation_type = args.get("variation_type", "combined")
        
        similarity = self.evaluator.calculate_tree_edit_distance_opt(
            json1, json2, variation_type=variation_type
        )
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "similarity": similarity,
                    "variation_type": variation_type
                }, indent=2)
            }]
        }
    
    def _evaluate_batch_consistency(self, args: dict) -> dict:
        json_list = args["json_list"]
        variation_type = args.get("variation_type", "combined")
        
        if len(json_list) < 2:
            return {"error": "Need at least 2 JSON structures"}
        
        # Pairwise comparisons
        scores = []
        for i in range(len(json_list)):
            for j in range(i + 1, len(json_list)):
                similarity = self.evaluator.calculate_tree_edit_distance_opt(
                    json_list[i], json_list[j], variation_type=variation_type
                )
                scores.append(similarity)
        
        avg_consistency = sum(scores) / len(scores)
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "average_consistency": avg_consistency,
                    "num_comparisons": len(scores),
                    "min_similarity": min(scores),
                    "max_similarity": max(scores),
                    "variation_type": variation_type
                }, indent=2)
            }]
        }
    
    def _evaluate_tool_calls(self, args: dict) -> dict:
        tool_calls = args["tool_calls"]
        variation_type = args.get("variation_type", "combined")
        
        if len(tool_calls) < 2:
            return {"error": "Need at least 2 tool calls"}
        
        # Evaluate tool selection consistency
        tool_names = [tc.get("tool") or tc.get("name") for tc in tool_calls]
        tool_consistency = len(set(tool_names)) == 1
        
        # Evaluate parameter consistency
        param_scores = []
        for i in range(len(tool_calls)):
            for j in range(i + 1, len(tool_calls)):
                params1 = tool_calls[i].get("parameters") or tool_calls[i].get("params", {})
                params2 = tool_calls[j].get("parameters") or tool_calls[j].get("params", {})
                
                similarity = self.evaluator.calculate_tree_edit_distance_opt(
                    params1, params2, variation_type=variation_type
                )
                param_scores.append(similarity)
        
        avg_param_consistency = sum(param_scores) / len(param_scores) if param_scores else 0
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "tool_selection_consistent": tool_consistency,
                    "unique_tools": list(set(tool_names)),
                    "parameter_consistency": avg_param_consistency,
                    "num_comparisons": len(param_scores),
                    "variation_type": variation_type
                }, indent=2)
            }]
        }
    
    def run(self):
        """Run MCP server on stdin/stdout"""
        for line in sys.stdin:
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    server = STEDMCPServer()
    server.run()
