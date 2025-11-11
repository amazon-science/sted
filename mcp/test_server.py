#!/usr/bin/env python3
"""Test script for STED MCP Server"""

import json
import subprocess
import sys


def test_mcp_server():
    """Test MCP server functionality"""
    
    tests = [
        {
            "name": "List Tools",
            "request": {"method": "tools/list", "params": {}},
        },
        {
            "name": "Evaluate Consistency",
            "request": {
                "method": "tools/call",
                "params": {
                    "name": "evaluate_consistency",
                    "arguments": {
                        "json1": {"tool": "search", "params": {"query": "test"}},
                        "json2": {"tool": "search", "params": {"q": "test"}},
                        "variation_type": "combined"
                    }
                }
            }
        },
        {
            "name": "Evaluate Tool Calls",
            "request": {
                "method": "tools/call",
                "params": {
                    "name": "evaluate_tool_calls",
                    "arguments": {
                        "tool_calls": [
                            {"tool": "search", "parameters": {"query": "test"}},
                            {"tool": "search", "parameters": {"query": "test"}},
                            {"tool": "search", "parameters": {"q": "test"}}
                        ],
                        "variation_type": "combined"
                    }
                }
            }
        }
    ]
    
    for test in tests:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"{'='*60}")
        
        request_json = json.dumps(test["request"])
        print(f"Request: {request_json}")
        
        try:
            result = subprocess.run(
                ["python", "server.py"],
                input=request_json + "\n",
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                response = json.loads(result.stdout.strip())
                print(f"Response: {json.dumps(response, indent=2)}")
            else:
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            print(f"Exception: {e}")


if __name__ == "__main__":
    test_mcp_server()
