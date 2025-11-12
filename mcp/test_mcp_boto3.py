#!/usr/bin/env python3
"""Test STED MCP server using boto3 (IAM authentication)"""

import boto3
import json

# Configuration
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:822507008821:runtime/sted_mcp_container-XAPUnaFuua"
REGION = "us-east-1"

client = boto3.client('bedrock-agentcore', region_name=REGION)

print("=" * 60)
print("Testing STED MCP Server with boto3")
print("=" * 60)
print(f"Runtime ARN: {RUNTIME_ARN}")
print()

# Test 1: List tools
print("Test 1: Listing tools...")
try:
    response = client.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        contentType='application/json',
        accept='application/json',
        payload=json.dumps({
            "method": "tools/list",
            "params": {}
        })
    )
    
    result = json.loads(response['payload'].read())
    print(f"✓ Found {len(result.get('tools', []))} tools:")
    for tool in result.get('tools', []):
        print(f"  - {tool['name']}: {tool['description']}")
    print()
except Exception as e:
    print(f"❌ Error: {e}\n")

# Test 2: Evaluate consistency
print("Test 2: Testing evaluate_consistency...")
try:
    response = client.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        contentType='application/json',
        accept='application/json',
        payload=json.dumps({
            "method": "tools/call",
            "params": {
                "name": "evaluate_consistency",
                "arguments": {
                    "json1": {"tool": "search", "params": {"query": "test"}},
                    "json2": {"tool": "search", "params": {"q": "test"}},
                    "variation_type": "combined"
                }
            }
        })
    )
    
    result = json.loads(response['payload'].read())
    print(f"✓ Result: {result}")
    print()
except Exception as e:
    print(f"❌ Error: {e}\n")

# Test 3: Evaluate tool calls
print("Test 3: Testing evaluate_tool_calls...")
try:
    response = client.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        contentType='application/json',
        accept='application/json',
        payload=json.dumps({
            "method": "tools/call",
            "params": {
                "name": "evaluate_tool_calls",
                "arguments": {
                    "tool_calls": [
                        {"tool": "search", "parameters": {"query": "AWS"}},
                        {"tool": "search", "parameters": {"query": "AWS"}},
                        {"tool": "search", "parameters": {"q": "AWS"}}
                    ],
                    "variation_type": "combined"
                }
            }
        })
    )
    
    result = json.loads(response['payload'].read())
    print(f"✓ Result: {result}")
    print()
except Exception as e:
    print(f"❌ Error: {e}\n")

print("=" * 60)
print("✅ Testing complete!")
print("=" * 60)
