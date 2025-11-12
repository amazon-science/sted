#!/usr/bin/env python3
"""
Invoke deployed STED MCP server on AWS Bedrock AgentCore Runtime

Usage:
    export AGENT_ARN="arn:aws:bedrock-agentcore:us-east-1:ACCOUNT:runtime/sted_mcp_server-xyz"
    export BEARER_TOKEN="your-oauth-token"  # Optional if using IAM
    python invoke_deployed_mcp.py
"""

import asyncio
import os
import sys
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def test_deployed_mcp():
    """Test the deployed STED MCP server"""
    
    # Get configuration from environment
    agent_arn = os.getenv('AGENT_ARN')
    bearer_token = os.getenv('BEARER_TOKEN')  # Optional for IAM auth
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not agent_arn:
        print("❌ Error: AGENT_ARN environment variable is not set")
        print("\nUsage:")
        print('  export AGENT_ARN="arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/NAME"')
        print('  export BEARER_TOKEN="token"  # Optional if using IAM')
        print('  python invoke_deployed_mcp.py')
        sys.exit(1)
    
    # Encode ARN for URL
    encoded_arn = agent_arn.replace(':', '%3A').replace('/', '%2F')
    mcp_url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{encoded_arn}/invocations?qualifier=DEFAULT"
    
    # Setup headers
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    
    print("=" * 60)
    print("Testing Deployed STED MCP Server")
    print("=" * 60)
    print(f"URL: {mcp_url}")
    print(f"Auth: {'OAuth' if bearer_token else 'IAM'}")
    print()
    
    try:
        async with streamablehttp_client(
            mcp_url, 
            headers, 
            timeout=120, 
            terminate_on_close=False
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize session
                print("Initializing MCP session...")
                await session.initialize()
                print("✓ Session initialized\n")
                
                # Test 1: List tools
                print("Test 1: Listing available tools...")
                tools = await session.list_tools()
                print(f"✓ Found {len(tools.tools)} tools:")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")
                print()
                
                # Test 2: Evaluate consistency
                print("Test 2: Testing evaluate_consistency...")
                result = await session.call_tool(
                    "evaluate_consistency",
                    arguments={
                        "json1": {"tool": "search", "params": {"query": "test"}},
                        "json2": {"tool": "search", "params": {"q": "test"}},
                        "variation_type": "combined"
                    }
                )
                print(f"✓ Result: {result.content[0].text}")
                print()
                
                # Test 3: Evaluate batch consistency
                print("Test 3: Testing evaluate_batch_consistency...")
                result = await session.call_tool(
                    "evaluate_batch_consistency",
                    arguments={
                        "json_list": [
                            {"tool": "search", "params": {"query": "test"}},
                            {"tool": "search", "params": {"query": "test"}},
                            {"tool": "search", "params": {"q": "test"}}
                        ],
                        "variation_type": "combined"
                    }
                )
                print(f"✓ Result: {result.content[0].text}")
                print()
                
                # Test 4: Evaluate tool calls
                print("Test 4: Testing evaluate_tool_calls...")
                result = await session.call_tool(
                    "evaluate_tool_calls",
                    arguments={
                        "tool_calls": [
                            {"tool": "search", "parameters": {"query": "AWS"}},
                            {"tool": "search", "parameters": {"query": "AWS"}},
                            {"tool": "search", "parameters": {"q": "AWS"}}
                        ],
                        "variation_type": "combined"
                    }
                )
                print(f"✓ Result: {result.content[0].text}")
                print()
                
                print("=" * 60)
                print("✅ All tests passed!")
                print("=" * 60)
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_deployed_mcp())
