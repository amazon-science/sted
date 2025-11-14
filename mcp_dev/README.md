# STED MCP Server

Model Context Protocol (MCP) server for real-time STED consistency evaluation in agentic systems.

## Overview

This MCP server exposes STED (Semantic Tree Edit Distance) evaluation capabilities as tools that can be used by AI agents and agentic systems to evaluate consistency of structured outputs in real-time.

Built with [FastMCP](https://github.com/jlowin/fastmcp), the server provides:
- **Decorator-based API**: Define tools with simple `@mcp.tool()` decorators
- **Automatic schema generation**: Type hints automatically generate JSON schemas
- **Type safety**: Runtime validation from Python type annotations
- **Multiple transports**: stdio (default), SSE, and HTTP streaming support
- **Clean code**: ~30% less code compared to manual implementation

## Features

Three evaluation tools are provided:

- **evaluate_consistency**: Compare two JSON structures
- **evaluate_batch_consistency**: Evaluate consistency across multiple JSON structures  
- **evaluate_tool_calls**: Evaluate consistency of agent tool calls

## Prerequisites

- Python 3.8+
- AWS credentials configured (for Bedrock embedding models)
- STED library installed
- MCP package (`pip install mcp`)

## Installation

```bash
# Install from parent directory
cd ..
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

### Test the Server

```bash
cd mcp
echo '{"method":"tools/list","params":{}}' | python server.py
```

### Integration with MCP Clients

Add to your MCP client configuration (e.g., Claude Desktop, Cline):

```json
{
  "mcpServers": {
    "sted-evaluator": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/field-aware-consistency-evaluation-framework/mcp",
      "env": {
        "AWS_REGION": "us-east-1"
      }
    }
  }
}
```

## Usage Examples

### Evaluate Consistency Between Two JSONs

```json
{
  "method": "tools/call",
  "params": {
    "name": "evaluate_consistency",
    "arguments": {
      "json1": {"name": "John", "age": 30},
      "json2": {"name": "John", "age": 30},
      "variation_type": "combined"
    }
  }
}
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "{\"similarity\": 1.0, \"variation_type\": \"combined\"}"
  }]
}
```

### Evaluate Batch Consistency

```json
{
  "method": "tools/call",
  "params": {
    "name": "evaluate_batch_consistency",
    "arguments": {
      "json_list": [
        {"tool": "search", "query": "test"},
        {"tool": "search", "query": "test"},
        {"tool": "search", "q": "test"}
      ],
      "variation_type": "combined"
    }
  }
}
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "{
      \"average_consistency\": 0.85,
      \"num_comparisons\": 3,
      \"min_similarity\": 0.75,
      \"max_similarity\": 1.0,
      \"variation_type\": \"combined\"
    }"
  }]
}
```

### Evaluate Tool Call Consistency

```json
{
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
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "{
      \"tool_selection_consistent\": true,
      \"unique_tools\": [\"search\"],
      \"parameter_consistency\": 0.85,
      \"num_comparisons\": 3,
      \"variation_type\": \"combined\"
    }"
  }]
}
```

## Tool Reference

### evaluate_consistency

Compare two JSON structures using STED.

**Parameters:**
- `json1` (object, required): First JSON structure
- `json2` (object, required): Second JSON structure
- `variation_type` (string, optional): "structural", "content", or "combined" (default: "combined")

**Returns:**
- `similarity`: Similarity score between 0 and 1
- `variation_type`: Type used for evaluation

### evaluate_batch_consistency

Evaluate consistency across multiple JSON structures using pairwise comparisons.

**Parameters:**
- `json_list` (array, required): List of JSON structures (minimum 2)
- `variation_type` (string, optional): Type of consistency evaluation

**Returns:**
- `average_consistency`: Average pairwise similarity
- `num_comparisons`: Number of pairwise comparisons
- `min_similarity`: Minimum similarity score
- `max_similarity`: Maximum similarity score
- `variation_type`: Type used for evaluation

### evaluate_tool_calls

Evaluate consistency of agent tool calls, checking both tool selection and parameter consistency.

**Parameters:**
- `tool_calls` (array, required): List of tool call objects (minimum 2)
  - Each object should have: `tool`/`name` and `parameters`/`params`
- `variation_type` (string, optional): Type of consistency evaluation

**Returns:**
- `tool_selection_consistent`: Whether the same tool was selected across all calls
- `unique_tools`: List of unique tools used
- `parameter_consistency`: Average parameter similarity across calls
- `num_comparisons`: Number of pairwise comparisons
- `variation_type`: Type used for evaluation

## Variation Types

- **structural**: Focus on JSON structure similarity (field names, nesting)
- **content**: Focus on semantic content similarity (field values)
- **combined**: Balanced evaluation of both structure and content (default)

## Use Cases

- **Real-time Agent Monitoring**: Track consistency of agent outputs during execution
- **Tool Call Validation**: Verify that agents consistently select and parameterize tools
- **Multi-Agent Coordination**: Ensure multiple agents produce consistent outputs
- **Prompt Robustness Testing**: Evaluate output stability across prompt variations
- **Quality Assurance**: Automated consistency checks in agentic workflows

## Testing

Run the test to verify the server is working:

```bash
python test_fastmcp.py
```

For interactive testing with MCP clients, the server runs on stdio and waits for JSON-RPC requests.

## Troubleshooting

**AWS Credentials Error:**
```bash
# Configure AWS credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_DEFAULT_REGION=us-east-1
```

**Import Error:**
Ensure the STED library is installed:
```bash
cd ..
pip install -e .
```

**JSON Decode Error:**
Ensure requests are valid JSON and properly formatted.

## Advanced: AWS Bedrock AgentCore Deployment

For production deployment to AWS Bedrock AgentCore Runtime, see [AGENTCORE_DEPLOYMENT_GUIDE.md](./AGENTCORE_DEPLOYMENT_GUIDE.md).

## License

See parent directory for license information.
