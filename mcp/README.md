# STED MCP Server

Model Context Protocol (MCP) server for STED consistency evaluation in agentic systems.

## Features

- **evaluate_consistency**: Compare two JSON structures
- **evaluate_batch_consistency**: Compare multiple JSON structures
- **evaluate_tool_calls**: Evaluate agent tool call consistency

## Installation

```bash
# Install dependencies
pip install -e ..

# Or with uv
uv pip install -e ..
```

## Usage

### Standalone Testing

```bash
# Test the server
echo '{"method":"tools/list","params":{}}' | python server.py
```

### Integration with MCP Clients

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "sted-evaluator": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/field-aware-consistency-evaluation-framework/mcp"
    }
  }
}
```

### Example Tool Calls

**Evaluate Consistency:**
```json
{
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
```

**Evaluate Tool Calls:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "evaluate_tool_calls",
    "arguments": {
      "tool_calls": [
        {"tool": "search", "parameters": {"query": "test"}},
        {"tool": "search", "parameters": {"query": "test"}}
      ],
      "variation_type": "combined"
    }
  }
}
```

## Tools

### evaluate_consistency
Compare two JSON structures using STED.

**Parameters:**
- `json1` (object): First JSON structure
- `json2` (object): Second JSON structure
- `variation_type` (string): "structural", "content", or "combined" (default)

**Returns:**
- `similarity`: Similarity score (0-1)
- `edit_distance`: Tree edit distance
- `variation_type`: Type used for evaluation

### evaluate_batch_consistency
Evaluate consistency across multiple JSON structures.

**Parameters:**
- `json_list` (array): List of JSON structures
- `variation_type` (string): Type of consistency

**Returns:**
- `average_consistency`: Average pairwise similarity
- `num_comparisons`: Number of comparisons made
- `min_similarity`: Minimum similarity score
- `max_similarity`: Maximum similarity score

### evaluate_tool_calls
Evaluate consistency of agent tool calls.

**Parameters:**
- `tool_calls` (array): List of tool call objects
- `variation_type` (string): Type of consistency

**Returns:**
- `tool_selection_consistent`: Whether same tool was selected
- `unique_tools`: List of unique tools used
- `parameter_consistency`: Average parameter similarity
- `num_comparisons`: Number of comparisons made

## Use Cases

- Real-time agent consistency monitoring
- Tool call validation in agentic workflows
- Multi-agent coordination verification
- Prompt robustness testing
