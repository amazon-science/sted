#!/bin/bash
# Script to prepare STED MCP server for AWS Bedrock AgentCore Runtime deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="${PROJECT_ROOT}/mcp_deployment"

echo "Preparing MCP deployment package for AgentCore Runtime..."
echo "Project root: ${PROJECT_ROOT}"
echo "Deployment directory: ${DEPLOY_DIR}"

# Clean up existing deployment directory
if [ -d "$DEPLOY_DIR" ]; then
    echo "Cleaning up existing deployment directory..."
    rm -rf "$DEPLOY_DIR"
fi

# Create deployment directory structure
echo "Creating deployment directory structure..."
mkdir -p "$DEPLOY_DIR"

# Copy STED library
echo "Copying STED library..."
cp -r "${PROJECT_ROOT}/sted" "$DEPLOY_DIR/"

# Copy MCP server
echo "Copying MCP server..."
mkdir -p "${DEPLOY_DIR}/mcp"
cp "${SCRIPT_DIR}/server.py" "${DEPLOY_DIR}/mcp/"

# Create FastMCP-compatible server for AgentCore
echo "Creating AgentCore-compatible MCP server..."
cat > "${DEPLOY_DIR}/mcp_server.py" << 'EOF'
#!/usr/bin/env python3
"""MCP Server for STED Consistency Evaluation - AgentCore Runtime Compatible"""

from mcp.server.fastmcp import FastMCP
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Initialize FastMCP with stateless HTTP for AgentCore Runtime
mcp = FastMCP(host="0.0.0.0", stateless_http=True)

# Initialize STED evaluator
evaluator = SemanticJsonTreeConsistencyEvaluator(
    model_id='amazon.titan-embed-text-v2:0'
)

@mcp.tool()
def evaluate_consistency(json1: dict, json2: dict, variation_type: str = "combined") -> dict:
    """Evaluate consistency between two JSON structures using STED
    
    Args:
        json1: First JSON structure
        json2: Second JSON structure
        variation_type: Type of consistency ("structural", "content", or "combined")
    
    Returns:
        Dictionary with similarity score and variation type
    """
    similarity = evaluator.calculate_tree_edit_distance_opt(
        json1, json2, variation_type=variation_type
    )
    return {
        "similarity": float(similarity),
        "variation_type": variation_type
    }

@mcp.tool()
def evaluate_batch_consistency(json_list: list, variation_type: str = "combined") -> dict:
    """Evaluate consistency across multiple JSON structures
    
    Args:
        json_list: List of JSON structures to compare
        variation_type: Type of consistency to evaluate
    
    Returns:
        Dictionary with average consistency and statistics
    """
    if len(json_list) < 2:
        return {"error": "Need at least 2 JSON structures"}
    
    scores = []
    for i in range(len(json_list)):
        for j in range(i + 1, len(json_list)):
            similarity = evaluator.calculate_tree_edit_distance_opt(
                json_list[i], json_list[j], variation_type=variation_type
            )
            scores.append(float(similarity))
    
    return {
        "average_consistency": sum(scores) / len(scores),
        "num_comparisons": len(scores),
        "min_similarity": min(scores),
        "max_similarity": max(scores),
        "variation_type": variation_type
    }

@mcp.tool()
def evaluate_tool_calls(tool_calls: list, variation_type: str = "combined") -> dict:
    """Evaluate consistency of agent tool calls
    
    Args:
        tool_calls: List of tool call objects with 'tool'/'name' and 'parameters'/'params'
        variation_type: Type of consistency to evaluate
    
    Returns:
        Dictionary with tool selection and parameter consistency metrics
    """
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
            
            similarity = evaluator.calculate_tree_edit_distance_opt(
                params1, params2, variation_type=variation_type
            )
            param_scores.append(float(similarity))
    
    avg_param_consistency = sum(param_scores) / len(param_scores) if param_scores else 0
    
    return {
        "tool_selection_consistent": tool_consistency,
        "unique_tools": list(set(tool_names)),
        "parameter_consistency": avg_param_consistency,
        "num_comparisons": len(param_scores),
        "variation_type": variation_type
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
EOF

# Create requirements.txt
echo "Creating requirements.txt..."
cat > "${DEPLOY_DIR}/requirements.txt" << 'EOF'
mcp
aiohttp>=3.8.0
boto3>=1.26.0
sentence-transformers>=4.1.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
deepdiff>=8.5.0
tenacity>=9.1.2
tqdm>=4.62.0
zss>=1.2.0
EOF

# Create __init__.py
echo "Creating __init__.py..."
touch "${DEPLOY_DIR}/__init__.py"

# Create README for deployment
echo "Creating deployment README..."
cat > "${DEPLOY_DIR}/README.md" << 'EOF'
# STED MCP Server - AgentCore Runtime Deployment Package

This package contains the STED MCP server configured for AWS Bedrock AgentCore Runtime.

## Contents

- `mcp_server.py` - FastMCP server compatible with AgentCore Runtime
- `sted/` - STED library for consistency evaluation
- `requirements.txt` - Python dependencies
- `mcp/server.py` - Original stdio-based MCP server (for reference)

## Deployment to AgentCore Runtime

### Prerequisites
```bash
pip install bedrock-agentcore-starter-toolkit
```

### Deploy
Follow the AWS Bedrock AgentCore MCP deployment guide:
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-mcp.html

### Key Configuration
- Server runs at: `0.0.0.0:8000/mcp`
- Transport: `streamable-http`
- Stateless: `True`

## Tools Available

1. **evaluate_consistency** - Compare two JSON structures
2. **evaluate_batch_consistency** - Compare multiple JSON structures
3. **evaluate_tool_calls** - Evaluate agent tool call consistency

## Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python mcp_server.py
```

Server will be available at http://0.0.0.0:8000/mcp
EOF

# Create deployment info file
echo "Creating deployment info..."
cat > "${DEPLOY_DIR}/DEPLOYMENT_INFO.txt" << EOF
STED MCP Server Deployment Package
Generated: $(date)
Source: ${PROJECT_ROOT}

Files included:
- mcp_server.py (AgentCore-compatible FastMCP server)
- sted/ (Complete STED library)
- requirements.txt (All dependencies)
- mcp/server.py (Original stdio server for reference)

Next steps:
1. Install bedrock-agentcore-starter-toolkit
2. Set up Cognito user pool for authentication
3. Deploy using the starter toolkit
4. Configure AgentCore Runtime

See README.md for detailed instructions.
EOF

echo ""
echo "✅ Deployment package prepared successfully!"
echo ""
echo "Location: ${DEPLOY_DIR}"
echo ""
echo "Contents:"
ls -lh "$DEPLOY_DIR"
echo ""
echo "Next steps:"
echo "1. cd ${DEPLOY_DIR}"
echo "2. Review README.md for deployment instructions"
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Test locally: python mcp_server.py"
echo "5. Deploy to AgentCore Runtime using bedrock-agentcore-starter-toolkit"
echo ""
