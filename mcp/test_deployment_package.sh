#!/bin/bash
# Test the MCP deployment package locally before deploying to AgentCore

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/../mcp_deployment"

echo "Testing MCP Deployment Package"
echo "==============================="
echo ""

# Check if deployment package exists
if [ ! -d "$DEPLOY_DIR" ]; then
    echo "❌ Deployment package not found. Run ./prepare_agentcore_deployment.sh first"
    exit 1
fi

echo "✓ Deployment package found at: $DEPLOY_DIR"
echo ""

# Check required files
echo "Checking required files..."
required_files=(
    "mcp_server.py"
    "requirements.txt"
    "README.md"
    "__init__.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$DEPLOY_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ❌ $file missing"
        exit 1
    fi
done

# Check sted directory
if [ -d "$DEPLOY_DIR/sted" ]; then
    echo "  ✓ sted/ directory"
else
    echo "  ❌ sted/ directory missing"
    exit 1
fi

echo ""

# Test Python imports
echo "Testing Python imports..."
cd "$DEPLOY_DIR"

python3 << 'EOF'
import sys
import os

try:
    # Test basic imports
    print("  Testing basic imports...")
    import json
    import asyncio
    print("    ✓ Standard library imports")
    
    # Test STED imports
    print("  Testing STED imports...")
    from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
    print("    ✓ STED library imports")
    
    # Test MCP server imports
    print("  Testing MCP server imports...")
    import mcp_server
    print("    ✓ MCP server imports")
    
    # Test evaluator initialization
    print("  Testing evaluator initialization...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(model_id='amazon.titan-embed-text-v2:0')
    print("    ✓ Evaluator initialized")
    
    # Test basic functionality
    print("  Testing basic STED functionality...")
    json1 = {"tool": "search", "params": {"query": "test"}}
    json2 = {"tool": "search", "params": {"q": "test"}}
    similarity = evaluator.calculate_tree_edit_distance_opt(json1, json2, "combined")
    print(f"    ✓ STED calculation successful (similarity: {similarity:.4f})")
    
    print("\n✅ All tests passed!")
    sys.exit(0)
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nMissing dependencies. Install with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

test_result=$?

echo ""
if [ $test_result -eq 0 ]; then
    echo "==============================="
    echo "✅ Deployment package is ready!"
    echo "==============================="
    echo ""
    echo "Next steps:"
    echo "1. Review AGENTCORE_DEPLOYMENT_GUIDE.md for deployment instructions"
    echo "2. Set up AWS Cognito user pool"
    echo "3. Deploy using bedrock-agentcore-starter-toolkit"
    echo ""
else
    echo "==============================="
    echo "❌ Tests failed"
    echo "==============================="
    echo ""
    echo "Fix the issues above before deploying"
    echo ""
    exit 1
fi
