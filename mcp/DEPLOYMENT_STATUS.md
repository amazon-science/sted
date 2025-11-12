# STED MCP Server - AgentCore Deployment Status

**Date:** November 11, 2025  
**Status:** ⚠️ Deployment Blocked - Dependency Issue

## Summary

Successfully installed `bedrock-agentcore-starter-toolkit` and configured the STED MCP server for deployment. However, deployment is blocked due to ARM64 compatibility issues with the `zss` library.

## What Was Accomplished

### ✅ 1. Toolkit Installation
```bash
pip install bedrock-agentcore-starter-toolkit
```
- Installed version: 0.1.32
- Includes `agentcore` CLI with full deployment capabilities

### ✅ 2. Server Configuration
```bash
agentcore configure \
  --name sted_mcp_server \
  --entrypoint mcp_server.py \
  --protocol MCP \
  --deployment-type direct_code_deploy \
  --runtime PYTHON_3_11 \
  --region us-east-1
```

**Configuration Created:**
- Agent Name: `sted_mcp_server`
- Deployment Type: `direct_code_deploy`
- Runtime: Python 3.11
- Protocol: MCP
- Region: us-east-1
- Account: 822507008821

**Resources Created:**
- Execution Role: `arn:aws:iam::822507008821:role/AmazonBedrockAgentCoreSDKRuntime-us-east-1-21b0e2f5b8`
- Memory Resource: `sted_mcp_server_mem-WUp3TZ7ILL` (ACTIVE)
- Config File: `.bedrock_agentcore.yaml`

### ❌ 3. Deployment Attempt
```bash
agentcore launch --agent sted_mcp_server
```

**Error:**
```
❌ Failed to install dependencies with uv: 
Because zss==1.2.0 has no usable wheels and only zss<=1.2.0 is available,
we can conclude that zss>=1.2.0 cannot be used.
```

**Root Cause:**
- `zss` library doesn't have ARM64 (manylinux2014_aarch64) wheels
- AgentCore Runtime uses ARM64 architecture
- `direct_code_deploy` requires pre-built wheels (no compilation)

## Workarounds

### Option 1: Use Container Deployment (RECOMMENDED)

Container deployment allows building from source:

```bash
# Reconfigure for container deployment
cd mcp_deployment
agentcore configure \
  --name sted_mcp_server \
  --entrypoint mcp_server.py \
  --protocol MCP \
  --deployment-type container \
  --region us-east-1 \
  --non-interactive

# Launch with container
agentcore launch --agent sted_mcp_server
```

This will:
1. Build a Docker container with all dependencies
2. Compile `zss` from source during build
3. Push to ECR
4. Deploy to AgentCore Runtime

### Option 2: Remove zss Dependency

The STED library imports `zss` but the MCP server uses `calculate_tree_edit_distance_opt` which sets `original_zss=False`, potentially avoiding the zss code path.

**Steps:**
1. Comment out `import zss` in `sted/semantic_json_tree_consistency.py`
2. Remove `zss` from `requirements.txt`
3. Test locally to ensure functionality
4. Deploy with `direct_code_deploy`

**Risk:** May break if zss is actually used in the code path.

### Option 3: Use Pre-built zss Wheel

Build zss wheel on ARM64 machine and include in deployment:

```bash
# On ARM64 machine or use QEMU
pip wheel zss --wheel-dir=./wheels
# Include wheels/ directory in deployment
```

## Invocation Script Ready

Created `invoke_deployed_mcp.py` for testing once deployed:

```bash
export AGENT_ARN="arn:aws:bedrock-agentcore:us-east-1:822507008821:runtime/sted_mcp_server-xyz"
export BEARER_TOKEN="token"  # Optional for IAM auth
python invoke_deployed_mcp.py
```

**Tests Included:**
1. List tools
2. Evaluate consistency
3. Evaluate batch consistency
4. Evaluate tool calls

## Next Steps

### Immediate (Choose One):

**A. Container Deployment (Recommended)**
```bash
cd mcp_deployment
agentcore configure --deployment-type container --name sted_mcp_server_container
agentcore launch
```

**B. Test Without zss**
```bash
# Modify code to remove zss dependency
# Test locally
# Deploy with direct_code_deploy
```

### After Successful Deployment:

1. Get runtime ARN from deployment output
2. Set environment variables:
   ```bash
   export AGENT_ARN="<runtime-arn>"
   ```
3. Test invocation:
   ```bash
   python mcp/invoke_deployed_mcp.py
   ```
4. Document results
5. Update README with deployment instructions

## Resources

- **AgentCore CLI Docs:** `agentcore --help`
- **AWS Documentation:** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-mcp.html
- **MCP Samples:** https://github.com/awslabs/amazon-bedrock-agentcore-samples

## Files Created

- `mcp/invoke_deployed_mcp.py` - Invocation test script
- `mcp_deployment/.bedrock_agentcore.yaml` - AgentCore configuration
- `DEPLOYMENT_STATUS.md` - This file

## AWS Resources

| Resource | ID/ARN | Status |
|----------|--------|--------|
| Execution Role | arn:aws:iam::822507008821:role/AmazonBedrockAgentCoreSDKRuntime-us-east-1-21b0e2f5b8 | ACTIVE |
| Memory Resource | sted_mcp_server_mem-WUp3TZ7ILL | ACTIVE |
| Runtime | N/A | NOT DEPLOYED |
| S3 Bucket | bedrock-agentcore-822507008821-us-east-1 | EXISTS |
| Cognito User Pool | us-east-1_uXNWClPGN | ACTIVE |

## Conclusion

The STED MCP server is fully configured and ready for deployment. The only blocker is the `zss` ARM64 compatibility issue. Using **container deployment** is the recommended solution as it allows building from source and will work with all dependencies.
