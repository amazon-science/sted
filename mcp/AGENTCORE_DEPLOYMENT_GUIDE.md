# AWS Bedrock AgentCore Runtime Deployment Guide

Complete guide for deploying and testing the STED MCP server on AWS Bedrock AgentCore Runtime.

## Prerequisites

1. **AWS Account Setup**
   - AWS account with appropriate permissions
   - AWS CLI configured with credentials
   - Python 3.10 or higher

2. **Install Required Tools**
   ```bash
   pip install bedrock-agentcore-starter-toolkit
   pip install boto3
   ```

## Step 1: Prepare Deployment Package

```bash
cd mcp
./prepare_agentcore_deployment.sh
```

This creates `mcp_deployment/` with all necessary files.

## Step 2: Test Locally (Before Deployment)

```bash
cd ../mcp_deployment

# Install dependencies
pip install -r requirements.txt

# Test import
python -c "from mcp_server import mcp, evaluator; print('✓ Imports successful')"

# Run server locally
python mcp_server.py
```

In another terminal, test the server:
```bash
# Test with MCP client
python << 'EOF'
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def test_local():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
            
            # Test evaluate_consistency
            result = await session.call_tool(
                "evaluate_consistency",
                arguments={
                    "json1": {"tool": "search", "params": {"query": "test"}},
                    "json2": {"tool": "search", "params": {"q": "test"}},
                    "variation_type": "combined"
                }
            )
            print(f"Result: {result}")

asyncio.run(test_local())
EOF
```

## Step 3: Set Up Cognito User Pool

```bash
# Create Cognito user pool for authentication
aws cognito-idp create-user-pool \
  --pool-name sted-mcp-user-pool \
  --policies "PasswordPolicy={MinimumLength=8,RequireUppercase=true,RequireLowercase=true,RequireNumbers=true}" \
  --auto-verified-attributes email

# Note the UserPoolId from output

# Create user pool client
aws cognito-idp create-user-pool-client \
  --user-pool-id <YOUR_USER_POOL_ID> \
  --client-name sted-mcp-client \
  --generate-secret \
  --explicit-auth-flows ALLOW_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH

# Note the ClientId and ClientSecret from output
```

## Step 4: Deploy to AgentCore Runtime

### Option A: Using Starter Toolkit (Recommended)

```bash
cd mcp_deployment

# Create deployment configuration
cat > agentcore_config.yaml << EOF
runtime:
  name: sted-mcp-runtime
  protocol: mcp
  authentication:
    type: oauth
    cognito:
      userPoolId: <YOUR_USER_POOL_ID>
      clientId: <YOUR_CLIENT_ID>
      clientSecret: <YOUR_CLIENT_SECRET>
  
server:
  entrypoint: mcp_server.py
  port: 8000
  path: /mcp
  
resources:
  memory: 2048
  cpu: 1024
EOF

# Deploy using starter toolkit
agentcore-deploy --config agentcore_config.yaml --region us-east-1
```

### Option B: Manual Deployment

```bash
# Package the deployment
zip -r sted-mcp-deployment.zip . -x "*.pyc" -x "__pycache__/*"

# Upload to S3
aws s3 cp sted-mcp-deployment.zip s3://your-bucket/mcp-servers/

# Create AgentCore Runtime (adjust parameters as needed)
aws bedrock-agentcore create-runtime \
  --runtime-name sted-mcp-runtime \
  --protocol mcp \
  --code-location s3://your-bucket/mcp-servers/sted-mcp-deployment.zip \
  --authentication-config file://auth-config.json \
  --region us-east-1
```

## Step 5: Invoke Deployed MCP Server

### Get OAuth Token

```bash
# Get OAuth token from Cognito
aws cognito-idp initiate-auth \
  --auth-flow USER_PASSWORD_AUTH \
  --client-id <YOUR_CLIENT_ID> \
  --auth-parameters USERNAME=<username>,PASSWORD=<password> \
  --region us-east-1

# Extract AccessToken from response
```

### Test with Python SDK

```python
import boto3
import json

# Initialize Bedrock AgentCore client
client = boto3.client('bedrock-agentcore', region_name='us-east-1')

# Get runtime ARN (from deployment output or list-runtimes)
runtime_arn = "arn:aws:bedrock-agentcore:us-east-1:ACCOUNT_ID:runtime/sted-mcp-runtime"

# Test 1: List tools
response = client.invoke_agent_runtime(
    runtimeArn=runtime_arn,
    payload=json.dumps({
        "method": "tools/list",
        "params": {}
    }),
    headers={
        "Authorization": f"Bearer {access_token}"
    }
)

print("Available tools:", response['payload'].read())

# Test 2: Evaluate consistency
response = client.invoke_agent_runtime(
    runtimeArn=runtime_arn,
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
    }),
    headers={
        "Authorization": f"Bearer {access_token}"
    }
)

result = json.loads(response['payload'].read())
print(f"Similarity: {result['content'][0]['text']}")

# Test 3: Evaluate tool calls
response = client.invoke_agent_runtime(
    runtimeArn=runtime_arn,
    payload=json.dumps({
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
    }),
    headers={
        "Authorization": f"Bearer {access_token}"
    }
)

result = json.loads(response['payload'].read())
print(f"Tool call consistency: {result['content'][0]['text']}")
```

### Test with AWS CLI

```bash
# Set variables
RUNTIME_ARN="arn:aws:bedrock-agentcore:us-east-1:ACCOUNT_ID:runtime/sted-mcp-runtime"
ACCESS_TOKEN="<your-oauth-token>"

# Test evaluate_consistency
aws bedrock-agentcore invoke-agent-runtime \
  --runtime-arn "$RUNTIME_ARN" \
  --payload '{
    "method": "tools/call",
    "params": {
      "name": "evaluate_consistency",
      "arguments": {
        "json1": {"tool": "search", "params": {"query": "test"}},
        "json2": {"tool": "search", "params": {"q": "test"}},
        "variation_type": "combined"
      }
    }
  }' \
  --headers "Authorization=Bearer $ACCESS_TOKEN" \
  --region us-east-1
```

## Step 6: Monitor and Debug

### View Logs

```bash
# Get runtime logs
aws logs tail /aws/bedrock-agentcore/sted-mcp-runtime --follow

# Filter for errors
aws logs filter-log-events \
  --log-group-name /aws/bedrock-agentcore/sted-mcp-runtime \
  --filter-pattern "ERROR"
```

### Check Runtime Status

```bash
# Get runtime details
aws bedrock-agentcore describe-runtime \
  --runtime-arn "$RUNTIME_ARN" \
  --region us-east-1

# List all runtimes
aws bedrock-agentcore list-runtimes --region us-east-1
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies in `requirements.txt` are correct
   - Check Python version compatibility (3.10+)

2. **Authentication Failures**
   - Verify Cognito user pool configuration
   - Check OAuth token expiration
   - Ensure correct client ID and secret

3. **Server Not Responding**
   - Verify server is listening on `0.0.0.0:8000/mcp`
   - Check CloudWatch logs for startup errors
   - Ensure memory/CPU resources are sufficient

4. **Tool Execution Errors**
   - Check that STED library is properly included
   - Verify AWS Bedrock access for embeddings
   - Review tool argument types and formats

## Performance Optimization

1. **Increase Memory**: For large JSON structures
   ```bash
   aws bedrock-agentcore update-runtime \
     --runtime-arn "$RUNTIME_ARN" \
     --memory 4096
   ```

2. **Enable Caching**: For embedding results
   - STED library already includes caching
   - Consider Redis for distributed caching

3. **Monitor Metrics**:
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace AWS/BedrockAgentCore \
     --metric-name Invocations \
     --dimensions Name=RuntimeName,Value=sted-mcp-runtime \
     --start-time 2025-01-01T00:00:00Z \
     --end-time 2025-01-02T00:00:00Z \
     --period 3600 \
     --statistics Sum
   ```

## Cost Estimation

- **AgentCore Runtime**: ~$0.10 per hour
- **Bedrock Embeddings**: ~$0.0001 per 1K tokens
- **Data Transfer**: Standard AWS rates

## Security Best Practices

1. Use IAM roles with least privilege
2. Enable CloudTrail logging
3. Rotate Cognito credentials regularly
4. Use VPC endpoints for private access
5. Enable encryption at rest and in transit

## References

- [AWS Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/)
- [MCP Protocol Specification](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-mcp-protocol-contract.html)
- [AgentCore Samples](https://github.com/awslabs/amazon-bedrock-agentcore-samples)
