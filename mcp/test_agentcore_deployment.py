#!/usr/bin/env python3
"""
Test script for deploying and invoking STED MCP server on AWS Bedrock AgentCore Runtime

Prerequisites:
1. AWS credentials configured
2. bedrock-agentcore-starter-toolkit installed
3. Cognito user pool created (or use script to create)
"""

import boto3
import json
import sys
import time
from pathlib import Path

# Configuration
REGION = "us-east-1"
RUNTIME_NAME = "sted-mcp-runtime-test"
USER_POOL_NAME = "sted-mcp-user-pool"
DEPLOYMENT_DIR = Path(__file__).parent.parent / "mcp_deployment"

def check_prerequisites():
    """Check if prerequisites are met"""
    print("Checking prerequisites...")
    
    # Check AWS credentials
    try:
        sts = boto3.client('sts', region_name=REGION)
        identity = sts.get_caller_identity()
        print(f"  ✓ AWS credentials configured (Account: {identity['Account']})")
    except Exception as e:
        print(f"  ❌ AWS credentials not configured: {e}")
        return False
    
    # Check deployment package
    if not DEPLOYMENT_DIR.exists():
        print(f"  ❌ Deployment package not found at {DEPLOYMENT_DIR}")
        print("     Run: cd mcp && ./prepare_agentcore_deployment.sh")
        return False
    print(f"  ✓ Deployment package found")
    
    # Check if bedrock-agentcore is available
    try:
        client = boto3.client('bedrock-agentcore', region_name=REGION)
        print(f"  ✓ Bedrock AgentCore service available")
    except Exception as e:
        print(f"  ⚠️  Bedrock AgentCore client error: {e}")
        print("     Note: Service may not be available in your region yet")
    
    return True

def create_cognito_user_pool():
    """Create Cognito user pool for authentication"""
    print("\nCreating Cognito user pool...")
    
    cognito = boto3.client('cognito-idp', region_name=REGION)
    
    try:
        # Check if pool already exists
        pools = cognito.list_user_pools(MaxResults=60)
        for pool in pools.get('UserPools', []):
            if pool['Name'] == USER_POOL_NAME:
                print(f"  ✓ User pool already exists: {pool['Id']}")
                return pool['Id']
        
        # Create user pool
        response = cognito.create_user_pool(
            PoolName=USER_POOL_NAME,
            Policies={
                'PasswordPolicy': {
                    'MinimumLength': 8,
                    'RequireUppercase': True,
                    'RequireLowercase': True,
                    'RequireNumbers': True,
                    'RequireSymbols': False
                }
            },
            AutoVerifiedAttributes=['email']
        )
        
        user_pool_id = response['UserPool']['Id']
        print(f"  ✓ User pool created: {user_pool_id}")
        
        # Create app client
        client_response = cognito.create_user_pool_client(
            UserPoolId=user_pool_id,
            ClientName=f"{USER_POOL_NAME}-client",
            GenerateSecret=True,
            ExplicitAuthFlows=[
                'ALLOW_USER_PASSWORD_AUTH',
                'ALLOW_REFRESH_TOKEN_AUTH'
            ]
        )
        
        client_id = client_response['UserPoolClient']['ClientId']
        print(f"  ✓ App client created: {client_id}")
        
        return user_pool_id, client_id
        
    except Exception as e:
        print(f"  ❌ Error creating Cognito user pool: {e}")
        return None

def package_deployment():
    """Package deployment for upload"""
    print("\nPackaging deployment...")
    
    import zipfile
    import tempfile
    
    zip_path = Path(tempfile.gettempdir()) / "sted-mcp-deployment.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in DEPLOYMENT_DIR.rglob('*'):
            if file_path.is_file() and '__pycache__' not in str(file_path):
                arcname = file_path.relative_to(DEPLOYMENT_DIR)
                zipf.write(file_path, arcname)
                
    print(f"  ✓ Package created: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return zip_path

def upload_to_s3(zip_path):
    """Upload deployment package to S3"""
    print("\nUploading to S3...")
    
    s3 = boto3.client('s3', region_name=REGION)
    
    # Create bucket name
    account_id = boto3.client('sts').get_caller_identity()['Account']
    bucket_name = f"bedrock-agentcore-{account_id}-{REGION}"
    
    try:
        # Create bucket if it doesn't exist
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"  ✓ Using existing bucket: {bucket_name}")
        except:
            s3.create_bucket(Bucket=bucket_name)
            print(f"  ✓ Created bucket: {bucket_name}")
        
        # Upload file
        key = f"mcp-servers/{RUNTIME_NAME}/deployment.zip"
        s3.upload_file(str(zip_path), bucket_name, key)
        
        s3_uri = f"s3://{bucket_name}/{key}"
        print(f"  ✓ Uploaded to: {s3_uri}")
        
        return s3_uri
        
    except Exception as e:
        print(f"  ❌ Error uploading to S3: {e}")
        return None

def deploy_to_agentcore(s3_uri, user_pool_id=None):
    """Deploy to AgentCore Runtime"""
    print("\nDeploying to AgentCore Runtime...")
    print("  ⚠️  Note: This requires AWS Bedrock AgentCore service to be available")
    print("  ⚠️  The service may not be generally available yet")
    
    try:
        client = boto3.client('bedrock-agentcore', region_name=REGION)
        
        # This is a placeholder - actual API may differ
        # Refer to official AWS documentation when service is GA
        print(f"  ℹ️  Would deploy from: {s3_uri}")
        print(f"  ℹ️  Runtime name: {RUNTIME_NAME}")
        print(f"  ℹ️  Protocol: MCP")
        
        print("\n  📝 Manual deployment steps:")
        print("     1. Use bedrock-agentcore-starter-toolkit")
        print("     2. Or use AWS Console when available")
        print("     3. Or use CloudFormation/Terraform")
        
        return None
        
    except Exception as e:
        print(f"  ⚠️  AgentCore deployment: {e}")
        return None

def test_invocation(runtime_arn):
    """Test invoking the deployed MCP server"""
    print("\nTesting MCP server invocation...")
    
    if not runtime_arn:
        print("  ⚠️  Skipping - no runtime ARN available")
        return
    
    try:
        client = boto3.client('bedrock-agentcore', region_name=REGION)
        
        # Test 1: List tools
        print("  Testing tools/list...")
        response = client.invoke_agent_runtime(
            runtimeArn=runtime_arn,
            payload=json.dumps({
                "method": "tools/list",
                "params": {}
            })
        )
        print(f"    ✓ Tools listed")
        
        # Test 2: Evaluate consistency
        print("  Testing evaluate_consistency...")
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
            })
        )
        
        result = json.loads(response['payload'].read())
        print(f"    ✓ Consistency evaluated: {result}")
        
    except Exception as e:
        print(f"  ❌ Invocation error: {e}")

def main():
    """Main test flow"""
    print("=" * 60)
    print("STED MCP Server - AgentCore Deployment Test")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Create Cognito user pool
    cognito_result = create_cognito_user_pool()
    if not cognito_result:
        print("\n⚠️  Continuing without Cognito setup...")
    
    # Package deployment
    zip_path = package_deployment()
    if not zip_path:
        print("\n❌ Failed to package deployment")
        sys.exit(1)
    
    # Upload to S3
    s3_uri = upload_to_s3(zip_path)
    if not s3_uri:
        print("\n❌ Failed to upload to S3")
        sys.exit(1)
    
    # Deploy to AgentCore
    runtime_arn = deploy_to_agentcore(s3_uri, cognito_result)
    
    # Test invocation
    if runtime_arn:
        test_invocation(runtime_arn)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"✓ Deployment package prepared")
    print(f"✓ Uploaded to S3: {s3_uri}")
    print(f"⚠️  AgentCore deployment: Manual steps required")
    print(f"\nNext steps:")
    print(f"1. Review AGENTCORE_DEPLOYMENT_GUIDE.md")
    print(f"2. Use bedrock-agentcore-starter-toolkit for deployment")
    print(f"3. Test with the provided invocation examples")
    print("=" * 60)

if __name__ == "__main__":
    main()
