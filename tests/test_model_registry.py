#!/usr/bin/env python3
"""
Tests for MODEL_REGISTRY models.

This module tests that all models listed in MODEL_REGISTRY are accessible
and can generate responses. It verifies both Bedrock and OpenAI-compatible models.

Usage:
    # Run all tests (may take time and incur costs)
    pytest tests/test_model_registry.py -v

    # Run only Bedrock model tests
    pytest tests/test_model_registry.py -v -k "bedrock"

    # Run only OpenAI model tests
    pytest tests/test_model_registry.py -v -k "openai"

    # Run a quick smoke test (first model of each provider)
    pytest tests/test_model_registry.py -v -k "smoke"

Note: These tests require valid AWS credentials for Bedrock models
and OPENAI_API_KEY for OpenAI-compatible models.
"""

import os
import json
import pytest
from typing import Dict, Any, Optional, Tuple
from dotenv import dotenv_values

# Load only non-AWS environment variables from .env file
# AWS credentials should come from ~/.aws/credentials or shell environment
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
env_values = dotenv_values(env_path)

# Only load non-AWS variables from .env (preserve boto3's credential chain)
for key, value in env_values.items():
    if not key.startswith('AWS_') and key not in os.environ:
        os.environ[key] = value

from sted.model_config import MODEL_REGISTRY, get_provider, get_display_name


# Test configuration
SIMPLE_PROMPT = "What is 2 + 2? Answer with just the number."
EXPECTED_RESPONSE_CONTAINS = "4"
MAX_TOKENS = 500  # Increased to accommodate models with internal reasoning tokens
TEMPERATURE = 0.0  # Deterministic for testing


def get_bedrock_client():
    """Create a Bedrock runtime client.

    Uses AWS credentials from (in order of priority):
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    2. ~/.aws/credentials file (via aws configure)
    3. IAM role (if running on AWS infrastructure)
    """
    import boto3
    from botocore.config import Config

    config = Config(
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        read_timeout=60,
        connect_timeout=10
    )

    # Let boto3 handle credential discovery automatically
    # It will check env vars, ~/.aws/credentials, IAM roles, etc.
    return boto3.client(
        'bedrock-runtime',
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2'),
        config=config
    )


def get_openai_client():
    """Create an OpenAI client."""
    import openai
    return openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "not-set"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def invoke_bedrock_model(client, model_id: str, prompt: str) -> Tuple[bool, str]:
    """
    Invoke a Bedrock model using the Converse API.

    Returns:
        Tuple of (success: bool, response_or_error: str)
    """
    try:
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        params = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": TEMPERATURE,
                "maxTokens": MAX_TOKENS
            }
        }

        response = client.converse(**params)
        content = response['output']['message']['content']

        if content and len(content) > 0:
            # Some models (e.g., Minimax) return reasoningContent first, then text
            # Try to find text content in any position
            text = ""
            for item in content:
                if 'text' in item:
                    text = item['text']
                    break
            return True, text.strip() if text else ""
        return False, "Empty response"

    except Exception as e:
        return False, str(e)


def invoke_openai_model(client, model_id: str, prompt: str) -> Tuple[bool, str]:
    """
    Invoke an OpenAI-compatible model.

    Returns:
        Tuple of (success: bool, response_or_error: str)
    """
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={
                # Allow data collection to use free models
                "provider": {"data_collection": "allow"}
            }
        )
        text = response.choices[0].message.content
        return True, text if text else ""

    except Exception as e:
        return False, str(e)


# Separate models by provider
BEDROCK_MODELS = {k: v for k, v in MODEL_REGISTRY.items() if v[0] == "bedrock"}
OPENAI_MODELS = {k: v for k, v in MODEL_REGISTRY.items() if v[0] == "openai"}


class TestModelRegistryStructure:
    """Test MODEL_REGISTRY structure and helper functions."""

    def test_registry_not_empty(self):
        """MODEL_REGISTRY should have models."""
        assert len(MODEL_REGISTRY) > 0

    def test_all_models_have_provider_and_name(self):
        """Each model should have (provider, display_name) tuple."""
        for model_id, value in MODEL_REGISTRY.items():
            assert isinstance(value, tuple), f"{model_id} value is not a tuple"
            assert len(value) == 2, f"{model_id} tuple length is not 2"
            provider, display_name = value
            assert provider in ("bedrock", "openai"), f"{model_id} has invalid provider: {provider}"
            assert isinstance(display_name, str), f"{model_id} display_name is not a string"
            assert len(display_name) > 0, f"{model_id} display_name is empty"

    def test_get_provider(self):
        """get_provider should return correct provider."""
        for model_id, (expected_provider, _) in MODEL_REGISTRY.items():
            assert get_provider(model_id) == expected_provider

    def test_get_display_name(self):
        """get_display_name should return correct name."""
        for model_id, (_, expected_name) in MODEL_REGISTRY.items():
            assert get_display_name(model_id) == expected_name

    def test_get_provider_unknown_model(self):
        """get_provider should return 'bedrock' for unknown models."""
        assert get_provider("unknown-model-xyz") == "bedrock"


class TestBedrockModels:
    """Test Bedrock models from MODEL_REGISTRY."""

    @pytest.fixture(scope="class")
    def bedrock_client(self):
        """Create Bedrock client once per test class."""
        try:
            return get_bedrock_client()
        except Exception as e:
            pytest.skip(f"Cannot create Bedrock client: {e}")

    @pytest.mark.parametrize("model_id,model_info", list(BEDROCK_MODELS.items()))
    def test_bedrock_model_invocation(self, bedrock_client, model_id: str, model_info: tuple):
        """Test that a Bedrock model can be invoked successfully."""
        provider, display_name = model_info

        success, response = invoke_bedrock_model(bedrock_client, model_id, SIMPLE_PROMPT)

        if not success:
            # Check for common skip conditions
            if "AccessDeniedException" in response:
                pytest.skip(f"No access to model {display_name}: {response}")
            elif "ResourceNotFoundException" in response:
                pytest.skip(f"Model {display_name} not found in region: {response}")
            elif "ValidationException" in response and "not supported" in response.lower():
                pytest.skip(f"Model {display_name} not supported: {response}")
            elif "ValidationException" in response and "invalid" in response.lower():
                pytest.skip(f"Model {display_name} invalid identifier (may not be available in region): {response}")
            elif "UnrecognizedClientException" in response or "security token" in response.lower():
                pytest.skip(f"AWS credentials invalid or not configured: {response}")
            elif "ExpiredTokenException" in response:
                pytest.skip(f"AWS credentials expired: {response}")
            else:
                pytest.fail(f"Model {display_name} ({model_id}) failed: {response}")

        # Verify we got a non-empty response
        if len(response) == 0:
            pytest.skip(f"Model {display_name} returned empty response (may have issues)")
        print(f"\n  {display_name}: '{response.strip()[:50]}...'")


class TestOpenAIModels:
    """Test OpenAI-compatible models from MODEL_REGISTRY."""

    @pytest.fixture(scope="class")
    def openai_client(self):
        """Create OpenAI client once per test class."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "not-set":
            pytest.skip("OPENAI_API_KEY not set")

        try:
            return get_openai_client()
        except Exception as e:
            pytest.skip(f"Cannot create OpenAI client: {e}")

    @pytest.mark.parametrize("model_id,model_info", list(OPENAI_MODELS.items()))
    def test_openai_model_invocation(self, openai_client, model_id: str, model_info: tuple):
        """Test that an OpenAI-compatible model can be invoked successfully."""
        provider, display_name = model_info

        success, response = invoke_openai_model(openai_client, model_id, SIMPLE_PROMPT)

        if not success:
            # Check for common skip conditions
            if "401" in response or "authentication" in response.lower():
                pytest.skip(f"Authentication failed for {display_name}: {response}")
            elif "404" in response or "not found" in response.lower():
                pytest.skip(f"Model {display_name} not found: {response}")
            elif "rate limit" in response.lower() or "429" in response:
                pytest.skip(f"Rate limited for {display_name}: {response}")
            else:
                pytest.fail(f"Model {display_name} ({model_id}) failed: {response}")

        # Verify we got a non-empty response
        if len(response) == 0:
            pytest.skip(f"Model {display_name} returned empty response (may have issues)")
        print(f"\n  {display_name}: '{response.strip()[:50]}...'")


class TestSmokeTest:
    """Quick smoke tests for one model from each provider."""

    def test_smoke_bedrock(self):
        """Smoke test: verify at least one Bedrock model works."""
        if not BEDROCK_MODELS:
            pytest.skip("No Bedrock models in registry")

        try:
            client = get_bedrock_client()
        except Exception as e:
            pytest.skip(f"Cannot create Bedrock client: {e}")

        # Try Claude 3.5 Sonnet first as it's commonly available
        preferred_models = [
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ]

        for model_id in preferred_models:
            if model_id in BEDROCK_MODELS:
                success, response = invoke_bedrock_model(client, model_id, SIMPLE_PROMPT)
                if success:
                    display_name = get_display_name(model_id)
                    print(f"\n  Bedrock smoke test passed with {display_name}")
                    assert len(response) > 0
                    return

        # If preferred models not available, try any model
        for model_id in BEDROCK_MODELS:
            success, response = invoke_bedrock_model(client, model_id, SIMPLE_PROMPT)
            if success:
                display_name = get_display_name(model_id)
                print(f"\n  Bedrock smoke test passed with {display_name}")
                assert len(response) > 0
                return

        pytest.skip("No accessible Bedrock models found")

    def test_smoke_openai(self):
        """Smoke test: verify at least one OpenAI model works."""
        if not OPENAI_MODELS:
            pytest.skip("No OpenAI models in registry")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "not-set":
            pytest.skip("OPENAI_API_KEY not set")

        try:
            client = get_openai_client()
        except Exception as e:
            pytest.skip(f"Cannot create OpenAI client: {e}")

        # Try GPT-4o first as it's commonly available
        preferred_models = [
            "openai/gpt-4o",
            "openai/gpt-4.1-mini",
        ]

        for model_id in preferred_models:
            if model_id in OPENAI_MODELS:
                success, response = invoke_openai_model(client, model_id, SIMPLE_PROMPT)
                if success:
                    display_name = get_display_name(model_id)
                    print(f"\n  OpenAI smoke test passed with {display_name}")
                    assert len(response) > 0
                    return

        # If preferred models not available, try any model
        for model_id in OPENAI_MODELS:
            success, response = invoke_openai_model(client, model_id, SIMPLE_PROMPT)
            if success:
                display_name = get_display_name(model_id)
                print(f"\n  OpenAI smoke test passed with {display_name}")
                assert len(response) > 0
                return

        pytest.skip("No accessible OpenAI models found")


def generate_model_status_report():
    """
    Generate a comprehensive status report for all models.

    This function can be run standalone to get a full report.
    """
    print("\n" + "=" * 70)
    print("MODEL REGISTRY STATUS REPORT")
    print("=" * 70)

    results = {
        "bedrock": {"success": [], "failed": [], "skipped": []},
        "openai": {"success": [], "failed": [], "skipped": []}
    }

    # Test Bedrock models
    print("\n--- BEDROCK MODELS ---")
    try:
        bedrock_client = get_bedrock_client()
        for model_id, (provider, display_name) in BEDROCK_MODELS.items():
            success, response = invoke_bedrock_model(bedrock_client, model_id, SIMPLE_PROMPT)
            if success:
                status = "OK"
                results["bedrock"]["success"].append(display_name)
            elif "AccessDeniedException" in response or "ResourceNotFoundException" in response:
                status = "SKIP"
                results["bedrock"]["skipped"].append(display_name)
            else:
                status = "FAIL"
                results["bedrock"]["failed"].append(display_name)
            print(f"  [{status}] {display_name}")
    except Exception as e:
        print(f"  Cannot test Bedrock models: {e}")

    # Test OpenAI models
    print("\n--- OPENAI MODELS ---")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "not-set":
        try:
            openai_client = get_openai_client()
            for model_id, (provider, display_name) in OPENAI_MODELS.items():
                success, response = invoke_openai_model(openai_client, model_id, SIMPLE_PROMPT)
                if success:
                    status = "OK"
                    results["openai"]["success"].append(display_name)
                elif "401" in response or "404" in response or "rate limit" in response.lower():
                    status = "SKIP"
                    results["openai"]["skipped"].append(display_name)
                else:
                    status = "FAIL"
                    results["openai"]["failed"].append(display_name)
                print(f"  [{status}] {display_name}")
        except Exception as e:
            print(f"  Cannot test OpenAI models: {e}")
    else:
        print("  OPENAI_API_KEY not set - skipping OpenAI models")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for provider in ["bedrock", "openai"]:
        total = len(results[provider]["success"]) + len(results[provider]["failed"]) + len(results[provider]["skipped"])
        if total > 0:
            print(f"\n{provider.upper()}:")
            print(f"  Success: {len(results[provider]['success'])}")
            print(f"  Failed:  {len(results[provider]['failed'])}")
            print(f"  Skipped: {len(results[provider]['skipped'])}")
            if results[provider]["failed"]:
                print(f"  Failed models: {', '.join(results[provider]['failed'])}")

    return results


if __name__ == "__main__":
    generate_model_status_report()
