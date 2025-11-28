"""Centralized model configuration for STED framework."""

# Model registry: model_id -> (provider, display_name)
MODEL_REGISTRY = {
    # Bedrock models
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": ("bedrock", "Claude-3.7-Sonnet"),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": ("bedrock", "Claude-Sonnet-4"),
    "us.anthropic.claude-3-haiku-20240307-v1:0": ("bedrock", "Claude-3-Haiku"),
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": ("bedrock", "Claude-3.5-Haiku"),
    "anthropic.claude-3-haiku-20240307-v1:0": ("bedrock", "Claude-3-Haiku"),
    "anthropic.claude-3-5-haiku-20241022-v1:0": ("bedrock", "Claude-3.5-Haiku"),
    "us.meta.llama3-3-70b-instruct-v1:0": ("bedrock", "Llama-3.3-70B"),
    "us.amazon.nova-pro-v1:0": ("bedrock", "Nova-Pro"),
    "us.qwen.qwen3-235b-a22b-2507-v1:0": ("bedrock", "Qwen3-235B-A22B"),
    "us.qwen.qwen3-32b-v1:0": ("bedrock", "Qwen3-32B"),
    "us.deepseek.v3-v1:0": ("bedrock", "DeepSeek-V3.1"),
    # OpenAI-compatible models
    "openai/gpt-5": ("openai", "GPT-5"),
    "openai/gpt-4o": ("openai", "GPT-4o"),
    "openai/gpt-4.1-mini": ("openai", "GPT-4.1-Mini"),
    "google/gemini-2.5-pro": ("openai", "Gemini-2.5-Pro"),
    "google/gemini-2.5-flash-lite": ("openai", "Gemini-2.5-Flash-Lite"),
    "x-ai/grok-4": ("openai", "Grok-4"),
}


def get_provider(model_id: str) -> str:
    """Get provider for a model ID."""
    return MODEL_REGISTRY.get(model_id, ("bedrock", None))[0]


def get_display_name(model_id: str) -> str:
    """Get display name from model ID."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id][1]
    # Fallback: extract short name from model_id
    return model_id.split('/')[-1].split(':')[0]
