"""Centralized model configuration for STED framework."""

# Model registry: model_id -> (provider, display_name, max_workers)
# max_workers is based on cross-region RPM quotas from AWS Bedrock
# Formula: max_workers ≈ RPM / 10 (assuming ~6 seconds avg per request)
MODEL_REGISTRY = {
    # Bedrock models - Claude 4.5 series
    # Claude Opus 4.5: 125 RPM -> 12 workers
    "us.anthropic.claude-opus-4-5-20251101-v1:0": ("bedrock", "Claude-Opus-4.5", 12),
    # Claude Sonnet 4.5: 200 RPM -> 20 workers
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": ("bedrock", "Claude-Sonnet-4.5", 20),
    # Claude Haiku 4.5: 125 RPM -> 12 workers
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": ("bedrock", "Claude-Haiku-4.5", 12),
    # Bedrock models - Claude 4 series
    # Claude Sonnet 4: 200 RPM -> 20 workers
    "us.anthropic.claude-sonnet-4-20250514-v1:0": ("bedrock", "Claude-Sonnet-4", 20),
    # Claude Opus 4: 200 RPM -> 20 workers
    "us.anthropic.claude-opus-4-20250514-v1:0": ("bedrock", "Claude-Opus-4", 20),
    # Bedrock models - Claude 3.7 series
    # Claude 3.7 Sonnet: 250 RPM -> 25 workers
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": ("bedrock", "Claude-3.7-Sonnet", 25),
    # Bedrock models - Claude 3.5 series
    # Claude 3.5 Sonnet V2: 100 RPM -> 10 workers
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": ("bedrock", "Claude-3.5-Sonnet", 10),
    # Claude 3.5 Haiku: 2000 RPM -> 40 workers (capped for stability)
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": ("bedrock", "Claude-3.5-Haiku", 40),
    # Bedrock models - Amazon Nova
    # Nova Pro: 500 RPM -> 50 workers
    "us.amazon.nova-pro-v1:0": ("bedrock", "Nova-Pro", 50),
    # Nova 2 Lite: 2000 RPM -> 50 workers (capped for stability)
    "us.amazon.nova-2-lite-v1:0": ("bedrock", "Nova-2-Lite", 50),
    # Bedrock models - Meta Llama
    # Llama 3.3 70B: 800 RPM -> 50 workers
    "us.meta.llama3-3-70b-instruct-v1:0": ("bedrock", "Llama-3.3-70B", 50),
    # Bedrock models - Other providers (no cross-region quota data, estimate conservatively)
    "minimax.minimax-m2": ("bedrock", "Minimax-M2", 20),
    "qwen.qwen3-235b-a22b-2507-v1:0": ("bedrock", "Qwen3-235B-A22B", 15),
    "qwen.qwen3-32b-v1:0": ("bedrock", "Qwen3-32B", 30),
    "deepseek.v3-v1:0": ("bedrock", "DeepSeek-V3.1", 20),
    "openai.gpt-oss-120b-1:0": ("bedrock", "GPT-OSS-120B", 10),
    # Mistral Large 3: 10000 RPM -> 100 workers (no cross-region support)
    "mistral.mistral-large-3-675b-instruct": ("bedrock", "Mistral-Large-3-675B", 10),
    "us.mistral.pixtral-large-2502-v1:0": ("bedrock", "Pixtral Large (25.02)", 10),
    # OpenAI-compatible models (via OpenRouter) - estimate based on tier
    "openai/gpt-4.1-mini": ("openai", "GPT-4.1-Mini", 30),
    "google/gemini-2.5-flash-lite": ("openai", "Gemini-2.5-Flash-Lite", 30),
    "x-ai/grok-4.1-fast": ("openai", "Grok-4.1-Fast", 20),
    "xiaomi/mimo-v2-flash:free": ("openai", "Mimo-V2-Flash:free", 10),
    "xiaomi/mimo-v2-flash": ("openai", "Mimo-V2-Flash", 10),
    "z-ai/glm-4.5-air:free": ("openai", "GLM-4.5-Air:free", 10),
    "nvidia/nemotron-3-nano-30b-a3b:free": ("openai", "NemoTron-3-Nano-30B-A3B:free", 10),
    "nvidia/nemotron-3-nano-30b-a3b": ("openai", "NemoTron-3-Nano", 30),  # Paid version
}


def get_provider(model_id: str) -> str:
    """Get provider for a model ID."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id][0]
    return "bedrock"  # Default


def get_display_name(model_id: str) -> str:
    """Get display name from model ID."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id][1]
    # Fallback: extract short name from model_id
    return model_id.split('/')[-1].split(':')[0]


def get_max_workers(model_id: str) -> int:
    """Get recommended max workers for a model based on rate limits."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id][2]
    return 20  # Default
