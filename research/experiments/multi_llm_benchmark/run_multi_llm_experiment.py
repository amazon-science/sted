#!/usr/bin/env python3
"""
Multi-LLM Benchmark for JSON Consistency Evaluation

This script generates JSON outputs from multiple LLMs on function calling datasets
to evaluate cross-model and within-model consistency using STED.

Usage:
    python run_multi_llm_experiment.py --sample-size 100 --runs-per-model 5
"""

import argparse
import json
import os
import sys
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import boto3
from dotenv import load_dotenv
load_dotenv()

# Try to import openai
try:
    import openai
    openai_client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "not-set"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None

# Import STED utilities
from sted.bedrock_utils import build_message, inference_with_converse_api

# ============================================================================
# Model Configuration
# ============================================================================

MODELS = {
    # Bedrock models
    "claude-sonnet": {
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "provider": "bedrock",
        "display_name": "Claude 3.5 Sonnet"
    },
    "claude-haiku": {
        "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "provider": "bedrock",
        "display_name": "Claude 3.5 Haiku"
    },
    "llama-70b": {
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0",
        "provider": "bedrock",
        "display_name": "Llama 3.3 70B"
    },
    "nova-pro": {
        "model_id": "us.amazon.nova-pro-v1:0",
        "provider": "bedrock",
        "display_name": "Nova Pro"
    },
    # OpenAI-compatible models (via OpenRouter or direct API)
    "gpt-4o": {
        "model_id": "openai/gpt-4o",
        "provider": "openai",
        "display_name": "GPT-4o"
    },
    "gemini-pro": {
        "model_id": "google/gemini-2.0-flash-001",
        "provider": "openai",
        "display_name": "Gemini 2.0 Flash"
    },
}

# ============================================================================
# Dataset Loading
# ============================================================================

DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"


def load_glaive_dataset(sample_size: int = 100) -> List[Dict]:
    """Load Glaive function calling dataset."""
    path = DATASETS_DIR / "glaiveai_glaive-function-calling-v2.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Convert to standardized format for generation
    samples = []
    for i, item in enumerate(data[:sample_size]):
        # Parse the system prompt to extract function definitions
        system_prompt = item.get("system", "")
        chat = item.get("chat", "")

        # Extract user message from chat
        user_msg = ""
        if "USER:" in chat:
            parts = chat.split("USER:")
            if len(parts) > 1:
                user_part = parts[1].split("ASSISTANT:")[0].strip()
                user_msg = user_part

        if not user_msg:
            continue

        samples.append({
            "id": f"glaive_{i:04d}",
            "source": "glaive_function_calling",
            "system_prompt": system_prompt,
            "user_prompt": user_msg,
            "full_chat": chat,
            "raw": item
        })

    return samples


def load_json_instruct_dataset(sample_size: int = 100) -> List[Dict]:
    """Load JSON instruction generation dataset."""
    path = DATASETS_DIR / "Maxscha_json-instruct-generation-large.json"
    if not path.exists():
        path = DATASETS_DIR / "Maxscha_json-instruct-generation.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found")

    with open(path) as f:
        data = json.load(f)

    samples = []
    for i, item in enumerate(data[:sample_size]):
        # Extract instruction and expected output
        instruction = item.get("instruction", item.get("input", ""))
        output = item.get("output", item.get("response", ""))

        if not instruction:
            continue

        # Try to parse output as JSON for ground truth
        try:
            ground_truth = json.loads(output) if isinstance(output, str) else output
        except:
            ground_truth = {"raw": output}

        samples.append({
            "id": f"json_instruct_{i:04d}",
            "source": "json_instruct",
            "system_prompt": "You are a helpful assistant that generates valid JSON responses.",
            "user_prompt": instruction,
            "ground_truth": ground_truth,
            "raw": item
        })

    return samples


def load_all_datasets(sample_size: int = 100) -> Dict[str, List[Dict]]:
    """Load all available datasets."""
    datasets = {}

    try:
        datasets["glaive"] = load_glaive_dataset(sample_size)
        print(f"Loaded glaive: {len(datasets['glaive'])} samples")
    except FileNotFoundError as e:
        print(f"Skipping glaive: {e}")

    try:
        datasets["json_instruct"] = load_json_instruct_dataset(sample_size)
        print(f"Loaded json_instruct: {len(datasets['json_instruct'])} samples")
    except FileNotFoundError as e:
        print(f"Skipping json_instruct: {e}")

    return datasets

# ============================================================================
# LLM Inference
# ============================================================================

def extract_json_from_response(response_text: str) -> Dict:
    """Extract JSON from LLM response."""
    import re

    # Try to find JSON in the response
    # Look for code blocks first
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except:
            pass

    # Try to find raw JSON
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass

    # Return empty dict if no JSON found
    return {}


def call_bedrock(model_id: str, system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: int = 2000) -> Dict:
    """Call AWS Bedrock model."""
    client = boto3.client(
        'bedrock-runtime',
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

    message = build_message(texts=[user_prompt])
    response = inference_with_converse_api(
        client,
        model_id=model_id,
        messages=[message],
        system_prompts=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9
    )

    if not response or not isinstance(response, list) or len(response) == 0:
        return {}

    response_text = response[0].get('text', '{}')
    return extract_json_from_response(response_text)


def call_openai(model_id: str, system_prompt: str, user_prompt: str,
                temperature: float = 0.7, max_tokens: int = 2000) -> Dict:
    """Call OpenAI-compatible model."""
    if not OPENAI_AVAILABLE or openai_client is None:
        raise RuntimeError("OpenAI client not available")

    response = openai_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    response_text = response.choices[0].message.content
    return extract_json_from_response(response_text)


def generate_output(model_config: Dict, system_prompt: str, user_prompt: str,
                    temperature: float = 0.7, max_tokens: int = 2000) -> Dict:
    """Generate output from a model."""
    provider = model_config["provider"]
    model_id = model_config["model_id"]

    try:
        if provider == "bedrock":
            return call_bedrock(model_id, system_prompt, user_prompt, temperature, max_tokens)
        elif provider == "openai":
            return call_openai(model_id, system_prompt, user_prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        print(f"Error generating output: {e}")
        return {}


def generate_multiple_outputs(model_config: Dict, system_prompt: str, user_prompt: str,
                              num_runs: int = 5, temperature: float = 0.7,
                              max_tokens: int = 2000) -> List[Dict]:
    """Generate multiple outputs from a model for consistency evaluation."""
    outputs = []

    for _ in range(num_runs):
        output = generate_output(model_config, system_prompt, user_prompt, temperature, max_tokens)
        outputs.append(output)
        time.sleep(0.2)  # Rate limiting

    return outputs

# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    models: List[str],
    datasets: Dict[str, List[Dict]],
    output_dir: Path,
    runs_per_model: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> Dict:
    """Run the multi-LLM benchmark experiment."""

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "runs_per_model": runs_per_model,
            "temperature": temperature,
        },
        "by_model": {},
        "by_sample": []
    }

    # Process each dataset
    for dataset_name, samples in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name} ({len(samples)} samples)")
        print(f"{'='*60}")

        for sample in tqdm(samples, desc=f"{dataset_name}"):
            sample_id = sample["id"]
            system_prompt = sample.get("system_prompt", "You are a helpful assistant.")
            user_prompt = sample.get("user_prompt", "")

            if not user_prompt:
                continue

            sample_result = {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "user_prompt": user_prompt[:500],  # Truncate for storage
                "outputs_by_model": {}
            }

            # Generate outputs from each model
            for model_name in models:
                if model_name not in MODELS:
                    print(f"Unknown model: {model_name}")
                    continue

                model_config = MODELS[model_name]

                try:
                    outputs = generate_multiple_outputs(
                        model_config,
                        system_prompt,
                        user_prompt,
                        num_runs=runs_per_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    sample_result["outputs_by_model"][model_name] = {
                        "model_id": model_config["model_id"],
                        "display_name": model_config["display_name"],
                        "outputs": outputs,
                        "valid_count": sum(1 for o in outputs if o)
                    }

                    # Track by model
                    if model_name not in results["by_model"]:
                        results["by_model"][model_name] = {
                            "model_config": model_config,
                            "total_samples": 0,
                            "total_valid": 0
                        }
                    results["by_model"][model_name]["total_samples"] += runs_per_model
                    results["by_model"][model_name]["total_valid"] += sum(1 for o in outputs if o)

                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    sample_result["outputs_by_model"][model_name] = {
                        "error": str(e),
                        "outputs": []
                    }

            results["by_sample"].append(sample_result)

            # Save intermediate results
            intermediate_path = output_dir / f"intermediate_{dataset_name}.json"
            with open(intermediate_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run multi-LLM benchmark for JSON consistency")

    parser.add_argument("--models", nargs="+",
                       default=["claude-sonnet", "claude-haiku"],
                       help="Models to evaluate")
    parser.add_argument("--sample-size", type=int, default=50,
                       help="Number of samples per dataset")
    parser.add_argument("--runs-per-model", type=int, default=5,
                       help="Number of runs per sample per model")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=2000,
                       help="Max tokens per generation")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=["glaive", "json_instruct"],
                       help="Datasets to use")

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results" / f"run_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Multi-LLM Benchmark for JSON Consistency")
    print("="*60)
    print(f"Models: {args.models}")
    print(f"Sample size: {args.sample_size}")
    print(f"Runs per model: {args.runs_per_model}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Load datasets
    print("\nLoading datasets...")
    all_datasets = load_all_datasets(args.sample_size)

    # Filter to requested datasets
    datasets = {k: v for k, v in all_datasets.items() if k in args.datasets}

    if not datasets:
        print("No datasets loaded. Run download_datasets.py first.")
        return

    # Run experiment
    results = run_experiment(
        models=args.models,
        datasets=datasets,
        output_dir=output_dir,
        runs_per_model=args.runs_per_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Save final results
    results_path = output_dir / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"\nSummary:")
    for model_name, model_data in results["by_model"].items():
        valid_rate = model_data["total_valid"] / max(model_data["total_samples"], 1) * 100
        print(f"  {model_name}: {model_data['total_valid']}/{model_data['total_samples']} valid ({valid_rate:.1f}%)")

    print(f"\nNext: Run evaluate_consistency.py to compute STED scores")


if __name__ == "__main__":
    main()
