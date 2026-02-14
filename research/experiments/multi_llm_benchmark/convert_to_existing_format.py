#!/usr/bin/env python3
"""
Convert Multi-LLM Benchmark results to existing temperature experiment format.

This allows results to be analyzed with existing evaluation scripts.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def convert_results(input_file: str, output_dir: str):
    """Convert multi-LLM results to per-model files in existing format."""

    with open(input_file) as f:
        data = json.load(f)

    # Get all models from the results
    models = list(data.get("by_model", {}).keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        model_config = data["by_model"][model_name].get("model_config", {})

        # Build results in existing format
        converted_results = []

        for sample in data.get("by_sample", []):
            sample_id = sample["sample_id"]
            user_prompt = sample.get("user_prompt", "")

            model_data = sample.get("outputs_by_model", {}).get(model_name, {})
            outputs = model_data.get("outputs", [])

            # Filter out empty outputs
            valid_outputs = [o for o in outputs if o and len(o) > 0]

            if not valid_outputs:
                continue

            converted_sample = {
                "sample_id": sample_id,
                "user_prompt": user_prompt,
                "system_prompt": "",  # Not stored in multi-LLM format
                "ground_truth": valid_outputs[0] if valid_outputs else {},  # Use first as reference
                "ground_truth_schema": {},
                "responses": valid_outputs,
                "schemas": [],
                "metadata": {
                    "model_id": model_config.get("model_id", model_name),
                    "temperature": data["metadata"].get("temperature", 0.7),
                    "run_num": data["metadata"].get("runs_per_model", 5)
                }
            }
            converted_results.append(converted_sample)

        # Create output in existing format
        output_data = {
            "metadata": {
                "model_id": model_config.get("model_id", model_name),
                "model_name": model_name,
                "display_name": model_config.get("display_name", model_name),
                "temperature": data["metadata"].get("temperature", 0.7),
                "run_num": data["metadata"].get("runs_per_model", 5),
                "timestamp": datetime.now().isoformat(),
                "source": "multi_llm_benchmark"
            },
            "results": converted_results
        }

        # Save per-model file
        model_output_file = output_path / f"all_results_{model_name}.json"
        with open(model_output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved {len(converted_results)} samples for {model_name} to {model_output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert multi-LLM results to existing format")
    parser.add_argument("--input", type=str, required=True,
                       help="Input experiment_results.json file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for converted files")

    args = parser.parse_args()
    convert_results(args.input, args.output_dir)


if __name__ == "__main__":
    main()
