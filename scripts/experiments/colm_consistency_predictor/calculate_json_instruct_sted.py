#!/usr/bin/env python3
"""
Calculate STED metrics for json-instruct generation results.

Uses the SemanticJsonTreeConsistencyEvaluator to compute proper c_mean, d_std
metrics for generated JSON outputs.

Usage:
    python calculate_json_instruct_sted.py --results-dir llm_gen_results/json_instruct
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import re

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


def calculate_sted_metrics(responses: List[Any]) -> Dict[str, float]:
    """Calculate STED consistency metrics for a set of responses."""
    # Filter valid JSON responses
    valid_responses = [r for r in responses if isinstance(r, (dict, list))]

    if len(valid_responses) < 2:
        return {
            'c_mean': 0.0,
            'd_std': 1.0,
            'stability_score': 0.0,
            'validity_rate': len(valid_responses) / len(responses) if responses else 0,
            'n_valid': len(valid_responses)
        }

    # Initialize STED evaluator
    evaluator = SemanticJsonTreeConsistencyEvaluator()

    # Calculate pairwise distances using correct method
    distances = []
    for i in range(len(valid_responses)):
        for j in range(i + 1, len(valid_responses)):
            try:
                dist = evaluator.calculate_tree_edit_distance_fast(valid_responses[i], valid_responses[j])
                distances.append(dist)
            except Exception:
                distances.append(1.0)  # Max distance on error

    if not distances:
        return {
            'c_mean': 0.0,
            'd_std': 1.0,
            'stability_score': 0.0,
            'validity_rate': len(valid_responses) / len(responses) if responses else 0,
            'n_valid': len(valid_responses)
        }

    import numpy as np
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # c_mean = 1 - mean_distance (higher = more consistent)
    c_mean = 1.0 - mean_dist

    return {
        'c_mean': float(c_mean),
        'd_std': float(std_dist),
        'stability_score': float(c_mean * (1 - std_dist)),
        'validity_rate': len(valid_responses) / len(responses) if responses else 0,
        'n_valid': len(valid_responses)
    }


def process_results_dir(results_dir: Path) -> List[Dict]:
    """Process all results in a directory."""
    all_results = []

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir() or not result_dir.name.startswith("llm_gen_results_"):
            continue

        # Parse model and temperature from directory name
        # Format: llm_gen_results_Claude-3.5-Haiku_temp_0_70_20260211_095317
        parts = result_dir.name.split('_')
        model = parts[3] if len(parts) > 3 else "unknown"

        # Find temperature
        temp = 0.5
        for i, p in enumerate(parts):
            if p == 'temp' and i + 2 < len(parts):
                try:
                    temp = float(f"{parts[i+1]}.{parts[i+2]}")
                except ValueError:
                    pass
                break

        # Load all_results.json
        all_results_file = result_dir / "all_results.json"
        if not all_results_file.exists():
            continue

        with open(all_results_file) as f:
            data = json.load(f)

        print(f"\nProcessing {result_dir.name}...")
        print(f"  Model: {model}, Temperature: {temp}")

        for result in data.get('results', []):
            responses = result.get('responses', [])
            if not responses:
                continue

            # Extract sample_idx
            sample_id = result.get('sample_id', 'sample_000')
            idx_match = re.search(r'(\d+)', sample_id)
            sample_idx = int(idx_match.group(1)) if idx_match else 0

            # Calculate STED metrics
            metrics = calculate_sted_metrics(responses)

            all_results.append({
                'model': model,
                'sample_idx': sample_idx,
                'sample_id': sample_id,
                'temperature': temp,
                'c_mean': metrics['c_mean'],
                'd_std': metrics['d_std'],
                'stability_score': metrics['stability_score'],
                'validity_rate': metrics['validity_rate'],
                'n_valid': metrics['n_valid'],
                'n_total': len(responses),
                'dataset': 'json_instruct'
            })

        print(f"  Processed {len(data.get('results', []))} samples")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Calculate STED metrics for json-instruct")
    parser.add_argument('--results-dir', type=str, default='llm_gen_results/json_instruct',
                        help='Directory containing generation results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: <results-dir>/sted_metrics.json)')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return

    print("=" * 70)
    print("STED Metrics Calculation for json-instruct")
    print("=" * 70)

    all_results = process_results_dir(results_dir)

    if not all_results:
        print("\nNo results found to process")
        return

    # Save results
    output_path = args.output or (results_dir / "sted_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total samples processed: {len(all_results)}")

    # Summary stats
    import numpy as np
    c_means = [r['c_mean'] for r in all_results]
    print(f"Average c_mean: {np.mean(c_means):.3f} (+/- {np.std(c_means):.3f})")

    validity_rates = [r['validity_rate'] for r in all_results]
    print(f"Average validity rate: {np.mean(validity_rates):.1%}")

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
