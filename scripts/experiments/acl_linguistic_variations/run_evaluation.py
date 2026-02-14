"""
ACL Paper: Linguistic Variation Evaluation Runner

Runs tool-calling experiments on linguistic variations and computes consistency metrics.

Usage:
    python run_evaluation.py --input data/acl_variations/linguistic_variations.json \
                             --output results/acl_linguistic/eval_results.json \
                             --model Claude-Sonnet-4 \
                             --temperatures 0.0 0.3 0.5 0.7 1.0 \
                             --runs 10
"""

import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sted.model_config import get_model_config, MODEL_CONFIGS
from sted.bedrock_utils import invoke_model_with_tools


@dataclass
class EvaluationResult:
    """Result of evaluating a single variation."""
    variation_id: str
    base_id: str
    variation_type: str
    variation_subtype: str
    linguistic_features: Dict[str, str]
    model: str
    temperature: float
    num_runs: int
    outputs: List[Dict]  # List of tool call outputs
    validity_rate: float
    consistency_metrics: Dict[str, float]  # c_mean, s_alpha, etc.


class ConsistencyCalculator:
    """Calculates consistency metrics for a set of outputs."""

    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha

    def compute_pairwise_similarity(self, outputs: List[Dict]) -> List[float]:
        """Compute pairwise Jaccard similarity of tool names."""
        similarities = []
        valid_outputs = [o for o in outputs if o.get('valid', False)]

        if len(valid_outputs) < 2:
            return []

        for i in range(len(valid_outputs)):
            for j in range(i + 1, len(valid_outputs)):
                sim = self._tool_similarity(valid_outputs[i], valid_outputs[j])
                similarities.append(sim)

        return similarities

    def _tool_similarity(self, o1: Dict, o2: Dict) -> float:
        """Compute similarity between two tool call outputs."""
        tools1 = set(tc.get('name', '') for tc in o1.get('tool_calls', []))
        tools2 = set(tc.get('name', '') for tc in o2.get('tool_calls', []))

        if not tools1 and not tools2:
            return 1.0
        if not tools1 or not tools2:
            return 0.0

        # Jaccard similarity
        intersection = len(tools1 & tools2)
        union = len(tools1 | tools2)
        return intersection / union if union > 0 else 0.0

    def compute_metrics(self, outputs: List[Dict]) -> Dict[str, float]:
        """Compute all consistency metrics."""
        valid_outputs = [o for o in outputs if o.get('valid', False)]
        validity_rate = len(valid_outputs) / len(outputs) if outputs else 0.0

        similarities = self.compute_pairwise_similarity(outputs)

        if not similarities:
            return {
                'validity_rate': validity_rate,
                'c_mean': 0.0,
                's_alpha': 0.0,
                'c_std': 0.0,
                'num_valid': len(valid_outputs)
            }

        import statistics
        c_mean = statistics.mean(similarities)
        c_std = statistics.stdev(similarities) if len(similarities) > 1 else 0.0

        # Stability score: S_alpha = (1 / (1 + 2*sigma_hat))^alpha
        sigma_max = 0.5  # Maximum std for values in [0,1]
        sigma_hat = c_std / sigma_max if sigma_max > 0 else 0.0
        s_alpha = (1 / (1 + 2 * sigma_hat)) ** self.alpha

        return {
            'validity_rate': validity_rate,
            'c_mean': c_mean,
            's_alpha': s_alpha,
            'c_std': c_std,
            'num_valid': len(valid_outputs)
        }


class LinguisticVariationEvaluator:
    """Evaluates linguistic variations using LLM tool calling."""

    def __init__(self, model_name: str, consistency_calculator: ConsistencyCalculator):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.calculator = consistency_calculator

    async def evaluate_variation(
        self,
        variation: Dict,
        temperature: float,
        num_runs: int
    ) -> EvaluationResult:
        """Evaluate a single variation."""
        outputs = []

        for run_idx in range(num_runs):
            try:
                # Prepare messages
                messages = [{"role": "user", "content": variation['varied_prompt']}]

                # Call model
                response = await asyncio.to_thread(
                    invoke_model_with_tools,
                    model_id=self.model_config['model_id'],
                    messages=messages,
                    tools=variation['tools'],
                    temperature=temperature,
                    region=self.model_config.get('region', 'us-east-1')
                )

                # Extract tool calls
                tool_calls = self._extract_tool_calls(response)
                outputs.append({
                    'run_idx': run_idx,
                    'valid': True,
                    'tool_calls': tool_calls,
                    'raw_response': response
                })

            except Exception as e:
                outputs.append({
                    'run_idx': run_idx,
                    'valid': False,
                    'error': str(e),
                    'tool_calls': []
                })

        # Compute metrics
        metrics = self.calculator.compute_metrics(outputs)

        return EvaluationResult(
            variation_id=variation['variation_id'],
            base_id=variation['base_id'],
            variation_type=variation['variation_type'],
            variation_subtype=variation['variation_subtype'],
            linguistic_features=variation['linguistic_features'],
            model=self.model_name,
            temperature=temperature,
            num_runs=num_runs,
            outputs=outputs,
            validity_rate=metrics['validity_rate'],
            consistency_metrics=metrics
        )

    def _extract_tool_calls(self, response: Dict) -> List[Dict]:
        """Extract tool calls from model response."""
        tool_calls = []

        # Handle different response formats
        if 'tool_calls' in response:
            tool_calls = response['tool_calls']
        elif 'content' in response:
            content = response['content']
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'tool_use':
                        tool_calls.append({
                            'name': item.get('name', ''),
                            'arguments': item.get('input', {})
                        })

        return tool_calls


class EvaluationRunner:
    """Runs full evaluation pipeline."""

    def __init__(
        self,
        model_name: str,
        temperatures: List[float],
        num_runs: int,
        output_dir: Path
    ):
        self.model_name = model_name
        self.temperatures = temperatures
        self.num_runs = num_runs
        self.output_dir = output_dir
        self.calculator = ConsistencyCalculator(alpha=20.0)
        self.evaluator = LinguisticVariationEvaluator(model_name, self.calculator)

    async def run(self, variations: List[Dict]) -> List[EvaluationResult]:
        """Run evaluation on all variations."""
        results = []
        total = len(variations) * len(self.temperatures)
        completed = 0

        for variation in variations:
            for temp in self.temperatures:
                print(f"\rProgress: {completed}/{total} ({100*completed/total:.1f}%)", end='')

                result = await self.evaluator.evaluate_variation(
                    variation,
                    temperature=temp,
                    num_runs=self.num_runs
                )
                results.append(result)
                completed += 1

        print(f"\rProgress: {total}/{total} (100.0%)")
        return results

    def save_results(self, results: List[EvaluationResult], output_path: Path):
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'metadata': {
                'model': self.model_name,
                'temperatures': self.temperatures,
                'num_runs': self.num_runs,
                'timestamp': datetime.now().isoformat(),
                'num_variations': len(set(r.variation_id for r in results)),
                'num_results': len(results)
            },
            'results': [asdict(r) for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"Saved {len(results)} results to {output_path}")

    def generate_summary(self, results: List[EvaluationResult]) -> Dict:
        """Generate summary statistics."""
        summary = defaultdict(lambda: defaultdict(list))

        for r in results:
            key = (r.variation_type, r.variation_subtype, r.temperature)
            summary[key]['c_mean'].append(r.consistency_metrics['c_mean'])
            summary[key]['s_alpha'].append(r.consistency_metrics['s_alpha'])
            summary[key]['validity'].append(r.validity_rate)

        # Aggregate
        import statistics
        aggregated = {}
        for key, metrics in summary.items():
            vtype, subtype, temp = key
            aggregated[f"{vtype}_{subtype}_T{temp}"] = {
                'c_mean': statistics.mean(metrics['c_mean']) if metrics['c_mean'] else 0,
                's_alpha': statistics.mean(metrics['s_alpha']) if metrics['s_alpha'] else 0,
                'validity': statistics.mean(metrics['validity']) if metrics['validity'] else 0,
                'n': len(metrics['c_mean'])
            }

        return aggregated


async def main():
    parser = argparse.ArgumentParser(description='Run linguistic variation evaluation')
    parser.add_argument('--input', type=str,
                        default='data/acl_variations/linguistic_variations.json',
                        help='Input variations file')
    parser.add_argument('--output', type=str,
                        default='results/acl_linguistic/eval_results.json',
                        help='Output results file')
    parser.add_argument('--model', type=str, default='Claude-Sonnet-4',
                        help='Model to evaluate')
    parser.add_argument('--temperatures', type=float, nargs='+',
                        default=[0.0, 0.3, 0.5, 0.7, 1.0],
                        help='Temperatures to test')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs per condition')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of variations to evaluate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without running')

    args = parser.parse_args()

    # Load variations
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    with open(input_path) as f:
        variations = json.load(f)

    if args.limit:
        variations = variations[:args.limit]

    print(f"Loaded {len(variations)} variations from {input_path}")
    print(f"Model: {args.model}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Runs per condition: {args.runs}")

    total_calls = len(variations) * len(args.temperatures) * args.runs
    print(f"Total API calls: {total_calls}")

    if args.dry_run:
        print("\n[DRY RUN] Would run evaluation but --dry-run specified")
        return

    # Run evaluation
    runner = EvaluationRunner(
        model_name=args.model,
        temperatures=args.temperatures,
        num_runs=args.runs,
        output_dir=Path(args.output).parent
    )

    results = await runner.run(variations)

    # Save results
    runner.save_results(results, Path(args.output))

    # Print summary
    summary = runner.generate_summary(results)
    print("\nSummary by variation type and temperature:")
    for key, stats in sorted(summary.items())[:20]:
        print(f"  {key}: c_mean={stats['c_mean']:.3f}, s_alpha={stats['s_alpha']:.3f}, validity={stats['validity']:.3f}")


if __name__ == '__main__':
    asyncio.run(main())
