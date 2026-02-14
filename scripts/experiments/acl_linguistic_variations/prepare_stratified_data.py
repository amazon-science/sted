#!/usr/bin/env python3
"""
ACL 2026 Paper: Stratified Sample Data Preparation

Prepares sample data stratified by schema complexity (from KDD findings):
- Simple: depth <= 2, <= 5 params
- Medium: depth 3-4, 6-10 params
- Complex: depth >= 5, >10 params

This addresses KDD's finding that schema complexity is the #1 factor (19% SHAP importance).

Statistical Power Considerations:
- Phase 2 ceiling effect: Need ~30 samples per difficulty stratum per variation
- Phase 3 interaction: Need ~15 samples per cell (variation × complexity)
- Minimum recommended: 50 samples per complexity stratum

Usage:
    python prepare_stratified_data.py --output data/acl_stratified/stratified_samples.json
    python prepare_stratified_data.py --use-all  # Use all available samples
    python prepare_stratified_data.py --relaxed-filter  # Include multi-tool samples
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any
import math


def compute_schema_complexity(tools: List[Dict]) -> Dict[str, Any]:
    """
    Compute schema complexity metrics for a tool set.

    Returns dict with:
    - depth: maximum nesting depth
    - breadth: total number of parameters
    - complexity_score: combined metric
    - complexity_level: 'simple', 'medium', 'complex'
    """
    if not tools:
        return {
            'depth': 0,
            'breadth': 0,
            'complexity_score': 0,
            'complexity_level': 'simple',
            'num_tools': 0,
            'total_params': 0,
            'max_params_per_tool': 0
        }

    def get_depth(obj, current_depth=0):
        """Recursively compute maximum depth of nested schema."""
        if not isinstance(obj, dict):
            return current_depth

        max_child_depth = current_depth

        if 'properties' in obj:
            for prop_name, prop_value in obj['properties'].items():
                if isinstance(prop_value, dict):
                    child_depth = get_depth(prop_value, current_depth + 1)
                    max_child_depth = max(max_child_depth, child_depth)

        if 'items' in obj:
            child_depth = get_depth(obj['items'], current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    def count_params(obj):
        """Count total parameters in schema."""
        if not isinstance(obj, dict):
            return 0

        count = 0
        if 'properties' in obj:
            count += len(obj['properties'])
            for prop_value in obj['properties'].values():
                count += count_params(prop_value)

        if 'items' in obj:
            count += count_params(obj['items'])

        return count

    # Analyze each tool
    max_depth = 0
    total_params = 0
    max_params_per_tool = 0

    for tool in tools:
        # Handle both OpenAI format and xLAM format
        if 'function' in tool:
            params = tool['function'].get('parameters', {})
        else:
            params = tool.get('parameters', {})

        tool_depth = get_depth(params)
        tool_params = count_params(params)

        max_depth = max(max_depth, tool_depth)
        total_params += tool_params
        max_params_per_tool = max(max_params_per_tool, tool_params)

    # Compute complexity score (normalized)
    complexity_score = max_depth * 2 + total_params * 0.5 + len(tools) * 0.3

    # Determine complexity level (following KDD paper criteria)
    if max_depth <= 2 and total_params <= 5:
        level = 'simple'
    elif max_depth >= 5 or total_params > 10:
        level = 'complex'
    else:
        level = 'medium'

    return {
        'depth': max_depth,
        'breadth': total_params,
        'complexity_score': complexity_score,
        'complexity_level': level,
        'num_tools': len(tools),
        'total_params': total_params,
        'max_params_per_tool': max_params_per_tool
    }


def filter_suitable_prompts(data: List[Dict], relaxed: bool = False) -> List[Dict]:
    """
    Filter prompts suitable for ACL linguistic experiments.

    Args:
        data: Raw dataset
        relaxed: If True, allow multi-tool samples to increase pool size
    """
    suitable = []

    for item in data:
        question = item.get('question', '')

        # Must have question
        if not question:
            continue

        # English only (ASCII)
        if not question.isascii():
            continue

        # Tool call constraint
        if relaxed:
            # Relaxed: allow 1-3 tool calls
            num_calls = item.get('num_tool_calls', 0)
            if num_calls < 1 or num_calls > 3:
                continue
        else:
            # Strict: single tool call for cleaner experiments
            if item.get('num_tool_calls', 0) != 1:
                continue

        # Reasonable length (relaxed allows slightly longer)
        min_len = 20 if relaxed else 30
        max_len = 600 if relaxed else 500
        if len(question) < min_len or len(question) > max_len:
            continue

        # Must have tools defined
        if not item.get('tools'):
            continue

        suitable.append(item)

    return suitable


def compute_power_analysis(n_samples: int, n_variations: int = 8,
                           n_complexity: int = 3, n_difficulty: int = 3) -> Dict:
    """
    Compute statistical power estimates for the experimental design.

    Returns recommendations and warnings about sample sizes.
    """
    # Phase 2: Ceiling effect analysis
    # Need samples per (variation × difficulty) cell
    # Assuming roughly equal difficulty distribution
    samples_per_difficulty = n_samples / n_difficulty
    phase2_samples_per_cell = samples_per_difficulty  # Each variation tested on same samples

    # Phase 3: Interaction analysis
    # Need samples per (variation × complexity) cell
    # Each complexity stratum has n_samples, tested with each variation
    phase3_samples_per_cell = n_samples  # Full stratum tested per variation

    # Power thresholds (based on common statistical guidelines)
    min_for_ttest = 20  # Minimum for reliable t-test
    min_for_anova = 15  # Minimum per cell for ANOVA
    recommended = 30    # For medium effect size detection (d=0.5, power=0.8)

    warnings = []

    if samples_per_difficulty < min_for_ttest:
        warnings.append(f"Phase 2: Only {samples_per_difficulty:.0f} samples per difficulty stratum "
                       f"(min recommended: {min_for_ttest})")

    if n_samples < min_for_anova:
        warnings.append(f"Phase 3: Only {n_samples} samples per complexity stratum "
                       f"(min recommended: {min_for_anova})")

    # Compute minimum detectable effect size (approximation)
    # For t-test: d = t * sqrt(2/n), with t~2 for p<0.05
    if samples_per_difficulty > 2:
        min_detectable_d = 2 * math.sqrt(2 / samples_per_difficulty)
    else:
        min_detectable_d = float('inf')

    return {
        'samples_per_complexity_stratum': n_samples,
        'phase2_samples_per_difficulty': samples_per_difficulty,
        'phase3_samples_per_cell': phase3_samples_per_cell,
        'min_detectable_effect_size': min_detectable_d,
        'adequate_for_phase2': samples_per_difficulty >= min_for_ttest,
        'adequate_for_phase3': n_samples >= min_for_anova,
        'recommended_minimum': recommended,
        'warnings': warnings
    }


def stratify_by_complexity(samples: List[Dict],
                           target_per_stratum: int = 40,
                           use_all: bool = False,
                           balance: bool = True) -> Dict[str, List[Dict]]:
    """
    Stratify samples by schema complexity.

    Args:
        samples: List of samples to stratify
        target_per_stratum: Target samples per stratum (if not use_all)
        use_all: If True, use all available samples (respecting balance)
        balance: If True, limit to minimum stratum size for balanced design

    Returns dict with keys: 'simple', 'medium', 'complex'
    """
    # Compute complexity for each sample
    for sample in samples:
        complexity = compute_schema_complexity(sample.get('tools', []))
        sample['schema_complexity'] = complexity

    # Group by complexity level
    stratified = defaultdict(list)
    for sample in samples:
        level = sample['schema_complexity']['complexity_level']
        stratified[level].append(sample)

    print(f"\nComplexity Distribution (available):")
    for level in ['simple', 'medium', 'complex']:
        print(f"  {level}: {len(stratified[level])} samples")

    # Determine sample size per stratum
    available_counts = [len(stratified[level]) for level in ['simple', 'medium', 'complex']]
    min_available = min(available_counts)

    if use_all:
        if balance:
            # Balanced design: use minimum available across all strata
            n_per_stratum = min_available
            print(f"\nUsing balanced design: {n_per_stratum} samples per stratum")
        else:
            # Unbalanced: use all available per stratum
            n_per_stratum = None  # Will use full available
            print(f"\nUsing all available samples (unbalanced design)")
    else:
        n_per_stratum = min(target_per_stratum, min_available)
        if n_per_stratum < target_per_stratum:
            print(f"\nWARNING: Reducing target from {target_per_stratum} to {n_per_stratum} "
                  f"(limited by medium stratum)")

    # Sample from each stratum
    result = {}
    for level in ['simple', 'medium', 'complex']:
        available = stratified[level]
        if n_per_stratum is None:
            # Unbalanced: use all
            result[level] = available.copy()
        else:
            n_sample = min(n_per_stratum, len(available))
            result[level] = random.sample(available, n_sample)
        print(f"  Selected {len(result[level])} {level} samples")

    return result


def create_experiment_config(stratified_data: Dict[str, List[Dict]]) -> Dict:
    """Create experiment configuration with sample metadata."""

    config = {
        'metadata': {
            'description': 'ACL 2026 Stratified Samples by Schema Complexity',
            'stratification_criteria': {
                'simple': 'depth <= 2, total_params <= 5',
                'medium': 'depth 3-4 OR total_params 6-10',
                'complex': 'depth >= 5 OR total_params > 10'
            },
            'source': 'Toucan benchmark filtered for single-tool ASCII prompts'
        },
        'statistics': {},
        'samples': {}
    }

    # Compute statistics per stratum
    for level, samples in stratified_data.items():
        if not samples:
            continue

        depths = [s['schema_complexity']['depth'] for s in samples]
        params = [s['schema_complexity']['total_params'] for s in samples]
        scores = [s['schema_complexity']['complexity_score'] for s in samples]

        config['statistics'][level] = {
            'n_samples': len(samples),
            'depth': {
                'mean': sum(depths) / len(depths),
                'min': min(depths),
                'max': max(depths)
            },
            'total_params': {
                'mean': sum(params) / len(params),
                'min': min(params),
                'max': max(params)
            },
            'complexity_score': {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores)
            }
        }

        # Store samples with IDs
        config['samples'][level] = [
            {
                'id': s['id'],
                'question': s['question'],
                'tools': s['tools'],
                'tool_calls': s.get('tool_calls', []),
                'schema_complexity': s['schema_complexity']
            }
            for s in samples
        ]

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Prepare stratified sample data for ACL experiments'
    )
    parser.add_argument(
        '--input', type=str,
        default='data/toucan/toucan_tool_calls_1006.json',
        help='Input Toucan dataset'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/acl_stratified/stratified_samples.json',
        help='Output stratified samples'
    )
    parser.add_argument(
        '--samples-per-stratum', type=int, default=36,
        help='Number of samples per complexity stratum (default: 36, the max balanced)'
    )
    parser.add_argument(
        '--use-all', action='store_true',
        help='Use all available samples (balanced by min stratum)'
    )
    parser.add_argument(
        '--unbalanced', action='store_true',
        help='Allow unbalanced strata (use with --use-all)'
    )
    parser.add_argument(
        '--relaxed-filter', action='store_true',
        help='Relaxed filtering: allow multi-tool samples to increase pool'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    with open(input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {input_path}")

    # Filter suitable prompts
    suitable = filter_suitable_prompts(data, relaxed=args.relaxed_filter)
    print(f"Filtered to {len(suitable)} suitable samples")
    if args.relaxed_filter:
        print("  (using relaxed filter: multi-tool samples allowed)")

    # Stratify by complexity
    stratified = stratify_by_complexity(
        suitable,
        target_per_stratum=args.samples_per_stratum,
        use_all=args.use_all,
        balance=not args.unbalanced
    )

    # Create config
    config = create_experiment_config(stratified)

    # Compute power analysis
    min_stratum_size = min(len(s) for s in stratified.values())
    power = compute_power_analysis(min_stratum_size)
    config['power_analysis'] = power

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved stratified samples to {output_path}")

    # Print summary
    total = sum(len(s) for s in config['samples'].values())
    print(f"\n{'='*60}")
    print("STRATIFIED SAMPLE SUMMARY")
    print('='*60)
    print(f"Total samples: {total}")
    for level in ['simple', 'medium', 'complex']:
        if level in config['statistics']:
            stats = config['statistics'][level]
            print(f"\n{level.upper()}:")
            print(f"  n = {stats['n_samples']}")
            print(f"  depth: {stats['depth']['mean']:.1f} (range: {stats['depth']['min']}-{stats['depth']['max']})")
            print(f"  params: {stats['total_params']['mean']:.1f} (range: {stats['total_params']['min']}-{stats['total_params']['max']})")

    # Print power analysis
    print(f"\n{'='*60}")
    print("STATISTICAL POWER ANALYSIS")
    print('='*60)
    print(f"Samples per complexity stratum: {power['samples_per_complexity_stratum']}")
    print(f"Phase 2 samples per difficulty: ~{power['phase2_samples_per_difficulty']:.0f}")
    print(f"Phase 3 samples per cell: {power['phase3_samples_per_cell']}")
    print(f"Min detectable effect size (d): {power['min_detectable_effect_size']:.2f}")
    print(f"Adequate for Phase 2 (ceiling): {'YES' if power['adequate_for_phase2'] else 'NO'}")
    print(f"Adequate for Phase 3 (interaction): {'YES' if power['adequate_for_phase3'] else 'NO'}")

    if power['warnings']:
        print(f"\nWARNINGS:")
        for warning in power['warnings']:
            print(f"  - {warning}")

    # Recommendations
    if not power['adequate_for_phase2'] or not power['adequate_for_phase3']:
        print(f"\nRECOMMENDATIONS:")
        print(f"  - Use --relaxed-filter to increase sample pool")
        print(f"  - Current minimum stratum: {min_stratum_size}")
        print(f"  - Recommended minimum: {power['recommended_minimum']}")


if __name__ == '__main__':
    main()
