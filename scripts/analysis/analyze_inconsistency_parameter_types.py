#!/usr/bin/env python3
"""
Analyze parameter types in inconsistent vs consistent tool calling samples.

This script examines whether inconsistent samples tend to have more complex
parameter types (dicts, lists) that would benefit from STED's semantic matching.

Key questions:
1. What parameter types are involved in inconsistent cases?
2. Are complex parameter types more common in inconsistent samples?
3. Does STED provide value for these complex cases?
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_param_type(value: Any) -> str:
    """Classify parameter value type."""
    if value is None:
        return 'null'
    elif isinstance(value, bool):
        return 'bool'
    elif isinstance(value, int):
        return 'int'
    elif isinstance(value, float):
        return 'float'
    elif isinstance(value, str):
        return 'string'
    elif isinstance(value, list):
        return 'list'
    elif isinstance(value, dict):
        return 'dict'
    else:
        return 'other'


def get_param_complexity(value: Any) -> Tuple[str, int]:
    """
    Determine parameter complexity level.
    Returns (complexity_level, depth)
    - 'primitive': str, int, float, bool, null
    - 'simple_list': list of primitives
    - 'simple_dict': dict with primitive values
    - 'nested': contains nested structures
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return ('primitive', 0)

    if isinstance(value, list):
        if not value:
            return ('simple_list', 1)
        max_depth = 0
        has_complex = False
        for item in value:
            item_complexity, item_depth = get_param_complexity(item)
            if item_complexity != 'primitive':
                has_complex = True
            max_depth = max(max_depth, item_depth)
        if has_complex:
            return ('nested', max_depth + 1)
        return ('simple_list', 1)

    if isinstance(value, dict):
        if not value:
            return ('simple_dict', 1)
        max_depth = 0
        has_complex = False
        for v in value.values():
            v_complexity, v_depth = get_param_complexity(v)
            if v_complexity != 'primitive':
                has_complex = True
            max_depth = max(max_depth, v_depth)
        if has_complex:
            return ('nested', max_depth + 1)
        return ('simple_dict', 1)

    return ('primitive', 0)


def analyze_tool_call_params(tool_call: Dict) -> Dict:
    """Analyze parameters of a single tool call."""
    args = tool_call.get('arguments', {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            return {'param_count': 0, 'types': [], 'complexities': [], 'max_depth': 0}

    if not isinstance(args, dict):
        return {'param_count': 0, 'types': [], 'complexities': [], 'max_depth': 0}

    types = []
    complexities = []
    max_depth = 0

    for key, value in args.items():
        types.append(get_param_type(value))
        complexity, depth = get_param_complexity(value)
        complexities.append(complexity)
        max_depth = max(max_depth, depth)

    return {
        'param_count': len(args),
        'types': types,
        'complexities': complexities,
        'max_depth': max_depth,
        'has_list': 'list' in types,
        'has_dict': 'dict' in types,
        'has_nested': 'nested' in complexities,
        'has_complex': any(c != 'primitive' for c in complexities)
    }


def analyze_sample_runs(runs: List) -> Dict:
    """Analyze all runs for a sample."""
    valid_runs = [r for r in runs if r and isinstance(r, list)]
    if len(valid_runs) < 2:
        return None

    # Check consistency
    run_strs = [json.dumps(r, sort_keys=True) for r in valid_runs]
    is_consistent = len(set(run_strs)) == 1
    num_unique = len(set(run_strs))

    # Analyze parameter types across all runs
    all_types = defaultdict(int)
    all_complexities = defaultdict(int)
    total_params = 0
    has_list = False
    has_dict = False
    has_nested = False
    has_complex = False
    max_depth = 0

    for run in valid_runs:
        for tool_call in run:
            if isinstance(tool_call, dict):
                analysis = analyze_tool_call_params(tool_call)
                for t in analysis['types']:
                    all_types[t] += 1
                for c in analysis['complexities']:
                    all_complexities[c] += 1
                total_params += analysis['param_count']
                has_list = has_list or analysis.get('has_list', False)
                has_dict = has_dict or analysis.get('has_dict', False)
                has_nested = has_nested or analysis.get('has_nested', False)
                has_complex = has_complex or analysis.get('has_complex', False)
                max_depth = max(max_depth, analysis.get('max_depth', 0))

    # Identify inconsistency type
    inconsistency_type = None
    if not is_consistent:
        # Check tool count variation
        tool_counts = [len(r) for r in valid_runs]
        if len(set(tool_counts)) > 1:
            inconsistency_type = 'tool_count'
        else:
            # Check tool selection variation
            tool_names_per_run = [tuple(t.get('name', '') for t in r if isinstance(t, dict)) for r in valid_runs]
            if len(set(tool_names_per_run)) > 1:
                inconsistency_type = 'tool_selection'
            else:
                inconsistency_type = 'parameter_value'

    return {
        'is_consistent': is_consistent,
        'num_unique': num_unique,
        'inconsistency_type': inconsistency_type,
        'types': dict(all_types),
        'complexities': dict(all_complexities),
        'total_params': total_params // len(valid_runs),  # Average per run
        'has_list': has_list,
        'has_dict': has_dict,
        'has_nested': has_nested,
        'has_complex': has_complex,
        'max_depth': max_depth
    }


def analyze_parameter_value_inconsistency(runs: List) -> Dict:
    """
    Detailed analysis of parameter value variations.
    Returns info about which parameters differ and their types.
    """
    valid_runs = [r for r in runs if r and isinstance(r, list)]
    if len(valid_runs) < 2:
        return None

    # Get tool names (assume they're the same for param value variation)
    first_run = valid_runs[0]

    differing_params = []

    for tool_idx, tool in enumerate(first_run):
        if not isinstance(tool, dict):
            continue

        tool_name = tool.get('name', '')
        args = tool.get('arguments', {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                continue

        if not isinstance(args, dict):
            continue

        # Compare this tool's params across runs
        for param_name, param_value in args.items():
            param_type = get_param_type(param_value)
            complexity, depth = get_param_complexity(param_value)

            # Check if this param varies across runs
            param_values_across_runs = []
            for run in valid_runs:
                if tool_idx < len(run) and isinstance(run[tool_idx], dict):
                    run_args = run[tool_idx].get('arguments', {})
                    if isinstance(run_args, str):
                        try:
                            run_args = json.loads(run_args)
                        except:
                            run_args = {}
                    if isinstance(run_args, dict):
                        param_values_across_runs.append(
                            json.dumps(run_args.get(param_name), sort_keys=True)
                        )

            if len(set(param_values_across_runs)) > 1:
                differing_params.append({
                    'tool': tool_name,
                    'param': param_name,
                    'type': param_type,
                    'complexity': complexity,
                    'depth': depth,
                    'unique_values': len(set(param_values_across_runs))
                })

    return {
        'differing_params': differing_params,
        'has_string_diff': any(p['type'] == 'string' for p in differing_params),
        'has_number_diff': any(p['type'] in ['int', 'float'] for p in differing_params),
        'has_list_diff': any(p['type'] == 'list' for p in differing_params),
        'has_dict_diff': any(p['type'] == 'dict' for p in differing_params),
        'has_complex_diff': any(p['complexity'] != 'primitive' for p in differing_params)
    }


def load_and_analyze_model(model_path: str, model_name: str, temperature: float = 0.5) -> List[Dict]:
    """Load and analyze a model's results."""
    temp_str = f"temp_0_{int(temperature * 100):02d}"

    dirs = os.listdir(model_path)
    temp_dirs = [d for d in dirs if temp_str in d]
    if not temp_dirs:
        return []

    results_path = os.path.join(model_path, temp_dirs[0], 'all_results.json')
    if not os.path.exists(results_path):
        return []

    with open(results_path) as f:
        data = json.load(f)

    samples = []
    for sample in data.get('results', []):
        runs = sample.get('generated_runs', [])
        analysis = analyze_sample_runs(runs)
        if analysis:
            analysis['model'] = model_name
            analysis['sample_id'] = sample.get('sample_id', '')

            # For parameter value inconsistencies, do detailed analysis
            if analysis['inconsistency_type'] == 'parameter_value':
                param_analysis = analyze_parameter_value_inconsistency(runs)
                if param_analysis:
                    analysis.update(param_analysis)

            samples.append(analysis)

    return samples


def print_analysis_report(all_samples: List[Dict]):
    """Print comprehensive analysis report."""
    print("=" * 80)
    print("INCONSISTENCY PARAMETER TYPE ANALYSIS")
    print("=" * 80)

    total = len(all_samples)
    consistent = [s for s in all_samples if s['is_consistent']]
    inconsistent = [s for s in all_samples if not s['is_consistent']]

    print(f"\nTotal samples: {total}")
    print(f"Consistent: {len(consistent)} ({len(consistent)/total*100:.1f}%)")
    print(f"Inconsistent: {len(inconsistent)} ({len(inconsistent)/total*100:.1f}%)")

    # Parameter complexity comparison
    print("\n" + "-" * 80)
    print("PARAMETER COMPLEXITY: CONSISTENT vs INCONSISTENT")
    print("-" * 80)

    metrics = ['has_list', 'has_dict', 'has_nested', 'has_complex']
    labels = ['Has List Params', 'Has Dict Params', 'Has Nested Structure', 'Has Any Complex Type']

    print(f"\n{'Metric':<30} {'Consistent':<15} {'Inconsistent':<15} {'Ratio':<10}")
    print("-" * 70)

    for metric, label in zip(metrics, labels):
        cons_rate = sum(1 for s in consistent if s.get(metric)) / len(consistent) * 100 if consistent else 0
        incons_rate = sum(1 for s in inconsistent if s.get(metric)) / len(inconsistent) * 100 if inconsistent else 0
        ratio = incons_rate / cons_rate if cons_rate > 0 else float('inf')
        print(f"{label:<30} {cons_rate:>12.1f}% {incons_rate:>12.1f}% {ratio:>8.2f}x")

    # Max depth comparison
    cons_depths = [s['max_depth'] for s in consistent]
    incons_depths = [s['max_depth'] for s in inconsistent]
    print(f"\n{'Mean Max Depth':<30} {np.mean(cons_depths):>12.2f} {np.mean(incons_depths):>12.2f}")

    # Parameter value inconsistency analysis
    param_value_samples = [s for s in inconsistent if s.get('inconsistency_type') == 'parameter_value']

    print("\n" + "-" * 80)
    print("PARAMETER VALUE INCONSISTENCY ANALYSIS")
    print(f"(Samples where same tools called but param values differ: {len(param_value_samples)})")
    print("-" * 80)

    if param_value_samples:
        # What types of parameters are differing?
        diff_metrics = ['has_string_diff', 'has_number_diff', 'has_list_diff', 'has_dict_diff', 'has_complex_diff']
        diff_labels = ['String params differ', 'Number params differ', 'List params differ',
                       'Dict params differ', 'Complex params differ']

        print(f"\n{'Differing Parameter Type':<30} {'Count':<10} {'Rate':<10}")
        print("-" * 50)

        for metric, label in zip(diff_metrics, diff_labels):
            count = sum(1 for s in param_value_samples if s.get(metric))
            rate = count / len(param_value_samples) * 100
            print(f"{label:<30} {count:<10} {rate:.1f}%")

        # Detailed breakdown of differing params
        all_differing_params = []
        for s in param_value_samples:
            all_differing_params.extend(s.get('differing_params', []))

        if all_differing_params:
            print(f"\nTotal differing parameters analyzed: {len(all_differing_params)}")

            # By type
            type_counts = defaultdict(int)
            for p in all_differing_params:
                type_counts[p['type']] += 1

            print(f"\n{'Parameter Type':<20} {'Count':<10} {'Rate':<10}")
            print("-" * 40)
            for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"{ptype:<20} {count:<10} {count/len(all_differing_params)*100:.1f}%")

            # By complexity
            complexity_counts = defaultdict(int)
            for p in all_differing_params:
                complexity_counts[p['complexity']] += 1

            print(f"\n{'Complexity Level':<20} {'Count':<10} {'Rate':<10}")
            print("-" * 40)
            for comp, count in sorted(complexity_counts.items(), key=lambda x: -x[1]):
                print(f"{comp:<20} {count:<10} {count/len(all_differing_params)*100:.1f}%")

    # STED value assessment
    print("\n" + "=" * 80)
    print("STED VALUE ASSESSMENT FOR TOOL CALLING")
    print("=" * 80)

    # Calculate STED benefits
    param_value_with_complex = sum(1 for s in param_value_samples if s.get('has_complex_diff')) if param_value_samples else 0
    param_value_with_string = sum(1 for s in param_value_samples if s.get('has_string_diff')) if param_value_samples else 0

    tool_selection = sum(1 for s in inconsistent if s.get('inconsistency_type') == 'tool_selection')
    tool_count = sum(1 for s in inconsistent if s.get('inconsistency_type') == 'tool_count')

    print(f"""
STED provides value through:

1. SEMANTIC STRING MATCHING:
   - {param_value_with_string} samples ({param_value_with_string/len(inconsistent)*100:.1f}% of inconsistent)
     have string parameter variations
   - STED's embedding-based comparison recognizes semantic equivalence
   - Example: "matrix_A" ≈ "matrix_2s", "user_email" ≈ "email_address"

2. COMPLEX TYPE HANDLING:
   - {param_value_with_complex} samples ({param_value_with_complex/len(inconsistent)*100:.1f}% of inconsistent)
     have complex parameter variations (lists/dicts)
   - STED recursively compares nested structures
   - Handles list reordering and dict key variations

3. TOOL ORDER INVARIANCE (via Hungarian matching):
   - {tool_selection} samples ({tool_selection/len(inconsistent)*100:.1f}% of inconsistent)
     have tool selection variations
   - STED provides proportional credit for partial matches

4. TOOL COUNT VARIATIONS:
   - {tool_count} samples ({tool_count/len(inconsistent)*100:.1f}% of inconsistent)
     have different tool counts
   - STED handles uneven comparisons with insert/delete costs
""")

    # Summary
    total_sted_benefits = param_value_with_string + param_value_with_complex + tool_selection + tool_count
    print(f"\nTotal samples where STED provides value: {len(inconsistent)}")
    print(f"  - String semantic matching: {param_value_with_string} ({param_value_with_string/len(inconsistent)*100:.1f}%)")
    print(f"  - Complex structure handling: {param_value_with_complex} ({param_value_with_complex/len(inconsistent)*100:.1f}%)")
    print(f"  - Tool selection/order: {tool_selection} ({tool_selection/len(inconsistent)*100:.1f}%)")
    print(f"  - Tool count variation: {tool_count} ({tool_count/len(inconsistent)*100:.1f}%)")


def print_example_cases(all_samples: List[Dict], toucan_base: Path):
    """Print example cases of complex parameter inconsistencies."""
    print("\n" + "=" * 80)
    print("EXAMPLE CASES: COMPLEX PARAMETER INCONSISTENCIES")
    print("=" * 80)

    # Find samples with complex parameter variations
    complex_cases = [s for s in all_samples
                     if s.get('inconsistency_type') == 'parameter_value'
                     and s.get('has_complex_diff')]

    if not complex_cases:
        print("\nNo complex parameter inconsistency cases found.")
        return

    print(f"\nFound {len(complex_cases)} samples with complex parameter variations.")
    print("\nExample cases (up to 5):")

    for i, sample in enumerate(complex_cases[:5]):
        print(f"\n--- Case {i+1}: {sample['sample_id']} ({sample['model']}) ---")
        differing = sample.get('differing_params', [])
        complex_diffs = [p for p in differing if p['complexity'] != 'primitive']
        for p in complex_diffs[:3]:
            print(f"  Tool: {p['tool']}")
            print(f"  Param: {p['param']}")
            print(f"  Type: {p['type']}, Complexity: {p['complexity']}, Depth: {p['depth']}")


def main():
    """Main analysis function."""
    toucan_base = PROJECT_ROOT / 'llm_gen_results' / 'toucan'

    # Model directories
    model_dirs = {
        'Claude-Opus-4': 'generations-claude-opus-4-20251222',
        'Claude-Opus-4.5': 'generations-claude-opus-4.5-20251224',
        'Claude-Sonnet-4': 'generations-claude-sonnet-4-20251223',
        'Claude-Sonnet-4.5': 'generations-claude-sonnet-4.5-20251224',
        'Claude-3.5-Sonnet': 'generations-claude-3.5-sonnet-20251223',
        'Claude-3.5-Haiku': 'generations-claude-3.5-haiku-20251224',
        'Claude-3.7-Sonnet': 'generations-claude37-sonnet-20251229',
        'Claude-Haiku-4.5': 'generations-claude-haiku-4.5-20251224',
        'Llama-3.3-70B': 'generations-llama-3.3-70b-20251223',
        'Nova-2-Lite': 'generations-nova2-lite-20251222_075929',
        'Qwen3-235B': 'generations-qwen3-235b-a22b-20251229',
        'Qwen3-32B': 'generations-qwen3-32b-20251224',
        'Mistral-Large-675B': 'generations-mistral.mistral-large-3-675b-instruct-20251224_155007',
        'Minimax-M2': 'generations-minimax-m2-20251229',
        'Mimo-V2-Flash': 'generations-mimo-v2-flash-20251229',
        'NemoTron-3-Nano': 'generations-nemotron3-nano-20251229',
        'GPT-OSS-120B': 'generations-gpt-oss-120b-20251229',
        'Gemini-2.5-Flash-Lite': 'generations-gemini-2.5-flash-lite-20251229',
    }

    print("Loading and analyzing model results...")
    all_samples = []

    for model_name, dir_name in model_dirs.items():
        model_path = toucan_base / dir_name
        if not model_path.exists():
            print(f"  Warning: {model_name} directory not found")
            continue

        samples = load_and_analyze_model(str(model_path), model_name, temperature=0.5)
        if samples:
            all_samples.extend(samples)
            print(f"  {model_name}: {len(samples)} samples")

    if not all_samples:
        print("No samples found!")
        return

    # Print analysis report
    print_analysis_report(all_samples)

    # Print example cases
    print_example_cases(all_samples, toucan_base)


if __name__ == '__main__':
    main()
