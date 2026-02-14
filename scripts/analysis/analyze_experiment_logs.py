#!/usr/bin/env python3
"""Analyze experiment logs for failure reasons."""

import os
import re
from pathlib import Path
from collections import defaultdict

def analyze_log_file(log_path: Path) -> dict:
    """Analyze a single log file for error patterns."""
    counts = defaultdict(int)

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Count different error types
    counts['empty_invalid'] = len(re.findall(r'Empty or invalid response', content))
    counts['json_failed'] = len(re.findall(r'JSON extraction failed', content))
    counts['throttling'] = len(re.findall(r'ThrottlingException|throttling', content, re.IGNORECASE))
    counts['timeout'] = len(re.findall(r'ReadTimeout|timed?\s*out|TimeoutError', content, re.IGNORECASE))
    counts['connection_error'] = len(re.findall(r'ConnectionError|connection refused', content, re.IGNORECASE))

    # Count valid/invalid from completion lines
    valid_pattern = re.findall(r'Completed.*\((\d+)/(\d+) valid\)', content)
    total_valid = sum(int(v) for v, t in valid_pattern)
    total_runs = sum(int(t) for v, t in valid_pattern)
    counts['total_valid'] = total_valid
    counts['total_runs'] = total_runs
    counts['total_invalid'] = total_runs - total_valid

    return counts

def analyze_experiment(exp_dir: Path) -> dict:
    """Analyze all logs in an experiment directory."""
    total_counts = defaultdict(int)

    log_files = list(exp_dir.glob('*.log'))
    for log_file in log_files:
        counts = analyze_log_file(log_file)
        for key, value in counts.items():
            total_counts[key] += value

    return dict(total_counts)

def main():
    sharegpt_dir = Path('/Users/guanghu/Documents/genai/projects/sted-internal/llm_gen_results/sharegpt')

    print("=" * 100)
    print(f"{'Model':<50} {'Valid':<12} {'Invalid':<10} {'Empty':<10} {'JSON':<10} {'Throttle':<10} {'Timeout':<10}")
    print("=" * 100)

    results = []
    for exp_dir in sorted(sharegpt_dir.glob('generations-*/')):
        model_name = exp_dir.name.replace('generations-', '').split('-202')[0]
        counts = analyze_experiment(exp_dir)

        results.append({
            'model': model_name,
            'dir': exp_dir.name,
            **counts
        })

        print(f"{model_name:<50} {counts.get('total_valid', 0):<12} {counts.get('total_invalid', 0):<10} "
              f"{counts.get('empty_invalid', 0):<10} {counts.get('json_failed', 0):<10} "
              f"{counts.get('throttling', 0):<10} {counts.get('timeout', 0):<10}")

    print("=" * 100)

    # Summary
    print("\n\nDetailed Failure Analysis:")
    print("-" * 80)
    for r in results:
        if r.get('total_invalid', 0) > 0:
            print(f"\n{r['model']}:")
            print(f"  Total Invalid: {r.get('total_invalid', 0)}")
            print(f"  - Empty/Invalid Response: {r.get('empty_invalid', 0)}")
            print(f"  - JSON Extraction Failed: {r.get('json_failed', 0)}")
            print(f"  - Throttling: {r.get('throttling', 0)}")
            print(f"  - Timeout: {r.get('timeout', 0)}")

if __name__ == '__main__':
    main()
