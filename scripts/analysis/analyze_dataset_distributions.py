"""
Analyze distribution of downloaded datasets for STED evaluation suitability.

This script analyzes:
1. Dataset sizes and structure
2. JSON complexity (depth, keys, field types)
3. Value distributions and diversity
4. Suitability for testing combined similarity approach
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = Path("/Users/guanghu/Documents/genai/projects/sted-internal/research/datasets")


def count_json_depth(obj: Any, current_depth: int = 0) -> int:
    """Recursively count the maximum depth of a JSON object."""
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(count_json_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(count_json_depth(item, current_depth + 1) for item in obj)
    else:
        return current_depth


def count_json_keys(obj: Any) -> int:
    """Count total number of keys in a JSON object."""
    if isinstance(obj, dict):
        count = len(obj)
        for v in obj.values():
            count += count_json_keys(v)
        return count
    elif isinstance(obj, list):
        return sum(count_json_keys(item) for item in obj)
    else:
        return 0


def get_value_types(obj: Any) -> Counter:
    """Get distribution of value types in a JSON object."""
    types = Counter()

    if isinstance(obj, dict):
        for v in obj.values():
            types.update(get_value_types(v))
    elif isinstance(obj, list):
        types["array"] += 1
        for item in obj:
            types.update(get_value_types(item))
    elif isinstance(obj, str):
        types["string"] += 1
    elif isinstance(obj, (int, float)):
        types["number"] += 1
    elif isinstance(obj, bool):
        types["boolean"] += 1
    elif obj is None:
        types["null"] += 1

    return types


def get_string_lengths(obj: Any) -> List[int]:
    """Get all string lengths in a JSON object."""
    lengths = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            lengths.append(len(k))  # Key length
            lengths.extend(get_string_lengths(v))
    elif isinstance(obj, list):
        for item in obj:
            lengths.extend(get_string_lengths(item))
    elif isinstance(obj, str):
        lengths.append(len(obj))

    return lengths


def analyze_dataset(dataset_path: Path, sample_size: int = 1000) -> Dict:
    """Analyze a single dataset."""
    print(f"\nAnalyzing: {dataset_path.name}")
    print("-" * 80)

    try:
        with open(dataset_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        # Sample if too large
        if len(data) > sample_size:
            print(f"  Sampling {sample_size} from {len(data)} examples")
            np.random.seed(42)
            indices = np.random.choice(len(data), sample_size, replace=False)
            data = [data[i] for i in indices]

        # Analyze structure
        depths = []
        key_counts = []
        type_distributions = Counter()
        all_string_lengths = []

        for item in data:
            depths.append(count_json_depth(item))
            key_counts.append(count_json_keys(item))
            type_distributions.update(get_value_types(item))
            all_string_lengths.extend(get_string_lengths(item))

        # Calculate statistics
        stats = {
            "dataset": dataset_path.stem,
            "total_examples": len(data),
            "depth": {
                "mean": np.mean(depths),
                "median": np.median(depths),
                "min": np.min(depths),
                "max": np.max(depths),
                "std": np.std(depths),
            },
            "keys_per_example": {
                "mean": np.mean(key_counts),
                "median": np.median(key_counts),
                "min": np.min(key_counts),
                "max": np.max(key_counts),
                "std": np.std(key_counts),
            },
            "type_distribution": dict(type_distributions),
            "string_lengths": {
                "mean": np.mean(all_string_lengths) if all_string_lengths else 0,
                "median": np.median(all_string_lengths) if all_string_lengths else 0,
                "min": np.min(all_string_lengths) if all_string_lengths else 0,
                "max": np.max(all_string_lengths) if all_string_lengths else 0,
                "short_strings_pct": sum(1 for l in all_string_lengths if l < 4) / len(all_string_lengths) * 100 if all_string_lengths else 0,
            }
        }

        # Print summary
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Depth: {stats['depth']['mean']:.2f} ± {stats['depth']['std']:.2f} (range: {stats['depth']['min']}-{stats['depth']['max']})")
        print(f"  Keys per example: {stats['keys_per_example']['mean']:.2f} ± {stats['keys_per_example']['std']:.2f}")
        print(f"  Type distribution: {dict(type_distributions)}")
        print(f"  String lengths: mean={stats['string_lengths']['mean']:.2f}, short(<4 chars)={stats['string_lengths']['short_strings_pct']:.1f}%")

        # Suitability assessment
        print(f"\n  Suitability for Combined Similarity Testing:")
        suitability_score = 0
        reasons = []

        # Check depth (good if varied and reasonable)
        if 2 <= stats['depth']['mean'] <= 6:
            suitability_score += 2
            reasons.append("✓ Good depth range for testing nested structures")
        else:
            reasons.append("⚠ Depth may be too shallow/deep")

        # Check key count (good if substantial)
        if stats['keys_per_example']['mean'] >= 5:
            suitability_score += 2
            reasons.append("✓ Sufficient keys for structural matching")
        else:
            reasons.append("⚠ Few keys per example")

        # Check type diversity
        if len(type_distributions) >= 3:
            suitability_score += 2
            reasons.append("✓ Good type diversity")
        else:
            reasons.append("⚠ Limited type diversity")

        # Check string variety (important for our short string fix)
        if 10 <= stats['string_lengths']['short_strings_pct'] <= 50:
            suitability_score += 2
            reasons.append("✓ Good mix of short/long strings")
        elif stats['string_lengths']['short_strings_pct'] > 50:
            suitability_score += 1
            reasons.append("⚠ Many short strings (good for testing short string fix)")
        else:
            reasons.append("⚠ Mostly long strings")

        # Check variability
        if stats['depth']['std'] > 0.5 and stats['keys_per_example']['std'] > 2:
            suitability_score += 2
            reasons.append("✓ Good structural variability")
        else:
            reasons.append("⚠ Low structural variability")

        for reason in reasons:
            print(f"    {reason}")

        print(f"  Overall suitability: {suitability_score}/10")

        return stats

    except Exception as e:
        print(f"  Error analyzing {dataset_path.name}: {e}")
        return None


def plot_distributions(all_stats: List[Dict], output_dir: Path):
    """Create visualization of dataset distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Dataset Distribution Analysis for STED Evaluation", fontsize=16)

    # 1. Depth distribution
    ax = axes[0, 0]
    depths_data = [(s['dataset'], s['depth']['mean'], s['depth']['std']) for s in all_stats if s]
    datasets = [d[0] for d in depths_data]
    means = [d[1] for d in depths_data]
    stds = [d[2] for d in depths_data]

    x = np.arange(len(datasets))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average JSON Depth')
    ax.set_title('JSON Depth Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # 2. Keys per example
    ax = axes[0, 1]
    keys_data = [(s['dataset'], s['keys_per_example']['mean'], s['keys_per_example']['std']) for s in all_stats if s]
    means = [d[1] for d in keys_data]
    stds = [d[2] for d in keys_data]

    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='coral')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average Keys per Example')
    ax.set_title('Structural Complexity (Keys)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # 3. Type distribution
    ax = axes[0, 2]
    type_data = defaultdict(list)
    for s in all_stats:
        if s:
            total = sum(s['type_distribution'].values())
            for t, count in s['type_distribution'].items():
                type_data[t].append(count / total * 100)

    bottom = np.zeros(len(datasets))
    colors = plt.cm.Set3(np.linspace(0, 1, len(type_data)))

    for (type_name, percentages), color in zip(type_data.items(), colors):
        ax.bar(x, percentages, bottom=bottom, label=type_name, alpha=0.8, color=color)
        bottom += percentages

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Type Distribution (%)')
    ax.set_title('Value Type Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # 4. String length distribution
    ax = axes[1, 0]
    short_pct = [s['string_lengths']['short_strings_pct'] for s in all_stats if s]
    long_pct = [100 - pct for pct in short_pct]

    width = 0.35
    ax.bar(x - width/2, short_pct, width, label='Short (<4 chars)', alpha=0.8, color='lightcoral')
    ax.bar(x + width/2, long_pct, width, label='Long (≥4 chars)', alpha=0.8, color='lightgreen')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('String Length Distribution (Short vs Long)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 5. Average string length
    ax = axes[1, 1]
    avg_lengths = [s['string_lengths']['mean'] for s in all_stats if s]
    ax.bar(x, avg_lengths, alpha=0.7, color='mediumpurple')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average String Length (chars)')
    ax.set_title('Average String Length')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # 6. Suitability scores (calculated from all factors)
    ax = axes[1, 2]
    # Recalculate suitability scores
    suitability_scores = []
    for s in all_stats:
        if s:
            score = 0
            if 2 <= s['depth']['mean'] <= 6:
                score += 2
            if s['keys_per_example']['mean'] >= 5:
                score += 2
            if len(s['type_distribution']) >= 3:
                score += 2
            if 10 <= s['string_lengths']['short_strings_pct'] <= 50:
                score += 2
            elif s['string_lengths']['short_strings_pct'] > 50:
                score += 1
            if s['depth']['std'] > 0.5 and s['keys_per_example']['std'] > 2:
                score += 2
            suitability_scores.append(score)

    colors_suit = ['green' if s >= 7 else 'orange' if s >= 5 else 'red' for s in suitability_scores]
    ax.bar(x, suitability_scores, alpha=0.7, color=colors_suit)
    ax.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Good (≥7)')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Fair (≥5)')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Suitability Score (0-10)')
    ax.set_title('Suitability for Combined Similarity Testing')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylim(0, 10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "dataset_distribution_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}")


def main():
    """Main analysis function."""
    print("="*80)
    print("DATASET DISTRIBUTION ANALYSIS FOR STED EVALUATION")
    print("="*80)

    # Find all JSON dataset files
    dataset_files = list(DATASET_DIR.glob("*.json"))
    dataset_files = [f for f in dataset_files if f.name not in ['data_loader.py']]

    if not dataset_files:
        print("No dataset files found!")
        return

    print(f"\nFound {len(dataset_files)} datasets")

    # Analyze each dataset
    all_stats = []
    for dataset_file in sorted(dataset_files):
        stats = analyze_dataset(dataset_file, sample_size=1000)
        if stats:
            all_stats.append(stats)

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    plot_distributions(all_stats, DATASET_DIR.parent / "experiments" / "dataset_analysis")

    # Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR EVALUATION")
    print("="*80)

    # Sort by suitability
    scored_datasets = []
    for s in all_stats:
        score = 0
        if 2 <= s['depth']['mean'] <= 6:
            score += 2
        if s['keys_per_example']['mean'] >= 5:
            score += 2
        if len(s['type_distribution']) >= 3:
            score += 2
        if 10 <= s['string_lengths']['short_strings_pct'] <= 50:
            score += 2
        elif s['string_lengths']['short_strings_pct'] > 50:
            score += 1
        if s['depth']['std'] > 0.5 and s['keys_per_example']['std'] > 2:
            score += 2
        scored_datasets.append((s['dataset'], score))

    scored_datasets.sort(key=lambda x: x[1], reverse=True)

    print("\nDatasets ranked by suitability:")
    for dataset, score in scored_datasets:
        rating = "EXCELLENT" if score >= 8 else "GOOD" if score >= 6 else "FAIR" if score >= 4 else "LIMITED"
        print(f"  {dataset:50} Score: {score}/10 ({rating})")

    print(f"\nTop recommendations:")
    for i, (dataset, score) in enumerate(scored_datasets[:3], 1):
        print(f"  {i}. {dataset} (score: {score}/10)")

    print("\n" + "="*80)
    print("TESTING OPPORTUNITIES")
    print("="*80)
    print("""
The new combined similarity approach can be evaluated on:

1. VALUE SWAP TESTING: Use datasets with many short strings to test the
   short string fix. Good candidates with high short string percentage.

2. STRUCTURAL MATCHING: Use datasets with complex nested structures
   (depth 3-6) to test structure-guided matching.

3. CROSS-FIELD VARIABILITY: Use datasets with high key count variability
   to test how combined similarity handles different structures.

4. TYPE DIVERSITY: Use datasets with multiple value types to test
   content similarity across different types.
""")


if __name__ == "__main__":
    main()
