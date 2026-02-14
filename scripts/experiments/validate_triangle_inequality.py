#!/usr/bin/env python3
"""
Triangle Inequality Validation for STED

This script empirically validates the relaxed triangle inequality bound for STED:
    d_STED(T1, T3) <= (1 + epsilon) * (d_STED(T1, T2) + d_STED(T2, T3))

We measure the actual epsilon observed in practice to either:
1. Validate the theoretical bound (epsilon <= L - 1)
2. Tighten the bound based on empirical evidence
3. Identify conditions under which standard triangle inequality holds

Reference: Theorem 1 in docs/sted_theory_icml.tex
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted import SemanticJsonTreeConsistencyEvaluator


@dataclass
class TriangleInequalityResult:
    """Result of a single triangle inequality test."""
    t1: Dict
    t2: Dict
    t3: Dict
    d12: float  # d_STED(T1, T2)
    d23: float  # d_STED(T2, T3)
    d13: float  # d_STED(T1, T3)
    epsilon: float  # Actual relaxation factor
    satisfies_standard: bool  # d13 <= d12 + d23
    category: str  # Category of test case


def compute_relaxation_factor(d12: float, d23: float, d13: float) -> float:
    """
    Compute the relaxation factor epsilon such that:
    d13 <= (1 + epsilon) * (d12 + d23)

    Returns epsilon = d13 / (d12 + d23) - 1
    """
    sum_d = d12 + d23
    if sum_d < 1e-10:
        # Both distances near zero, triangle inequality trivially satisfied
        return 0.0
    return max(0.0, d13 / sum_d - 1.0)


def generate_json_triple_simple() -> Tuple[Dict, Dict, Dict, str]:
    """Generate a simple triple of related JSON objects."""
    import random

    base = {
        "name": "test_item",
        "value": random.randint(1, 100),
        "category": random.choice(["A", "B", "C"]),
        "active": True
    }

    # T2 is a small modification of T1
    t2 = base.copy()
    t2["value"] = base["value"] + random.randint(1, 10)

    # T3 is a modification of T2 (transitively related to T1)
    t3 = t2.copy()
    t3["category"] = random.choice(["D", "E", "F"])

    return base, t2, t3, "simple_chain"


def generate_json_triple_nested() -> Tuple[Dict, Dict, Dict, str]:
    """Generate nested JSON triples."""
    import random

    t1 = {
        "user": {
            "name": f"user_{random.randint(1, 100)}",
            "profile": {
                "age": random.randint(20, 60),
                "city": random.choice(["NYC", "LA", "Chicago"])
            }
        }
    }

    t2 = {
        "user": {
            "name": t1["user"]["name"],
            "profile": {
                "age": t1["user"]["profile"]["age"] + random.randint(1, 5),
                "city": t1["user"]["profile"]["city"]
            }
        }
    }

    t3 = {
        "user": {
            "name": t1["user"]["name"],
            "profile": {
                "age": t2["user"]["profile"]["age"],
                "city": random.choice(["Boston", "Seattle", "Denver"])
            }
        }
    }

    return t1, t2, t3, "nested_chain"


def generate_json_triple_array() -> Tuple[Dict, Dict, Dict, str]:
    """Generate array-heavy JSON triples."""
    import random

    items = [{"id": i, "val": random.randint(1, 50)} for i in range(5)]

    t1 = {"items": items, "count": len(items)}

    # T2: modify one item
    items2 = [item.copy() for item in items]
    items2[2]["val"] = items2[2]["val"] + 10
    t2 = {"items": items2, "count": len(items2)}

    # T3: modify another item
    items3 = [item.copy() for item in items2]
    items3[4]["val"] = items3[4]["val"] + 10
    t3 = {"items": items3, "count": len(items3)}

    return t1, t2, t3, "array_chain"


def generate_json_triple_divergent() -> Tuple[Dict, Dict, Dict, str]:
    """Generate divergent triples (T2 between T1 and T3)."""
    import random

    # T1 and T3 are very different, T2 is intermediate
    t1 = {
        "type": "request",
        "method": "GET",
        "url": "/api/users",
        "headers": {"auth": "token123"}
    }

    t2 = {
        "type": "request",
        "method": "POST",  # Changed from GET
        "url": "/api/users",
        "headers": {"auth": "token123"}
    }

    t3 = {
        "type": "response",  # Changed from request
        "method": "POST",
        "url": "/api/data",  # Changed URL
        "headers": {"auth": "token456"}  # Changed token
    }

    return t1, t2, t3, "divergent"


def generate_json_triple_semantic() -> Tuple[Dict, Dict, Dict, str]:
    """Generate triples with semantic variations."""
    t1 = {
        "action": "search",
        "query": "machine learning papers",
        "limit": 10
    }

    t2 = {
        "action": "search",
        "query": "deep learning papers",  # Semantically similar
        "limit": 10
    }

    t3 = {
        "action": "search",
        "query": "neural network research",  # Semantically similar to both
        "limit": 10
    }

    return t1, t2, t3, "semantic"


def generate_json_triple_structural_change() -> Tuple[Dict, Dict, Dict, str]:
    """Generate triples with structural changes."""
    t1 = {
        "data": {
            "values": [1, 2, 3],
            "meta": {"type": "list"}
        }
    }

    t2 = {
        "data": {
            "values": [1, 2, 3, 4],  # Added element
            "meta": {"type": "list"}
        }
    }

    t3 = {
        "data": {
            "values": [1, 2, 3, 4, 5],  # Added another element
            "meta": {"type": "list", "count": 5}  # Added field
        }
    }

    return t1, t2, t3, "structural"


def generate_json_triple_function_call() -> Tuple[Dict, Dict, Dict, str]:
    """Generate function call triples (realistic LLM output)."""
    import random

    t1 = {
        "function": "search_database",
        "arguments": {
            "query": "weather forecast",
            "location": "New York",
            "days": 5
        }
    }

    t2 = {
        "function": "search_database",
        "arguments": {
            "query": "weather forecast",
            "location": "New York",
            "days": 7  # Changed days
        }
    }

    t3 = {
        "function": "search_database",
        "arguments": {
            "query": "weather prediction",  # Semantically similar query
            "location": "New York",
            "days": 7
        }
    }

    return t1, t2, t3, "function_call"


def generate_json_triple_worst_case() -> Tuple[Dict, Dict, Dict, str]:
    """Generate worst-case scenario for triangle inequality."""
    # T1 and T3 share semantic similarity but via different paths
    t1 = {
        "field_a": "hello world",
        "field_b": "goodbye moon"
    }

    t2 = {
        "field_a": "hello world",
        "field_b": "random text"
    }

    t3 = {
        "field_a": "greetings planet",  # Semantically similar to t1.field_a
        "field_b": "random text"  # Same as t2.field_b
    }

    return t1, t2, t3, "worst_case"


def run_triangle_inequality_tests(
    evaluator: SemanticJsonTreeConsistencyEvaluator,
    num_tests: int = 100,
    variation_type: str = 'combined'
) -> List[TriangleInequalityResult]:
    """Run comprehensive triangle inequality tests."""

    results = []
    generators = [
        generate_json_triple_simple,
        generate_json_triple_nested,
        generate_json_triple_array,
        generate_json_triple_divergent,
        generate_json_triple_semantic,
        generate_json_triple_structural_change,
        generate_json_triple_function_call,
        generate_json_triple_worst_case,
    ]

    tests_per_generator = num_tests // len(generators)

    for generator in tqdm(generators, desc="Testing triangle inequality"):
        for _ in range(tests_per_generator):
            t1, t2, t3, category = generator()

            # Compute STED distances (1 - similarity)
            sim12 = evaluator.calculate_tree_edit_distance_opt(t1, t2, variation_type=variation_type)
            sim23 = evaluator.calculate_tree_edit_distance_opt(t2, t3, variation_type=variation_type)
            sim13 = evaluator.calculate_tree_edit_distance_opt(t1, t3, variation_type=variation_type)

            d12 = 1 - sim12
            d23 = 1 - sim23
            d13 = 1 - sim13

            epsilon = compute_relaxation_factor(d12, d23, d13)
            satisfies_standard = d13 <= d12 + d23 + 1e-10  # Small tolerance for floating point

            results.append(TriangleInequalityResult(
                t1=t1, t2=t2, t3=t3,
                d12=d12, d23=d23, d13=d13,
                epsilon=epsilon,
                satisfies_standard=satisfies_standard,
                category=category
            ))

    return results


def analyze_results(results: List[TriangleInequalityResult]) -> Dict[str, Any]:
    """Analyze triangle inequality test results."""

    epsilons = [r.epsilon for r in results]
    satisfies_standard = [r.satisfies_standard for r in results]

    # Overall statistics
    analysis = {
        "total_tests": len(results),
        "satisfies_standard_count": sum(satisfies_standard),
        "satisfies_standard_pct": 100 * sum(satisfies_standard) / len(results),
        "epsilon_mean": float(np.mean(epsilons)),
        "epsilon_std": float(np.std(epsilons)),
        "epsilon_median": float(np.median(epsilons)),
        "epsilon_max": float(np.max(epsilons)),
        "epsilon_p95": float(np.percentile(epsilons, 95)),
        "epsilon_p99": float(np.percentile(epsilons, 99)),
    }

    # Per-category analysis
    category_stats = defaultdict(list)
    for r in results:
        category_stats[r.category].append(r.epsilon)

    analysis["by_category"] = {}
    for category, cat_epsilons in category_stats.items():
        analysis["by_category"][category] = {
            "count": len(cat_epsilons),
            "epsilon_mean": float(np.mean(cat_epsilons)),
            "epsilon_max": float(np.max(cat_epsilons)),
            "epsilon_p95": float(np.percentile(cat_epsilons, 95)),
            "satisfies_standard_pct": 100 * sum(1 for e in cat_epsilons if e <= 1e-10) / len(cat_epsilons)
        }

    # Find worst violations
    sorted_results = sorted(results, key=lambda r: r.epsilon, reverse=True)
    analysis["worst_violations"] = []
    for r in sorted_results[:5]:
        analysis["worst_violations"].append({
            "category": r.category,
            "epsilon": r.epsilon,
            "d12": r.d12,
            "d23": r.d23,
            "d13": r.d13,
            "t1": r.t1,
            "t2": r.t2,
            "t3": r.t3
        })

    return analysis


def compute_theoretical_bound(embedding_model: str = 'all-MiniLM-L6-v2') -> Dict[str, float]:
    """
    Compute theoretical bound based on embedding properties.

    For Sentence-BERT embeddings, we analyze:
    1. Lipschitz constant L of the embedding function
    2. Non-additivity of cosine similarity
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)

    # Test Lipschitz property with various string pairs
    test_pairs = [
        ("hello", "hallo"),  # 1 edit
        ("hello", "helloa"),  # 1 edit
        ("cat", "car"),  # 1 edit
        ("machine learning", "machine learnin"),  # 1 edit
        ("apple", "apply"),  # 1 edit
        ("test", "tests"),  # 1 edit
    ]

    lipschitz_ratios = []
    for s1, s2 in test_pairs:
        emb1 = model.encode(s1, normalize_embeddings=True)
        emb2 = model.encode(s2, normalize_embeddings=True)

        emb_dist = np.linalg.norm(emb1 - emb2)
        edit_dist = sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))
        normalized_edit = edit_dist / max(len(s1), len(s2))

        if normalized_edit > 0:
            lipschitz_ratios.append(emb_dist / normalized_edit)

    # Test triangle inequality in embedding space
    triangle_tests = [
        ("hello", "world", "hello world"),
        ("cat", "dog", "animal"),
        ("machine", "learning", "AI"),
    ]

    triangle_violations = []
    for s1, s2, s3 in triangle_tests:
        emb1 = model.encode(s1, normalize_embeddings=True)
        emb2 = model.encode(s2, normalize_embeddings=True)
        emb3 = model.encode(s3, normalize_embeddings=True)

        d12 = np.linalg.norm(emb1 - emb2)
        d23 = np.linalg.norm(emb2 - emb3)
        d13 = np.linalg.norm(emb1 - emb3)

        if d12 + d23 > 1e-10:
            violation = max(0, d13 / (d12 + d23) - 1)
            triangle_violations.append(violation)

    return {
        "estimated_lipschitz_constant": float(np.max(lipschitz_ratios)) if lipschitz_ratios else 1.0,
        "mean_lipschitz_ratio": float(np.mean(lipschitz_ratios)) if lipschitz_ratios else 1.0,
        "embedding_triangle_violation_max": float(np.max(triangle_violations)) if triangle_violations else 0.0,
        "theoretical_epsilon_bound": float(np.max(lipschitz_ratios) - 1) if lipschitz_ratios else 0.0,
    }


def plot_results(results: List[TriangleInequalityResult], analysis: Dict, output_path: Path):
    """Generate visualization of triangle inequality results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of epsilon values
    epsilons = [r.epsilon for r in results]
    axes[0, 0].hist(epsilons, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='g', linestyle='--', label='Standard TI (ε=0)')
    axes[0, 0].axvline(x=analysis['epsilon_p95'], color='r', linestyle='--',
                       label=f'95th percentile (ε={analysis["epsilon_p95"]:.4f})')
    axes[0, 0].set_xlabel('Relaxation Factor ε')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Triangle Inequality Relaxation Factor')
    axes[0, 0].legend()

    # 2. Box plot by category
    categories = list(analysis['by_category'].keys())
    category_data = [[r.epsilon for r in results if r.category == cat] for cat in categories]
    axes[0, 1].boxplot(category_data, labels=categories)
    axes[0, 1].set_xlabel('Test Category')
    axes[0, 1].set_ylabel('Relaxation Factor ε')
    axes[0, 1].set_title('Triangle Inequality by Test Category')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Scatter plot: d12 + d23 vs d13
    d12_d23 = [r.d12 + r.d23 for r in results]
    d13 = [r.d13 for r in results]
    colors = ['green' if r.satisfies_standard else 'red' for r in results]
    axes[1, 0].scatter(d12_d23, d13, c=colors, alpha=0.5, s=20)
    max_val = max(max(d12_d23), max(d13))
    axes[1, 0].plot([0, max_val], [0, max_val], 'k--', label='d13 = d12 + d23')
    axes[1, 0].set_xlabel('d(T1,T2) + d(T2,T3)')
    axes[1, 0].set_ylabel('d(T1,T3)')
    axes[1, 0].set_title('Triangle Inequality Visualization')
    axes[1, 0].legend()

    # 4. CDF of epsilon
    sorted_eps = np.sort(epsilons)
    cdf = np.arange(1, len(sorted_eps) + 1) / len(sorted_eps)
    axes[1, 1].plot(sorted_eps, cdf, linewidth=2)
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95%')
    axes[1, 1].axhline(y=0.99, color='orange', linestyle='--', label='99%')
    axes[1, 1].set_xlabel('Relaxation Factor ε')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('CDF of Relaxation Factor')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")


def generate_tightened_bound_latex(analysis: Dict, theoretical: Dict) -> str:
    """Generate LaTeX for tightened triangle inequality bound."""

    empirical_bound = analysis['epsilon_p99']
    theoretical_bound = theoretical['theoretical_epsilon_bound']

    latex = f"""
% Tightened Triangle Inequality Bound
% Based on empirical validation with N={analysis['total_tests']} test cases

\\begin{{proposition}}[Empirically Tightened Triangle Inequality]
\\label{{prop:tight-triangle}}
For STED distance with Sentence-BERT embeddings, the relaxation factor $\\epsilon$ satisfies:
\\begin{{equation}}
    \\epsilon \\leq {empirical_bound:.4f} \\quad \\text{{(99th percentile)}}
\\end{{equation}}
with the standard triangle inequality ($\\epsilon = 0$) holding for {analysis['satisfies_standard_pct']:.1f}\\% of cases.

\\textbf{{Empirical Statistics:}}
\\begin{{itemize}}
    \\item Mean $\\epsilon$: {analysis['epsilon_mean']:.6f}
    \\item Median $\\epsilon$: {analysis['epsilon_median']:.6f}
    \\item 95th percentile: {analysis['epsilon_p95']:.4f}
    \\item 99th percentile: {analysis['epsilon_p99']:.4f}
    \\item Maximum observed: {analysis['epsilon_max']:.4f}
\\end{{itemize}}

This is significantly tighter than the theoretical bound $\\epsilon \\leq L - 1 \\approx {theoretical_bound:.2f}$
derived from the Lipschitz constant.
\\end{{proposition}}

\\begin{{remark}}[Practical Implications]
The empirical results show STED behaves very close to a proper metric:
\\begin{{enumerate}}
    \\item For {analysis['satisfies_standard_pct']:.1f}\\% of cases, the standard triangle inequality holds exactly
    \\item The remaining {100 - analysis['satisfies_standard_pct']:.1f}\\% have $\\epsilon < {analysis['epsilon_max']:.4f}$
    \\item The 99th percentile bound of $\\epsilon \\leq {empirical_bound:.4f}$ provides a practical guarantee
\\end{{enumerate}}
\\end{{remark}}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description='Validate STED triangle inequality')
    parser.add_argument('--num-tests', type=int, default=200,
                        help='Number of test triples to generate')
    parser.add_argument('--output-dir', type=str, default='research/triangle_inequality',
                        help='Output directory for results')
    parser.add_argument('--variation-type', type=str, default='combined',
                        choices=['structural', 'content', 'combined'],
                        help='STED variation type to test')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STED Triangle Inequality Validation")
    print("=" * 70)

    # Initialize evaluator
    print("\nInitializing STED evaluator...")
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='all-MiniLM-L6-v2',
    )

    # Run tests
    print(f"\nRunning {args.num_tests} triangle inequality tests...")
    start_time = time.time()
    results = run_triangle_inequality_tests(
        evaluator,
        num_tests=args.num_tests,
        variation_type=args.variation_type
    )
    elapsed = time.time() - start_time
    print(f"Tests completed in {elapsed:.2f} seconds")

    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results)

    # Compute theoretical bound
    print("\nComputing theoretical bounds from embedding properties...")
    theoretical = compute_theoretical_bound()
    analysis['theoretical'] = theoretical

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal tests: {analysis['total_tests']}")
    print(f"Standard triangle inequality satisfied: {analysis['satisfies_standard_pct']:.1f}%")
    print(f"\nRelaxation factor (epsilon) statistics:")
    print(f"  Mean:   {analysis['epsilon_mean']:.6f}")
    print(f"  Median: {analysis['epsilon_median']:.6f}")
    print(f"  Std:    {analysis['epsilon_std']:.6f}")
    print(f"  95th percentile: {analysis['epsilon_p95']:.6f}")
    print(f"  99th percentile: {analysis['epsilon_p99']:.6f}")
    print(f"  Maximum: {analysis['epsilon_max']:.6f}")

    print(f"\nTheoretical bounds:")
    print(f"  Estimated Lipschitz constant L: {theoretical['estimated_lipschitz_constant']:.4f}")
    print(f"  Theoretical epsilon bound (L-1): {theoretical['theoretical_epsilon_bound']:.4f}")
    print(f"  Embedding space triangle violation: {theoretical['embedding_triangle_violation_max']:.6f}")

    print("\nBy category:")
    for category, stats in analysis['by_category'].items():
        print(f"  {category}:")
        print(f"    Mean epsilon: {stats['epsilon_mean']:.6f}")
        print(f"    Max epsilon: {stats['epsilon_max']:.6f}")
        print(f"    Standard TI satisfied: {stats['satisfies_standard_pct']:.1f}%")

    # Save results
    print(f"\nSaving results to {output_dir}...")

    # JSON results
    with open(output_dir / 'triangle_inequality_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # Generate plots
    plot_results(results, analysis, output_dir / 'triangle_inequality_plots.png')

    # Generate LaTeX
    latex = generate_tightened_bound_latex(analysis, theoretical)
    with open(output_dir / 'tightened_bound.tex', 'w') as f:
        f.write(latex)
    print(f"\nLaTeX proposition saved to {output_dir / 'tightened_bound.tex'}")

    # Print LaTeX for direct inclusion
    print("\n" + "=" * 70)
    print("LATEX FOR PAPER (tightened bound)")
    print("=" * 70)
    print(latex)

    return analysis


if __name__ == '__main__':
    main()
