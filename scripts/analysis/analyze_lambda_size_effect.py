#!/usr/bin/env python3
"""
Analyze the effect of lambda_size parameter on STED similarity calculations.

This script tests how different lambda_size values (0.0 to 1.0) affect
the similarity scores for JSON pairs with varying size mismatches.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import json

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


def create_test_cases() -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Create test JSON pairs with different characteristics."""

    test_cases = []

    # Case 1: Identical structures (no size mismatch)
    test_cases.append((
        "Identical (5 vs 5 fields)",
        {
            "name": "John",
            "age": 30,
            "city": "NYC",
            "country": "USA",
            "email": "john@example.com"
        },
        {
            "name": "John",
            "age": 30,
            "city": "NYC",
            "country": "USA",
            "email": "john@example.com"
        }
    ))

    # Case 2: Small size mismatch (1 extra field)
    test_cases.append((
        "Small mismatch (5 vs 6 fields)",
        {
            "name": "John",
            "age": 30,
            "city": "NYC",
            "country": "USA",
            "email": "john@example.com"
        },
        {
            "name": "John",
            "age": 30,
            "city": "NYC",
            "country": "USA",
            "email": "john@example.com",
            "phone": "123-456-7890"
        }
    ))

    # Case 3: Medium size mismatch (3 extra fields)
    test_cases.append((
        "Medium mismatch (5 vs 8 fields)",
        {
            "name": "John",
            "age": 30,
            "city": "NYC",
            "country": "USA",
            "email": "john@example.com"
        },
        {
            "name": "John",
            "age": 30,
            "city": "NYC",
            "country": "USA",
            "email": "john@example.com",
            "phone": "123-456-7890",
            "occupation": "Engineer",
            "company": "TechCorp"
        }
    ))

    # Case 4: Large size mismatch (5 extra fields)
    test_cases.append((
        "Large mismatch (5 vs 10 fields)",
        {
            "name": "John",
            "age": 30,
            "city": "NYC",
            "country": "USA",
            "email": "john@example.com"
        },
        {
            "name": "John",
            "age": 30,
            "city": "NYC",
            "country": "USA",
            "email": "john@example.com",
            "phone": "123-456-7890",
            "occupation": "Engineer",
            "company": "TechCorp",
            "department": "R&D",
            "salary": 100000
        }
    ))

    # Case 5: Nested structure with size mismatch
    test_cases.append((
        "Nested mismatch (2 vs 4 nested)",
        {
            "user": {
                "name": "John",
                "profile": {
                    "age": 30
                }
            }
        },
        {
            "user": {
                "name": "John",
                "profile": {
                    "age": 30,
                    "bio": "Developer",
                    "skills": ["Python", "ML"],
                    "experience": 5
                }
            }
        }
    ))

    # Case 6: Array size mismatch
    test_cases.append((
        "Array mismatch (3 vs 6 items)",
        {
            "items": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
                {"id": 3, "name": "C"}
            ]
        },
        {
            "items": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
                {"id": 3, "name": "C"},
                {"id": 4, "name": "D"},
                {"id": 5, "name": "E"},
                {"id": 6, "name": "F"}
            ]
        }
    ))

    # Case 7: Completely different structures
    test_cases.append((
        "Different structures",
        {
            "type": "request",
            "action": "create",
            "data": {"id": 1}
        },
        {
            "status": "success",
            "result": {"created": True},
            "timestamp": "2024-01-01",
            "metadata": {"version": "1.0"}
        }
    ))

    # Case 8: Similar content, different keys
    test_cases.append((
        "Similar content, diff keys",
        {
            "user_name": "John",
            "user_age": 30,
            "user_email": "john@test.com"
        },
        {
            "userName": "John",
            "userAge": 30,
            "userEmail": "john@test.com"
        }
    ))

    return test_cases


def analyze_lambda_effect(
    lambda_values: List[float],
    test_cases: List[Tuple[str, Dict, Dict]]
) -> Dict[str, List[float]]:
    """
    Analyze how lambda_size affects similarity for each test case.

    Returns:
        Dictionary mapping test case names to list of similarity scores
    """
    results = {name: [] for name, _, _ in test_cases}

    for lambda_size in lambda_values:
        print(f"Testing lambda_size = {lambda_size:.2f}")

        # Create evaluator with specific lambda_size
        evaluator = SemanticJsonTreeConsistencyEvaluator(
            lambda_size=lambda_size,
            alpha=0.5  # Keep alpha fixed
        )

        for name, json1, json2 in test_cases:
            # Calculate similarity (STED already returns similarity score)
            similarity = evaluator.calculate_tree_edit_distance(
                json1, json2,
                original_zss=False,  # Use optimized Hungarian algorithm
                variation_type="combined"
            )
            results[name].append(similarity)

    return results


def plot_results(
    lambda_values: List[float],
    results: Dict[str, List[float]],
    output_path: str = None
):
    """Create visualization of lambda_size effect on similarity."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Effect of λ (lambda_size) on STED Similarity Scores', fontsize=14, fontweight='bold')

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot 1: All test cases
    ax1 = axes[0, 0]
    for (name, similarities), color in zip(results.items(), colors):
        ax1.plot(lambda_values, similarities, marker='o', markersize=4,
                label=name, color=color, linewidth=2)
    ax1.set_xlabel('λ (lambda_size)', fontsize=11)
    ax1.set_ylabel('Similarity Score', fontsize=11)
    ax1.set_title('All Test Cases', fontsize=12)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)

    # Plot 2: Size mismatch cases only
    ax2 = axes[0, 1]
    size_cases = ['Identical (5 vs 5 fields)', 'Small mismatch (5 vs 6 fields)',
                  'Medium mismatch (5 vs 8 fields)', 'Large mismatch (5 vs 10 fields)']
    for name in size_cases:
        if name in results:
            ax2.plot(lambda_values, results[name], marker='o', markersize=4,
                    label=name, linewidth=2)
    ax2.set_xlabel('λ (lambda_size)', fontsize=11)
    ax2.set_ylabel('Similarity Score', fontsize=11)
    ax2.set_title('Size Mismatch Comparison', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)

    # Plot 3: Similarity drop from lambda=0 to lambda=1
    ax3 = axes[1, 0]
    drops = []
    names = []
    for name, similarities in results.items():
        drop = similarities[0] - similarities[-1]  # Drop from λ=0 to λ=1
        drops.append(drop)
        names.append(name)

    bars = ax3.barh(range(len(names)), drops, color=colors)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=9)
    ax3.set_xlabel('Similarity Drop (λ=0.0 → λ=1.0)', fontsize=11)
    ax3.set_title('Impact of λ on Each Test Case', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for bar, drop in zip(bars, drops):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{drop:.3f}', va='center', fontsize=9)

    # Plot 4: Heatmap of similarities
    ax4 = axes[1, 1]
    similarity_matrix = np.array([results[name] for name, _, _ in
                                   [(n, None, None) for n in results.keys()]])

    im = ax4.imshow(similarity_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_xticks(np.arange(0, len(lambda_values), 2))
    ax4.set_xticklabels([f'{lambda_values[i]:.1f}' for i in range(0, len(lambda_values), 2)])
    ax4.set_yticks(range(len(results)))
    ax4.set_yticklabels(list(results.keys()), fontsize=8)
    ax4.set_xlabel('λ (lambda_size)', fontsize=11)
    ax4.set_title('Similarity Heatmap', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Similarity', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()


def print_summary(lambda_values: List[float], results: Dict[str, List[float]]):
    """Print summary statistics."""

    print("\n" + "="*80)
    print("SUMMARY: Effect of lambda_size on STED Similarity")
    print("="*80)

    # Key lambda values to show
    key_lambdas = [0.0, 0.1, 0.3, 0.5, 1.0]
    key_indices = [lambda_values.index(l) for l in key_lambdas if l in lambda_values]

    print(f"\n{'Test Case':<35} | " + " | ".join([f"λ={lambda_values[i]:.1f}" for i in key_indices]))
    print("-"*80)

    for name, similarities in results.items():
        values = " | ".join([f"{similarities[i]:.3f}" for i in key_indices])
        print(f"{name:<35} | {values}")

    print("\n" + "-"*80)
    print("Key Observations:")
    print("-"*80)

    # Find cases most affected by lambda
    max_drop = 0
    max_drop_case = ""
    min_drop = float('inf')
    min_drop_case = ""

    for name, similarities in results.items():
        drop = similarities[0] - similarities[-1]
        if drop > max_drop:
            max_drop = drop
            max_drop_case = name
        if drop < min_drop:
            min_drop = drop
            min_drop_case = name

    print(f"  - Most affected by λ: '{max_drop_case}' (drop: {max_drop:.3f})")
    print(f"  - Least affected by λ: '{min_drop_case}' (drop: {min_drop:.3f})")

    # Default lambda (0.1) analysis
    default_idx = lambda_values.index(0.1) if 0.1 in lambda_values else 1
    print(f"\n  - At default λ=0.1:")
    for name, similarities in results.items():
        print(f"      {name}: {similarities[default_idx]:.3f}")


def main():
    """Main function to run the analysis."""

    print("="*60)
    print("Lambda Size Effect Analysis")
    print("="*60)

    # Define lambda values to test
    lambda_values = [round(x * 0.1, 1) for x in range(11)]  # 0.0 to 1.0 in 0.1 steps
    print(f"\nTesting lambda_size values: {lambda_values}")

    # Create test cases
    test_cases = create_test_cases()
    print(f"Number of test cases: {len(test_cases)}")

    # Run analysis
    print("\nRunning similarity calculations...")
    results = analyze_lambda_effect(lambda_values, test_cases)

    # Print summary
    print_summary(lambda_values, results)

    # Create visualization
    output_path = Path(__file__).parent.parent.parent / "docs" / "ICML_paper" / "figures" / "lambda_size_effect.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualization...")
    plot_results(lambda_values, results, str(output_path))

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
