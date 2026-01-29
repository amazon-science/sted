#!/usr/bin/env python
"""
Analyze best-match selection human validation results.

Computes:
- Win Rate: % of times each method's pick was chosen by humans
- Mean Reciprocal Rank (MRR): Average 1/rank of each method's pick
- Accuracy@1: % of times method's pick matches human's top choice
- Statistical significance tests (chi-squared, bootstrap CI)
"""

import json
import os
import argparse
import csv
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy import stats
from datetime import datetime


def load_annotations(annotation_path: str) -> Dict[str, Dict]:
    """Load annotations from CSV or JSON file."""
    annotations = {}

    if annotation_path.endswith(".json"):
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
    elif annotation_path.endswith(".csv"):
        with open(annotation_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = row.get("item_id", "")
                if item_id and row.get("choice"):
                    annotations[item_id] = {
                        "choice": row.get("choice", ""),
                        "confidence": int(row["confidence"]) if row.get("confidence") else None,
                        "notes": row.get("notes", ""),
                    }
    else:
        raise ValueError(f"Unsupported annotation file format: {annotation_path}")

    return annotations


def load_method_mapping(mapping_path: str) -> Dict[str, Dict[str, str]]:
    """Load method -> label mapping from JSON file."""
    with open(mapping_path, "r") as f:
        return json.load(f)


def compute_win_rates(
    annotations: Dict[str, Dict],
    method_mapping: Dict[str, Dict[str, str]],
    methods: List[str] = ["sted", "deepdiff", "ted", "bertscore"],
) -> Dict[str, float]:
    """
    Compute win rate for each method.

    Win rate = % of times method's pick was chosen by human
    """
    wins = {m: 0 for m in methods}
    total = 0

    for item_id, ann in annotations.items():
        if not ann.get("choice") or item_id not in method_mapping:
            continue

        human_choice = ann["choice"]
        picks = method_mapping[item_id]

        for method in methods:
            if method in picks and picks[method] == human_choice:
                wins[method] += 1

        total += 1

    win_rates = {m: wins[m] / total if total > 0 else 0 for m in methods}
    return win_rates, wins, total


def compute_mrr(
    annotations: Dict[str, Dict],
    method_mapping: Dict[str, Dict[str, str]],
    methods: List[str] = ["sted", "deepdiff", "ted", "bertscore"],
) -> Dict[str, float]:
    """
    Compute Mean Reciprocal Rank for each method.

    MRR = average(1/rank) where rank is human's ranking of method's pick
    For best-match (single choice), rank is 1 if method's pick = human's choice, else infinity (0 reciprocal)
    """
    reciprocal_ranks = {m: [] for m in methods}

    for item_id, ann in annotations.items():
        if not ann.get("choice") or item_id not in method_mapping:
            continue

        human_choice = ann["choice"]
        picks = method_mapping[item_id]

        for method in methods:
            if method in picks:
                if picks[method] == human_choice:
                    reciprocal_ranks[method].append(1.0)  # Rank 1 -> RR = 1.0
                else:
                    reciprocal_ranks[method].append(0.0)  # Not chosen -> RR = 0

    mrr = {m: np.mean(rrs) if rrs else 0 for m, rrs in reciprocal_ranks.items()}
    return mrr


def compute_accuracy_at_1(
    annotations: Dict[str, Dict],
    method_mapping: Dict[str, Dict[str, str]],
    methods: List[str] = ["sted", "deepdiff", "ted", "bertscore"],
) -> Dict[str, float]:
    """
    Compute Accuracy@1 for each method.

    Same as win rate for single-choice tasks.
    """
    return compute_win_rates(annotations, method_mapping, methods)[0]


def bootstrap_confidence_interval(
    annotations: Dict[str, Dict],
    method_mapping: Dict[str, Dict[str, str]],
    methods: List[str],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for win rates.
    """
    # Get per-item outcomes
    item_outcomes = defaultdict(list)  # method -> list of 0/1 outcomes

    for item_id, ann in annotations.items():
        if not ann.get("choice") or item_id not in method_mapping:
            continue

        human_choice = ann["choice"]
        picks = method_mapping[item_id]

        for method in methods:
            if method in picks:
                outcome = 1 if picks[method] == human_choice else 0
                item_outcomes[method].append(outcome)

    # Bootstrap
    cis = {}
    for method in methods:
        outcomes = np.array(item_outcomes[method])
        if len(outcomes) == 0:
            cis[method] = (0.0, 0.0)
            continue

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(outcomes, size=len(outcomes), replace=True)
            bootstrap_means.append(np.mean(sample))

        lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
        cis[method] = (lower, upper)

    return cis


def chi_squared_test(
    annotations: Dict[str, Dict],
    method_mapping: Dict[str, Dict[str, str]],
    methods: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Perform chi-squared tests comparing each method pair.

    Returns dict of {method1: {method2: p_value}}
    """
    # Get win counts
    _, wins, total = compute_win_rates(annotations, method_mapping, methods)

    results = {}
    for m1 in methods:
        results[m1] = {}
        for m2 in methods:
            if m1 == m2:
                results[m1][m2] = 1.0
                continue

            # 2x2 contingency: m1 wins vs m2 wins
            # This is a simplified test; more rigorous would use McNemar's test
            observed = [wins[m1], wins[m2]]
            expected = [total / 2, total / 2]  # Expected under null (equal performance)

            # Chi-squared test
            if sum(observed) > 0:
                chi2, p_value = stats.chisquare(observed, expected)
                results[m1][m2] = p_value
            else:
                results[m1][m2] = 1.0

    return results


def mcnemar_test(
    annotations: Dict[str, Dict],
    method_mapping: Dict[str, Dict[str, str]],
    method1: str,
    method2: str,
) -> float:
    """
    Perform McNemar's test comparing two methods.

    More appropriate for paired comparisons than chi-squared.
    """
    # Count discordant pairs
    n_01 = 0  # method1 wrong, method2 correct
    n_10 = 0  # method1 correct, method2 wrong

    for item_id, ann in annotations.items():
        if not ann.get("choice") or item_id not in method_mapping:
            continue

        human_choice = ann["choice"]
        picks = method_mapping[item_id]

        m1_correct = picks.get(method1) == human_choice
        m2_correct = picks.get(method2) == human_choice

        if m1_correct and not m2_correct:
            n_10 += 1
        elif not m1_correct and m2_correct:
            n_01 += 1

    # McNemar's test (with continuity correction)
    if n_01 + n_10 == 0:
        return 1.0

    chi2 = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return p_value


def analyze_by_confidence(
    annotations: Dict[str, Dict],
    method_mapping: Dict[str, Dict[str, str]],
    methods: List[str],
) -> Dict[int, Dict[str, float]]:
    """
    Analyze win rates by annotator confidence level.
    """
    results = {}

    for conf_level in [1, 2, 3, 4, 5]:
        # Filter to items with this confidence
        filtered_ann = {
            k: v for k, v in annotations.items()
            if v.get("confidence") == conf_level
        }

        if len(filtered_ann) >= 5:  # Need minimum samples
            win_rates, _, _ = compute_win_rates(filtered_ann, method_mapping, methods)
            results[conf_level] = win_rates

    return results


def generate_latex_table(
    win_rates: Dict[str, float],
    cis: Dict[str, Tuple[float, float]],
    p_values: Dict[str, float],  # p-values vs STED
) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Best-Match Selection Study Results}",
        r"\label{tab:best-match-results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Win Rate & 95\% CI & p-value vs STED \\",
        r"\midrule",
    ]

    methods_order = ["sted", "bertscore", "deepdiff", "ted"]
    for method in methods_order:
        wr = win_rates.get(method, 0)
        ci = cis.get(method, (0, 0))
        p = p_values.get(method, 1.0)

        # Format
        wr_str = f"{wr*100:.1f}\\%"
        ci_str = f"[{ci[0]*100:.1f}\\%, {ci[1]*100:.1f}\\%]"

        if method == "sted":
            p_str = "-"
            wr_str = f"\\textbf{{{wr_str}}}"
        else:
            if p < 0.001:
                p_str = "$<$0.001***"
            elif p < 0.01:
                p_str = f"{p:.3f}**"
            elif p < 0.05:
                p_str = f"{p:.3f}*"
            else:
                p_str = f"{p:.3f}"

        method_display = method.upper() if method != "bertscore" else "BERTScore"
        lines.append(f"{method_display} & {wr_str} & {ci_str} & {p_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def analyze_best_match_results(
    dataset_path: str,
    annotations_path: str,
    output_path: str,
    n_bootstrap: int = 1000,
):
    """Main analysis function."""

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Try to find method mapping
    method_mapping_path = os.path.join(
        os.path.dirname(annotations_path),
        "_method_mapping.json"
    )

    if os.path.exists(method_mapping_path):
        print(f"Loading method mapping from {method_mapping_path}...")
        method_mapping = load_method_mapping(method_mapping_path)
    else:
        # Extract from dataset
        print("Extracting method mapping from dataset...")
        method_mapping = {}
        for item in dataset.get("items", dataset.get("samples", [])):
            item_id = item.get("id", "")
            method_picks = item.get("metadata", {}).get("method_picks", {})
            if method_picks:
                method_mapping[item_id] = method_picks

    print(f"Loading annotations from {annotations_path}...")
    annotations = load_annotations(annotations_path)
    print(f"Loaded {len(annotations)} annotations")

    methods = ["sted", "bertscore", "deepdiff", "ted"]

    # Compute metrics
    print("\nComputing metrics...")
    win_rates, wins, total = compute_win_rates(annotations, method_mapping, methods)
    mrr = compute_mrr(annotations, method_mapping, methods)
    cis = bootstrap_confidence_interval(annotations, method_mapping, methods, n_bootstrap)

    # Statistical tests
    print("Running statistical tests...")
    p_values_vs_sted = {}
    for method in methods:
        if method != "sted":
            p = mcnemar_test(annotations, method_mapping, "sted", method)
            p_values_vs_sted[method] = p

    # Analyze by confidence
    conf_results = analyze_by_confidence(annotations, method_mapping, methods)

    # Generate report
    report = []
    report.append("=" * 60)
    report.append("BEST-MATCH SELECTION STUDY RESULTS")
    report.append("=" * 60)
    report.append(f"\nAnalysis date: {datetime.now().isoformat()}")
    report.append(f"Total annotated samples: {total}")
    report.append(f"Total annotations loaded: {len(annotations)}")

    report.append("\n" + "-" * 60)
    report.append("WIN RATES")
    report.append("-" * 60)
    for method in methods:
        wr = win_rates[method]
        ci = cis[method]
        report.append(f"  {method.upper():12s}: {wr*100:5.1f}% (95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%])")

    report.append("\n" + "-" * 60)
    report.append("MEAN RECIPROCAL RANK (MRR)")
    report.append("-" * 60)
    for method in methods:
        report.append(f"  {method.upper():12s}: {mrr[method]:.3f}")

    report.append("\n" + "-" * 60)
    report.append("STATISTICAL SIGNIFICANCE (McNemar's test vs STED)")
    report.append("-" * 60)
    for method, p in p_values_vs_sted.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        report.append(f"  STED vs {method.upper():10s}: p = {p:.4f} {sig}")

    if conf_results:
        report.append("\n" + "-" * 60)
        report.append("WIN RATES BY CONFIDENCE LEVEL")
        report.append("-" * 60)
        for conf, rates in sorted(conf_results.items()):
            n_items = sum(1 for a in annotations.values() if a.get("confidence") == conf)
            report.append(f"\n  Confidence {conf} (n={n_items}):")
            for method in methods:
                report.append(f"    {method.upper():12s}: {rates.get(method, 0)*100:5.1f}%")

    report.append("\n" + "-" * 60)
    report.append("RAW WIN COUNTS")
    report.append("-" * 60)
    for method in methods:
        report.append(f"  {method.upper():12s}: {wins[method]:4d} / {total}")

    # LaTeX table
    report.append("\n" + "=" * 60)
    report.append("LATEX TABLE")
    report.append("=" * 60)
    latex = generate_latex_table(win_rates, cis, p_values_vs_sted)
    report.append(latex)

    # Write report
    report_text = "\n".join(report)
    print(report_text)

    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to {output_path}")

    # Also save structured results as JSON
    json_output = output_path.replace(".txt", "_results.json")
    results_data = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "total_samples": total,
            "n_bootstrap": n_bootstrap,
        },
        "win_rates": win_rates,
        "mrr": mrr,
        "confidence_intervals_95": {m: list(ci) for m, ci in cis.items()},
        "p_values_vs_sted": p_values_vs_sted,
        "win_counts": wins,
        "by_confidence": conf_results,
    }

    with open(json_output, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Structured results saved to {json_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze best-match selection human validation results"
    )
    parser.add_argument(
        "--dataset",
        default="ranking_validation_dataset.json",
        help="Original dataset JSON file",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Annotations file (CSV or JSON)",
    )
    parser.add_argument(
        "--output",
        default="best_match_report.txt",
        help="Output report file",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI (default: 1000)",
    )

    args = parser.parse_args()

    analyze_best_match_results(
        dataset_path=args.dataset,
        annotations_path=args.annotations,
        output_path=args.output,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
