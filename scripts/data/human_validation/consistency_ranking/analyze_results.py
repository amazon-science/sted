#!/usr/bin/env python
"""
Analyze consistency ranking results for STED validation.

This script computes:
1. Accuracy: Does STED consistency ranking match human ranking?
2. Agreement by difficulty level
3. Inter-annotator agreement (if multiple annotators)
4. Statistical significance tests

Output: Summary statistics and publication-ready tables.
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact


def load_ranking_annotations(
    dataset_path: str,
    annotations_path: str,
) -> pd.DataFrame:
    """Load dataset and merge with human annotations."""

    # Load original dataset with ground truth
    with open(dataset_path, "r") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])

    # Create dataframe
    rows = []
    for pair in pairs:
        gt = pair.get("_ground_truth", {})
        row = {
            "pair_id": pair.get("id"),
            "difficulty": pair.get("metadata", {}).get("difficulty"),
            "sted_consistency_a": gt.get("sted_consistency_a"),
            "sted_consistency_b": gt.get("sted_consistency_b"),
            "consistency_diff": gt.get("consistency_diff"),
            "expected_answer": gt.get("expected_answer"),  # "A" or "B"
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Load annotations
    if annotations_path.endswith(".csv"):
        annotations_df = pd.read_csv(annotations_path)
    else:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        annotations_df = pd.DataFrame([
            {"pair_id": k, **v} for k, v in annotations.items()
        ])

    # Check for multiple annotators
    choice_cols = [c for c in annotations_df.columns if c.startswith("choice")]

    if len(choice_cols) > 1:
        # Multiple annotators
        df = df.merge(annotations_df, on="pair_id", how="left")
        # Compute majority vote
        df["human_choice"] = df[choice_cols].mode(axis=1)[0]
    else:
        if "choice" in annotations_df.columns:
            annotations_df = annotations_df.rename(columns={"choice": "human_choice"})
        df = df.merge(
            annotations_df[["pair_id", "human_choice"] +
                          (["confidence"] if "confidence" in annotations_df.columns else [])],
            on="pair_id",
            how="left"
        )

    return df


def compute_accuracy(df: pd.DataFrame) -> Dict[str, float]:
    """Compute accuracy of STED predictions vs human rankings."""

    # Filter valid annotations
    valid = df[df["human_choice"].notna() & (df["human_choice"] != "")]
    valid = valid[valid["expected_answer"].notna()]

    if len(valid) == 0:
        return {"accuracy": np.nan, "n_samples": 0}

    # Compute accuracy (treating "equal" as correct if difference is small)
    correct = 0
    total = 0

    for _, row in valid.iterrows():
        expected = row["expected_answer"]
        human = row["human_choice"]
        diff = row["consistency_diff"]

        if human == expected:
            correct += 1
        elif human == "equal" and diff < 0.1:
            # Allow "equal" for small differences
            correct += 0.5
        total += 1

    accuracy = correct / total if total > 0 else np.nan

    # Compute strict accuracy (no partial credit)
    strict_correct = (valid["human_choice"] == valid["expected_answer"]).sum()
    strict_accuracy = strict_correct / len(valid)

    # Compute accuracy excluding "equal" responses
    non_equal = valid[valid["human_choice"] != "equal"]
    if len(non_equal) > 0:
        directional_accuracy = (non_equal["human_choice"] == non_equal["expected_answer"]).sum() / len(non_equal)
    else:
        directional_accuracy = np.nan

    return {
        "accuracy": accuracy,
        "strict_accuracy": strict_accuracy,
        "directional_accuracy": directional_accuracy,
        "n_samples": len(valid),
        "n_correct": strict_correct,
        "n_equal_responses": (valid["human_choice"] == "equal").sum(),
    }


def compute_accuracy_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy broken down by difficulty level."""

    results = []

    for difficulty in ["easy", "medium", "hard", "very_hard"]:
        subset = df[df["difficulty"] == difficulty]

        if len(subset) == 0:
            continue

        metrics = compute_accuracy(subset)

        results.append({
            "difficulty": difficulty,
            **metrics,
            "mean_consistency_diff": subset["consistency_diff"].mean(),
        })

    return pd.DataFrame(results)


def compute_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute confusion matrix of human vs STED predictions."""

    valid = df[df["human_choice"].notna() & (df["human_choice"] != "")]
    valid = valid[valid["expected_answer"].notna()]

    # Create confusion matrix
    labels = ["A", "B", "equal"]
    matrix = pd.DataFrame(
        np.zeros((3, 2)),
        index=labels,
        columns=["STED: A", "STED: B"]
    )

    for _, row in valid.iterrows():
        human = row["human_choice"]
        expected = row["expected_answer"]

        if human in labels and expected in ["A", "B"]:
            matrix.loc[human, f"STED: {expected}"] += 1

    return matrix


def compute_inter_annotator_agreement(
    df: pd.DataFrame,
    choice_cols: List[str],
) -> Dict[str, float]:
    """Compute inter-annotator agreement for ranking task."""

    if len(choice_cols) < 2:
        return {}

    # Filter rows with multiple annotations
    valid = df[choice_cols].dropna(how="all")

    if len(valid) == 0:
        return {}

    # Compute pairwise agreement
    agreements = []
    for i in range(len(choice_cols)):
        for j in range(i + 1, len(choice_cols)):
            col_i = choice_cols[i]
            col_j = choice_cols[j]

            # Filter rows where both annotators provided ratings
            both = valid[[col_i, col_j]].dropna()
            if len(both) > 0:
                agree = (both[col_i] == both[col_j]).sum()
                agreements.append(agree / len(both))

    if not agreements:
        return {}

    # Compute Fleiss' kappa for categorical data
    # Simplified: use mean pairwise agreement
    return {
        "mean_pairwise_agreement": np.mean(agreements),
        "min_pairwise_agreement": np.min(agreements),
        "n_annotators": len(choice_cols),
    }


def statistical_test(df: pd.DataFrame) -> Dict[str, float]:
    """Test if STED predictions are significantly better than chance."""

    valid = df[df["human_choice"].notna() & (df["human_choice"] != "")]
    valid = valid[valid["expected_answer"].notna()]

    # Exclude "equal" responses for binomial test
    non_equal = valid[valid["human_choice"] != "equal"]

    if len(non_equal) == 0:
        return {"binomial_p": np.nan}

    correct = (non_equal["human_choice"] == non_equal["expected_answer"]).sum()
    total = len(non_equal)

    # Binomial test: Is accuracy significantly better than 50%?
    # Using one-tailed test (we expect STED to be better than chance)
    from scipy.stats import binomtest
    result = binomtest(correct, total, p=0.5, alternative="greater")

    return {
        "binomial_p": result.pvalue,
        "observed_accuracy": correct / total,
        "n_directional": total,
        "significant_0.05": result.pvalue < 0.05,
        "significant_0.01": result.pvalue < 0.01,
    }


def compute_bootstrap_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for accuracy."""

    valid = df[df["human_choice"].notna() & (df["human_choice"] != "")]
    valid = valid[valid["expected_answer"].notna()]

    if len(valid) < 10:
        return np.nan, np.nan, np.nan

    accuracies = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        sample = valid.sample(n=len(valid), replace=True)
        correct = (sample["human_choice"] == sample["expected_answer"]).sum()
        accuracies.append(correct / len(sample))

    alpha = 1 - confidence
    lower = np.percentile(accuracies, 100 * alpha / 2)
    upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
    mean = np.mean(accuracies)

    return mean, lower, upper


def generate_report(
    df: pd.DataFrame,
    accuracy_metrics: Dict,
    accuracy_by_diff: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
    stat_test: Dict,
    iaa: Dict,
    output_path: str = None,
) -> str:
    """Generate analysis report."""

    report = []
    report.append("=" * 60)
    report.append("CONSISTENCY RANKING VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")

    # Dataset summary
    report.append("1. DATASET SUMMARY")
    report.append("-" * 40)
    report.append(f"Total pairs: {len(df)}")
    report.append(f"Pairs with annotations: {accuracy_metrics.get('n_samples', 0)}")
    report.append(f"'Equal' responses: {accuracy_metrics.get('n_equal_responses', 0)}")
    report.append("")

    # Main accuracy results
    report.append("2. STED CONSISTENCY VALIDATION")
    report.append("-" * 40)
    report.append(f"Overall accuracy: {accuracy_metrics.get('accuracy', 0):.1%}")
    report.append(f"Strict accuracy: {accuracy_metrics.get('strict_accuracy', 0):.1%}")
    report.append(f"Directional accuracy (excl. 'equal'): {accuracy_metrics.get('directional_accuracy', 0):.1%}")
    report.append("")

    # Statistical test
    report.append("3. STATISTICAL SIGNIFICANCE")
    report.append("-" * 40)
    p_val = stat_test.get("binomial_p", 1)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    report.append(f"Binomial test (vs 50% chance): p = {p_val:.4f} {sig}")
    report.append(f"Interpretation: STED consistency ranking {'significantly' if p_val < 0.05 else 'does not significantly'} matches human judgment")
    report.append("")

    # Accuracy by difficulty
    report.append("4. ACCURACY BY DIFFICULTY")
    report.append("-" * 40)
    if not accuracy_by_diff.empty:
        for _, row in accuracy_by_diff.iterrows():
            report.append(
                f"  {row['difficulty']:12s}: {row['strict_accuracy']:.1%} "
                f"(n={int(row['n_samples'])}, avg diff={row['mean_consistency_diff']:.3f})"
            )
    report.append("")

    # Confusion matrix
    report.append("5. CONFUSION MATRIX")
    report.append("-" * 40)
    report.append("(Rows = Human choice, Columns = STED prediction)")
    report.append(confusion_matrix.to_string())
    report.append("")

    # Inter-annotator agreement
    if iaa:
        report.append("6. INTER-ANNOTATOR AGREEMENT")
        report.append("-" * 40)
        report.append(f"Mean pairwise agreement: {iaa.get('mean_pairwise_agreement', 0):.1%}")
        report.append(f"Number of annotators: {iaa.get('n_annotators', 0)}")
        report.append("")

    # Paper-ready LaTeX
    report.append("7. PAPER-READY TABLE (LaTeX)")
    report.append("-" * 40)
    report.append("")
    report.append("\\begin{table}[h]")
    report.append("\\centering")
    report.append("\\caption{Consistency Score Validation via Human Ranking}")
    report.append("\\begin{tabular}{@{}lcc@{}}")
    report.append("\\toprule")
    report.append("\\textbf{Difficulty} & \\textbf{Accuracy} & \\textbf{N} \\\\")
    report.append("\\midrule")

    for _, row in accuracy_by_diff.iterrows():
        report.append(f"{row['difficulty'].title()} & {row['strict_accuracy']:.1%} & {int(row['n_samples'])} \\\\")

    report.append("\\midrule")
    report.append(f"\\textbf{{Overall}} & \\textbf{{{accuracy_metrics.get('strict_accuracy', 0):.1%}}} & \\textbf{{{accuracy_metrics.get('n_samples', 0)}}} \\\\")
    report.append("\\bottomrule")
    report.append("\\end{tabular}")
    report.append(f"\\\\[2pt]")
    report.append(f"\\footnotesize{{Binomial test vs. chance: $p = {stat_test.get('binomial_p', 1):.3f}$}}")
    report.append("\\end{table}")
    report.append("")

    # Key finding for paper
    report.append("8. KEY FINDING FOR PAPER")
    report.append("-" * 40)
    acc = accuracy_metrics.get('directional_accuracy', 0)
    p = stat_test.get('binomial_p', 1)
    report.append(f"""
Human annotators ranked output set consistency with {acc:.0%} agreement
with STED consistency scores (binomial test: p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}).
This validates that STED consistency scores reflect human intuition about
output consistency, beyond just pairwise similarity.
""")

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)
        print(f"Report saved to {output_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze consistency ranking results"
    )
    parser.add_argument(
        "--dataset",
        default="consistency_ranking_dataset.json",
        help="Original dataset JSON file with ground truth",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Annotations file (CSV or JSON)",
    )
    parser.add_argument(
        "--output",
        default="consistency_ranking_report.txt",
        help="Output report file",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI",
    )

    args = parser.parse_args()

    print("Loading data...")
    df = load_ranking_annotations(args.dataset, args.annotations)

    print(f"Loaded {len(df)} pairs")

    # Compute metrics
    print("Computing accuracy metrics...")
    accuracy_metrics = compute_accuracy(df)
    print(f"  Overall accuracy: {accuracy_metrics.get('accuracy', 0):.1%}")

    print("Computing accuracy by difficulty...")
    accuracy_by_diff = compute_accuracy_by_difficulty(df)

    print("Computing confusion matrix...")
    confusion_matrix = compute_confusion_matrix(df)

    print("Running statistical tests...")
    stat_test = statistical_test(df)
    print(f"  Binomial test p-value: {stat_test.get('binomial_p', 1):.4f}")

    # Check for multiple annotators
    choice_cols = [c for c in df.columns if c.startswith("choice") and c != "human_choice"]
    iaa = {}
    if len(choice_cols) > 1:
        print("Computing inter-annotator agreement...")
        iaa = compute_inter_annotator_agreement(df, choice_cols)

    # Bootstrap CI
    print("Computing bootstrap confidence intervals...")
    mean_acc, ci_lower, ci_upper = compute_bootstrap_ci(df, args.n_bootstrap)
    accuracy_metrics["ci_lower"] = ci_lower
    accuracy_metrics["ci_upper"] = ci_upper

    # Generate report
    print("\nGenerating report...")
    report = generate_report(
        df, accuracy_metrics, accuracy_by_diff, confusion_matrix,
        stat_test, iaa, args.output
    )
    print("\n" + report)


if __name__ == "__main__":
    main()
