#!/usr/bin/env python
"""
Analyze human validation results for STED.

This script computes:
1. Inter-annotator agreement (Krippendorff's alpha, Fleiss' kappa)
2. Correlation between STED and human ratings (Spearman, Pearson, Kendall)
3. Comparison of STED vs baselines (BERTScore, DeepDiff, TED)
4. Statistical significance tests (Williams' test for comparing correlations)

Output: Summary statistics and publication-ready tables.
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats
from scipy.stats import spearmanr, pearsonr, kendalltau, wilcoxon
import warnings


def load_annotations(
    dataset_path: str,
    annotations_path: str,
) -> pd.DataFrame:
    """Load dataset and merge with human annotations."""

    # Load original dataset with STED scores
    with open(dataset_path, "r") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])

    # Create dataframe from pairs
    rows = []
    for pair in pairs:
        row = {
            "pair_id": pair.get("id"),
            "source": pair.get("metadata", {}).get("source"),
            "sted_score": pair.get("metadata", {}).get("sted_score"),
        }
        # Add baseline scores
        all_scores = pair.get("metadata", {}).get("all_scores", {})
        row["sted_combined"] = all_scores.get("sted_combined")
        row["sted_structural"] = all_scores.get("sted_structural")
        row["sted_semantic"] = all_scores.get("sted_semantic")
        row["deepdiff"] = all_scores.get("deepdiff")
        row["ted"] = all_scores.get("ted")
        row["bertscore"] = all_scores.get("bertscore")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Load annotations (CSV format: pair_id, rating, notes OR multiple annotators)
    if annotations_path.endswith(".csv"):
        annotations_df = pd.read_csv(annotations_path)
    else:
        # JSON format from annotation interface
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        annotations_df = pd.DataFrame([
            {"pair_id": k, **v} for k, v in annotations.items()
        ])

    # Check if multiple annotators (columns like rating_1, rating_2, etc.)
    rating_cols = [c for c in annotations_df.columns if c.startswith("rating")]

    if len(rating_cols) > 1:
        # Multiple annotators - compute mean and track individual ratings
        df = df.merge(annotations_df, on="pair_id", how="left")
        df["human_rating"] = df[rating_cols].mean(axis=1)
        df["rating_std"] = df[rating_cols].std(axis=1)
    else:
        # Single annotator or aggregated
        if "rating" in annotations_df.columns:
            annotations_df = annotations_df.rename(columns={"rating": "human_rating"})
        df = df.merge(annotations_df[["pair_id", "human_rating"]], on="pair_id", how="left")

    return df


def compute_krippendorff_alpha(
    ratings_matrix: np.ndarray,
    level: str = "ordinal",
) -> float:
    """
    Compute Krippendorff's alpha for inter-annotator agreement.

    Args:
        ratings_matrix: (n_items, n_annotators) matrix, NaN for missing
        level: "nominal", "ordinal", "interval", or "ratio"
    """
    # Filter out items with fewer than 2 ratings
    valid_mask = np.sum(~np.isnan(ratings_matrix), axis=1) >= 2
    ratings = ratings_matrix[valid_mask]

    if len(ratings) == 0:
        return np.nan

    n_items, n_annotators = ratings.shape

    # Flatten to get all observed values
    all_values = ratings[~np.isnan(ratings)]
    if len(all_values) == 0:
        return np.nan

    # Get unique values
    unique_values = np.unique(all_values)
    n_values = len(unique_values)

    # Compute coincidence matrix
    coincidence = np.zeros((n_values, n_values))

    for item in ratings:
        valid_ratings = item[~np.isnan(item)]
        n_valid = len(valid_ratings)
        if n_valid < 2:
            continue

        for i, v1 in enumerate(valid_ratings):
            for j, v2 in enumerate(valid_ratings):
                if i != j:
                    idx1 = np.where(unique_values == v1)[0][0]
                    idx2 = np.where(unique_values == v2)[0][0]
                    coincidence[idx1, idx2] += 1.0 / (n_valid - 1)

    # Compute observed disagreement
    if level == "nominal":
        # 0 if same, 1 if different
        delta = 1 - np.eye(n_values)
    elif level == "ordinal":
        # Squared rank difference
        ranks = np.arange(n_values)
        delta = (ranks[:, None] - ranks[None, :]) ** 2
    elif level == "interval":
        # Squared value difference
        delta = (unique_values[:, None] - unique_values[None, :]) ** 2
    else:  # ratio
        delta = (unique_values[:, None] - unique_values[None, :]) ** 2 / (
            (unique_values[:, None] + unique_values[None, :]) ** 2 + 1e-10
        )

    # Normalize delta
    delta = delta / (delta.max() + 1e-10)

    # Observed disagreement
    n_total = coincidence.sum()
    if n_total == 0:
        return np.nan

    D_o = np.sum(coincidence * delta) / n_total

    # Expected disagreement
    marginals = coincidence.sum(axis=1)
    D_e = np.sum(marginals[:, None] * marginals[None, :] * delta) / (n_total * (n_total - 1) + 1e-10)

    if D_e == 0:
        return 1.0 if D_o == 0 else 0.0

    alpha = 1 - D_o / D_e
    return alpha


def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' kappa for multiple annotators.

    Args:
        ratings_matrix: (n_items, n_annotators) matrix with integer ratings
    """
    # Remove items with missing ratings
    valid_mask = ~np.any(np.isnan(ratings_matrix), axis=1)
    ratings = ratings_matrix[valid_mask].astype(int)

    if len(ratings) == 0:
        return np.nan

    n_items, n_annotators = ratings.shape

    # Get unique categories
    categories = np.unique(ratings)
    n_categories = len(categories)

    # Build count matrix: (n_items, n_categories)
    counts = np.zeros((n_items, n_categories))
    for i, cat in enumerate(categories):
        counts[:, i] = np.sum(ratings == cat, axis=1)

    # Proportion of all assignments to each category
    p_j = counts.sum(axis=0) / (n_items * n_annotators)

    # Agreement for each item
    P_i = (np.sum(counts ** 2, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))

    # Mean agreement
    P_bar = P_i.mean()

    # Expected agreement by chance
    P_e = np.sum(p_j ** 2)

    if P_e == 1:
        return 1.0 if P_bar == 1 else 0.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def compute_correlations(
    df: pd.DataFrame,
    method_cols: List[str],
    human_col: str = "human_rating",
) -> Dict[str, Dict[str, float]]:
    """Compute correlations between methods and human ratings."""

    results = {}

    for method in method_cols:
        if method not in df.columns:
            continue

        # Filter valid pairs
        valid = df[[method, human_col]].dropna()
        if len(valid) < 10:
            continue

        x = valid[method].values
        y = valid[human_col].values

        # Normalize to same scale if needed
        # Human ratings: 1-5, STED: 0-1
        # Convert human ratings to 0-1 scale
        y_normalized = (y - 1) / 4  # 1->0, 5->1

        results[method] = {
            "spearman_rho": spearmanr(x, y_normalized)[0],
            "spearman_p": spearmanr(x, y_normalized)[1],
            "pearson_r": pearsonr(x, y_normalized)[0],
            "pearson_p": pearsonr(x, y_normalized)[1],
            "kendall_tau": kendalltau(x, y_normalized)[0],
            "kendall_p": kendalltau(x, y_normalized)[1],
            "n_samples": len(valid),
        }

    return results


def williams_test(
    r12: float, r13: float, r23: float, n: int
) -> Tuple[float, float]:
    """
    Williams' test for comparing two dependent correlations.

    Tests H0: rho(method1, human) = rho(method2, human)

    Args:
        r12: correlation between method1 and human
        r13: correlation between method2 and human
        r23: correlation between method1 and method2
        n: sample size

    Returns:
        t_stat: test statistic
        p_value: two-tailed p-value
    """
    if n < 4:
        return np.nan, np.nan

    # Williams' formula
    r_bar = (r12 + r13) / 2

    numerator = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))

    denominator_term1 = 2 * ((n - 1) / (n - 3)) * (1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23)
    denominator_term2 = (r_bar**2) * (1 - r23)**3

    denominator = np.sqrt(denominator_term1 + denominator_term2)

    if denominator == 0:
        return np.nan, np.nan

    t_stat = numerator / denominator

    # Two-tailed p-value with n-3 degrees of freedom
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-3))

    return t_stat, p_value


def compare_correlations(
    df: pd.DataFrame,
    baseline_methods: List[str] = ["deepdiff", "ted", "bertscore"],
    target_method: str = "sted_combined",
    human_col: str = "human_rating",
) -> Dict[str, Dict[str, float]]:
    """Compare correlations between STED and baselines using Williams' test."""

    # Get valid data for all methods
    all_methods = [target_method] + baseline_methods
    valid_cols = [m for m in all_methods if m in df.columns]
    valid = df[[human_col] + valid_cols].dropna()

    if len(valid) < 10:
        return {}

    n = len(valid)
    y = (valid[human_col].values - 1) / 4  # Normalize to 0-1

    # Compute all pairwise correlations
    correlations = {}
    for method in valid_cols:
        x = valid[method].values
        correlations[method] = spearmanr(x, y)[0]

    # Compare target vs each baseline
    results = {}
    target_corr = correlations.get(target_method)

    if target_corr is None:
        return {}

    for baseline in baseline_methods:
        if baseline not in correlations:
            continue

        baseline_corr = correlations[baseline]

        # Correlation between target and baseline methods
        r_methods = spearmanr(valid[target_method], valid[baseline])[0]

        t_stat, p_value = williams_test(
            target_corr, baseline_corr, r_methods, n
        )

        results[baseline] = {
            f"{target_method}_corr": target_corr,
            f"{baseline}_corr": baseline_corr,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant_0.05": p_value < 0.05 if not np.isnan(p_value) else False,
            "significant_0.01": p_value < 0.01 if not np.isnan(p_value) else False,
            "sted_better": target_corr > baseline_corr,
        }

    return results


def compute_bootstrap_ci(
    df: pd.DataFrame,
    method: str,
    human_col: str = "human_rating",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for correlation."""

    valid = df[[method, human_col]].dropna()
    if len(valid) < 10:
        return np.nan, np.nan, np.nan

    x = valid[method].values
    y = (valid[human_col].values - 1) / 4

    correlations = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(x), size=len(x), replace=True)
        r, _ = spearmanr(x[indices], y[indices])
        correlations.append(r)

    correlations = np.array(correlations)
    alpha = 1 - confidence

    lower = np.percentile(correlations, 100 * alpha / 2)
    upper = np.percentile(correlations, 100 * (1 - alpha / 2))
    mean = np.mean(correlations)

    return mean, lower, upper


def analyze_by_stratum(
    df: pd.DataFrame,
    strata: List[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Analyze correlations by STED score stratum."""

    if strata is None:
        strata = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    results = []

    for low, high in strata:
        mask = (df["sted_score"] >= low) & (df["sted_score"] < high)
        if high == 1.0:
            mask = mask | (df["sted_score"] == 1.0)

        stratum_df = df[mask]

        if len(stratum_df) < 5:
            continue

        valid = stratum_df[["sted_combined", "human_rating"]].dropna()
        if len(valid) < 5:
            continue

        x = valid["sted_combined"].values
        y = (valid["human_rating"].values - 1) / 4

        rho, p = spearmanr(x, y)

        results.append({
            "stratum": f"[{low:.1f}, {high:.1f})",
            "n_samples": len(valid),
            "spearman_rho": rho,
            "p_value": p,
            "mean_sted": valid["sted_combined"].mean(),
            "mean_human": valid["human_rating"].mean(),
        })

    return pd.DataFrame(results)


def generate_report(
    df: pd.DataFrame,
    correlations: Dict,
    comparisons: Dict,
    iaa: Dict,
    output_path: str = None,
) -> str:
    """Generate analysis report."""

    report = []
    report.append("=" * 60)
    report.append("HUMAN VALIDATION ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")

    # Dataset summary
    report.append("1. DATASET SUMMARY")
    report.append("-" * 40)
    report.append(f"Total pairs: {len(df)}")
    report.append(f"Pairs with human ratings: {df['human_rating'].notna().sum()}")

    sources = df["source"].value_counts()
    report.append("\nBy source:")
    for source, count in sources.items():
        report.append(f"  {source}: {count}")
    report.append("")

    # Inter-annotator agreement
    report.append("2. INTER-ANNOTATOR AGREEMENT")
    report.append("-" * 40)
    if iaa:
        report.append(f"Krippendorff's alpha (ordinal): {iaa.get('krippendorff_alpha', 'N/A'):.3f}")
        report.append(f"Fleiss' kappa: {iaa.get('fleiss_kappa', 'N/A'):.3f}")
        report.append(f"Number of annotators: {iaa.get('n_annotators', 'N/A')}")
    else:
        report.append("Single annotator (no IAA computed)")
    report.append("")

    # Correlations
    report.append("3. CORRELATIONS WITH HUMAN RATINGS")
    report.append("-" * 40)
    report.append("")
    report.append(f"{'Method':<20} {'Spearman rho':>12} {'95% CI':>16} {'p-value':>10}")
    report.append("-" * 60)

    for method, stats in correlations.items():
        ci_str = f"[{stats.get('ci_lower', 0):.3f}, {stats.get('ci_upper', 0):.3f}]"
        rho = stats.get('spearman_rho', 0)
        p = stats.get('spearman_p', 1)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        report.append(f"{method:<20} {rho:>12.3f} {ci_str:>16} {p:>10.4f}{sig}")

    report.append("")
    report.append("* p<0.05, ** p<0.01, *** p<0.001")
    report.append("")

    # Statistical comparisons
    report.append("4. STED VS BASELINES (Williams' test)")
    report.append("-" * 40)
    report.append("")

    if comparisons:
        for baseline, result in comparisons.items():
            sted_corr = result.get('sted_combined_corr', 0)
            base_corr = result.get(f'{baseline}_corr', 0)
            p = result.get('p_value', 1)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            report.append(f"STED ({sted_corr:.3f}) vs {baseline} ({base_corr:.3f}): p={p:.4f} {sig}")
    report.append("")

    # Paper-ready table
    report.append("5. PAPER-READY TABLE (LaTeX)")
    report.append("-" * 40)
    report.append("")
    report.append("\\begin{table}[h]")
    report.append("\\centering")
    report.append("\\caption{Correlation with Human Judgments}")
    report.append("\\begin{tabular}{@{}lccc@{}}")
    report.append("\\toprule")
    report.append("\\textbf{Method} & \\textbf{Spearman $\\rho$} & \\textbf{95\\% CI} & \\textbf{p-value} \\\\")
    report.append("\\midrule")

    for method, stats in correlations.items():
        rho = stats.get('spearman_rho', 0)
        ci_low = stats.get('ci_lower', 0)
        ci_high = stats.get('ci_upper', 0)
        p = stats.get('spearman_p', 1)

        method_name = method.replace("_", " ").title()
        p_str = f"{p:.3f}" if p >= 0.001 else "$<$0.001"

        report.append(f"{method_name} & {rho:.3f} & [{ci_low:.3f}, {ci_high:.3f}] & {p_str} \\\\")

    report.append("\\bottomrule")
    report.append("\\end{tabular}")
    report.append("\\end{table}")
    report.append("")

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)
        print(f"Report saved to {output_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze human validation results for STED"
    )
    parser.add_argument(
        "--dataset",
        default="human_validation_dataset.json",
        help="Original dataset JSON file with STED scores",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Annotations file (CSV or JSON from annotation interface)",
    )
    parser.add_argument(
        "--output",
        default="human_validation_report.txt",
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
    df = load_annotations(args.dataset, args.annotations)

    print(f"Loaded {len(df)} pairs, {df['human_rating'].notna().sum()} with ratings")

    # Check for multiple annotators
    rating_cols = [c for c in df.columns if c.startswith("rating") and c != "rating_std"]
    n_annotators = len(rating_cols) if len(rating_cols) > 1 else 1

    # Compute inter-annotator agreement
    iaa = {}
    if n_annotators > 1:
        print(f"Computing inter-annotator agreement ({n_annotators} annotators)...")
        ratings_matrix = df[rating_cols].values

        iaa["krippendorff_alpha"] = compute_krippendorff_alpha(ratings_matrix, level="ordinal")
        iaa["fleiss_kappa"] = compute_fleiss_kappa(ratings_matrix)
        iaa["n_annotators"] = n_annotators

        print(f"  Krippendorff's alpha: {iaa['krippendorff_alpha']:.3f}")
        print(f"  Fleiss' kappa: {iaa['fleiss_kappa']:.3f}")

    # Compute correlations
    print("Computing correlations...")
    methods = ["sted_combined", "sted_structural", "sted_semantic", "deepdiff", "ted", "bertscore"]
    correlations = compute_correlations(df, methods)

    # Add bootstrap CIs
    for method in correlations:
        mean, lower, upper = compute_bootstrap_ci(df, method, n_bootstrap=args.n_bootstrap)
        correlations[method]["ci_lower"] = lower
        correlations[method]["ci_upper"] = upper

    # Compare STED vs baselines
    print("Comparing STED vs baselines...")
    comparisons = compare_correlations(df, ["deepdiff", "ted", "bertscore"])

    # Analyze by stratum
    print("Analyzing by stratum...")
    stratum_results = analyze_by_stratum(df)
    if not stratum_results.empty:
        print("\nCorrelations by STED score stratum:")
        print(stratum_results.to_string(index=False))

    # Generate report
    print("\nGenerating report...")
    report = generate_report(df, correlations, comparisons, iaa, args.output)
    print("\n" + report)


if __name__ == "__main__":
    main()
