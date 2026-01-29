#!/usr/bin/env python
"""
Analyze best-match selection results with multiple annotators using majority voting.

Computes:
- Fleiss' kappa (3+ raters) or Cohen's kappa (2 raters) for inter-annotator agreement
- Method alignment with human majority vote
- McNemar's tests for pairwise method comparisons
- Bootstrap confidence intervals

Usage:
    python analyze_multi_annotator.py --users 1 2 4
    python analyze_multi_annotator.py --users 3 4 --output report.txt
"""

import json
import csv
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from scipy.stats import binom


def load_csv_annotations(filepath: str) -> Dict[str, Dict]:
    """Load annotations from CSV file."""
    annotations = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get('item_id')
            choice = row.get('choice')
            skipped = row.get('skipped', 'false').lower() == 'true'
            if item_id and choice and not skipped:
                annotations[item_id] = {
                    'choice': choice.strip().upper(),
                    'confidence': int(row['confidence']) if row.get('confidence') else None,
                }
    return annotations


def load_json_annotations(filepath: str) -> Dict[str, Dict]:
    """Load annotations from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    annotations = {}
    for sample_id, entry in data.items():
        if not entry.get('skipped') and entry.get('choice'):
            annotations[sample_id] = {
                'choice': entry['choice'].upper(),
                'confidence': entry.get('confidence'),
            }
    return annotations


def load_annotations(filepath: str) -> Dict[str, Dict]:
    """Load annotations from CSV or JSON file."""
    if filepath.endswith('.json'):
        return load_json_annotations(filepath)
    elif filepath.endswith('.csv'):
        return load_csv_annotations(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def fleiss_kappa(ratings: List[Dict[str, int]]) -> float:
    """
    Compute Fleiss' kappa for inter-rater reliability.

    Args:
        ratings: List of dicts, where each dict maps category -> count for one item

    Returns:
        Fleiss' kappa value
    """
    n_items = len(ratings)
    if n_items == 0:
        return 0.0

    # Get all categories and number of raters
    categories = set()
    for r in ratings:
        categories.update(r.keys())
    n_raters = sum(ratings[0].values())

    if n_raters < 2:
        return 0.0

    # Compute P_i for each item
    P_i = []
    for r in ratings:
        total = sum(r.values())
        sum_sq = sum(v * v for v in r.values())
        P_i.append((sum_sq - total) / (total * (total - 1)) if total > 1 else 0)
    P_bar = sum(P_i) / n_items

    # Compute P_j for each category (proportion of assignments to each category)
    p_j = {}
    for cat in categories:
        total = sum(r.get(cat, 0) for r in ratings)
        p_j[cat] = total / (n_items * n_raters)
    P_e = sum(p * p for p in p_j.values())

    # Compute kappa
    if P_e >= 1:
        return 0.0
    return (P_bar - P_e) / (1 - P_e)


def cohens_kappa(ratings1: List[str], ratings2: List[str]) -> float:
    """
    Compute Cohen's kappa for two raters.

    Args:
        ratings1: List of choices from rater 1
        ratings2: List of choices from rater 2

    Returns:
        Cohen's kappa value
    """
    assert len(ratings1) == len(ratings2)
    n = len(ratings1)
    if n == 0:
        return 0.0

    # Count agreements
    agree = sum(1 for a, b in zip(ratings1, ratings2) if a == b)
    p_o = agree / n  # Observed agreement

    # Count category frequencies
    freq1 = Counter(ratings1)
    freq2 = Counter(ratings2)
    all_cats = set(freq1.keys()) | set(freq2.keys())

    # Expected agreement by chance
    p_e = sum((freq1[c] / n) * (freq2[c] / n) for c in all_cats)

    if p_e >= 1:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


def mcnemar_test(outcomes1: List[bool], outcomes2: List[bool]) -> Tuple[int, int, float]:
    """
    Perform McNemar's test comparing two methods.

    Args:
        outcomes1: List of bool (correct/incorrect) for method 1
        outcomes2: List of bool (correct/incorrect) for method 2

    Returns:
        Tuple of (method1_only_correct, method2_only_correct, p_value)
    """
    assert len(outcomes1) == len(outcomes2)

    # Count discordant pairs
    m1_yes_m2_no = sum(1 for o1, o2 in zip(outcomes1, outcomes2) if o1 and not o2)
    m1_no_m2_yes = sum(1 for o1, o2 in zip(outcomes1, outcomes2) if not o1 and o2)

    n = m1_yes_m2_no + m1_no_m2_yes
    if n == 0:
        return m1_yes_m2_no, m1_no_m2_yes, 1.0

    # Exact binomial test (two-tailed)
    k = min(m1_yes_m2_no, m1_no_m2_yes)
    p_value = 2 * binom.cdf(k, n, 0.5)
    p_value = min(p_value, 1.0)

    return m1_yes_m2_no, m1_no_m2_yes, p_value


def bootstrap_ci(
    outcomes: List[bool],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for accuracy."""
    if len(outcomes) == 0:
        return 0.0, 0.0

    outcomes = np.array(outcomes, dtype=float)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(outcomes, size=len(outcomes), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return lower, upper


def analyze_multi_annotator(
    results_dir: str,
    user_ids: List[int],
    method_mapping_path: Optional[str] = None,
    majority_threshold: Optional[int] = None,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Analyze multi-annotator results with majority voting.

    Args:
        results_dir: Directory containing annotation files
        user_ids: List of user IDs to include (e.g., [1, 2, 4])
        method_mapping_path: Path to method mapping JSON (default: _method_mapping.json in results_dir)
        majority_threshold: Minimum votes for majority (default: ceil(n_users/2))
        n_bootstrap: Number of bootstrap samples for CI

    Returns:
        Dict containing all computed statistics
    """
    results_dir = Path(results_dir)

    # Load method mapping
    if method_mapping_path is None:
        method_mapping_path = results_dir / "_method_mapping.json"
    with open(method_mapping_path) as f:
        method_mapping = json.load(f)

    # Load annotations for each user
    users = {}
    for uid in user_ids:
        # Try JSON first, then CSV
        json_path = results_dir / f"best_match_annotations_user0{uid}.json"
        csv_path = results_dir / f"best_match_annotations_user0{uid}.csv"

        if json_path.exists():
            users[uid] = load_annotations(str(json_path))
        elif csv_path.exists():
            users[uid] = load_annotations(str(csv_path))
        else:
            raise FileNotFoundError(f"No annotation file found for user {uid}")

    n_users = len(user_ids)
    if majority_threshold is None:
        majority_threshold = (n_users // 2) + 1  # Strict majority

    # Find common samples (all users have annotated AND in method_mapping)
    common_samples = set(method_mapping.keys())
    for uid in user_ids:
        common_samples &= set(users[uid].keys())
    common_samples = sorted(common_samples)

    # Compute inter-annotator agreement
    if n_users == 2:
        # Cohen's kappa for 2 raters
        r1 = [users[user_ids[0]][s]['choice'] for s in common_samples]
        r2 = [users[user_ids[1]][s]['choice'] for s in common_samples]
        kappa = cohens_kappa(r1, r2)
        kappa_type = "Cohen's"
    else:
        # Fleiss' kappa for 3+ raters
        ratings = []
        for sample_id in common_samples:
            counts = Counter()
            for uid in user_ids:
                choice = users[uid][sample_id]['choice']
                counts[choice] += 1
            ratings.append(dict(counts))
        kappa = fleiss_kappa(ratings)
        kappa_type = "Fleiss'"

    # Compute method alignment with majority vote
    methods = ['sted', 'bertscore', 'deepdiff', 'ted']
    method_outcomes = {m: [] for m in methods}  # List of bool for each method

    samples_with_majority = 0
    unanimous_samples = []

    for sample_id in common_samples:
        choices = [users[uid][sample_id]['choice'] for uid in user_ids]
        counter = Counter(choices)
        most_common = counter.most_common()

        if most_common[0][1] >= majority_threshold:
            human_majority = most_common[0][0]
            samples_with_majority += 1

            # Check if unanimous
            if most_common[0][1] == n_users:
                unanimous_samples.append(sample_id)

            # Record outcomes for each method
            mapping = method_mapping[sample_id]
            for method in methods:
                correct = mapping.get(method) == human_majority
                method_outcomes[method].append(correct)

    # Compute win rates and CIs
    win_rates = {}
    win_counts = {}
    confidence_intervals = {}

    for method in methods:
        outcomes = method_outcomes[method]
        win_counts[method] = sum(outcomes)
        win_rates[method] = sum(outcomes) / len(outcomes) if outcomes else 0
        confidence_intervals[method] = bootstrap_ci(outcomes, n_bootstrap)

    # Compute unanimous alignment
    unanimous_alignment = {}
    for method in methods:
        count = 0
        for sample_id in unanimous_samples:
            human_choice = users[user_ids[0]][sample_id]['choice']  # All agree, so any user works
            mapping = method_mapping[sample_id]
            if mapping.get(method) == human_choice:
                count += 1
        unanimous_alignment[method] = count / len(unanimous_samples) if unanimous_samples else 0

    # McNemar's tests (STED vs others)
    mcnemar_results = {}
    for other in methods:
        if other != 'sted':
            m1_yes, m2_yes, p_value = mcnemar_test(
                method_outcomes['sted'],
                method_outcomes[other]
            )
            mcnemar_results[other] = {
                'sted_only_correct': m1_yes,
                'other_only_correct': m2_yes,
                'p_value': p_value,
            }

    # Compile results
    results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'user_ids': user_ids,
            'n_users': n_users,
            'majority_threshold': majority_threshold,
            'n_bootstrap': n_bootstrap,
        },
        'sample_counts': {
            'total_in_mapping': len(method_mapping),
            'common_samples': len(common_samples),
            'samples_with_majority': samples_with_majority,
            'unanimous_samples': len(unanimous_samples),
        },
        'agreement': {
            'kappa': kappa,
            'kappa_type': kappa_type,
            'interpretation': interpret_kappa(kappa),
        },
        'win_rates': win_rates,
        'win_counts': win_counts,
        'confidence_intervals_95': {m: list(ci) for m, ci in confidence_intervals.items()},
        'unanimous_alignment': unanimous_alignment,
        'mcnemar_vs_sted': mcnemar_results,
    }

    return results


def interpret_kappa(kappa: float) -> str:
    """Interpret kappa value according to Landis & Koch (1977)."""
    if kappa < 0:
        return "poor"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost perfect"


def generate_report(results: Dict) -> str:
    """Generate human-readable report from results."""
    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-ANNOTATOR HUMAN VALIDATION ANALYSIS")
    lines.append("=" * 70)

    meta = results['metadata']
    lines.append(f"\nAnalysis date: {meta['analysis_date']}")
    lines.append(f"Users included: {meta['user_ids']}")
    lines.append(f"Majority threshold: {meta['majority_threshold']}/{meta['n_users']}")

    counts = results['sample_counts']
    lines.append(f"\n{'Sample Counts':-^50}")
    lines.append(f"  Total in method mapping: {counts['total_in_mapping']}")
    lines.append(f"  Common across all users: {counts['common_samples']}")
    lines.append(f"  With clear majority:     {counts['samples_with_majority']}")
    lines.append(f"  Unanimous agreement:     {counts['unanimous_samples']}")

    agr = results['agreement']
    lines.append(f"\n{'Inter-Annotator Agreement':-^50}")
    lines.append(f"  {agr['kappa_type']} kappa: {agr['kappa']:.3f} ({agr['interpretation']})")

    lines.append(f"\n{'Method Alignment with Human Majority':-^50}")
    n = counts['samples_with_majority']
    for method in ['sted', 'bertscore', 'deepdiff', 'ted']:
        wins = results['win_counts'][method]
        rate = results['win_rates'][method]
        ci = results['confidence_intervals_95'][method]
        lines.append(f"  {method.upper():12s}: {wins:3d}/{n} = {rate*100:5.1f}%  "
                    f"(95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%])")

    lines.append(f"\n{'Unanimous Sample Alignment':-^50}")
    n_unan = counts['unanimous_samples']
    for method in ['sted', 'bertscore', 'deepdiff', 'ted']:
        rate = results['unanimous_alignment'][method]
        lines.append(f"  {method.upper():12s}: {rate*100:5.1f}%")

    lines.append(f"\n{'McNemar Tests (STED vs Others)':-^50}")
    for other, res in results['mcnemar_vs_sted'].items():
        p = res['p_value']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(f"  STED vs {other.upper():10s}: "
                    f"STED+/{other}- = {res['sted_only_correct']:2d}, "
                    f"STED-/{other}+ = {res['other_only_correct']:2d}, "
                    f"p = {p:.4f} {sig}")

    # Summary for paper
    lines.append("\n" + "=" * 70)
    lines.append("SUMMARY FOR PAPER")
    lines.append("=" * 70)
    sted_rate = results['win_rates']['sted'] * 100
    bert_rate = results['win_rates']['bertscore'] * 100
    gap = sted_rate - bert_rate
    p_bert = results['mcnemar_vs_sted']['bertscore']['p_value']

    lines.append(f"  {meta['n_users']} annotators, {n} samples")
    lines.append(f"  {agr['kappa_type']} kappa = {agr['kappa']:.3f} ({agr['interpretation']} agreement)")
    lines.append(f"  STED: {sted_rate:.1f}% ({results['win_counts']['sted']}/{n})")
    lines.append(f"  BERTScore: {bert_rate:.1f}%")
    lines.append(f"  DeepDiff: {results['win_rates']['deepdiff']*100:.1f}%")
    lines.append(f"  TED: {results['win_rates']['ted']*100:.1f}%")
    lines.append(f"  STED-BERTScore gap: {gap:.1f}pp, p = {p_bert:.3f}")
    lines.append(f"  Unanimous: {n_unan} items, STED = {results['unanimous_alignment']['sted']*100:.1f}%")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-annotator human validation results"
    )
    parser.add_argument(
        "--results-dir",
        default="human_validation_results",
        help="Directory containing annotation files",
    )
    parser.add_argument(
        "--users",
        type=int,
        nargs="+",
        required=True,
        help="User IDs to include (e.g., --users 1 2 4)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report file (default: stdout)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Output JSON file with structured results",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI (default: 1000)",
    )

    args = parser.parse_args()

    # Run analysis
    results = analyze_multi_annotator(
        results_dir=args.results_dir,
        user_ids=args.users,
        n_bootstrap=args.n_bootstrap,
    )

    # Generate report
    report = generate_report(results)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to {args.json_output}")


if __name__ == "__main__":
    main()
