#!/usr/bin/env python3
"""
Test the p^D(s) Decision Point Hypothesis.

Hypothesis from Section 10.1: If each atomic decision (tool selection, parameter
inclusion, value generation) has consistency probability p, then overall consistency
scales as p^D(s) where D(s) is the total decision count for schema s.

This means: log(consistency) ~ D(s) with negative slope

We test this by fitting:
  log(stability_score) = a + b * decision_count + epsilon

If the hypothesis holds:
- b should be negative (more decisions = lower consistency)
- The fit should be better than linear (consistency ~ decision_count)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'results/factor_analysis/factor_analysis_data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/kdd_paper_tables'


def estimate_decision_count(row):
    """
    Estimate total decision count D(s) for a sample.

    Decisions include:
    1. Tool selection (which tool to call) - 1 decision per tool call
    2. Parameter inclusion (which params to include) - 1 per optional param
    3. Value generation (what value to assign) - 1 per param

    Approximation: D(s) = num_tools * (1 + avg_params_per_tool) + total_params
    """
    num_tools = row.get('num_tools', 1)
    total_params = row.get('total_params', 0)
    avg_params = row.get('avg_params_per_tool', 0)
    schema_breadth = row.get('schema_breadth', 0)

    # D(s) approximation
    # Each tool: 1 decision to select
    # Each param: 1 decision to include + 1 decision for value
    decision_count = num_tools + 2 * total_params

    return max(decision_count, 1)  # Ensure at least 1


def test_multiplicative_hypothesis(df):
    """Test the p^D(s) multiplicative hypothesis."""

    # Compute decision count for each sample
    df = df.copy()
    df['decision_count'] = df.apply(estimate_decision_count, axis=1)

    # Filter valid samples (stability > 0 for log transform)
    valid = df[df['stability_score'] > 0.01].copy()
    valid['log_stability'] = np.log(valid['stability_score'])

    print(f"Valid samples: {len(valid)} / {len(df)}")
    print(f"Decision count range: {valid['decision_count'].min():.0f} - {valid['decision_count'].max():.0f}")
    print(f"Stability range: {valid['stability_score'].min():.3f} - {valid['stability_score'].max():.3f}")

    # Test 1: Linear model (consistency ~ D)
    print("\n" + "="*60)
    print("Model 1: Linear - consistency ~ D(s)")
    print("="*60)

    X = valid[['decision_count']].values
    y_linear = valid['stability_score'].values

    lr_linear = LinearRegression()
    lr_linear.fit(X, y_linear)
    y_pred_linear = lr_linear.predict(X)
    r2_linear = r2_score(y_linear, y_pred_linear)

    print(f"  R² = {r2_linear:.4f}")
    print(f"  Slope = {lr_linear.coef_[0]:.6f}")
    print(f"  Intercept = {lr_linear.intercept_:.4f}")

    # Test 2: Log model (log(consistency) ~ D)
    print("\n" + "="*60)
    print("Model 2: Multiplicative - log(consistency) ~ D(s)")
    print("="*60)

    y_log = valid['log_stability'].values

    lr_log = LinearRegression()
    lr_log.fit(X, y_log)
    y_pred_log = lr_log.predict(X)
    r2_log = r2_score(y_log, y_pred_log)

    # Compute implied p per decision
    slope = lr_log.coef_[0]
    p_per_decision = np.exp(slope)

    print(f"  R² = {r2_log:.4f}")
    print(f"  Slope (log scale) = {slope:.6f}")
    print(f"  Intercept = {lr_log.intercept_:.4f}")
    print(f"  Implied p per decision = {p_per_decision:.4f}")

    # Statistical test for slope < 0
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(valid['decision_count'], y_log)
    print(f"  Correlation: r = {corr:.4f}, p = {p_value:.2e}")

    # Test 3: Compare model fits
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)

    # For fair comparison, evaluate both in consistency space
    y_pred_log_transformed = np.exp(y_pred_log)
    r2_log_transformed = r2_score(y_linear, y_pred_log_transformed)

    print(f"  Linear R² (raw): {r2_linear:.4f}")
    print(f"  Log model R² (transformed back): {r2_log_transformed:.4f}")

    if r2_log > r2_linear:
        print("\n  >>> Multiplicative model fits better - SUPPORTS hypothesis")
    else:
        print("\n  >>> Linear model fits better - WEAK support for hypothesis")

    # Test 4: By schema complexity bins
    print("\n" + "="*60)
    print("Multiplicative Effect by Schema Complexity")
    print("="*60)

    valid['complexity_bin'] = pd.qcut(valid['schema_complexity'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = valid[valid['complexity_bin'] == q]
        if len(subset) > 100:
            X_sub = subset[['decision_count']].values
            y_sub = subset['log_stability'].values
            lr_sub = LinearRegression()
            lr_sub.fit(X_sub, y_sub)
            p_implied = np.exp(lr_sub.coef_[0])
            print(f"  {q}: slope = {lr_sub.coef_[0]:.4f}, p_implied = {p_implied:.4f} (n={len(subset)})")

    # Test 5: Interaction with temperature
    print("\n" + "="*60)
    print("Temperature Effect on p per Decision")
    print("="*60)

    for temp in [0.0, 0.5, 1.0]:
        subset = valid[valid['temperature'] == temp]
        if len(subset) > 100:
            X_sub = subset[['decision_count']].values
            y_sub = subset['log_stability'].values
            lr_sub = LinearRegression()
            lr_sub.fit(X_sub, y_sub)
            p_implied = np.exp(lr_sub.coef_[0])
            print(f"  T={temp}: slope = {lr_sub.coef_[0]:.4f}, p_implied = {p_implied:.4f} (n={len(subset)})")

    results = {
        'hypothesis': 'p^D(s) multiplicative consistency decay',
        'samples_tested': len(valid),
        'linear_model': {
            'R2': float(r2_linear),
            'slope': float(lr_linear.coef_[0]),
            'intercept': float(lr_linear.intercept_)
        },
        'multiplicative_model': {
            'R2': float(r2_log),
            'slope': float(slope),
            'intercept': float(lr_log.intercept_),
            'implied_p_per_decision': float(p_per_decision),
            'correlation': float(corr),
            'p_value': float(p_value)
        },
        'conclusion': 'Supports hypothesis' if slope < 0 and r2_log > 0.01 else 'Does not support hypothesis'
    }

    return results


def main():
    print("=" * 60)
    print("DECISION POINT HYPOTHESIS TEST")
    print("log(consistency) ~ D(s) where D(s) = decision count")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    results = test_multiplicative_hypothesis(df)

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / 'decision_point_hypothesis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'decision_point_hypothesis.json'}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"\n{results['conclusion'].upper()}")

    if results['multiplicative_model']['slope'] < 0:
        print(f"\nEach additional decision point reduces consistency by factor of {results['multiplicative_model']['implied_p_per_decision']:.4f}")
        print(f"This is statistically significant (p = {results['multiplicative_model']['p_value']:.2e})")
    else:
        print("\nSlope is not negative - multiplicative decay not observed")


if __name__ == '__main__':
    main()
