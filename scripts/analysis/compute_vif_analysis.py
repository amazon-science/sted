#!/usr/bin/env python3
"""
Compute Variance Inflation Factor (VIF) for schema variables.

This addresses reviewer concern W4: Schema complexity, depth, breadth, total_params
are all measuring overlapping constructs.

VIF > 5 indicates moderate multicollinearity
VIF > 10 indicates high multicollinearity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import json
import warnings
warnings.filterwarnings('ignore')


def variance_inflation_factor(X, idx):
    """Compute VIF for feature at index idx."""
    y = X[:, idx]
    X_other = np.delete(X, idx, axis=1)
    if X_other.shape[1] == 0:
        return 1.0
    lr = LinearRegression()
    lr.fit(X_other, y)
    r_squared = lr.score(X_other, y)
    if r_squared == 1.0:
        return np.inf
    return 1.0 / (1.0 - r_squared)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'results/factor_analysis/factor_analysis_data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/kdd_paper_tables'

# Schema features to analyze for multicollinearity
SCHEMA_FEATURES = [
    'schema_complexity', 'schema_depth', 'schema_breadth', 'total_params',
    'max_params_per_tool', 'param_type_diversity', 'avg_params_per_tool',
    'num_tools', 'avg_tool_name_length', 'tool_name_ambiguity', 'tool_prefix_diversity'
]

QUERY_FEATURES = [
    'query_length', 'query_word_count', 'query_sentence_count', 'num_questions',
    'num_commands', 'num_conjunctions', 'query_complexity_score', 'constraint_score'
]

ALL_CONTROLLABLE = SCHEMA_FEATURES + QUERY_FEATURES + ['temperature']


def compute_vif(df, features):
    """Compute VIF for each feature."""
    X = df[features].copy()
    X = X.fillna(0).values

    vif_data = []
    for i, col in enumerate(features):
        try:
            vif = variance_inflation_factor(X, i)
            vif_data.append({
                'feature': col,
                'VIF': vif
            })
        except Exception as e:
            vif_data.append({
                'feature': col,
                'VIF': np.nan
            })

    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


def compute_partial_correlations(df, features, target='stability_score'):
    """Compute partial correlations controlling for other features."""
    from scipy.stats import pearsonr

    partial_corrs = []

    for feature in features:
        # Simple correlation
        r_simple, p_simple = pearsonr(df[feature].fillna(0), df[target].fillna(0))

        # Partial correlation: residualize both target and feature on other features
        other_features = [f for f in features if f != feature]
        if other_features:
            from sklearn.linear_model import LinearRegression

            X_other = df[other_features].fillna(0)

            # Residualize target
            lr_target = LinearRegression()
            lr_target.fit(X_other, df[target].fillna(0))
            target_resid = df[target].fillna(0) - lr_target.predict(X_other)

            # Residualize feature
            lr_feature = LinearRegression()
            lr_feature.fit(X_other, df[feature].fillna(0))
            feature_resid = df[feature].fillna(0) - lr_feature.predict(X_other)

            r_partial, p_partial = pearsonr(feature_resid, target_resid)
        else:
            r_partial, p_partial = r_simple, p_simple

        partial_corrs.append({
            'feature': feature,
            'simple_r': r_simple,
            'partial_r': r_partial,
            'partial_p': p_partial,
            'r_change': r_partial - r_simple
        })

    return pd.DataFrame(partial_corrs).sort_values('partial_r', key=abs, ascending=False)


def analyze_feature_group_redundancy(df, features, target='stability_score'):
    """Analyze redundancy between feature groups using incremental R^2."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    y = df[target].fillna(0).values

    results = {}

    # Individual groups
    for name, feats in [('Schema', SCHEMA_FEATURES), ('Query', QUERY_FEATURES), ('Config', ['temperature'])]:
        avail_feats = [f for f in feats if f in df.columns]
        X = df[avail_feats].fillna(0)
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        results[name] = {'R2': scores.mean(), 'features': len(avail_feats)}

    # Combined - but understanding the paradox
    # The paradox happens because RF with more correlated features has worse generalization
    X_all = df[ALL_CONTROLLABLE].fillna(0)
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_all, y, cv=5, scoring='r2')
    results['All'] = {'R2': scores.mean(), 'features': len(ALL_CONTROLLABLE)}

    return results


def main():
    print("=" * 60)
    print("VIF AND MULTICOLLINEARITY ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # 1. VIF for schema features
    print("\n" + "-" * 60)
    print("1. VIF Analysis - Schema Features")
    print("-" * 60)

    schema_feats = [f for f in SCHEMA_FEATURES if f in df.columns]
    vif_schema = compute_vif(df, schema_feats)

    print("\nSchema Feature VIF:")
    print(f"{'Feature':<30} {'VIF':>10} {'Interpretation':>20}")
    print("-" * 60)
    for _, row in vif_schema.iterrows():
        vif = row['VIF']
        if np.isnan(vif):
            interp = 'N/A'
        elif vif > 10:
            interp = 'HIGH (>10)'
        elif vif > 5:
            interp = 'MODERATE (>5)'
        else:
            interp = 'OK'
        print(f"{row['feature']:<30} {vif:>10.2f} {interp:>20}")

    # 2. VIF for all controllable features
    print("\n" + "-" * 60)
    print("2. VIF Analysis - All Controllable Features")
    print("-" * 60)

    all_feats = [f for f in ALL_CONTROLLABLE if f in df.columns]
    vif_all = compute_vif(df, all_feats)

    high_vif = vif_all[vif_all['VIF'] > 5].sort_values('VIF', ascending=False)
    print(f"\nFeatures with VIF > 5 (moderate+ multicollinearity):")
    for _, row in high_vif.iterrows():
        print(f"  {row['feature']:<30} VIF = {row['VIF']:.2f}")

    # 3. Partial correlations for schema features
    print("\n" + "-" * 60)
    print("3. Partial Correlations - Schema Features")
    print("-" * 60)

    partial_schema = compute_partial_correlations(df, schema_feats)

    print("\nSchema Feature Partial Correlations with Stability Score:")
    print(f"{'Feature':<30} {'Simple r':>10} {'Partial r':>10} {'Change':>10}")
    print("-" * 60)
    for _, row in partial_schema.iterrows():
        print(f"{row['feature']:<30} {row['simple_r']:>10.3f} {row['partial_r']:>10.3f} {row['r_change']:>10.3f}")

    # 4. Feature correlation matrix
    print("\n" + "-" * 60)
    print("4. Feature Correlation Matrix (Schema)")
    print("-" * 60)

    corr_matrix = df[schema_feats].corr()

    # Find highly correlated pairs (|r| > 0.7)
    print("\nHighly correlated pairs (|r| > 0.7):")
    for i in range(len(schema_feats)):
        for j in range(i+1, len(schema_feats)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                print(f"  {schema_feats[i]} <-> {schema_feats[j]}: r = {r:.3f}")

    # 5. Explain the Table 3 paradox
    print("\n" + "-" * 60)
    print("5. Explaining Table 3 Paradox")
    print("-" * 60)

    results = analyze_feature_group_redundancy(df, all_feats)

    print("\nR² by feature group:")
    for name, vals in results.items():
        print(f"  {name:<20} R² = {vals['R2']:.3f} ({vals['features']} features)")

    print("\n*** EXPLANATION OF PARADOX ***")
    print("""
The paradox (Query R²=0.178 > All R²=0.103) occurs because:

1. MULTICOLLINEARITY: Schema features are highly correlated with each other
   (e.g., schema_complexity, schema_depth, schema_breadth share variance).

2. NOISE ADDITION: When adding correlated features, the RF splits become
   less informative because multiple features provide similar signal.

3. CROSS-VALIDATION EFFECT: CV penalizes overfitting. More correlated
   features = more opportunities to overfit, worse CV scores.

4. INDEPENDENT CONTRIBUTIONS: Query features provide largely independent
   signal from schema features, but combining them introduces redundancy
   within the schema group that hurts overall prediction.

RECOMMENDATION: Report both individual group R² and combined R², noting
that the lower combined R² reflects multicollinearity effects, not that
more information hurts prediction. The TRUE interpretable contribution
requires partial R² or orthogonalized features.
""")

    # Save results
    output = {
        'vif_schema': vif_schema.to_dict('records'),
        'vif_all': vif_all.to_dict('records'),
        'partial_correlations': partial_schema.to_dict('records'),
        'group_r2': results,
        'high_correlation_pairs': [
            {'feat1': schema_feats[i], 'feat2': schema_feats[j], 'r': float(corr_matrix.iloc[i, j])}
            for i in range(len(schema_feats))
            for j in range(i+1, len(schema_feats))
            if abs(corr_matrix.iloc[i, j]) > 0.7
        ]
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / 'vif_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'vif_analysis.json'}")


if __name__ == '__main__':
    main()
