#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis.

Compares SHAP values vs MDI (Mean Decrease in Impurity) for feature importance
across different models.

Usage:
    python scripts/analysis/shap_feature_importance.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Install shap if needed
try:
    import shap
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap', '-q'])
    import shap

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data():
    """Load the factor analysis data."""
    data_path = PROJECT_ROOT / 'results/factor_analysis/factor_analysis_data.csv'
    df = pd.read_csv(data_path)
    return df


def get_feature_columns(df):
    """Get feature columns for importance analysis."""
    exclude = ['model', 'sample_idx', 'temperature', 'c_mean', 'stability_score',
               'validity_rate', 'ranking_score', 'subset_name', 'is_english',
               'language', 'model_family', 'is_consistent', 'is_highly_consistent',
               'has_json_example', 'has_numbered_list', 'has_bullet_list',
               'has_code_block', 'has_vague_terms', 'has_conditional',
               'has_optional_params', 'has_complex_params', 'has_nested_params']

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def compute_shap_importance(df, feature_cols, model_name, target='stability_score', n_samples=2000):
    """Compute SHAP feature importance for a model."""
    print(f"\n{'='*60}")
    print(f"SHAP Analysis: {model_name}")
    print('='*60)

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    print(f"  Total samples: {len(X)}")

    # Subsample for SHAP (it's slow)
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
        print(f"  Using {n_samples} samples for SHAP")
    else:
        X_sample = X
        y_sample = y

    # Train Random Forest
    print("  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_sample, y_sample)

    # MDI importance
    mdi_importance = pd.DataFrame({
        'feature': feature_cols,
        'mdi_importance': rf.feature_importances_,
    }).sort_values('mdi_importance', ascending=False)

    # SHAP values
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP value per feature
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': np.abs(shap_values).mean(axis=0),
    }).sort_values('shap_importance', ascending=False)

    # Merge
    importance_df = mdi_importance.merge(shap_importance, on='feature')
    importance_df['mdi_rank'] = importance_df['mdi_importance'].rank(ascending=False)
    importance_df['shap_rank'] = importance_df['shap_importance'].rank(ascending=False)
    importance_df['rank_diff'] = importance_df['mdi_rank'] - importance_df['shap_rank']

    # Sort by SHAP importance
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    # Compute correlation
    rho, p = spearmanr(importance_df['mdi_rank'], importance_df['shap_rank'])

    print(f"\n  MDI vs SHAP Rank Correlation: rho={rho:.3f} (p={p:.4f})")

    print(f"\n  Top 10 Features by SHAP:")
    print(f"  {'Feature':<30} {'SHAP':>10} {'MDI':>10} {'SHAP_Rank':>10} {'MDI_Rank':>10} {'Diff':>6}")
    print("  " + "-"*78)
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['shap_importance']:>10.4f} {row['mdi_importance']:>10.4f} "
              f"{row['shap_rank']:>10.0f} {row['mdi_rank']:>10.0f} {row['rank_diff']:>6.0f}")

    # Features with biggest rank changes
    importance_df['abs_rank_diff'] = np.abs(importance_df['rank_diff'])
    biggest_changes = importance_df.sort_values('abs_rank_diff', ascending=False).head(5)

    print(f"\n  Features with Biggest Rank Changes (MDI vs SHAP):")
    for _, row in biggest_changes.iterrows():
        direction = "↑" if row['rank_diff'] > 0 else "↓"
        print(f"    {row['feature']:<30} MDI rank {row['mdi_rank']:.0f} -> SHAP rank {row['shap_rank']:.0f} ({direction}{abs(row['rank_diff']):.0f})")

    return importance_df, shap_values, X_sample, rho


def plot_shap_summary(shap_values, X_sample, feature_cols, model_name, output_dir):
    """Create SHAP summary plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False, max_display=15)
    plt.title(f'SHAP Summary: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'shap_summary_{model_name.replace(":", "_").replace("-", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved SHAP plot to {output_dir}")


def compare_all_models(results):
    """Compare SHAP rankings across all models."""
    print("\n" + "="*60)
    print("CROSS-MODEL SHAP RANKING COMPARISON")
    print("="*60)

    # Create comparison dataframe
    all_shap = {}
    for model, data in results.items():
        imp_df = data['importance'].set_index('feature')['shap_importance']
        all_shap[model] = imp_df

    combined = pd.DataFrame(all_shap)

    # Add ranks
    for col in combined.columns:
        combined[f'{col}_rank'] = combined[col].rank(ascending=False)

    # Pairwise correlations
    models = list(results.keys())
    n = len(models)

    print("\nPairwise SHAP Rank Correlations:")
    corr_matrix = np.zeros((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i <= j:
                rho, _ = spearmanr(
                    combined[f'{m1}_rank'].values,
                    combined[f'{m2}_rank'].values
                )
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    corr_df = pd.DataFrame(corr_matrix, index=models, columns=models)
    print(corr_df.round(3).to_string())

    # Most variable features by SHAP
    rank_cols = [c for c in combined.columns if '_rank' in str(c)]
    combined['rank_std'] = combined[rank_cols].std(axis=1)

    print("\n\nMost Variable Features (SHAP ranks):")
    most_variable = combined.sort_values('rank_std', ascending=False).head(10)
    for feat, row in most_variable.iterrows():
        ranks = [f"{row[f'{m}_rank']:.0f}" for m in models]
        print(f"  {feat:<30} ranks: {', '.join(ranks)} (std={row['rank_std']:.1f})")

    return combined, corr_df


def main():
    print("="*60)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")

    # All 18 FINAL_MODELS
    FINAL_MODELS = [
        'Qwen3-235B-A22B',
        'Claude-3.5-Sonnet',
        'Claude-Haiku-4.5',
        'Claude-3.7-Sonnet',
        'Claude-3.5-Haiku',
        'Claude-Opus-4.5',
        'Claude-Sonnet-4',
        'Claude-Sonnet-4.5',
        'Qwen3-32B',
        'Llama-3.3-70B',
        'Nova-2-Lite',
        'Mimo-V2-Flash',
        'Grok-4.1-Fast',
        'Minimax-M2',
        'GPT-4.1-Mini',
        'Gemini-2.5-Flash-Lite',
        'GPT-OSS-120B',
    ]
    key_models = FINAL_MODELS

    output_dir = PROJECT_ROOT / 'results/factor_analysis/shap'

    results = {}
    mdi_vs_shap_correlations = {}

    for model in key_models:
        model_df = df[df['model'] == model]
        if len(model_df) < 100:
            # Try partial match
            model_df = df[df['model'].str.contains(model, case=False, na=False)]

        if len(model_df) < 100:
            print(f"\nSkipping {model} - insufficient data")
            continue

        importance_df, shap_values, X_sample, rho = compute_shap_importance(
            model_df, feature_cols, model
        )

        plot_shap_summary(shap_values, X_sample, feature_cols, model, output_dir)

        results[model] = {
            'importance': importance_df,
            'shap_values': shap_values,
            'X_sample': X_sample,
        }
        mdi_vs_shap_correlations[model] = rho

    # Compare across models
    if len(results) > 1:
        combined, corr_df = compare_all_models(results)
        combined.to_csv(output_dir / 'shap_importance_comparison.csv')
        corr_df.to_csv(output_dir / 'shap_rank_correlation.csv')

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: MDI vs SHAP Correlation per Model")
    print("="*60)
    for model, rho in mdi_vs_shap_correlations.items():
        status = "HIGH" if rho > 0.8 else "MEDIUM" if rho > 0.6 else "LOW"
        print(f"  {model:<30} rho={rho:.3f} ({status})")

    avg_corr = np.mean(list(mdi_vs_shap_correlations.values()))
    print(f"\n  Average MDI-SHAP correlation: {avg_corr:.3f}")

    if avg_corr > 0.8:
        print("\n  Conclusion: MDI and SHAP rankings are HIGHLY consistent")
        print("             -> MDI results are reliable")
    elif avg_corr > 0.6:
        print("\n  Conclusion: MDI and SHAP rankings are MODERATELY consistent")
        print("             -> Some features may have different true importance")
    else:
        print("\n  Conclusion: MDI and SHAP rankings DIFFER significantly")
        print("             -> SHAP values should be used for reliable importance")

    print(f"\n\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
