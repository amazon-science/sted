#!/usr/bin/env python3
"""
Run SHAP analysis for missing models in the per-model table.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap', '-q'])
    import shap

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data():
    data_path = PROJECT_ROOT / 'results/factor_analysis/factor_analysis_data.csv'
    df = pd.read_csv(data_path)
    return df


def get_feature_columns(df):
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
    print(f"\n{'='*60}")
    print(f"SHAP Analysis: {model_name}")
    print('='*60)

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    print(f"  Total samples: {len(X)}")

    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
        print(f"  Using {n_samples} samples for SHAP")
    else:
        X_sample = X
        y_sample = y

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

    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': np.abs(shap_values).mean(axis=0),
    }).sort_values('shap_importance', ascending=False)

    importance_df = mdi_importance.merge(shap_importance, on='feature')
    importance_df['mdi_rank'] = importance_df['mdi_importance'].rank(ascending=False)
    importance_df['shap_rank'] = importance_df['shap_importance'].rank(ascending=False)

    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    rho, p = spearmanr(importance_df['mdi_rank'], importance_df['shap_rank'])

    print(f"\n  MDI vs SHAP Rank Correlation: rho={rho:.3f}")
    print(f"\n  Top Feature by SHAP: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['shap_importance']:.4f})")

    return importance_df, rho


def main():
    print("="*60)
    print("SHAP ANALYSIS FOR MISSING MODELS")
    print("="*60)

    df = load_data()
    feature_cols = get_feature_columns(df)

    # Models missing from the appendix table
    missing_models = [
        ('Claude-Opus-4', 'us.anthropic.claude-opus-4-20250514-v1'),
        ('Claude-3.7-Sonnet', 'Claude-3.7-Sonnet'),
        ('Claude-3.5-Haiku', 'Claude-3.5-Haiku'),
        ('Qwen3-32B', 'Qwen3-32B'),
        ('GPT-OSS-120B', 'GPT-OSS-120B'),
    ]

    print("\nModels in data:")
    for m in df['model'].unique():
        print(f"  - {m}")

    results = {}

    for display_name, data_name in missing_models:
        model_df = df[df['model'] == data_name]

        if len(model_df) < 100:
            # Try partial match
            model_df = df[df['model'].str.contains(data_name, case=False, na=False)]

        if len(model_df) < 100:
            print(f"\nSkipping {display_name} - insufficient data (found {len(model_df)} samples)")
            continue

        importance_df, rho = compute_shap_importance(model_df, feature_cols, display_name)

        top_feature = importance_df.iloc[0]['feature']
        top_shap = importance_df.iloc[0]['shap_importance']

        results[display_name] = {
            'top_feature': top_feature,
            'shap_value': top_shap,
            'mdi_shap_rho': rho,
            'importance_df': importance_df,
        }

    # Now compute cross-model correlations with existing results
    print("\n" + "="*60)
    print("CROSS-MODEL CORRELATION WITH EXISTING MODELS")
    print("="*60)

    # Load existing SHAP results
    existing_shap_path = PROJECT_ROOT / 'results/factor_analysis/shap/shap_importance_comparison.csv'
    if existing_shap_path.exists():
        existing_df = pd.read_csv(existing_shap_path, index_col=0)

        for model_name, data in results.items():
            imp_df = data['importance_df'].set_index('feature')['shap_importance']

            # Compute correlation with each existing model
            correlations = []
            for col in existing_df.columns:
                if '_rank' not in col and col not in ['rank_std', 'rank_mean']:
                    existing_imp = existing_df[col]
                    # Align indices
                    common_features = imp_df.index.intersection(existing_imp.index)
                    if len(common_features) > 5:
                        rho, _ = spearmanr(
                            imp_df.loc[common_features].rank(ascending=False),
                            existing_imp.loc[common_features].rank(ascending=False)
                        )
                        correlations.append(rho)

            avg_rho = np.mean(correlations) if correlations else 0
            print(f"\n{model_name}:")
            print(f"  Top SHAP Feature: {data['top_feature']} ({data['shap_value']:.3f})")
            print(f"  MDI-SHAP correlation: {data['mdi_shap_rho']:.3f}")
            print(f"  Avg correlation with other models: {avg_rho:.3f}")

            results[model_name]['avg_cross_model_rho'] = avg_rho

    # Print LaTeX table rows
    print("\n" + "="*60)
    print("LATEX TABLE ROWS TO ADD")
    print("="*60)

    for model_name, data in results.items():
        avg_rho = data.get('avg_cross_model_rho', 0)
        top_feat = data['top_feature'].replace('_', r'\_')
        print(f"{model_name} & {top_feat} & {data['shap_value']:.3f} & {avg_rho:.2f} \\\\")


if __name__ == '__main__':
    main()
