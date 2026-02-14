#!/usr/bin/env python3
"""
Per-Model Feature Importance Analysis.

Analyzes whether feature importance rankings change across different models
and temperatures. This addresses the question: Do we need model-specific
feature importance analysis?

Usage:
    python scripts/analysis/per_model_feature_importance.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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


def compute_feature_importance(df, feature_cols, target='stability_score'):
    """Compute Random Forest feature importance."""
    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_,
    }).sort_values('importance', ascending=False)

    return importance_df, cv_scores.mean(), cv_scores.std()


def analyze_per_model(df, models, feature_cols):
    """Run feature importance for each model separately."""
    results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Analyzing: {model}")
        print('='*60)

        model_df = df[df['model'] == model]
        print(f"  Samples: {len(model_df)}")

        if len(model_df) < 100:
            print(f"  Skipping - insufficient samples")
            continue

        importance_df, r2_mean, r2_std = compute_feature_importance(
            model_df, feature_cols
        )

        print(f"  R² (CV): {r2_mean:.3f} ± {r2_std:.3f}")
        print(f"\n  Top 10 Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:30s} {row['importance']:.4f}")

        results[model] = {
            'importance': importance_df,
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'n_samples': len(model_df),
        }

    return results


def analyze_per_temperature(df, temperatures, feature_cols):
    """Run feature importance for each temperature separately."""
    results = {}

    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Analyzing: Temperature = {temp}")
        print('='*60)

        temp_df = df[df['temperature'] == temp]
        print(f"  Samples: {len(temp_df)}")

        if len(temp_df) < 100:
            print(f"  Skipping - insufficient samples")
            continue

        importance_df, r2_mean, r2_std = compute_feature_importance(
            temp_df, feature_cols
        )

        print(f"  R² (CV): {r2_mean:.3f} ± {r2_std:.3f}")
        print(f"\n  Top 10 Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:30s} {row['importance']:.4f}")

        results[temp] = {
            'importance': importance_df,
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'n_samples': len(temp_df),
        }

    return results


def compare_rankings(results, name_field='model'):
    """Compare feature importance rankings across models/temperatures."""
    print(f"\n{'='*60}")
    print("RANKING COMPARISON")
    print('='*60)

    # Get all importance dataframes
    all_importances = {}
    for key, data in results.items():
        imp_df = data['importance'].set_index('feature')['importance']
        all_importances[key] = imp_df

    # Create combined dataframe
    combined = pd.DataFrame(all_importances)

    # Add rank columns
    for col in combined.columns:
        combined[f'{str(col)}_rank'] = combined[col].rank(ascending=False)

    # Compute pairwise Spearman correlations
    keys = list(results.keys())
    n = len(keys)

    print("\nPairwise Spearman Rank Correlations:")
    corr_matrix = np.zeros((n, n))

    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i <= j:
                rho, p = spearmanr(
                    combined[f'{k1}_rank'].values,
                    combined[f'{k2}_rank'].values
                )
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    corr_df = pd.DataFrame(corr_matrix, index=keys, columns=keys)
    print(corr_df.round(3).to_string())

    # Find most variable features
    rank_cols = [c for c in combined.columns if str(c).endswith('_rank')]
    combined['rank_std'] = combined[rank_cols].std(axis=1)
    combined['rank_mean'] = combined[rank_cols].mean(axis=1)

    print("\n\nMost Variable Features (highest rank std):")
    most_variable = combined.sort_values('rank_std', ascending=False).head(10)
    for feat, row in most_variable.iterrows():
        ranks = [f"{row[f'{str(k)}_rank']:.0f}" for k in keys]
        print(f"  {feat:30s} ranks: {', '.join(ranks)} (std={row['rank_std']:.1f})")

    print("\n\nMost Stable Features (lowest rank std):")
    most_stable = combined.sort_values('rank_std', ascending=True).head(10)
    for feat, row in most_stable.iterrows():
        ranks = [f"{row[f'{str(k)}_rank']:.0f}" for k in keys]
        print(f"  {feat:30s} ranks: {', '.join(ranks)} (std={row['rank_std']:.1f})")

    return combined, corr_df


def plot_comparison(results, output_dir, name='models'):
    """Create comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get top features from each model
    all_top_features = set()
    for key, data in results.items():
        top_features = data['importance'].head(10)['feature'].tolist()
        all_top_features.update(top_features)

    # Create heatmap data
    heatmap_data = {}
    for key, data in results.items():
        imp_df = data['importance'].set_index('feature')['importance']
        # Convert key to string for column name
        heatmap_data[str(key)] = imp_df

    combined = pd.DataFrame(heatmap_data)
    combined = combined.loc[list(all_top_features)]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(combined, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title(f'Feature Importance by {name.title()}', fontsize=14, fontweight='bold')
    ax.set_xlabel(name.title(), fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_importance_by_{name}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'feature_importance_by_{name}.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to {output_dir / f'feature_importance_by_{name}.png'}")


def main():
    print("="*60)
    print("PER-MODEL FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Total samples: {len(df)}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Temperatures: {sorted(df['temperature'].unique())}")

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")

    # Use FINAL_MODELS from model_config.py
    FINAL_MODELS = [
        'Qwen3-235B-A22B',
        'Claude-3.5-Sonnet',
        'Claude-Haiku-4.5',
        'Claude-3.7-Sonnet',
        'Claude-3.5-Haiku',
        'Claude-Opus-4.5',
        'Claude-Opus-4',
        'Claude-Sonnet-4',
        'Claude-Sonnet-4.5',
        'Qwen3-32B',
        'Llama-3.3-70B',
        'Nova-2-Lite',
        'Mimo-V2-Flash',  # Also matches 'Mimo-V2-Flash:free'
        'Grok-4.1-Fast',
        'Minimax-M2',
        'GPT-4.1-Mini',
        'Gemini-2.5-Flash-Lite',
        'GPT-OSS-120B',
    ]

    # Match models in data (handle suffix like ':free')
    available_models = df['model'].unique().tolist()
    selected_models = []
    for fm in FINAL_MODELS:
        for am in available_models:
            if fm in am or am in fm:
                selected_models.append(am)
                break

    print(f"\nAnalyzing {len(selected_models)} FINAL_MODELS: {selected_models}")

    # Run global analysis first
    print("\n" + "="*60)
    print("GLOBAL ANALYSIS (ALL DATA)")
    print("="*60)
    global_imp, global_r2, global_std = compute_feature_importance(df, feature_cols)
    print(f"R² (CV): {global_r2:.3f} ± {global_std:.3f}")
    print("\nTop 10 Features (Global):")
    for i, row in global_imp.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    # Run per-model analysis
    model_results = analyze_per_model(df, selected_models, feature_cols)

    # Compare rankings across models
    if len(model_results) > 1:
        combined_models, corr_models = compare_rankings(model_results, 'model')

        # Plot comparison
        output_dir = PROJECT_ROOT / 'results/factor_analysis/per_model'
        plot_comparison(model_results, output_dir, 'models')

        # Save results
        combined_models.to_csv(output_dir / 'feature_importance_comparison_models.csv')
        corr_models.to_csv(output_dir / 'rank_correlation_models.csv')

    # Run per-temperature analysis
    print("\n\n" + "="*60)
    print("PER-TEMPERATURE ANALYSIS")
    print("="*60)

    temperatures = [0.0, 0.3, 0.5, 0.7, 1.0]
    temp_results = analyze_per_temperature(df, temperatures, feature_cols)

    if len(temp_results) > 1:
        combined_temps, corr_temps = compare_rankings(temp_results, 'temperature')

        output_dir = PROJECT_ROOT / 'results/factor_analysis/per_model'
        plot_comparison(temp_results, output_dir, 'temperatures')

        combined_temps.to_csv(output_dir / 'feature_importance_comparison_temps.csv')
        corr_temps.to_csv(output_dir / 'rank_correlation_temps.csv')

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Average correlation
    if len(model_results) > 1:
        # Get off-diagonal elements
        mask = ~np.eye(corr_models.shape[0], dtype=bool)
        avg_model_corr = corr_models.values[mask].mean()
        print(f"\nAverage rank correlation across MODELS: {avg_model_corr:.3f}")

    if len(temp_results) > 1:
        mask = ~np.eye(corr_temps.shape[0], dtype=bool)
        avg_temp_corr = corr_temps.values[mask].mean()
        print(f"Average rank correlation across TEMPERATURES: {avg_temp_corr:.3f}")

    print("\nConclusion:")
    if len(model_results) > 1 and avg_model_corr > 0.7:
        print("  -> Feature importance rankings are STABLE across models (rho > 0.7)")
        print("  -> Per-model analysis NOT required for practical purposes")
    elif len(model_results) > 1 and avg_model_corr > 0.5:
        print("  -> Feature importance rankings are MODERATELY stable (0.5 < rho < 0.7)")
        print("  -> Per-model analysis may reveal useful nuances")
    else:
        print("  -> Feature importance rankings VARY significantly across models (rho < 0.5)")
        print("  -> Per-model analysis IS recommended")


if __name__ == '__main__':
    main()
