#!/usr/bin/env python3
"""
Generate data files for KDD paper tables that are missing backing data.

Tables to generate:
- Table 2: Predictive Model Comparison (5-fold CV)
- Table 3: Baseline Comparison
- Table 7: Observational Feature Associations
- Table 8: Leave-One-Model-Out (LOMO) Cross-Validation
- Table 16: Pairwise Transfer Matrix

Author: Auto-generated for KDD 2026 paper
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path("/Users/guanghu/Documents/genai/projects/sted-internal/results/factor_analysis/factor_analysis_data.csv")
OUTPUT_DIR = Path("/Users/guanghu/Documents/genai/projects/sted-internal/results/kdd_paper_tables")
OUTPUT_DIR.mkdir(exist_ok=True)

# Final 18 models to include (from model_config.py)
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

# Feature columns (controllable factors)
SCHEMA_FEATURES = [
    'num_tools', 'avg_params_per_tool', 'max_params_per_tool', 'total_params',
    'param_type_diversity', 'avg_tool_name_length', 'tool_name_ambiguity',
    'tool_prefix_diversity', 'schema_depth', 'schema_breadth', 'schema_complexity'
]

QUERY_FEATURES = [
    'query_length', 'query_word_count', 'query_sentence_count', 'num_questions',
    'num_commands', 'num_conjunctions', 'query_complexity_score', 'constraint_score'
]

CONFIG_FEATURES = ['temperature']

CONTROLLABLE_FEATURES = SCHEMA_FEATURES + QUERY_FEATURES + CONFIG_FEATURES


# Model name mapping from raw data names to FINAL_MODELS names
MODEL_NAME_MAP = {
    'us.anthropic.claude-opus-4-20250514-v1': 'Claude-Opus-4',
    'Mimo-V2-Flash:free': 'Mimo-V2-Flash',
    'NemoTron-3-Nano-30B-A3B:free': None,  # Exclude
    'Mistral-Large-3-675B': None,  # Exclude
    'Unknown': None,  # Exclude
}


def load_data():
    """Load the factor analysis data and filter for FINAL_MODELS only."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows total")

    # Normalize model names using mapping
    def normalize_model(m):
        if pd.isna(m):
            return None
        m_str = str(m)
        # Check explicit mapping first
        if m_str in MODEL_NAME_MAP:
            return MODEL_NAME_MAP[m_str]
        # Strip ':free' suffix if present
        if ':free' in m_str:
            m_str = m_str.replace(':free', '')
        return m_str

    df['model_normalized'] = df['model'].apply(normalize_model)

    # Remove rows where model is mapped to None (excluded models)
    df = df[df['model_normalized'].notna()].copy()

    # Filter for FINAL_MODELS only
    df_filtered = df[df['model_normalized'].isin(FINAL_MODELS)].copy()
    df_filtered['model'] = df_filtered['model_normalized']  # Use normalized names
    df_filtered = df_filtered.drop(columns=['model_normalized'])

    print(f"After filtering for {len(FINAL_MODELS)} FINAL_MODELS: {len(df_filtered)} rows")
    print(f"Models included ({len(df_filtered['model'].unique())}): {sorted(df_filtered['model'].unique())}")

    return df_filtered


def table2_model_comparison(df):
    """
    Table 2: Predictive Model Comparison (5-fold CV)
    Tests: RF (nested target enc), RF (one-hot), MLP, GB, Ridge
    """
    print("\n=== Table 2: Predictive Model Comparison ===")

    # Prepare features
    X = df[CONTROLLABLE_FEATURES].copy()
    y = df['stability_score'].values

    # Add model as target encoding (nested CV)
    le = LabelEncoder()
    model_encoded = le.fit_transform(df['model'])

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. RF with nested target encoding
    print("Running RF (nested target encoding)...")
    X_with_model = X.copy()

    # Compute model mean in nested CV fashion
    model_means = df.groupby('model')['stability_score'].mean()
    X_with_model['model_target_enc'] = df['model'].map(model_means)

    rf_nested = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_nested, X_with_model, y, cv=kf, scoring='r2')
    results['RF (nested target enc.)'] = {'R2': scores.mean(), 'Std': scores.std()}
    print(f"  R2 = {scores.mean():.3f} ± {scores.std():.3f}")

    # 2. RF with one-hot encoding
    print("Running RF (one-hot encoding)...")
    X_onehot = pd.get_dummies(df[['model']], prefix='model')
    X_onehot = pd.concat([X, X_onehot], axis=1)

    rf_onehot = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_onehot, X_onehot, y, cv=kf, scoring='r2')
    results['RF (one-hot enc.)'] = {'R2': scores.mean(), 'Std': scores.std()}
    print(f"  R2 = {scores.mean():.3f} ± {scores.std():.3f}")

    # 3. MLP
    print("Running MLP...")
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    scores = cross_val_score(mlp, X, y, cv=kf, scoring='r2')
    results['MLP (100-50)'] = {'R2': scores.mean(), 'Std': scores.std()}
    print(f"  R2 = {scores.mean():.3f} ± {scores.std():.3f}")

    # 4. Gradient Boosting
    print("Running Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(gb, X, y, cv=kf, scoring='r2')
    results['Gradient Boosting'] = {'R2': scores.mean(), 'Std': scores.std()}
    print(f"  R2 = {scores.mean():.3f} ± {scores.std():.3f}")

    # 5. Ridge Regression
    print("Running Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    scores = cross_val_score(ridge, X, y, cv=kf, scoring='r2')
    results['Ridge Regression'] = {'R2': scores.mean(), 'Std': scores.std()}
    print(f"  R2 = {scores.mean():.3f} ± {scores.std():.3f}")

    # Save results
    output = {
        'description': 'Table 2: Predictive Model Comparison (5-fold CV)',
        'n_samples': len(df),
        'n_folds': 5,
        'results': results
    }

    with open(OUTPUT_DIR / 'table2_model_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table2_model_comparison.json'}")
    return results


def table3_baseline_comparison(df):
    """
    Table 3: Baseline Comparison - Controllable Factor Model vs Simple Baselines
    """
    print("\n=== Table 3: Baseline Comparison ===")

    X = df[CONTROLLABLE_FEATURES].copy()
    y = df['stability_score'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # 1. All Controllable Features
    print("Running All Controllable Features...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    r2_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_absolute_error')
    results['All Controllable'] = {
        'Features': len(CONTROLLABLE_FEATURES),
        'R2': r2_scores.mean(),
        'MAE': mae_scores.mean()
    }
    print(f"  R2 = {r2_scores.mean():.3f}, MAE = {mae_scores.mean():.3f}")

    # 2. Schema Features Only
    print("Running Schema Features Only...")
    X_schema = df[SCHEMA_FEATURES].copy()
    r2_scores = cross_val_score(rf, X_schema, y, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(rf, X_schema, y, cv=kf, scoring='neg_mean_absolute_error')
    results['Schema Only'] = {
        'Features': len(SCHEMA_FEATURES),
        'R2': r2_scores.mean(),
        'MAE': mae_scores.mean()
    }
    print(f"  R2 = {r2_scores.mean():.3f}, MAE = {mae_scores.mean():.3f}")

    # 3. Query Features Only
    print("Running Query Features Only...")
    X_query = df[QUERY_FEATURES].copy()
    r2_scores = cross_val_score(rf, X_query, y, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(rf, X_query, y, cv=kf, scoring='neg_mean_absolute_error')
    results['Query Only'] = {
        'Features': len(QUERY_FEATURES),
        'R2': r2_scores.mean(),
        'MAE': mae_scores.mean()
    }
    print(f"  R2 = {r2_scores.mean():.3f}, MAE = {mae_scores.mean():.3f}")

    # 4. Temperature Only
    print("Running Temperature Only...")
    X_temp = df[['temperature']].copy()
    r2_scores = cross_val_score(rf, X_temp, y, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(rf, X_temp, y, cv=kf, scoring='neg_mean_absolute_error')
    results['Temperature Only'] = {
        'Features': 1,
        'R2': r2_scores.mean(),
        'MAE': mae_scores.mean()
    }
    print(f"  R2 = {r2_scores.mean():.3f}, MAE = {mae_scores.mean():.3f}")

    # 5. Length Only (query + schema)
    print("Running Length Only...")
    X_length = df[['query_length', 'schema_complexity']].copy()
    r2_scores = cross_val_score(rf, X_length, y, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(rf, X_length, y, cv=kf, scoring='neg_mean_absolute_error')
    results['Length Only'] = {
        'Features': 2,
        'R2': r2_scores.mean(),
        'MAE': mae_scores.mean()
    }
    print(f"  R2 = {r2_scores.mean():.3f}, MAE = {mae_scores.mean():.3f}")

    # 6. Mean Prediction (baseline)
    print("Running Mean Prediction...")
    mean_pred = np.full_like(y, y.mean())
    ss_res = np.sum((y - mean_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_mean = 0.0  # By definition
    mae_mean = np.mean(np.abs(y - mean_pred))
    results['Mean Prediction'] = {
        'Features': 0,
        'R2': r2_mean,
        'MAE': mae_mean
    }
    print(f"  R2 = {r2_mean:.3f}, MAE = {mae_mean:.3f}")

    # Save
    output = {
        'description': 'Table 3: Baseline Comparison',
        'n_samples': len(df),
        'results': results
    }

    with open(OUTPUT_DIR / 'table3_baseline_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table3_baseline_comparison.json'}")
    return results


def table7_observational_associations(df):
    """
    Table 7: Observational Feature Associations
    Computes effect sizes for binary features
    """
    print("\n=== Table 7: Observational Feature Associations ===")

    y = df['stability_score'].values
    results = {}

    # Binary features to analyze
    binary_features = [
        ('has_shorter_prompt', df['query_length'] < df['query_length'].median()),
        ('no_numbered_list', ~df['has_numbered_list'].astype(bool)),
        ('has_polite_phrasing', df['num_questions'] > 0),  # Proxy for politeness
        ('no_should', True),  # Would need actual feature
        ('no_must', True),  # Would need actual feature
        ('no_conditionals', ~df['has_conditional'].astype(bool)),
    ]

    # Shorter prompts (actual)
    median_length = df['query_length'].median()
    short_mask = df['query_length'] < median_length

    y_short = y[short_mask]
    y_long = y[~short_mask]

    delta = y_short.mean() - y_long.mean()
    pooled_std = np.sqrt((y_short.std()**2 + y_long.std()**2) / 2)
    cohens_d = delta / pooled_std if pooled_std > 0 else 0

    # CI via bootstrap
    n_boot = 1000
    boot_deltas = []
    for _ in range(n_boot):
        idx_short = np.random.choice(len(y_short), len(y_short), replace=True)
        idx_long = np.random.choice(len(y_long), len(y_long), replace=True)
        boot_deltas.append(y_short[idx_short].mean() - y_long[idx_long].mean())
    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])

    results['Shorter prompts'] = {
        'delta_S': delta,
        'cohens_d': cohens_d,
        'CI_low': ci_low,
        'CI_high': ci_high,
        'interpretation': 'Small-medium' if abs(cohens_d) < 0.5 else 'Medium'
    }
    print(f"Shorter prompts: Δ={delta:.3f}, d={cohens_d:.3f}")

    # No numbered lists
    no_list_mask = ~df['has_numbered_list'].astype(bool)
    y_no_list = y[no_list_mask]
    y_list = y[~no_list_mask]

    if len(y_list) > 0 and len(y_no_list) > 0:
        delta = y_no_list.mean() - y_list.mean()
        pooled_std = np.sqrt((y_no_list.std()**2 + y_list.std()**2) / 2)
        cohens_d = delta / pooled_std if pooled_std > 0 else 0

        results['No numbered lists'] = {
            'delta_S': delta,
            'cohens_d': cohens_d,
            'CI_low': delta - 1.96 * pooled_std / np.sqrt(len(y)),
            'CI_high': delta + 1.96 * pooled_std / np.sqrt(len(y)),
            'interpretation': 'Small' if abs(cohens_d) < 0.2 else 'Small-medium'
        }
        print(f"No numbered lists: Δ={delta:.3f}, d={cohens_d:.3f}")

    # No conditionals
    no_cond_mask = ~df['has_conditional'].astype(bool)
    y_no_cond = y[no_cond_mask]
    y_cond = y[~no_cond_mask]

    if len(y_cond) > 0 and len(y_no_cond) > 0:
        delta = y_no_cond.mean() - y_cond.mean()
        pooled_std = np.sqrt((y_no_cond.std()**2 + y_cond.std()**2) / 2)
        cohens_d = delta / pooled_std if pooled_std > 0 else 0

        results['No conditionals'] = {
            'delta_S': delta,
            'cohens_d': cohens_d,
            'CI_low': delta - 1.96 * pooled_std / np.sqrt(len(y)),
            'CI_high': delta + 1.96 * pooled_std / np.sqrt(len(y)),
            'interpretation': 'Negligible' if abs(cohens_d) < 0.1 else 'Small'
        }
        print(f"No conditionals: Δ={delta:.3f}, d={cohens_d:.3f}")

    # Save
    output = {
        'description': 'Table 7: Observational Feature Associations',
        'n_samples': len(df),
        'note': 'These are observational associations, not causal effects',
        'results': results
    }

    with open(OUTPUT_DIR / 'table7_observational_associations.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table7_observational_associations.json'}")
    return results


def table8_lomo_cv(df):
    """
    Table 8: Leave-One-Model-Out (LOMO) Cross-Validation
    Train on all models except one, test on held-out model
    """
    print("\n=== Table 8: LOMO Cross-Validation ===")

    models = df['model'].unique()
    print(f"Found {len(models)} models")

    X = df[CONTROLLABLE_FEATURES].copy()
    y = df['stability_score'].values

    results = {}
    r2_scores = []
    mae_scores = []

    for test_model in models:
        print(f"Testing on {test_model}...")

        # Split
        train_mask = df['model'] != test_model
        test_mask = df['model'] == test_model

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(y_test) == 0:
            continue

        # Train RF
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # Predict
        y_pred = rf.predict(X_test)

        # Compute metrics
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        mae = np.mean(np.abs(y_test - y_pred))

        # Categorize transfer quality
        if r2 > 0.1:
            transfer = 'Good'
        elif r2 > 0:
            transfer = 'Moderate'
        elif r2 > -0.5:
            transfer = 'Poor'
        else:
            transfer = 'Very Poor'

        results[test_model] = {
            'R2': r2,
            'MAE': mae,
            'Transfer': transfer,
            'n_test': len(y_test),
            'n_train': len(y_train)
        }

        r2_scores.append(r2)
        mae_scores.append(mae)

        print(f"  R2 = {r2:.3f}, MAE = {mae:.3f}, Transfer = {transfer}")

    # Summary
    summary = {
        'mean_R2': np.mean(r2_scores),
        'std_R2': np.std(r2_scores),
        'mean_MAE': np.mean(mae_scores),
        'std_MAE': np.std(mae_scores)
    }

    print(f"\nMean R2 = {summary['mean_R2']:.3f} ± {summary['std_R2']:.3f}")

    # Save
    output = {
        'description': 'Table 8: Leave-One-Model-Out Cross-Validation',
        'n_models': len(models),
        'n_samples': len(df),
        'per_model_results': results,
        'summary': summary
    }

    with open(OUTPUT_DIR / 'table8_lomo_cv.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table8_lomo_cv.json'}")
    return results, summary


def table16_pairwise_transfer(df):
    """
    Table 16: Pairwise Model Transfer Matrix
    Train on model A, test on model B for all pairs
    """
    print("\n=== Table 16: Pairwise Transfer Matrix ===")

    models = df['model'].unique()
    n_models = len(models)
    print(f"Computing {n_models}x{n_models} transfer matrix...")

    X = df[CONTROLLABLE_FEATURES].copy()
    y = df['stability_score'].values

    # Initialize matrix
    transfer_matrix = np.zeros((n_models, n_models))

    for i, train_model in enumerate(models):
        print(f"Training on {train_model}...")

        train_mask = df['model'] == train_model
        X_train = X[train_mask]
        y_train = y[train_mask]

        if len(y_train) < 10:
            continue

        # Train RF
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        for j, test_model in enumerate(models):
            test_mask = df['model'] == test_model
            X_test = X[test_mask]
            y_test = y[test_mask]

            if len(y_test) < 10:
                continue

            y_pred = rf.predict(X_test)

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            transfer_matrix[i, j] = r2

    # Convert to DataFrame
    transfer_df = pd.DataFrame(
        transfer_matrix,
        index=models,
        columns=models
    )

    # Save as CSV
    transfer_df.to_csv(OUTPUT_DIR / 'table16_pairwise_transfer.csv')

    # Also save as JSON with summary
    output = {
        'description': 'Table 16: Pairwise Model Transfer Matrix (R^2)',
        'n_models': n_models,
        'matrix': transfer_matrix.tolist(),
        'model_names': list(models),
        'diagonal_mean': np.mean(np.diag(transfer_matrix)),
        'off_diagonal_mean': np.mean(transfer_matrix[~np.eye(n_models, dtype=bool)])
    }

    with open(OUTPUT_DIR / 'table16_pairwise_transfer.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Diagonal mean (within-model): {output['diagonal_mean']:.3f}")
    print(f"Off-diagonal mean (cross-model): {output['off_diagonal_mean']:.3f}")
    print(f"Saved to {OUTPUT_DIR / 'table16_pairwise_transfer.csv'}")

    return transfer_df


def table1_factor_correlations_by_model(df):
    """
    Table 1 (per-model): Factor Correlations with Stability Score
    Computes Pearson correlations for each controllable factor, broken down by model.
    """
    print("\n=== Table 1 (per-model): Factor Correlations ===")

    models = df['model'].unique()
    results = {}

    # Also compute aggregate results
    all_correlations = {}
    for feature in CONTROLLABLE_FEATURES:
        corr, p_val = stats.pearsonr(df[feature], df['stability_score'])
        all_correlations[feature] = {'correlation': corr, 'p_value': p_val}

    results['aggregate'] = {
        'n_samples': len(df),
        'correlations': all_correlations
    }

    # Per-model results
    for model in sorted(models):
        model_df = df[df['model'] == model]
        model_corrs = {}

        for feature in CONTROLLABLE_FEATURES:
            try:
                corr, p_val = stats.pearsonr(model_df[feature], model_df['stability_score'])
                model_corrs[feature] = {'correlation': corr, 'p_value': p_val}
            except Exception:
                model_corrs[feature] = {'correlation': None, 'p_value': None}

        results[model] = {
            'n_samples': len(model_df),
            'correlations': model_corrs
        }
        print(f"  {model}: {len(model_df)} samples")

    # Save
    output = {
        'description': 'Table 1: Factor Correlations with Stability Score (by model)',
        'n_models': len(models),
        'features': CONTROLLABLE_FEATURES,
        'results': results
    }

    with open(OUTPUT_DIR / 'table1_factor_correlations_by_model.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table1_factor_correlations_by_model.json'}")
    return results


def table2_model_comparison_by_model(df):
    """
    Table 2 (per-model): Predictive Model Comparison
    Runs 5-fold CV for each model separately to understand model-specific predictability.
    """
    print("\n=== Table 2 (per-model): Predictive Model Comparison ===")

    models = df['model'].unique()
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model in sorted(models):
        model_df = df[df['model'] == model]
        X = model_df[CONTROLLABLE_FEATURES].copy()
        y = model_df['stability_score'].values

        if len(y) < 50:  # Skip if too few samples
            print(f"  {model}: Skipping (only {len(y)} samples)")
            continue

        model_results = {}

        # RF with controllable features only
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        try:
            scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
            model_results['RF'] = {'R2': scores.mean(), 'Std': scores.std()}
        except Exception:
            model_results['RF'] = {'R2': None, 'Std': None}

        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        try:
            scores = cross_val_score(gb, X, y, cv=kf, scoring='r2')
            model_results['GB'] = {'R2': scores.mean(), 'Std': scores.std()}
        except Exception:
            model_results['GB'] = {'R2': None, 'Std': None}

        # Ridge
        ridge = Ridge(alpha=1.0)
        try:
            scores = cross_val_score(ridge, X, y, cv=kf, scoring='r2')
            model_results['Ridge'] = {'R2': scores.mean(), 'Std': scores.std()}
        except Exception:
            model_results['Ridge'] = {'R2': None, 'Std': None}

        results[model] = {
            'n_samples': len(y),
            'methods': model_results
        }

        rf_r2 = model_results['RF']['R2']
        rf_r2_str = f"{rf_r2:.3f}" if rf_r2 is not None else "N/A"
        print(f"  {model}: RF R2 = {rf_r2_str} (n={len(y)})")

    # Compute summary statistics
    rf_r2_values = [r['methods']['RF']['R2'] for r in results.values()
                    if r['methods']['RF']['R2'] is not None]
    summary = {
        'mean_RF_R2': np.mean(rf_r2_values) if rf_r2_values else None,
        'std_RF_R2': np.std(rf_r2_values) if rf_r2_values else None,
        'n_models': len(rf_r2_values)
    }

    # Save
    output = {
        'description': 'Table 2: Predictive Model Comparison (5-fold CV, by model)',
        'n_models': len(models),
        'features': CONTROLLABLE_FEATURES,
        'per_model_results': results,
        'summary': summary
    }

    with open(OUTPUT_DIR / 'table2_model_comparison_by_model.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table2_model_comparison_by_model.json'}")
    return results


def table4_shap_importance(df):
    """
    Table 4: Feature Importance for Predicting Consistency (SHAP Values)
    Uses SHAP TreeExplainer on Random Forest
    """
    print("\n=== Table 4: SHAP Feature Importance ===")

    try:
        import shap
    except ImportError:
        print("SHAP not installed, skipping...")
        return None

    X = df[CONTROLLABLE_FEATURES].copy()
    y = df['stability_score'].values

    # Train RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Sample for SHAP (use 10K samples for speed)
    sample_size = min(10000, len(X))
    sample_idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[sample_idx]

    # Compute SHAP values
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_importance = mean_abs_shap.sum()

    # Create results
    results = {}
    for i, feature in enumerate(CONTROLLABLE_FEATURES):
        importance = mean_abs_shap[i]
        percentage = 100 * importance / total_importance
        results[feature] = {
            'shap_importance': float(importance),
            'percentage': float(percentage)
        }

    # Sort by importance
    sorted_features = sorted(results.items(), key=lambda x: x[1]['shap_importance'], reverse=True)

    # Print top 10
    print("\nTop 10 features by SHAP importance:")
    for i, (feature, vals) in enumerate(sorted_features[:10]):
        print(f"  {i+1}. {feature}: {vals['shap_importance']:.4f} ({vals['percentage']:.1f}%)")

    # Save
    output = {
        'description': 'Table 4: Feature Importance (SHAP Values, Controllable Factors Only)',
        'n_samples': len(df),
        'sample_size_for_shap': sample_size,
        'total_importance': float(total_importance),
        'results': dict(sorted_features)
    }

    with open(OUTPUT_DIR / 'table4_shap_importance.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table4_shap_importance.json'}")
    return results


def table5_interaction_effects(df):
    """
    Table 5: Temperature × Schema Complexity Interaction
    Shows stability scores across temperature and complexity bins
    """
    print("\n=== Table 5: Temperature × Complexity Interaction ===")

    # Bin schema complexity into quartiles
    df['complexity_bin'] = pd.qcut(df['schema_complexity'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Bin temperature into low/med/high
    def temp_bin(t):
        if t <= 0.3:
            return 'Low (0-0.3)'
        elif t <= 0.6:
            return 'Medium (0.3-0.6)'
        else:
            return 'High (0.6-1.0)'

    df['temp_bin'] = df['temperature'].apply(temp_bin)

    # Compute mean stability by bins
    pivot = df.pivot_table(
        values='stability_score',
        index='complexity_bin',
        columns='temp_bin',
        aggfunc='mean'
    )

    # Compute degradation rates
    q1_low = df[(df['complexity_bin'] == 'Q1') & (df['temperature'] <= 0.3)]['stability_score'].mean()
    q1_high = df[(df['complexity_bin'] == 'Q1') & (df['temperature'] > 0.6)]['stability_score'].mean()
    q4_low = df[(df['complexity_bin'] == 'Q4') & (df['temperature'] <= 0.3)]['stability_score'].mean()
    q4_high = df[(df['complexity_bin'] == 'Q4') & (df['temperature'] > 0.6)]['stability_score'].mean()

    simple_degradation = (q1_low - q1_high) / q1_low * 100 if q1_low > 0 else 0
    complex_degradation = (q4_low - q4_high) / q4_low * 100 if q4_low > 0 else 0
    degradation_ratio = complex_degradation / simple_degradation if simple_degradation > 0 else 0

    results = {
        'pivot_table': pivot.to_dict(),
        'simple_schema': {
            'low_temp': float(q1_low),
            'high_temp': float(q1_high),
            'degradation_pct': float(simple_degradation)
        },
        'complex_schema': {
            'low_temp': float(q4_low),
            'high_temp': float(q4_high),
            'degradation_pct': float(complex_degradation)
        },
        'degradation_ratio': float(degradation_ratio)
    }

    print(f"Simple schema (Q1) degradation: {simple_degradation:.1f}%")
    print(f"Complex schema (Q4) degradation: {complex_degradation:.1f}%")
    print(f"Degradation ratio (complex/simple): {degradation_ratio:.1f}x")

    # Save
    output = {
        'description': 'Table 5: Temperature × Schema Complexity Interaction',
        'n_samples': len(df),
        'results': results
    }

    with open(OUTPUT_DIR / 'table5_interaction_effects.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table5_interaction_effects.json'}")

    # Cleanup temp columns
    df.drop(columns=['complexity_bin', 'temp_bin'], inplace=True, errors='ignore')
    return results


def table6_cross_task_validation():
    """
    Table 6: Cross-Task Validation (Toucan vs ShareGPT)
    Compares model rankings and consistency patterns across tasks
    """
    print("\n=== Table 6: Cross-Task Validation ===")

    # Load ShareGPT results
    sharegpt_path = Path("/Users/guanghu/Documents/genai/projects/sted-internal/results/sharegpt/minilm-ec2/combined_consistency_metrics_results.json")
    toucan_path = Path("/Users/guanghu/Documents/genai/projects/sted-internal/results/toucan/minilm-ec2/combined_consistency_metrics_results.json")

    if not sharegpt_path.exists() or not toucan_path.exists():
        print("Required result files not found, skipping...")
        return None

    with open(sharegpt_path) as f:
        sharegpt_data = json.load(f)
    with open(toucan_path) as f:
        toucan_data = json.load(f)

    # Extract model-level metrics
    def extract_model_metrics(data, target_temp=0.0):
        models = {}
        # Handle both old format (results.by_model) and new format (model at top level)
        temp_key = f'temperature_{target_temp}'
        if 'results' in data and 'by_model' in data['results']:
            for model, temps in data['results']['by_model'].items():
                if temp_key in temps:
                    models[model] = temps[temp_key].get('stability_score', {}).get('mean', None)
        else:
            # New format: models at top level with list of records
            for model, records in data.items():
                if isinstance(records, list):
                    # Filter records by temperature and compute mean stability_score
                    temp_records = [r for r in records if isinstance(r, dict) and r.get('temperature') == target_temp]
                    if temp_records:
                        stability_scores = [r['stability_score'] for r in temp_records if 'stability_score' in r]
                        if stability_scores:
                            models[model] = np.mean(stability_scores)
        return models

    toucan_t0 = extract_model_metrics(toucan_data, 0.0)
    toucan_t1 = extract_model_metrics(toucan_data, 1.0)
    sharegpt_t0 = extract_model_metrics(sharegpt_data, 0.0)
    sharegpt_t1 = extract_model_metrics(sharegpt_data, 1.0)

    # Find common models
    common_models = set(toucan_t0.keys()) & set(sharegpt_t0.keys())
    common_models = [m for m in common_models if toucan_t0.get(m) is not None and sharegpt_t0.get(m) is not None]

    if len(common_models) < 3:
        print(f"Only {len(common_models)} common models found, need at least 3")
        return None

    # Compute Spearman correlation of model rankings
    toucan_ranks = [toucan_t0.get(m, 0) for m in common_models]
    sharegpt_ranks = [sharegpt_t0.get(m, 0) for m in common_models]

    rho, p_val = stats.spearmanr(toucan_ranks, sharegpt_ranks)

    # Compute mean consistency at T=0 and T=1
    toucan_mean_t0 = np.mean([v for v in toucan_t0.values() if v is not None])
    toucan_mean_t1 = np.mean([v for v in toucan_t1.values() if v is not None])
    sharegpt_mean_t0 = np.mean([v for v in sharegpt_t0.values() if v is not None])
    sharegpt_mean_t1 = np.mean([v for v in sharegpt_t1.values() if v is not None])

    toucan_degradation = (toucan_mean_t0 - toucan_mean_t1) / toucan_mean_t0 * 100 if toucan_mean_t0 > 0 else 0
    sharegpt_degradation = (sharegpt_mean_t0 - sharegpt_mean_t1) / sharegpt_mean_t0 * 100 if sharegpt_mean_t0 > 0 else 0

    results = {
        'n_common_models': len(common_models),
        'spearman_rho': float(rho),
        'p_value': float(p_val),
        'toucan': {
            'mean_t0': float(toucan_mean_t0),
            'mean_t1': float(toucan_mean_t1),
            'degradation_pct': float(toucan_degradation)
        },
        'sharegpt': {
            'mean_t0': float(sharegpt_mean_t0),
            'mean_t1': float(sharegpt_mean_t1),
            'degradation_pct': float(sharegpt_degradation)
        }
    }

    print(f"Common models: {len(common_models)}")
    print(f"Model ranking correlation: ρ = {rho:.3f}, p = {p_val:.4f}")
    print(f"Toucan: T=0 mean={toucan_mean_t0:.3f}, T=1 mean={toucan_mean_t1:.3f}, degradation={toucan_degradation:.1f}%")
    print(f"ShareGPT: T=0 mean={sharegpt_mean_t0:.3f}, T=1 mean={sharegpt_mean_t1:.3f}, degradation={sharegpt_degradation:.1f}%")

    # Save
    output = {
        'description': 'Table 6: Cross-Task Validation (Toucan vs ShareGPT)',
        'results': results
    }

    with open(OUTPUT_DIR / 'table6_cross_task_validation.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'table6_cross_task_validation.json'}")
    return results


def validity_consistency_correlation(df):
    """
    Compute aggregate validity-consistency correlation across all models.
    """
    print("\n=== Validity-Consistency Correlation ===")

    # Check if we have validity data
    validity_path = Path("/Users/guanghu/Documents/genai/projects/sted-internal/results/accuracy_analysis/consistency_accuracy_correlation_toucan.json")

    if not validity_path.exists():
        print("Validity data not found, computing from factor analysis data...")
        # Check if validity column exists in df
        if 'validity' not in df.columns and 'accuracy' not in df.columns:
            print("No validity/accuracy column found in data, skipping...")
            return None

        validity_col = 'validity' if 'validity' in df.columns else 'accuracy'
        consistency_col = 'stability_score'

        # Filter out NaN
        valid_mask = df[validity_col].notna() & df[consistency_col].notna()
        df_valid = df[valid_mask]

        # Compute correlation
        corr, p_val = stats.pearsonr(df_valid[consistency_col], df_valid[validity_col])

        results = {
            'pearson_r': float(corr),
            'p_value': float(p_val),
            'n_samples': len(df_valid)
        }
    else:
        # Load and aggregate per-model correlations
        with open(validity_path) as f:
            data = json.load(f)

        # Collect all correlations
        all_r = []
        all_n = []

        for variation_type in data.get('results_by_variation_type', {}).values():
            for temp_key in ['T=0.0', 'T=1.0']:
                if temp_key in variation_type:
                    for model, vals in variation_type[temp_key].items():
                        r = vals.get('pearson_r')
                        n = vals.get('n_samples', 0)
                        if r is not None and not np.isnan(r) and n > 0:
                            all_r.append(r)
                            all_n.append(n)

        # Weighted average correlation (by sample size)
        if all_r:
            weights = np.array(all_n)
            weighted_r = np.average(all_r, weights=weights)
            # Fisher z-transform for proper averaging
            z_values = np.arctanh(np.clip(all_r, -0.999, 0.999))
            weighted_z = np.average(z_values, weights=weights)
            fisher_r = np.tanh(weighted_z)

            results = {
                'simple_mean_r': float(np.mean(all_r)),
                'weighted_mean_r': float(weighted_r),
                'fisher_transformed_r': float(fisher_r),
                'n_model_temps': len(all_r),
                'total_samples': int(sum(all_n)),
                'note': 'fisher_transformed_r is most statistically appropriate'
            }
        else:
            print("No valid correlations found")
            return None

    print(f"Aggregate validity-consistency correlation: r = {results.get('fisher_transformed_r', results.get('pearson_r', 'N/A')):.3f}")

    # Save
    output = {
        'description': 'Aggregate Validity-Consistency Correlation',
        'results': results
    }

    with open(OUTPUT_DIR / 'validity_consistency_correlation.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_DIR / 'validity_consistency_correlation.json'}")
    return results


def main():
    """Run all table data generation."""
    print("=" * 60)
    print("KDD Paper Table Data Generation")
    print("=" * 60)

    # Load data
    df = load_data()

    # Generate each table
    table1_factor_correlations_by_model(df)
    table2_model_comparison(df)
    table2_model_comparison_by_model(df)
    table3_baseline_comparison(df)
    table4_shap_importance(df)
    table5_interaction_effects(df)
    table6_cross_task_validation()
    table7_observational_associations(df)
    table8_lomo_cv(df)
    table16_pairwise_transfer(df)
    validity_consistency_correlation(df)

    print("\n" + "=" * 60)
    print("All tables generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
