#!/usr/bin/env python3
"""
ACL 2026 Paper: Cross-Model Transfer Analysis (LOMO)

Analyzes whether linguistic intervention effects transfer across models.
Extends KDD's LOMO analysis (R² = -0.009) to linguistic features.

Key questions:
1. Do intervention effects generalize across models?
2. Which variations transfer best?
3. Does model family predict transferability?

Uses Leave-One-Model-Out (LOMO) cross-validation.

Usage:
    python analyze_cross_model_transfer.py --results-dir results/acl_linguistic/phase2_interventions
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import statistics

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


def load_all_intervention_results(results_dir: str) -> Dict[str, Dict]:
    """Load all Phase 2 intervention results by model."""
    results_path = Path(results_dir)
    model_results = {}

    for json_file in results_path.glob("*_interventions.json"):
        with open(json_file) as f:
            data = json.load(f)
            model_name = data['metadata']['display_name']
            model_results[model_name] = data
        print(f"Loaded: {model_name}")

    return model_results


def extract_features(result: Dict) -> Dict[str, float]:
    """Extract features from a single result for prediction."""
    features = {
        'baseline_consistency': result.get('baseline_consistency', 0.5),
        'schema_depth': result.get('schema_complexity', {}).get('depth', 0),
        'schema_params': result.get('schema_complexity', {}).get('total_params', 0),
    }

    # One-hot encode variation
    variation = result.get('variation', 'baseline')
    for var in ['baseline', 'polite_please', 'polite_bald', 'modal_must',
                'modal_might', 'hedge_conditional', 'speech_directive', 'speech_hint']:
        features[f'var_{var}'] = 1.0 if variation == var else 0.0

    # One-hot encode difficulty
    difficulty = result.get('difficulty_stratum', 'medium')
    for diff in ['difficult', 'medium', 'easy']:
        features[f'diff_{diff}'] = 1.0 if difficulty == diff else 0.0

    return features


def prepare_model_data(model_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and targets for a single model."""
    X_list = []
    y_list = []

    for result in model_data.get('results', []):
        if 'metrics' not in result:
            continue

        features = extract_features(result)
        target = result['metrics']['c_mean']

        # Convert features to array (consistent order)
        feature_names = sorted(features.keys())
        X_list.append([features[f] for f in feature_names])
        y_list.append(target)

    return np.array(X_list), np.array(y_list), feature_names


def lomo_analysis(model_results: Dict[str, Dict]) -> Dict:
    """
    Leave-One-Model-Out cross-validation.

    Train on N-1 models, test on held-out model.
    """
    print("\n" + "=" * 70)
    print("LOMO CROSS-VALIDATION ANALYSIS")
    print("=" * 70)

    models = list(model_results.keys())
    n_models = len(models)

    if n_models < 2:
        print("Need at least 2 models for LOMO analysis")
        return {}

    # Prepare data for all models
    model_data = {}
    for model_name, data in model_results.items():
        X, y, feature_names = prepare_model_data(data)
        if len(X) > 0:
            model_data[model_name] = {'X': X, 'y': y, 'features': feature_names}
            print(f"  {model_name}: {len(y)} samples")

    # LOMO CV
    lomo_results = {}

    print("\n{:<25} {:>10} {:>10} {:>10}".format(
        "Held-Out Model", "R²", "MAE", "Samples"
    ))
    print("-" * 60)

    for held_out in model_data.keys():
        # Combine training data from all other models
        X_train = []
        y_train = []
        for model_name, data in model_data.items():
            if model_name != held_out:
                X_train.append(data['X'])
                y_train.append(data['y'])

        if not X_train:
            continue

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        # Test data
        X_test = model_data[held_out]['X']
        y_test = model_data[held_out]['y']

        # Train model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print("{:<25} {:>+9.3f} {:>10.3f} {:>10}".format(
            held_out[:25], r2, mae, len(y_test)
        ))

        lomo_results[held_out] = {
            'r2': r2,
            'mae': mae,
            'n_test': len(y_test),
            'n_train': len(y_train)
        }

    # Summary statistics
    r2_values = [r['r2'] for r in lomo_results.values()]
    mae_values = [r['mae'] for r in lomo_results.values()]

    print("-" * 60)
    print("{:<25} {:>+9.3f} {:>10.3f} {:>10}".format(
        "MEAN", np.mean(r2_values), np.mean(mae_values), ""
    ))
    print("{:<25} {:>+9.3f} {:>10.3f} {:>10}".format(
        "STD", np.std(r2_values), np.std(mae_values), ""
    ))

    return {
        'per_model': lomo_results,
        'summary': {
            'mean_r2': np.mean(r2_values),
            'std_r2': np.std(r2_values),
            'mean_mae': np.mean(mae_values),
            'std_mae': np.std(mae_values)
        }
    }


def pairwise_transfer_matrix(model_results: Dict[str, Dict]) -> Dict:
    """
    Compute pairwise transfer matrix.

    Train on model A, test on model B for all pairs.
    """
    print("\n" + "=" * 70)
    print("PAIRWISE TRANSFER MATRIX")
    print("=" * 70)

    # Prepare data
    model_data = {}
    for model_name, data in model_results.items():
        X, y, feature_names = prepare_model_data(data)
        if len(X) > 0:
            model_data[model_name] = {'X': X, 'y': y}

    models = list(model_data.keys())
    n_models = len(models)

    # Compute pairwise R²
    transfer_matrix = {}

    print("\nPairwise Transfer R² (row=train, col=test):")
    print("-" * (15 + 12 * n_models))

    # Header
    header = "{:<15}".format("")
    for test_model in models:
        header += "{:>12}".format(test_model[:10])
    print(header)
    print("-" * (15 + 12 * n_models))

    for train_model in models:
        transfer_matrix[train_model] = {}
        row = "{:<15}".format(train_model[:15])

        X_train = model_data[train_model]['X']
        y_train = model_data[train_model]['y']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)

        for test_model in models:
            X_test = model_data[test_model]['X']
            y_test = model_data[test_model]['y']

            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)

            r2 = r2_score(y_test, y_pred)
            transfer_matrix[train_model][test_model] = r2

            row += "{:>+11.2f}".format(r2)

        print(row)

    return transfer_matrix


def variation_transfer_analysis(model_results: Dict[str, Dict]) -> Dict:
    """
    Analyze which linguistic variations transfer best across models.
    """
    print("\n" + "=" * 70)
    print("VARIATION-SPECIFIC TRANSFER ANALYSIS")
    print("=" * 70)

    # Collect effect sizes by variation and model
    variation_effects = defaultdict(lambda: defaultdict(list))

    for model_name, data in model_results.items():
        for result in data.get('results', []):
            if 'delta_consistency' not in result:
                continue

            variation = result['variation']
            delta = result['delta_consistency']
            variation_effects[variation][model_name].append(delta)

    # Compute cross-model correlation for each variation
    print("\n{:<20} {:>12} {:>12} {:>15}".format(
        "Variation", "ICC", "Mean Effect", "Transfer?"
    ))
    print("-" * 65)

    transfer_results = {}

    for variation in sorted(variation_effects.keys()):
        model_means = {}
        for model_name, deltas in variation_effects[variation].items():
            if deltas:
                model_means[model_name] = np.mean(deltas)

        if len(model_means) < 2:
            continue

        # Compute consistency across models (ICC-like measure)
        values = list(model_means.values())
        mean_effect = np.mean(values)
        std_effect = np.std(values)

        # ICC approximation: ratio of between-model to total variance
        icc = 1 - (std_effect / (abs(mean_effect) + 0.01)) if abs(mean_effect) > 0.01 else 0

        # High ICC = consistent effect across models = good transfer
        transfers_well = icc > 0.5 and std_effect < abs(mean_effect)

        print("{:<20} {:>+11.2f} {:>+11.3f} {:>15}".format(
            variation[:20], icc, mean_effect, "YES" if transfers_well else "no"
        ))

        transfer_results[variation] = {
            'icc': icc,
            'mean_effect': mean_effect,
            'std_effect': std_effect,
            'transfers_well': transfers_well,
            'per_model': model_means
        }

    return transfer_results


def model_family_analysis(model_results: Dict[str, Dict]) -> Dict:
    """
    Analyze transfer within vs across model families.
    """
    print("\n" + "=" * 70)
    print("MODEL FAMILY TRANSFER ANALYSIS")
    print("=" * 70)

    # Infer model families from names
    def get_family(model_name: str) -> str:
        name_lower = model_name.lower()
        if 'claude' in name_lower or 'anthropic' in name_lower:
            return 'Claude'
        elif 'gpt' in name_lower or 'openai' in name_lower:
            return 'GPT'
        elif 'llama' in name_lower or 'meta' in name_lower:
            return 'Llama'
        elif 'qwen' in name_lower:
            return 'Qwen'
        elif 'gemini' in name_lower or 'google' in name_lower:
            return 'Gemini'
        elif 'nova' in name_lower or 'amazon' in name_lower:
            return 'Nova'
        else:
            return 'Other'

    # Group models by family
    family_models = defaultdict(list)
    for model_name in model_results.keys():
        family = get_family(model_name)
        family_models[family].append(model_name)

    print("\nModel Family Distribution:")
    for family, models in sorted(family_models.items()):
        print(f"  {family}: {', '.join(models)}")

    # Compute within-family vs cross-family effect correlations
    # Collect all (variation, model, effect) tuples
    all_effects = []
    for model_name, data in model_results.items():
        for result in data.get('results', []):
            if 'delta_consistency' not in result:
                continue
            all_effects.append({
                'model': model_name,
                'family': get_family(model_name),
                'variation': result['variation'],
                'delta': result['delta_consistency']
            })

    # Compute correlations
    # Group by variation, then correlate across models
    family_transfer = {
        'within_family': [],
        'cross_family': []
    }

    # For each pair of models, compute effect correlation
    model_effects = defaultdict(dict)  # model -> variation -> mean_delta
    for effect in all_effects:
        key = effect['variation']
        if key not in model_effects[effect['model']]:
            model_effects[effect['model']][key] = []
        model_effects[effect['model']][key].append(effect['delta'])

    # Average effects per variation per model
    for model_name in model_effects:
        for var in model_effects[model_name]:
            model_effects[model_name][var] = np.mean(model_effects[model_name][var])

    # Compute correlations between model pairs
    models = list(model_effects.keys())
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # Get common variations
            common_vars = set(model_effects[model1].keys()) & set(model_effects[model2].keys())
            if len(common_vars) < 3:
                continue

            effects1 = [model_effects[model1][v] for v in common_vars]
            effects2 = [model_effects[model2][v] for v in common_vars]

            r, _ = stats.pearsonr(effects1, effects2)

            family1 = get_family(model1)
            family2 = get_family(model2)

            if family1 == family2:
                family_transfer['within_family'].append(r)
            else:
                family_transfer['cross_family'].append(r)

    print("\nEffect Correlation (mean Pearson r):")
    if family_transfer['within_family']:
        print(f"  Within-family:  {np.mean(family_transfer['within_family']):+.3f} "
              f"(n={len(family_transfer['within_family'])})")
    if family_transfer['cross_family']:
        print(f"  Cross-family:   {np.mean(family_transfer['cross_family']):+.3f} "
              f"(n={len(family_transfer['cross_family'])})")

    return {
        'family_distribution': dict(family_models),
        'within_family_r': np.mean(family_transfer['within_family']) if family_transfer['within_family'] else None,
        'cross_family_r': np.mean(family_transfer['cross_family']) if family_transfer['cross_family'] else None
    }


def main():
    parser = argparse.ArgumentParser(
        description='Cross-model transfer analysis for linguistic interventions'
    )
    parser.add_argument('--results-dir', type=str,
                        default='results/acl_linguistic/phase2_interventions',
                        help='Directory containing intervention results')
    parser.add_argument('--output', type=str,
                        default='results/acl_linguistic/analysis/cross_model_transfer.json',
                        help='Output file for analysis results')

    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        print("Run phase2_linguistic_interventions.py first.")
        sys.exit(1)

    # Load all results
    model_results = load_all_intervention_results(args.results_dir)

    if len(model_results) < 2:
        print("Need at least 2 models for cross-model analysis.")
        sys.exit(1)

    # Run analyses
    lomo = lomo_analysis(model_results)
    pairwise = pairwise_transfer_matrix(model_results)
    variation_transfer = variation_transfer_analysis(model_results)
    family_analysis = model_family_analysis(model_results)

    # Save results
    output = {
        'lomo_results': lomo,
        'pairwise_transfer': pairwise,
        'variation_transfer': variation_transfer,
        'family_analysis': family_analysis
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n\nSaved analysis results to {output_path}")


if __name__ == '__main__':
    main()
