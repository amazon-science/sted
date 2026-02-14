#!/usr/bin/env python3
"""
COLM 2026: Theoretical Consistency Bounds Estimation

Derives and validates formal bounds on expected consistency based on:
1. Schema complexity (depth, breadth, constraints)
2. Temperature settings
3. Model family characteristics

Key contribution: C_expected >= 1 - alpha * schema_complexity

Usage:
    python bound_estimation.py --results-base llm_gen_results/toucan
    python bound_estimation.py --validate --holdout 0.2
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import random

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# Schema Complexity Computation
# =============================================================================

def compute_schema_complexity(tools: List[Dict]) -> Dict[str, Any]:
    """Compute schema complexity features for tool definitions."""
    if not tools:
        return {
            'depth': 0,
            'breadth': 0,
            'required_count': 0,
            'complexity_score': 0
        }

    def get_depth(obj, current_depth=0):
        """Recursively compute maximum depth."""
        if not isinstance(obj, dict):
            return current_depth

        max_child = current_depth
        if 'properties' in obj:
            for prop_value in obj['properties'].values():
                if isinstance(prop_value, dict):
                    max_child = max(max_child, get_depth(prop_value, current_depth + 1))
        if 'items' in obj:
            max_child = max(max_child, get_depth(obj['items'], current_depth + 1))
        return max_child

    def count_params(obj):
        """Count total parameters."""
        if not isinstance(obj, dict):
            return 0
        count = 0
        if 'properties' in obj:
            count += len(obj['properties'])
            for prop_value in obj['properties'].values():
                count += count_params(prop_value)
        if 'items' in obj:
            count += count_params(obj['items'])
        return count

    def count_required(obj):
        """Count required fields."""
        if not isinstance(obj, dict):
            return 0
        count = len(obj.get('required', []))
        if 'properties' in obj:
            for prop_value in obj['properties'].values():
                count += count_required(prop_value)
        if 'items' in obj:
            count += count_required(obj['items'])
        return count

    max_depth = 0
    total_params = 0
    total_required = 0

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if 'function' in tool:
            params = tool['function'].get('parameters', {})
        else:
            params = tool.get('parameters', {})

        max_depth = max(max_depth, get_depth(params))
        total_params += count_params(params)
        total_required += count_required(params)

    # Composite complexity score (normalized)
    complexity_score = 2 * max_depth + 0.5 * total_params + 0.3 * total_required + 0.3 * len(tools)

    return {
        'depth': max_depth,
        'breadth': total_params,
        'required_count': total_required,
        'num_tools': len(tools),
        'complexity_score': complexity_score
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_consistency_data(results_base: str, temperature: float = 0.7) -> List[Dict]:
    """Load consistency data with schema features."""
    results_path = Path(results_base)
    data = []

    # Load original Toucan data for schema information
    toucan_path = Path("data/toucan/toucan_tool_calls_1006.json")
    schema_by_id = {}
    if toucan_path.exists():
        with open(toucan_path) as f:
            toucan_data = json.load(f)
        for item in toucan_data:
            schema_by_id[item['id']] = item.get('tools', [])

    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Find temperature directory
        temp_str = f"temp_{int(temperature)}_{int((temperature % 1) * 100):02d}"
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir() or temp_str not in run_dir.name:
                continue

            results_file = run_dir / "all_results.json"
            if not results_file.exists():
                continue

            with open(results_file) as f:
                results = json.load(f)

            for sample in results.get('results', []):
                sample_id = sample.get('sample_id')
                runs = sample.get('generated_runs', [])
                valid_runs = [r for r in runs if r]

                if len(valid_runs) < 2:
                    continue

                # Compute consistency
                sims = []
                for i in range(len(valid_runs)):
                    for j in range(i + 1, len(valid_runs)):
                        tools1 = set(tc.get('name', '') for tc in valid_runs[i])
                        tools2 = set(tc.get('name', '') for tc in valid_runs[j])
                        if tools1 or tools2:
                            sim = len(tools1 & tools2) / len(tools1 | tools2)
                            sims.append(sim)

                if not sims:
                    continue

                c_mean = np.mean(sims)

                # Get schema complexity
                tools = schema_by_id.get(sample_id, [])
                schema = compute_schema_complexity(tools)

                data.append({
                    'sample_id': sample_id,
                    'model': model_name,
                    'temperature': temperature,
                    'consistency': c_mean,
                    'schema': schema
                })

    return data


def load_multi_temp_data(results_base: str, temperatures: List[float] = None) -> Dict[float, List[Dict]]:
    """Load data across multiple temperatures."""
    if temperatures is None:
        temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    data_by_temp = {}
    for temp in temperatures:
        data_by_temp[temp] = load_consistency_data(results_base, temp)
        print(f"  T={temp}: {len(data_by_temp[temp])} samples")

    return data_by_temp


# =============================================================================
# Bound Estimation Functions
# =============================================================================

def estimate_complexity_bound(data: List[Dict], model_filter: str = None) -> Dict:
    """
    Estimate bound: C_expected >= 1 - alpha * complexity

    Returns estimated alpha and validation metrics.
    """
    if model_filter:
        data = [d for d in data if model_filter.lower() in d['model'].lower()]

    if len(data) < 10:
        return {"error": "insufficient_data", "n": len(data)}

    # Extract features
    X = np.array([[d['schema']['complexity_score']] for d in data])
    y = np.array([d['consistency'] for d in data])

    # Fit linear model: C = 1 - alpha * complexity
    # Rearranged: C = beta0 - alpha * complexity
    # Where beta0 should be close to 1

    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # Extract bound parameters
    beta0 = model.intercept_
    alpha = -model.coef_[0]  # Negative because we expect negative slope

    # Compute residuals for bound estimation
    residuals = y - y_pred
    bound_offset = np.percentile(residuals, 5)  # 5th percentile for conservative bound

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    return {
        'alpha': float(alpha),
        'intercept': float(beta0),
        'bound_formula': f"C >= {beta0:.3f} - {alpha:.4f} * complexity + {bound_offset:.3f}",
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'cv_r2_mean': float(np.mean(cv_scores)),
        'cv_r2_std': float(np.std(cv_scores)),
        'n_samples': len(data),
        'complexity_range': [float(X.min()), float(X.max())],
        'consistency_range': [float(y.min()), float(y.max())]
    }


def estimate_temperature_decay(data_by_temp: Dict[float, List[Dict]], sample_id: str = None) -> Dict:
    """
    Estimate temperature decay: C(T) = C(0) * exp(-beta * T)

    If sample_id provided, estimate for specific sample.
    Otherwise, estimate aggregate decay.
    """
    temps = sorted(data_by_temp.keys())

    if sample_id:
        # Sample-specific decay
        c_by_temp = {}
        for temp, data in data_by_temp.items():
            for d in data:
                if d['sample_id'] == sample_id:
                    c_by_temp[temp] = d['consistency']
                    break

        if len(c_by_temp) < 3:
            return {"error": "insufficient_temperatures", "sample_id": sample_id}

        temps_arr = np.array(sorted(c_by_temp.keys()))
        c_arr = np.array([c_by_temp[t] for t in temps_arr])

    else:
        # Aggregate decay
        c_by_temp = {}
        for temp, data in data_by_temp.items():
            if data:
                c_by_temp[temp] = np.mean([d['consistency'] for d in data])

        temps_arr = np.array(sorted(c_by_temp.keys()))
        c_arr = np.array([c_by_temp[t] for t in temps_arr])

    # Fit exponential decay: C(T) = C0 * exp(-beta * T)
    def exp_decay(T, C0, beta):
        return C0 * np.exp(-beta * T)

    try:
        popt, pcov = curve_fit(
            exp_decay, temps_arr, c_arr,
            p0=[c_arr[0], 0.1],
            bounds=([0.5, 0], [1.0, 2.0]),
            maxfev=5000
        )
        C0, beta = popt
        perr = np.sqrt(np.diag(pcov))

        # Goodness of fit
        y_pred = exp_decay(temps_arr, C0, beta)
        r2 = r2_score(c_arr, y_pred)

        return {
            'C0': float(C0),
            'beta': float(beta),
            'C0_std': float(perr[0]),
            'beta_std': float(perr[1]),
            'r2': float(r2),
            'formula': f"C(T) = {C0:.3f} * exp(-{beta:.3f} * T)",
            'observed': {float(t): float(c) for t, c in zip(temps_arr, c_arr)},
            'predicted': {float(t): float(exp_decay(t, C0, beta)) for t in temps_arr}
        }
    except Exception as e:
        return {"error": str(e)}


def estimate_family_bounds(data: List[Dict]) -> Dict:
    """Estimate bounds per model family."""
    # Group by family
    by_family = defaultdict(list)
    for d in data:
        family = infer_family(d['model'])
        by_family[family].append(d)

    family_bounds = {}
    for family, family_data in by_family.items():
        if len(family_data) >= 50:
            bound = estimate_complexity_bound(family_data)
            family_bounds[family] = bound

    return family_bounds


def infer_family(model_name: str) -> str:
    """Infer model family from name."""
    name_lower = model_name.lower()
    if 'claude' in name_lower:
        return 'Claude'
    elif 'gpt' in name_lower:
        return 'GPT'
    elif 'llama' in name_lower:
        return 'Llama'
    elif 'qwen' in name_lower:
        return 'Qwen'
    elif 'gemini' in name_lower:
        return 'Gemini'
    elif 'nova' in name_lower:
        return 'Nova'
    elif 'mistral' in name_lower:
        return 'Mistral'
    else:
        return 'Other'


# =============================================================================
# Validation Functions
# =============================================================================

def validate_bounds(data: List[Dict], bounds: Dict, holdout: float = 0.2) -> Dict:
    """Validate bounds on held-out data."""
    random.seed(42)

    # Split data
    train_data, test_data = train_test_split(data, test_size=holdout, random_state=42)

    # Re-estimate on training data
    train_bounds = estimate_complexity_bound(train_data)

    if 'error' in train_bounds:
        return {"error": "training_failed"}

    # Validate on test data
    alpha = train_bounds['alpha']
    intercept = train_bounds['intercept']

    X_test = np.array([[d['schema']['complexity_score']] for d in test_data])
    y_test = np.array([d['consistency'] for d in test_data])

    y_pred = intercept - alpha * X_test.flatten()

    # Metrics
    r2_test = r2_score(y_test, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    # Bound satisfaction (how often actual >= predicted - margin)
    margin = 0.05
    bound_satisfied = np.mean(y_test >= (y_pred - margin))

    return {
        'train_r2': train_bounds['r2'],
        'test_r2': float(r2_test),
        'test_rmse': float(rmse_test),
        'bound_satisfaction_rate': float(bound_satisfied),
        'n_train': len(train_data),
        'n_test': len(test_data),
        'generalization_gap': float(train_bounds['r2'] - r2_test)
    }


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='COLM 2026: Theoretical Consistency Bounds Estimation'
    )
    parser.add_argument('--results-base', type=str,
                        default='llm_gen_results/toucan',
                        help='Base directory for results')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for complexity bounds')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation with holdout')
    parser.add_argument('--holdout', type=float, default=0.2,
                        help='Holdout fraction for validation')
    parser.add_argument('--output', type=str,
                        default='results/colm_architecture/bound_estimation.json',
                        help='Output file')

    args = parser.parse_args()

    print("=" * 70)
    print("COLM 2026: THEORETICAL BOUNDS ESTIMATION")
    print("=" * 70)

    results_path = Path(args.results_base)
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from {results_path}...")
    data = load_consistency_data(args.results_base, args.temperature)
    print(f"Loaded {len(data)} samples at T={args.temperature}")

    results = {
        "metadata": {
            "results_base": str(results_path),
            "temperature": args.temperature,
            "n_samples": len(data)
        }
    }

    # ==========================================================================
    # 1. Overall Complexity Bound
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. SCHEMA COMPLEXITY BOUND")
    print("=" * 70)

    overall_bound = estimate_complexity_bound(data)

    print(f"\nBound Formula: {overall_bound.get('bound_formula', 'N/A')}")
    print(f"R² = {overall_bound.get('r2', 0):.4f}")
    print(f"RMSE = {overall_bound.get('rmse', 0):.4f}")
    print(f"CV R² = {overall_bound.get('cv_r2_mean', 0):.4f} (+/- {overall_bound.get('cv_r2_std', 0):.4f})")

    results["complexity_bound"] = overall_bound

    # ==========================================================================
    # 2. Family-Specific Bounds
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. FAMILY-SPECIFIC BOUNDS")
    print("=" * 70)

    family_bounds = estimate_family_bounds(data)

    print("\n{:<15} {:>10} {:>10} {:>12}".format(
        "Family", "Alpha", "R²", "n"
    ))
    print("-" * 50)

    for family in sorted(family_bounds.keys()):
        bound = family_bounds[family]
        if 'error' not in bound:
            print("{:<15} {:>10.4f} {:>10.4f} {:>12}".format(
                family, bound['alpha'], bound['r2'], bound['n_samples']
            ))

    results["family_bounds"] = family_bounds

    # ==========================================================================
    # 3. Temperature Decay
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. TEMPERATURE-CONSISTENCY DECAY")
    print("=" * 70)

    print("\nLoading multi-temperature data...")
    data_by_temp = load_multi_temp_data(args.results_base, [0.3, 0.5, 0.7, 1.0])

    decay = estimate_temperature_decay(data_by_temp)

    if 'error' not in decay:
        print(f"\nDecay Formula: {decay.get('formula', 'N/A')}")
        print(f"R² = {decay.get('r2', 0):.4f}")
        print(f"C(0) = {decay.get('C0', 0):.4f}")
        print(f"Beta = {decay.get('beta', 0):.4f}")

        print("\nObserved vs Predicted:")
        for temp in sorted(decay.get('observed', {}).keys()):
            obs = decay['observed'][temp]
            pred = decay['predicted'][temp]
            print(f"  T={temp}: observed={obs:.4f}, predicted={pred:.4f}")

    results["temperature_decay"] = decay

    # ==========================================================================
    # 4. Validation
    # ==========================================================================
    if args.validate:
        print("\n" + "=" * 70)
        print("4. BOUND VALIDATION")
        print("=" * 70)

        validation = validate_bounds(data, overall_bound, args.holdout)

        print(f"\nTrain R² = {validation.get('train_r2', 0):.4f}")
        print(f"Test R² = {validation.get('test_r2', 0):.4f}")
        print(f"Generalization Gap = {validation.get('generalization_gap', 0):.4f}")
        print(f"Bound Satisfaction = {validation.get('bound_satisfaction_rate', 0):.1%}")

        results["validation"] = validation

    # ==========================================================================
    # 5. Practical Guidelines
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PRACTICAL GUIDELINES")
    print("=" * 70)

    alpha = overall_bound.get('alpha', 0.02)
    intercept = overall_bound.get('intercept', 0.95)

    print("\nGiven schema complexity C, expected consistency:")
    for complexity in [5, 10, 15, 20, 25]:
        expected = intercept - alpha * complexity
        print(f"  Complexity={complexity:2d}: E[C] >= {max(0, expected):.3f}")

    print("\nOptimal temperature for target consistency:")
    if 'error' not in decay:
        C0 = decay.get('C0', 0.95)
        beta = decay.get('beta', 0.1)
        for target in [0.95, 0.90, 0.85, 0.80]:
            if C0 > 0 and beta > 0 and target < C0:
                opt_temp = -np.log(target / C0) / beta
                if 0 <= opt_temp <= 1:
                    print(f"  Target C={target:.2f}: T* = {opt_temp:.2f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
