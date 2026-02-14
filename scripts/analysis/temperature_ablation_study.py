#!/usr/bin/env python3
"""
Temperature Ablation Study for KDD 2026 Paper

This script computes how much temperature contributes to per-model R²
by comparing:
1. Per-model R² with all features (including temperature)
2. Per-model R² without temperature
3. Per-model R² with temperature only

Results are saved to results/kdd_paper_tables/temperature_ablation.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "results/factor_analysis/factor_analysis_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "results/kdd_paper_tables/temperature_ablation.json"

# Exact features from paper's table2_model_comparison_by_model.json
ALL_FEATURES = [
    'num_tools', 'avg_params_per_tool', 'max_params_per_tool', 'total_params',
    'param_type_diversity', 'avg_tool_name_length', 'tool_name_ambiguity',
    'tool_prefix_diversity', 'schema_depth', 'schema_breadth', 'schema_complexity',
    'query_length', 'query_word_count', 'query_sentence_count', 'num_questions',
    'num_commands', 'num_conjunctions', 'query_complexity_score', 'constraint_score',
    'temperature'
]

TARGET = 'stability_score'

# Final 18 models used in the paper (must match generate_kdd_table_data.py)
FINAL_MODELS = [
    "Claude-3.5-Haiku", "Claude-3.5-Sonnet", "Claude-3.7-Sonnet",
    "Claude-Haiku-4.5", "Claude-Sonnet-4", "Claude-Sonnet-4.5",
    "Claude-Opus-4.5", "us.anthropic.claude-opus-4-20250514-v1",
    "GPT-4.1-Mini", "Qwen3-32B", "Qwen3-235B-A22B",
    "Llama-3.3-70B", "Gemini-2.5-Flash-Lite", "Grok-4.1-Fast",
    "GPT-OSS-120B", "Nova-2-Lite", "Minimax-M2", "Mimo-V2-Flash:free"
]


def run_ablation():
    """Run temperature ablation study."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Filter to available features
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing = set(ALL_FEATURES) - set(available)

    print(f"Available features: {len(available)} of {len(ALL_FEATURES)}")
    if missing:
        print(f"Missing features: {missing}")

    # Results storage
    results = {
        "description": "Temperature Ablation Study for Per-Model R²",
        "features_used": available,
        "features_missing": list(missing),
        "target": TARGET,
        "per_model_results": {},
        "summary": {}
    }

    r2_with_temp = []
    r2_without_temp = []
    r2_temp_only = []

    # Filter to final 18 models only
    models = [m for m in df['model'].unique() if m in FINAL_MODELS]
    print(f"\nAnalyzing {len(models)} models (filtered to FINAL_MODELS)...")

    for model in models:
        model_df = df[df['model'] == model].dropna(subset=available + [TARGET])

        if len(model_df) < 50:
            print(f"  Skipping {model}: only {len(model_df)} samples")
            continue

        print(f"  Processing {model} (n={len(model_df)})...")

        X_all = model_df[available]
        X_no_temp = model_df[[f for f in available if f != 'temperature']]
        X_temp_only = model_df[['temperature']]
        y = model_df[TARGET]

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        # With all features (including temperature)
        scores_with = cross_val_score(rf, X_all, y, cv=5, scoring='r2')
        r2_with = float(np.mean(scores_with))

        # Without temperature
        scores_without = cross_val_score(rf, X_no_temp, y, cv=5, scoring='r2')
        r2_without = float(np.mean(scores_without))

        # Temperature only
        scores_temp = cross_val_score(rf, X_temp_only, y, cv=5, scoring='r2')
        r2_temp = float(np.mean(scores_temp))

        # Store per-model results
        results["per_model_results"][model] = {
            "n_samples": len(model_df),
            "r2_with_temperature": r2_with,
            "r2_without_temperature": r2_without,
            "r2_temperature_only": r2_temp,
            "delta": r2_with - r2_without
        }

        r2_with_temp.append(r2_with)
        r2_without_temp.append(r2_without)
        r2_temp_only.append(r2_temp)

    # Compute summary statistics
    mean_with = float(np.mean(r2_with_temp))
    mean_without = float(np.mean(r2_without_temp))
    mean_temp_only = float(np.mean(r2_temp_only))
    delta = mean_with - mean_without

    results["summary"] = {
        "n_models_analyzed": len(r2_with_temp),
        "r2_with_temperature": {
            "mean": mean_with,
            "std": float(np.std(r2_with_temp))
        },
        "r2_without_temperature": {
            "mean": mean_without,
            "std": float(np.std(r2_without_temp))
        },
        "r2_temperature_only": {
            "mean": mean_temp_only,
            "std": float(np.std(r2_temp_only))
        },
        "delta": delta,
        "temperature_contribution_pct": (delta / mean_with * 100) if mean_with > 0 else 0
    }

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("TEMPERATURE ABLATION RESULTS")
    print(f"{'='*50}")
    print(f"Models analyzed: {len(r2_with_temp)}")
    print(f"\nPer-model R² with all features:    {mean_with:.3f} ± {np.std(r2_with_temp):.3f}")
    print(f"Per-model R² WITHOUT temperature:  {mean_without:.3f} ± {np.std(r2_without_temp):.3f}")
    print(f"Per-model R² temperature ONLY:     {mean_temp_only:.3f} ± {np.std(r2_temp_only):.3f}")
    print(f"\nDelta (with - without): {delta:.3f}")
    print(f"Temperature contribution: {results['summary']['temperature_contribution_pct']:.1f}%")
    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    run_ablation()
