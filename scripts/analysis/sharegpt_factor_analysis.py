#!/usr/bin/env python3
"""
Factor Analysis for ShareGPT Structured Output Consistency.

Adapted from comprehensive_factor_analysis.py for ShareGPT data format.
ShareGPT doesn't have tool schemas, so this focuses on query features only.

Usage:
    python scripts/analysis/sharegpt_factor_analysis.py \
        --sharegpt-data data/sharegpt/sharegpt-structured-output-json/all_conversations.json \
        --metrics-path results/sharegpt/minilm-ec2/combined_consistency_metrics_results.json \
        --output-dir results/factor_analysis_sharegpt
"""

import json
import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available, skipping ML analysis")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sharegpt_data(sharegpt_path: str) -> Dict[str, Dict]:
    """Load ShareGPT dataset with metadata from both subsets."""
    sharegpt_dict = {}

    # Load from both ShareGPT subsets
    base_dir = Path(sharegpt_path).parent.parent
    data_files = [
        base_dir / 'sharegpt-structured-output-json' / 'all_conversations.json',
        base_dir / 'sharegpt-quizz-generation-json-output' / 'all_conversations.json',
    ]

    # If specific file provided, use it; otherwise combine both
    if Path(sharegpt_path).exists():
        data_files = [Path(sharegpt_path)]
        # Check if we should also load the other subset
        parent = Path(sharegpt_path).parent
        if 'structured-output' in str(parent):
            other = parent.parent / 'sharegpt-quizz-generation-json-output' / 'all_conversations.json'
            if other.exists():
                data_files.append(other)
        elif 'quizz-generation' in str(parent):
            other = parent.parent / 'sharegpt-structured-output-json' / 'all_conversations.json'
            if other.exists():
                data_files.append(other)

    idx = 0
    for data_file in data_files:
        if not data_file.exists():
            print(f"Warning: {data_file} not found")
            continue

        with open(data_file, 'r') as f:
            data = json.load(f)

        print(f"Loading {len(data)} samples from {data_file.name}")

        for item in data:
            sample_id = str(idx)
            conversations = item.get('conversations', [])

            # Extract human query and expected output
            human_query = ""
            expected_output = ""
            system_prompt = ""

            for conv in conversations:
                if conv.get('from') == 'system':
                    system_prompt = conv.get('value', '')
                elif conv.get('from') == 'human':
                    human_query = conv.get('value', '')
                elif conv.get('from') == 'gpt':
                    expected_output = conv.get('value', '')

            sharegpt_dict[sample_id] = {
                'question': human_query,
                'system_prompt': system_prompt,
                'expected_output': expected_output,
                'contributor': item.get('contributor', ''),
                'source_file': data_file.name,
            }
            idx += 1

    return sharegpt_dict


def load_consistency_metrics(metrics_path: str) -> Dict:
    """Load consistency metrics results."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_query_features(question: str) -> Dict[str, float]:
    """Extract query-level features."""
    features = {}

    # Basic length features
    features['query_length'] = len(question)
    features['query_word_count'] = len(question.split())
    features['query_sentence_count'] = len(re.split(r'[.!?]+', question))

    # Complexity features
    words = question.split()
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['vocabulary_diversity'] = len(set(words)) / len(words) if words else 0

    # Punctuation
    features['punctuation_count'] = sum(1 for c in question if c in '.,;:!?')
    features['question_marks'] = question.count('?')

    # Non-ASCII (potential non-English)
    non_ascii = sum(1 for c in question if ord(c) > 127)
    features['non_ascii_ratio'] = non_ascii / len(question) if question else 0

    # Linguistic markers
    question_lower = question.lower()
    features['has_please'] = 1.0 if 'please' in question_lower else 0.0
    features['has_must'] = 1.0 if 'must' in question_lower else 0.0
    features['has_should'] = 1.0 if 'should' in question_lower else 0.0
    features['has_if'] = 1.0 if ' if ' in question_lower else 0.0
    features['has_can_you'] = 1.0 if 'can you' in question_lower else 0.0
    features['num_conjunctions'] = sum(1 for w in ['and', 'or', 'but', 'then']
                                       if f' {w} ' in question_lower)

    # Query complexity score
    features['query_complexity_score'] = (
        features['query_length'] * 0.01 +
        features['query_sentence_count'] * 0.5 +
        features['num_conjunctions'] * 0.3
    )

    return features


def extract_output_features(expected_output: str) -> Dict[str, float]:
    """Extract expected output features."""
    features = {}

    features['output_length'] = len(expected_output)

    # Try to parse as JSON and extract structure features
    try:
        parsed = json.loads(expected_output)
        features['is_valid_json'] = 1.0
        features['json_depth'] = get_json_depth(parsed)
        features['json_keys'] = count_json_keys(parsed)
    except:
        features['is_valid_json'] = 0.0
        features['json_depth'] = 0
        features['json_keys'] = 0

    return features


def get_json_depth(obj, current_depth=0):
    """Get maximum depth of JSON object."""
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(get_json_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(get_json_depth(item, current_depth + 1) for item in obj)
    return current_depth


def count_json_keys(obj):
    """Count total keys in JSON object."""
    if isinstance(obj, dict):
        return len(obj) + sum(count_json_keys(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_json_keys(item) for item in obj)
    return 0


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def build_analysis_dataframe(
    sharegpt_data: Dict[str, Dict],
    metrics_data: Dict,
) -> pd.DataFrame:
    """Build combined dataframe for analysis."""
    rows = []

    for model_name, model_results in metrics_data.items():
        # Handle both formats: list of items or dict of temps
        if isinstance(model_results, list):
            # New format: list of all results with temperature field
            for sample_result in model_results:
                sample_idx = str(sample_result.get('sample_idx', ''))
                temperature = sample_result.get('temperature', 0)

                if sample_idx not in sharegpt_data:
                    continue

                sample_info = sharegpt_data[sample_idx]

                # Extract features
                query_features = extract_query_features(sample_info['question'])
                output_features = extract_output_features(sample_info['expected_output'])

                row = {
                    'sample_id': sample_idx,
                    'model': model_name,
                    'temperature': temperature,
                    'stability_score': sample_result.get('stability_score', 0),
                    'c_mean': sample_result.get('c_mean', 0),
                    'validity_rate': sample_result.get('validity_rate', 0),
                    'source_file': sample_info.get('source_file', ''),
                    **query_features,
                    **output_features,
                }
                rows.append(row)
        elif isinstance(model_results, dict):
            # Old format: dict keyed by temp_X_XX
            for temp_key, temp_results in model_results.items():
                if not isinstance(temp_results, list):
                    continue

                # Parse temperature
                temp_match = re.search(r'temp_(\d+)_(\d+)', temp_key)
                if temp_match:
                    temperature = float(f"{temp_match.group(1)}.{temp_match.group(2)}")
                else:
                    continue

                for sample_result in temp_results:
                    sample_idx = str(sample_result.get('sample_idx', ''))

                    if sample_idx not in sharegpt_data:
                        continue

                    sample_info = sharegpt_data[sample_idx]

                    # Extract features
                    query_features = extract_query_features(sample_info['question'])
                    output_features = extract_output_features(sample_info['expected_output'])

                    row = {
                        'sample_id': sample_idx,
                        'model': model_name,
                        'temperature': temperature,
                        'stability_score': sample_result.get('stability_score', 0),
                        'c_mean': sample_result.get('c_mean', 0),
                        'validity_rate': sample_result.get('validity_rate', 0),
                        'source_file': sample_info.get('source_file', ''),
                        **query_features,
                        **output_features,
                    }
                    rows.append(row)

    return pd.DataFrame(rows)


def run_correlation_analysis(df: pd.DataFrame, output_dir: Path):
    """Run correlation analysis."""
    target = 'stability_score'

    # Exclude non-numeric and identifier columns
    exclude_cols = ['sample_id', 'model', 'stability_score', 'c_mean', 'validity_rate', 'source_file']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    results = []
    for col in feature_cols:
        if df[col].std() == 0:
            continue
        try:
            pearson_r, pearson_p = pearsonr(df[col], df[target])
            spearman_r, spearman_p = spearmanr(df[col], df[target])
            results.append({
                'feature': col,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
            })
        except Exception as e:
            print(f"Error computing correlation for {col}: {e}")

    corr_df = pd.DataFrame(results)
    corr_df = corr_df.sort_values('pearson_r', key=abs, ascending=False)
    corr_df.to_csv(output_dir / 'correlations_sharegpt.csv', index=False)

    print("\n=== Top Correlations with Consistency ===")
    print(corr_df.head(15).to_string(index=False))

    return corr_df


def run_feature_importance(df: pd.DataFrame, output_dir: Path):
    """Run Random Forest feature importance analysis."""
    if not SKLEARN_AVAILABLE:
        print("sklearn not available, skipping feature importance")
        return None

    target = 'stability_score'
    exclude_cols = ['sample_id', 'model', 'stability_score', 'c_mean', 'validity_rate', 'source_file']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    X = df[feature_cols].fillna(0)
    y = df[target]

    # Train RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Cross-validation score
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"\n=== Random Forest R² (5-fold CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f} ===")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_df.to_csv(output_dir / 'feature_importance_sharegpt.csv', index=False)

    print("\n=== Top Feature Importance ===")
    print(importance_df.head(15).to_string(index=False))

    return importance_df


def main():
    parser = argparse.ArgumentParser(description='ShareGPT Factor Analysis')
    parser.add_argument('--sharegpt-data', type=str,
                       default='data/sharegpt/sharegpt-structured-output-json/all_conversations.json')
    parser.add_argument('--metrics-path', type=str,
                       default='results/sharegpt/minilm-ec2/combined_consistency_metrics_results.json')
    parser.add_argument('--output-dir', type=str, default='results/factor_analysis_sharegpt')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ShareGPT data...")
    sharegpt_data = load_sharegpt_data(args.sharegpt_data)
    print(f"Loaded {len(sharegpt_data)} ShareGPT samples")

    print("Loading consistency metrics...")
    metrics_data = load_consistency_metrics(args.metrics_path)
    print(f"Loaded metrics for {len(metrics_data)} models")

    print("Building analysis dataframe...")
    df = build_analysis_dataframe(sharegpt_data, metrics_data)
    print(f"Built dataframe with {len(df)} rows")

    # Save combined data
    df.to_csv(output_dir / 'factor_analysis_data_sharegpt.csv', index=False)

    # Run analyses
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    run_correlation_analysis(df, output_dir)

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    run_feature_importance(df, output_dir)

    # Temperature interaction
    print("\n" + "="*60)
    print("TEMPERATURE EFFECT")
    print("="*60)
    temp_groups = df.groupby('temperature')['stability_score'].agg(['mean', 'std', 'count'])
    print(temp_groups)
    temp_groups.to_csv(output_dir / 'temperature_effect_sharegpt.csv')

    # Model comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    model_groups = df.groupby('model')['stability_score'].agg(['mean', 'std', 'count'])
    model_groups = model_groups.sort_values('mean', ascending=False)
    print(model_groups)
    model_groups.to_csv(output_dir / 'model_comparison_sharegpt.csv')

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
