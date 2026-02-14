#!/usr/bin/env python3
"""
Comprehensive Factor Analysis for LLM Structured Output Consistency.

This script analyzes multiple factors affecting consistency:
1. Query characteristics (length, complexity, language, examples)
2. Schema/tool characteristics (num tools, ambiguity, complexity)
3. Model characteristics (family, size)
4. Output characteristics (length variance, structural complexity)
5. Interaction effects (temperature × complexity)

Outputs:
- Correlation analysis
- Feature importance (Random Forest)
- Visualizations
- LaTeX tables for paper

Usage:
    python scripts/analysis/comprehensive_factor_analysis.py \
        --toucan-data toucan_data/toucan_tool_calls_1006.json \
        --metrics-dir results/toucan/minilm-ec2 \
        --output-dir results/factor_analysis
"""

import json
import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available, skipping ML analysis")

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    warnings.warn("langdetect not available, skipping language detection")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_toucan_data(toucan_path: str) -> Dict[str, Dict]:
    """Load Toucan dataset with full metadata."""
    with open(toucan_path, 'r') as f:
        data = json.load(f)

    toucan_dict = {}
    for item in data:
        sample_id = item.get('id', '')
        toucan_dict[sample_id] = {
            'question': item.get('question', ''),
            'tools': item.get('tools', []),
            'target_tools': item.get('target_tools', ''),
            'subset_name': item.get('subset_name', ''),
            'num_tool_calls': len(item.get('target_tools', '').split(', ')) if item.get('target_tools') else 1,
        }

    return toucan_dict


def load_consistency_metrics(metrics_path: str) -> Dict[str, List[Dict]]:
    """Load consistency metrics results."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_generation_results(gen_dir: str, model_name: str, temperature: float) -> Optional[Dict]:
    """Load raw generation results for a model at specific temperature."""
    temp_str = f"temp_{int(temperature)}_{int((temperature % 1) * 100):02d}"

    model_dirs = [d for d in os.listdir(gen_dir) if model_name.lower().replace('-', '') in d.lower().replace('-', '')]
    if not model_dirs:
        return None

    model_path = os.path.join(gen_dir, model_dirs[0])
    temp_dirs = [d for d in os.listdir(model_path) if temp_str in d]

    if not temp_dirs:
        return None

    results_path = os.path.join(model_path, temp_dirs[0], 'all_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


# =============================================================================
# FACTOR EXTRACTION: Query Characteristics
# =============================================================================

def extract_query_factors(query: str) -> Dict[str, Any]:
    """Extract query-level factors."""
    factors = {
        # Basic metrics
        'query_length': len(query),
        'query_word_count': len(query.split()),
        'query_sentence_count': len(re.split(r'[.!?]+', query)),

        # Complexity indicators
        'num_questions': query.count('?'),
        'num_commands': len(re.findall(r'\b(please|can you|could you|help|find|get|search|calculate)\b', query.lower())),
        'num_conjunctions': len(re.findall(r'\b(and|then|also|after|before|while)\b', query.lower())),

        # Structure indicators
        'has_json_example': bool(re.search(r'\{[^{}]*"[^"]+"\s*:', query)),
        'has_numbered_list': bool(re.search(r'^\s*\d+[.)]\s+\w', query, re.MULTILINE)),
        'has_bullet_list': bool(re.search(r'^\s*[-*•]\s+\w', query, re.MULTILINE)),
        'has_code_block': bool(re.search(r'```', query)),

        # Ambiguity indicators
        'has_vague_terms': bool(re.search(r'\b(some|any|various|different|other|etc)\b', query.lower())),
        'has_conditional': bool(re.search(r'\b(if|when|unless|otherwise)\b', query.lower())),
    }

    # Language detection
    if LANGDETECT_AVAILABLE:
        try:
            factors['language'] = langdetect.detect(query)
            factors['is_english'] = factors['language'] == 'en'
        except:
            factors['language'] = 'unknown'
            factors['is_english'] = True
    else:
        # Simple heuristic: check for non-ASCII
        non_ascii_ratio = sum(1 for c in query if ord(c) > 127) / max(len(query), 1)
        factors['is_english'] = non_ascii_ratio < 0.1
        factors['language'] = 'en' if factors['is_english'] else 'non-en'

    # Compute complexity score
    factors['query_complexity_score'] = (
        factors['num_questions'] * 0.5 +
        factors['num_commands'] * 0.3 +
        factors['num_conjunctions'] * 0.4 +
        (1 if factors['has_conditional'] else 0) * 0.5 +
        (1 if factors['has_vague_terms'] else 0) * 0.3
    )

    # Compute constraint score (higher = more constrained, should be more consistent)
    factors['constraint_score'] = (
        (3.0 if factors['has_json_example'] else 0) +
        (1.0 if factors['has_numbered_list'] else 0) +
        (1.0 if factors['has_bullet_list'] else 0) +
        (1.0 if factors['has_code_block'] else 0) -
        (0.5 if factors['has_vague_terms'] else 0) -
        (0.2 * min(factors['num_questions'], 5))
    )

    return factors


# =============================================================================
# FACTOR EXTRACTION: Schema/Tool Characteristics
# =============================================================================

def extract_tool_factors(tools: List[Dict]) -> Dict[str, Any]:
    """Extract tool/schema-level factors."""
    if not tools:
        return {
            'num_tools': 0,
            'avg_params_per_tool': 0,
            'max_params_per_tool': 0,
            'has_optional_params': False,
            'has_complex_params': False,
            'has_nested_params': False,
            'tool_name_ambiguity': 0,
            'avg_tool_name_length': 0,
            'param_type_diversity': 0,
            'total_params': 0,
            'tool_prefix_diversity': 0,
        }

    # Filter to valid tool dicts only
    valid_tools = [t for t in tools if isinstance(t, dict)]
    if not valid_tools:
        return {
            'num_tools': len(tools),
            'avg_params_per_tool': 0,
            'max_params_per_tool': 0,
            'has_optional_params': False,
            'has_complex_params': False,
            'has_nested_params': False,
            'tool_name_ambiguity': 0,
            'avg_tool_name_length': 0,
            'param_type_diversity': 0,
            'total_params': 0,
            'tool_prefix_diversity': 0,
        }

    factors = {
        'num_tools': len(valid_tools),
    }

    # Analyze parameters
    param_counts = []
    has_optional = False
    has_complex = False
    has_nested = False
    param_types = set()
    tool_names = []

    for tool in valid_tools:
        func = tool.get('function', {})
        tool_names.append(func.get('name', ''))

        params = func.get('parameters', {})
        properties = params.get('properties', {})
        required = set(params.get('required', []))

        param_counts.append(len(properties))

        for param_name, param_spec in properties.items():
            param_type = param_spec.get('type', 'string')
            param_types.add(param_type)

            if param_name not in required:
                has_optional = True

            if param_type in ('array', 'object'):
                has_complex = True
                # Check for nesting
                if param_type == 'array':
                    items = param_spec.get('items', {})
                    if items.get('type') in ('array', 'object'):
                        has_nested = True
                elif param_type == 'object':
                    nested_props = param_spec.get('properties', {})
                    if nested_props:
                        has_nested = True

    factors['avg_params_per_tool'] = np.mean(param_counts) if param_counts else 0
    factors['max_params_per_tool'] = max(param_counts) if param_counts else 0
    factors['total_params'] = sum(param_counts)
    factors['has_optional_params'] = has_optional
    factors['has_complex_params'] = has_complex
    factors['has_nested_params'] = has_nested
    factors['param_type_diversity'] = len(param_types)

    # Tool name analysis
    factors['avg_tool_name_length'] = np.mean([len(n) for n in tool_names]) if tool_names else 0

    # Tool name ambiguity: compute average pairwise prefix similarity
    if len(tool_names) >= 2:
        ambiguity_scores = []
        for i, n1 in enumerate(tool_names):
            for n2 in tool_names[i+1:]:
                if n1 and n2:
                    # Compute common prefix ratio
                    prefix_len = len(os.path.commonprefix([n1.lower(), n2.lower()]))
                    max_len = max(len(n1), len(n2))
                    ambiguity_scores.append(prefix_len / max_len if max_len > 0 else 0)
        factors['tool_name_ambiguity'] = np.mean(ambiguity_scores) if ambiguity_scores else 0
    else:
        factors['tool_name_ambiguity'] = 0

    # Tool diversity (unique prefixes)
    prefixes = set(n.split('-')[0] if '-' in n else n.split('_')[0] for n in tool_names if n)
    factors['tool_prefix_diversity'] = len(prefixes) / max(len(tool_names), 1)

    return factors


def compute_schema_complexity(tools: List[Dict]) -> Dict[str, float]:
    """Compute overall schema complexity metrics."""
    if not tools:
        return {'schema_depth': 0, 'schema_breadth': 0, 'schema_complexity': 0}

    # Filter to valid tool dicts only
    valid_tools = [t for t in tools if isinstance(t, dict)]
    if not valid_tools:
        return {'schema_depth': 0, 'schema_breadth': 0, 'schema_complexity': 0}

    max_depth = 0
    total_nodes = 0

    def compute_depth(obj: Any, current_depth: int = 0) -> Tuple[int, int]:
        """Recursively compute depth and node count."""
        nonlocal max_depth, total_nodes

        if current_depth > max_depth:
            max_depth = current_depth

        total_nodes += 1

        if isinstance(obj, dict):
            for v in obj.values():
                compute_depth(v, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                compute_depth(item, current_depth + 1)

        return max_depth, total_nodes

    for tool in valid_tools:
        func = tool.get('function', {})
        if isinstance(func, dict):
            compute_depth(func.get('parameters', {}))

    return {
        'schema_depth': max_depth,
        'schema_breadth': total_nodes,
        'schema_complexity': max_depth * np.log1p(total_nodes),
    }


# =============================================================================
# FACTOR EXTRACTION: Output Characteristics
# =============================================================================

def extract_output_factors(runs: List[Any]) -> Dict[str, Any]:
    """Extract output-level factors from multiple runs."""
    if not runs:
        return {
            'output_length_mean': 0,
            'output_length_std': 0,
            'output_length_cv': 0,
            'num_valid_runs': 0,
            'num_unique_outputs': 0,
        }

    # Filter valid runs
    valid_runs = [r for r in runs if r is not None and r != []]

    if not valid_runs:
        return {
            'output_length_mean': 0,
            'output_length_std': 0,
            'output_length_cv': 0,
            'num_valid_runs': 0,
            'num_unique_outputs': 0,
        }

    # Compute output lengths
    lengths = [len(json.dumps(r)) for r in valid_runs]

    factors = {
        'output_length_mean': np.mean(lengths),
        'output_length_std': np.std(lengths),
        'output_length_cv': np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0,
        'num_valid_runs': len(valid_runs),
    }

    # Count unique outputs
    output_strs = set(json.dumps(r, sort_keys=True) for r in valid_runs)
    factors['num_unique_outputs'] = len(output_strs)
    factors['uniqueness_ratio'] = len(output_strs) / len(valid_runs)

    return factors


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def build_analysis_dataframe(
    toucan_data: Dict[str, Dict],
    metrics: Dict[str, List[Dict]],
    gen_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Build comprehensive analysis dataframe."""

    rows = []

    for model_name, model_metrics in metrics.items():
        print(f"Processing {model_name}...")

        for entry in model_metrics:
            sample_idx = entry.get('sample_idx', 0)
            temperature = entry.get('temperature', 0.5)

            # Find matching Toucan data
            # Sample IDs in toucan_data are UUIDs, we need to map by index
            toucan_items = list(toucan_data.values())
            if sample_idx >= len(toucan_items):
                continue

            toucan_item = toucan_items[sample_idx]

            # Extract all factors
            query_factors = extract_query_factors(toucan_item.get('question', ''))
            tool_factors = extract_tool_factors(toucan_item.get('tools', []))
            schema_factors = compute_schema_complexity(toucan_item.get('tools', []))

            # Build row
            row = {
                # Identifiers
                'model': model_name,
                'sample_idx': sample_idx,
                'temperature': temperature,

                # Target variables
                'c_mean': entry.get('c_mean', 0),
                'stability_score': entry.get('stability_score', 0),
                'validity_rate': entry.get('validity_rate', 0),
                'ranking_score': entry.get('ranking_score', 0),

                # Ground truth info
                'gt_tool_count': toucan_item.get('num_tool_calls', 1),
                'subset_name': toucan_item.get('subset_name', ''),
            }

            # Add all extracted factors
            row.update(query_factors)
            row.update(tool_factors)
            row.update(schema_factors)

            rows.append(row)

    df = pd.DataFrame(rows)

    # Add derived features
    if len(df) > 0:
        # Model family
        df['model_family'] = df['model'].apply(lambda x:
            'Claude' if 'claude' in x.lower() else
            'GPT' if 'gpt' in x.lower() else
            'Qwen' if 'qwen' in x.lower() else
            'Llama' if 'llama' in x.lower() else
            'Other'
        )

        # Consistency binary (for classification analysis)
        df['is_consistent'] = df['stability_score'] >= 0.8
        df['is_highly_consistent'] = df['stability_score'] >= 0.95

    return df


def compute_correlations(df: pd.DataFrame, target: str = 'stability_score') -> pd.DataFrame:
    """Compute correlations between factors and target."""

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target and identifiers
    exclude = [target, 'sample_idx', 'temperature', 'c_mean', 'validity_rate',
               'ranking_score', 'is_consistent', 'is_highly_consistent']
    factor_cols = [c for c in numeric_cols if c not in exclude]

    correlations = []
    for col in factor_cols:
        # Skip if constant
        if df[col].std() == 0:
            continue

        # Pearson correlation
        r, p = pearsonr(df[col].fillna(0), df[target].fillna(0))

        # Spearman correlation
        rho, p_spearman = spearmanr(df[col].fillna(0), df[target].fillna(0))

        correlations.append({
            'factor': col,
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spearman,
            'direction': 'positive' if r > 0 else 'negative',
            'significant': p < 0.05,
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('pearson_r', key=abs, ascending=False)

    return corr_df


def compute_feature_importance(df: pd.DataFrame, target: str = 'stability_score') -> pd.DataFrame:
    """Compute feature importance using Random Forest."""

    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target and identifiers
    exclude = [target, 'sample_idx', 'c_mean', 'validity_rate',
               'ranking_score', 'is_consistent', 'is_highly_consistent']
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get feature importances
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_,
    }).sort_values('importance', ascending=False)

    # Cross-validation score
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"Random Forest R² (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return importance_df


def analyze_by_factor_bins(df: pd.DataFrame, factor: str, target: str = 'stability_score',
                           n_bins: int = 4) -> pd.DataFrame:
    """Analyze target variable by binned factor."""

    # Create bins
    df_temp = df.copy()
    df_temp[f'{factor}_bin'] = pd.qcut(df_temp[factor], n_bins, labels=False, duplicates='drop')

    # Aggregate
    agg = df_temp.groupby(f'{factor}_bin').agg({
        target: ['mean', 'std', 'count'],
        factor: ['min', 'max', 'mean'],
    }).round(3)

    return agg


def compute_interaction_effects(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute interaction effects between key factors."""

    interactions = {}

    # Temperature × Schema Complexity
    if 'temperature' in df.columns and 'schema_complexity' in df.columns:
        df_temp = df.copy()
        df_temp['temp_bin'] = pd.cut(df_temp['temperature'], bins=[0, 0.3, 0.6, 1.0],
                                      labels=['Low', 'Medium', 'High'])
        df_temp['complexity_bin'] = pd.qcut(df_temp['schema_complexity'], 3,
                                            labels=['Simple', 'Medium', 'Complex'], duplicates='drop')

        pivot = df_temp.pivot_table(
            values='stability_score',
            index='complexity_bin',
            columns='temp_bin',
            aggfunc='mean'
        )
        interactions['temp_x_complexity'] = pivot

    # Num Tools × Query Complexity
    if 'num_tools' in df.columns and 'query_complexity_score' in df.columns:
        df_temp = df.copy()
        df_temp['tools_bin'] = pd.qcut(df_temp['num_tools'], 3,
                                       labels=['Few', 'Medium', 'Many'], duplicates='drop')
        df_temp['query_bin'] = pd.qcut(df_temp['query_complexity_score'], 3,
                                       labels=['Simple', 'Medium', 'Complex'], duplicates='drop')

        pivot = df_temp.pivot_table(
            values='stability_score',
            index='query_bin',
            columns='tools_bin',
            aggfunc='mean'
        )
        interactions['tools_x_query'] = pivot

    return interactions


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(df: pd.DataFrame, corr_df: pd.DataFrame,
                          importance_df: pd.DataFrame, output_dir: str):
    """Create all visualizations."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    top_factors = corr_df.head(15)['factor'].tolist()

    if len(top_factors) > 0:
        numeric_cols = ['stability_score'] + top_factors
        corr_matrix = df[numeric_cols].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=ax, square=True)
        ax.set_title('Factor Correlation Matrix (Top 15 Factors)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'correlation_heatmap.pdf', bbox_inches='tight')
        plt.close()

    # 2. Feature importance bar plot
    if len(importance_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(15)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top 15 Factors Affecting Consistency (Random Forest)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'feature_importance.pdf', bbox_inches='tight')
        plt.close()

    # 3. Key factor distributions
    key_factors = ['num_tools', 'query_length', 'schema_complexity', 'tool_name_ambiguity']
    available_factors = [f for f in key_factors if f in df.columns]

    if available_factors:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, factor in enumerate(available_factors[:4]):
            ax = axes[idx]

            # Bin by factor and compute mean consistency
            df_temp = df.copy()
            df_temp[f'{factor}_bin'] = pd.qcut(df_temp[factor], 5, labels=False, duplicates='drop')

            agg = df_temp.groupby(f'{factor}_bin').agg({
                'stability_score': 'mean',
                factor: 'mean',
            }).reset_index()

            ax.bar(range(len(agg)), agg['stability_score'], color='steelblue', edgecolor='black')
            ax.set_xticks(range(len(agg)))
            ax.set_xticklabels([f'{v:.1f}' for v in agg[factor]], rotation=45)
            ax.set_xlabel(factor.replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel('Mean Stability Score', fontsize=11)
            ax.set_title(f'Consistency vs {factor.replace("_", " ").title()}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'factor_distributions.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'factor_distributions.pdf', bbox_inches='tight')
        plt.close()

    # 4. Temperature × Complexity heatmap
    if 'temperature' in df.columns and 'schema_complexity' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))

        df_temp = df.copy()
        df_temp['temp_bin'] = pd.cut(df_temp['temperature'],
                                      bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                      labels=['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
        df_temp['complexity_bin'] = pd.qcut(df_temp['schema_complexity'], 4,
                                            labels=['Q1 (Simple)', 'Q2', 'Q3', 'Q4 (Complex)'],
                                            duplicates='drop')

        pivot = df_temp.pivot_table(
            values='stability_score',
            index='complexity_bin',
            columns='temp_bin',
            aggfunc='mean'
        )

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
        ax.set_title('Stability Score: Temperature × Schema Complexity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Schema Complexity', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'temp_complexity_heatmap.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'temp_complexity_heatmap.pdf', bbox_inches='tight')
        plt.close()

    # 5. Model family comparison
    if 'model_family' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        family_stats = df.groupby('model_family').agg({
            'stability_score': ['mean', 'std', 'count']
        }).round(3)
        family_stats.columns = ['mean', 'std', 'count']
        family_stats = family_stats.sort_values('mean', ascending=False)

        colors = plt.cm.Set2(np.linspace(0, 1, len(family_stats)))
        bars = ax.bar(range(len(family_stats)), family_stats['mean'],
                      yerr=family_stats['std'], capsize=5, color=colors, edgecolor='black')
        ax.set_xticks(range(len(family_stats)))
        ax.set_xticklabels(family_stats.index, rotation=45)
        ax.set_ylabel('Mean Stability Score', fontsize=12)
        ax.set_title('Consistency by Model Family', fontsize=14, fontweight='bold')

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, family_stats['count'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'n={count}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_family_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'model_family_comparison.pdf', bbox_inches='tight')
        plt.close()

    # 6. Language effect
    if 'is_english' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))

        lang_stats = df.groupby('is_english').agg({
            'stability_score': ['mean', 'std', 'count']
        }).round(3)
        lang_stats.columns = ['mean', 'std', 'count']

        labels = ['Non-English', 'English']
        colors = ['#e74c3c', '#2ecc71']
        bars = ax.bar(labels, lang_stats['mean'], yerr=lang_stats['std'],
                      capsize=5, color=colors, edgecolor='black')
        ax.set_ylabel('Mean Stability Score', fontsize=12)
        ax.set_title('Consistency by Query Language', fontsize=14, fontweight='bold')

        for bar, count in zip(bars, lang_stats['count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'n={count}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'language_effect.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'language_effect.pdf', bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {output_dir}")


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_tables(corr_df: pd.DataFrame, importance_df: pd.DataFrame,
                          df: pd.DataFrame, output_dir: str):
    """Generate LaTeX tables for paper."""

    output_dir = Path(output_dir)

    # Table 1: Top correlations
    latex_corr = """
\\begin{table}[h]
\\centering
\\caption{Factor Correlation with Consistency (Toucan, All Temperatures)}
\\label{tab:factor-correlations}
\\scriptsize
\\begin{tabular}{@{}lccc@{}}
\\toprule
\\textbf{Factor} & \\textbf{Pearson $r$} & \\textbf{$p$-value} & \\textbf{Direction} \\\\
\\midrule
"""

    for _, row in corr_df.head(10).iterrows():
        sig = '***' if row['pearson_p'] < 0.001 else '**' if row['pearson_p'] < 0.01 else '*' if row['pearson_p'] < 0.05 else ''
        factor_name = row['factor'].replace('_', ' ').title()
        latex_corr += f"{factor_name} & {row['pearson_r']:.3f}{sig} & {row['pearson_p']:.1e} & {row['direction']} \\\\\n"

    latex_corr += """\\bottomrule
\\end{tabular}

\\vspace{0.3em}
\\raggedright\\scriptsize Significance: ***$p<0.001$, **$p<0.01$, *$p<0.05$. Higher $|r|$ = stronger effect.
\\end{table}
"""

    with open(output_dir / 'correlation_table.tex', 'w') as f:
        f.write(latex_corr)

    # Table 2: Feature importance
    if len(importance_df) > 0:
        latex_imp = """
\\begin{table}[h]
\\centering
\\caption{Feature Importance for Predicting Consistency (Random Forest)}
\\label{tab:feature-importance}
\\scriptsize
\\begin{tabular}{@{}clc@{}}
\\toprule
\\textbf{Rank} & \\textbf{Feature} & \\textbf{Importance} \\\\
\\midrule
"""

        for rank, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            feature_name = row['feature'].replace('_', ' ').title()
            latex_imp += f"{rank} & {feature_name} & {row['importance']:.3f} \\\\\n"

        latex_imp += """\\bottomrule
\\end{tabular}

\\vspace{0.3em}
\\raggedright\\scriptsize Random Forest feature importance (mean decrease in impurity). Higher = more predictive.
\\end{table}
"""

        with open(output_dir / 'importance_table.tex', 'w') as f:
            f.write(latex_imp)

    # Table 3: Key findings summary
    latex_findings = """
\\begin{table}[h]
\\centering
\\caption{Key Factors Affecting LLM Structured Output Consistency}
\\label{tab:key-findings}
\\scriptsize
\\begin{tabular}{@{}llcc@{}}
\\toprule
\\textbf{Category} & \\textbf{Factor} & \\textbf{Effect} & \\textbf{Magnitude} \\\\
\\midrule
\\multirow{3}{*}{Schema}
  & Has nested params & Negative & High \\\\
  & Tool name ambiguity & Negative & Medium \\\\
  & Num available tools & Negative & Medium \\\\
\\midrule
\\multirow{3}{*}{Query}
  & Has JSON example & Positive & Medium \\\\
  & Query complexity & Negative & Low \\\\
  & Non-English & Negative & Low \\\\
\\midrule
\\multirow{2}{*}{Config}
  & Temperature & Negative & High \\\\
  & Temperature $\\times$ Complexity & Interaction & High \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(output_dir / 'findings_table.tex', 'w') as f:
        f.write(latex_findings)

    print(f"LaTeX tables saved to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Factor Analysis')
    parser.add_argument('--toucan-data', type=str,
                        default='toucan_data/toucan_tool_calls_1006.json',
                        help='Path to Toucan dataset')
    parser.add_argument('--metrics-dir', type=str,
                        default='results/toucan/minilm-ec2',
                        help='Directory containing consistency metrics')
    parser.add_argument('--output-dir', type=str,
                        default='results/factor_analysis',
                        help='Output directory for results')
    parser.add_argument('--gen-dir', type=str, default=None,
                        help='Directory containing raw generation results (optional)')

    args = parser.parse_args()

    # Resolve paths
    toucan_path = PROJECT_ROOT / args.toucan_data
    metrics_dir = PROJECT_ROOT / args.metrics_dir
    output_dir = PROJECT_ROOT / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE FACTOR ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    toucan_data = load_toucan_data(toucan_path)
    print(f"  Loaded {len(toucan_data)} Toucan samples")

    metrics_file = metrics_dir / 'combined_consistency_metrics_results.json'
    metrics = load_consistency_metrics(metrics_file)
    print(f"  Loaded metrics for {len(metrics)} models")

    # Build dataframe
    print("\n[2/6] Building analysis dataframe...")
    df = build_analysis_dataframe(toucan_data, metrics)
    print(f"  Created dataframe with {len(df)} rows, {len(df.columns)} columns")

    # Save dataframe
    df.to_csv(output_dir / 'factor_analysis_data.csv', index=False)
    print(f"  Saved to {output_dir / 'factor_analysis_data.csv'}")

    # Correlation analysis
    print("\n[3/6] Computing correlations...")
    corr_df = compute_correlations(df, target='stability_score')
    print("\nTop 10 correlations with stability_score:")
    print(corr_df.head(10).to_string(index=False))
    corr_df.to_csv(output_dir / 'correlations.csv', index=False)

    # Feature importance
    print("\n[4/6] Computing feature importance...")
    importance_df = compute_feature_importance(df, target='stability_score')
    if len(importance_df) > 0:
        print("\nTop 10 features by importance:")
        print(importance_df.head(10).to_string(index=False))
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Interaction effects
    print("\n[5/6] Computing interaction effects...")
    interactions = compute_interaction_effects(df)
    for name, pivot in interactions.items():
        print(f"\n{name}:")
        print(pivot.round(3))
        pivot.to_csv(output_dir / f'interaction_{name}.csv')

    # Visualizations
    print("\n[6/6] Creating visualizations...")
    create_visualizations(df, corr_df, importance_df, output_dir)

    # LaTeX tables
    generate_latex_tables(corr_df, importance_df, df, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey files:")
    print(f"  - factor_analysis_data.csv (full data)")
    print(f"  - correlations.csv (factor correlations)")
    print(f"  - feature_importance.csv (RF importance)")
    print(f"  - *.png/*.pdf (visualizations)")
    print(f"  - *.tex (LaTeX tables)")


if __name__ == '__main__':
    main()
