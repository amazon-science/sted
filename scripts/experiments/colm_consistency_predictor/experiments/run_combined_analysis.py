#!/usr/bin/env python3
"""
Combined Analysis: Toucan + ShareGPT + json-instruct Datasets

Runs feature extraction and correlation analysis on all datasets:
- Toucan: Tool calling (1000 prompts, 21 models)
- ShareGPT: General structured output (80 prompts, 25 models)
- json-instruct: Complex JSON schemas (100+ prompts, 74% complex schemas)

Total: ~1180+ unique prompts with diverse schema complexity
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

from features import extract_all_features, AllFeatures


def load_toucan_prompts(base_dir: str = "llm_gen_results/toucan") -> List[Dict]:
    """Load Toucan prompts (tool calling)."""
    prompts = []
    base_path = PROJECT_ROOT / base_dir
    seen_ids = set()

    for model_dir in base_path.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("generations-"):
            for run_dir in model_dir.iterdir():
                if run_dir.is_dir():
                    results_file = run_dir / "intermediate_results.json"
                    if results_file.exists():
                        with open(results_file) as f:
                            data = json.load(f)
                        for idx, item in enumerate(data):
                            sample_id = item.get('sample_id', f'toucan_{idx}')
                            if sample_id not in seen_ids:
                                seen_ids.add(sample_id)
                                prompts.append({
                                    'sample_id': sample_id,
                                    'sample_idx': idx,
                                    'dataset': 'toucan',
                                    'prompt': item.get('query', ''),
                                    'tools': item.get('tools', []),
                                    'has_tools': True
                                })
                        if prompts:
                            return prompts
    return prompts


def extract_json_schema_from_prompt(prompt: str) -> Optional[Dict]:
    """Extract JSON schema from ShareGPT prompt if present."""
    # Look for JSON schema in the prompt
    schema_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', prompt)
    if schema_match:
        try:
            schema = json.loads(schema_match.group(1))
            return schema
        except json.JSONDecodeError:
            pass
    return None


def count_schema_properties(schema: Dict, depth: int = 0) -> Dict[str, Any]:
    """Count properties and nesting in JSON schema."""
    if not isinstance(schema, dict):
        return {'n_properties': 0, 'max_depth': depth, 'has_array': False, 'has_object': False}

    n_props = 0
    max_depth = depth
    has_array = schema.get('type') == 'array'
    has_object = schema.get('type') == 'object'

    if 'properties' in schema:
        n_props += len(schema['properties'])
        for prop_schema in schema['properties'].values():
            sub = count_schema_properties(prop_schema, depth + 1)
            n_props += sub['n_properties']
            max_depth = max(max_depth, sub['max_depth'])
            has_array = has_array or sub['has_array']
            has_object = has_object or sub['has_object']

    if 'items' in schema:
        sub = count_schema_properties(schema['items'], depth + 1)
        n_props += sub['n_properties']
        max_depth = max(max_depth, sub['max_depth'])
        has_array = True
        has_object = has_object or sub['has_object']

    return {
        'n_properties': n_props,
        'max_depth': max_depth,
        'has_array': has_array,
        'has_object': has_object
    }


def load_sharegpt_prompts(base_dir: str = "llm_gen_results/sharegpt") -> List[Dict]:
    """Load ShareGPT prompts (general structured output)."""
    prompts = []
    base_path = PROJECT_ROOT / base_dir
    seen_ids = set()

    for model_dir in base_path.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("generations-"):
            for temp_dir in model_dir.iterdir():
                if temp_dir.is_dir() and temp_dir.name.startswith("llm_gen_results_"):
                    # Load sample files
                    for sample_file in sorted(temp_dir.glob("sample_*.json")):
                        if "_modified_prompt" in sample_file.name:
                            continue

                        try:
                            with open(sample_file) as f:
                                data = json.load(f)

                            sample_id = data.get('sample_id', sample_file.stem)
                            if sample_id in seen_ids:
                                continue
                            seen_ids.add(sample_id)

                            # Use modified_prompt which includes the schema
                            prompt = data.get('modified_prompt', data.get('original_prompt', ''))

                            # Extract schema from prompt
                            schema = extract_json_schema_from_prompt(prompt)
                            schema_info = count_schema_properties(schema) if schema else {
                                'n_properties': 0, 'max_depth': 0, 'has_array': False, 'has_object': False
                            }

                            # Extract idx from sample_id (e.g., "sample_000" -> 0)
                            idx_match = re.search(r'(\d+)', sample_id)
                            idx = int(idx_match.group(1)) if idx_match else len(prompts)

                            prompts.append({
                                'sample_id': f'sharegpt_{sample_id}',
                                'sample_idx': idx,
                                'dataset': 'sharegpt',
                                'prompt': prompt,
                                'tools': [],  # No tools
                                'has_tools': False,
                                'schema_n_properties': schema_info['n_properties'],
                                'schema_max_depth': schema_info['max_depth'],
                                'schema_has_array': schema_info['has_array'],
                                'schema_has_object': schema_info['has_object'],
                            })

                        except (json.JSONDecodeError, Exception) as e:
                            continue

                    if prompts:
                        return prompts  # Return after first model to get unique prompts
    return prompts


def load_json_instruct_prompts(base_dir: str = "llm_gen_results/json_instruct") -> List[Dict]:
    """Load json-instruct prompts (complex JSON schemas)."""
    prompts = []
    base_path = PROJECT_ROOT / base_dir

    # Load the converted data
    converted_path = base_path / "converted_data" / "json_instruct" / "all_conversations.json"
    if converted_path.exists():
        with open(converted_path) as f:
            data = json.load(f)

        for i, item in enumerate(data):
            prompt = item['conversations'][1]['value']  # User prompt with schema
            original_schema = item.get('original_schema', '')

            # Parse schema for complexity metrics
            try:
                schema = json.loads(original_schema)
                schema_info = count_schema_properties(schema)
            except (json.JSONDecodeError, TypeError):
                schema_info = {'n_properties': 0, 'max_depth': 0, 'has_array': False, 'has_object': False}

            prompts.append({
                'sample_id': f'json_instruct_{i}',
                'sample_idx': i,
                'dataset': 'json_instruct',
                'prompt': prompt,
                'tools': [],  # No tools, but has complex schema
                'has_tools': False,
                'schema_n_properties': schema_info['n_properties'],
                'schema_max_depth': schema_info['max_depth'],
                'schema_has_array': schema_info['has_array'],
                'schema_has_object': schema_info['has_object'],
            })

    return prompts


def load_json_instruct_metrics(base_dir: str = "llm_gen_results/json_instruct") -> pd.DataFrame:
    """Load consistency metrics from json-instruct generation results.

    First tries to load pre-computed STED metrics. If not available,
    falls back to validity_rate as proxy.
    """
    base_path = PROJECT_ROOT / base_dir

    # Try to load pre-computed STED metrics
    sted_metrics_file = base_path / "sted_metrics.json"
    if sted_metrics_file.exists():
        with open(sted_metrics_file) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    # Fall back to computing from raw results
    rows = []
    for result_dir in base_path.iterdir():
        if result_dir.is_dir() and result_dir.name.startswith("llm_gen_results_"):
            all_results = result_dir / "all_results.json"
            if all_results.exists():
                with open(all_results) as f:
                    data = json.load(f)

                metadata = data.get('metadata', {})
                model = metadata.get('model_id', 'unknown')
                temperature = metadata.get('temperature', 0.5)

                for result in data.get('results', []):
                    responses = result.get('responses', [])
                    if len(responses) < 2:
                        continue

                    n_valid = len([r for r in responses if isinstance(r, (dict, list))])
                    validity_rate = n_valid / len(responses) if responses else 0

                    sample_id = result.get('sample_id', 'sample_000')
                    idx_match = re.search(r'(\d+)', sample_id)
                    sample_idx = int(idx_match.group(1)) if idx_match else 0

                    rows.append({
                        'model': model,
                        'sample_idx': sample_idx,
                        'temperature': temperature,
                        'c_mean': validity_rate,  # Proxy for consistency
                        'd_std': 0,
                        'stability_score': validity_rate,
                        'validity_rate': validity_rate,
                        'dataset': 'json_instruct'
                    })

    return pd.DataFrame(rows)


def load_consistency_metrics(dataset: str) -> pd.DataFrame:
    """Load pre-computed consistency metrics for a dataset."""
    if dataset == 'toucan':
        path = PROJECT_ROOT / "results" / "toucan_exact_final" / "combined_consistency_metrics_results.json"
    else:
        path = PROJECT_ROOT / "results" / "sharegpt_exact_final" / "combined_consistency_metrics_results.json"

    with open(path) as f:
        data = json.load(f)

    rows = []
    for model, samples in data.items():
        for sample in samples:
            rows.append({
                'model': model,
                'sample_idx': sample['sample_idx'],
                'temperature': sample.get('temperature', 0.5),
                'c_mean': sample.get('c_mean', sample.get('mean_similarity', 0)),
                'd_std': sample.get('d_std', 0),
                'stability_score': sample.get('stability_score', 0),
                'validity_rate': sample.get('validity_rate', 1),
                'dataset': dataset
            })

    return pd.DataFrame(rows)


def extract_features_batch(prompts: List[Dict], show_progress: bool = True) -> pd.DataFrame:
    """Extract features for all prompts."""
    rows = []
    n = len(prompts)

    for i, item in enumerate(prompts):
        if show_progress and (i + 1) % 50 == 0:
            print(f"  Extracting features: {i+1}/{n}...")

        features = extract_all_features(item['prompt'], item.get('tools', []))
        feature_dict = features.to_dict()

        # Override schema features for ShareGPT if we extracted them
        if item['dataset'] == 'sharegpt' and 'schema_n_properties' in item:
            feature_dict['schema_num_tools'] = 1  # Treat output schema as 1 "tool"
            feature_dict['schema_total_params'] = item['schema_n_properties']
            feature_dict['schema_max_params'] = item['schema_n_properties']
            feature_dict['schema_max_nesting_depth'] = item['schema_max_depth']
            feature_dict['schema_has_array_params'] = 1 if item['schema_has_array'] else 0
            feature_dict['schema_has_object_params'] = 1 if item['schema_has_object'] else 0

        feature_dict['sample_idx'] = item['sample_idx']
        feature_dict['sample_id'] = item['sample_id']
        feature_dict['dataset'] = item['dataset']

        rows.append(feature_dict)

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("COLM 2026: Combined Dataset Analysis (Toucan + ShareGPT + json-instruct)")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446" / "results" / "combined_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts from all datasets
    print("\n[1/5] Loading prompts...")
    toucan_prompts = load_toucan_prompts()
    print(f"  Toucan: {len(toucan_prompts)} prompts")

    sharegpt_prompts = load_sharegpt_prompts()
    print(f"  ShareGPT: {len(sharegpt_prompts)} prompts")

    json_instruct_prompts = load_json_instruct_prompts()
    print(f"  json-instruct: {len(json_instruct_prompts)} prompts")

    all_prompts = toucan_prompts + sharegpt_prompts + json_instruct_prompts
    print(f"  Total: {len(all_prompts)} prompts")

    # Extract features
    print("\n[2/5] Extracting 67 features...")
    features_df = extract_features_batch(all_prompts)
    features_df.to_csv(output_dir / "all_features.csv", index=False)

    # Load consistency metrics
    print("\n[3/5] Loading consistency metrics...")
    toucan_metrics = load_consistency_metrics('toucan')
    sharegpt_metrics = load_consistency_metrics('sharegpt')
    json_instruct_metrics = load_json_instruct_metrics()
    all_metrics = pd.concat([toucan_metrics, sharegpt_metrics, json_instruct_metrics], ignore_index=True)
    print(f"  Toucan: {len(toucan_metrics)} records")
    print(f"  ShareGPT: {len(sharegpt_metrics)} records")
    print(f"  json-instruct: {len(json_instruct_metrics)} records")
    print(f"  Total: {len(all_metrics)} records")

    # Analyze feature distributions by dataset
    print("\n[4/5] Analyzing feature distributions...")
    feature_cols = [c for c in features_df.columns if c not in ['sample_idx', 'sample_id', 'dataset']]

    dist_stats = []
    for col in feature_cols:
        for ds in ['toucan', 'sharegpt', 'json_instruct', 'combined']:
            if ds == 'combined':
                values = features_df[col].values
            else:
                values = features_df[features_df['dataset'] == ds][col].values

            if len(values) == 0:
                continue

            nonzero = values[values != 0]
            dist_stats.append({
                'feature': col,
                'dataset': ds,
                'n_samples': len(values),
                'nonzero_rate': len(nonzero) / len(values) if len(values) > 0 else 0,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            })

    dist_df = pd.DataFrame(dist_stats)
    dist_df.to_csv(output_dir / "feature_distributions_by_dataset.csv", index=False)

    # Compute correlations
    print("\n[5/5] Computing correlations...")

    def compute_corrs(features, metrics, dataset_name):
        """Compute correlations for a dataset."""
        agg = metrics.groupby('sample_idx')['c_mean'].mean().reset_index()
        merged = features.merge(agg, on='sample_idx', how='inner')

        results = []
        for col in feature_cols:
            if merged[col].std() == 0:
                continue
            corr, pval = spearmanr(merged[col], merged['c_mean'])
            results.append({
                'feature': col,
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05,
                'dataset': dataset_name,
                'category': col.split('_')[0]
            })
        return pd.DataFrame(results)

    # Per-dataset correlations
    toucan_corrs = compute_corrs(
        features_df[features_df['dataset'] == 'toucan'],
        toucan_metrics,
        'toucan'
    )
    sharegpt_corrs = compute_corrs(
        features_df[features_df['dataset'] == 'sharegpt'],
        sharegpt_metrics,
        'sharegpt'
    )
    json_instruct_corrs = compute_corrs(
        features_df[features_df['dataset'] == 'json_instruct'],
        json_instruct_metrics,
        'json_instruct'
    ) if len(json_instruct_metrics) > 0 else pd.DataFrame()
    combined_corrs = compute_corrs(features_df, all_metrics, 'combined')

    all_corrs = pd.concat([toucan_corrs, sharegpt_corrs, json_instruct_corrs, combined_corrs], ignore_index=True)
    all_corrs.to_csv(output_dir / "correlations_by_dataset.csv", index=False)

    # Print results
    print("\n" + "=" * 70)
    print("TOP 10 FEATURES BY DATASET")
    print("=" * 70)

    for ds in ['toucan', 'sharegpt', 'json_instruct', 'combined']:
        df = all_corrs[all_corrs['dataset'] == ds].sort_values('correlation', key=abs, ascending=False)
        if len(df) == 0:
            continue
        print(f"\n{ds.upper()} (n={len(features_df[features_df['dataset'] == ds]) if ds != 'combined' else len(features_df)}):")
        for _, row in df.head(5).iterrows():
            print(f"  {row['feature']}: r={row['correlation']:.3f} (p={row['p_value']:.4f})")

    # Category comparison
    print("\n" + "=" * 70)
    print("CATEGORY IMPORTANCE BY DATASET")
    print("=" * 70)

    cat_summary = all_corrs.groupby(['dataset', 'category']).agg({
        'correlation': lambda x: np.abs(x).mean()
    }).reset_index()
    cat_summary.columns = ['dataset', 'category', 'avg_abs_corr']
    cat_pivot = cat_summary.pivot(index='category', columns='dataset', values='avg_abs_corr')
    print(cat_pivot.round(3).to_string())

    # Feature coverage comparison
    print("\n" + "=" * 70)
    print("FEATURE COVERAGE COMPARISON")
    print("=" * 70)

    coverage_pivot = dist_df.pivot(index='feature', columns='dataset', values='nonzero_rate')
    coverage_diff = coverage_pivot['sharegpt'] - coverage_pivot['toucan']
    most_different = coverage_diff.abs().sort_values(ascending=False).head(10)

    print("\nFeatures with most different coverage between datasets:")
    for feat in most_different.index:
        toucan_cov = coverage_pivot.loc[feat, 'toucan']
        sharegpt_cov = coverage_pivot.loc[feat, 'sharegpt']
        print(f"  {feat}: Toucan={toucan_cov:.2%}, ShareGPT={sharegpt_cov:.2%}")

    # Save summary
    summary = {
        'datasets': {
            'toucan': {'n_prompts': len(toucan_prompts), 'n_metrics': len(toucan_metrics)},
            'sharegpt': {'n_prompts': len(sharegpt_prompts), 'n_metrics': len(sharegpt_metrics)},
            'json_instruct': {'n_prompts': len(json_instruct_prompts), 'n_metrics': len(json_instruct_metrics)},
            'combined': {'n_prompts': len(all_prompts), 'n_metrics': len(all_metrics)}
        },
        'top_features': {
            ds: all_corrs[all_corrs['dataset'] == ds].sort_values(
                'correlation', key=abs, ascending=False
            ).head(5)[['feature', 'correlation']].to_dict('records')
            for ds in ['toucan', 'sharegpt', 'json_instruct', 'combined'] if len(all_corrs[all_corrs['dataset'] == ds]) > 0
        },
        'category_importance': cat_pivot.to_dict()
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    print(f"  Output: {output_dir}")

    return all_corrs, dist_df


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    all_corrs, dist_df = main()
