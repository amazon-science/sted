#!/usr/bin/env python3
"""
Combined Feature Extractor for COLM 2026 Consistency Predictor

Extracts all 67 features from prompt + tools:
- Surface Linguistic: 25 features
- Semantic: 19 features
- Pragmatic: 17 features
- Schema: 6 features

Total: 67 features
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.surface_features import extract_surface_features, SurfaceFeatures
from features.semantic_features import extract_semantic_features, SemanticFeatures
from features.pragmatic_features import extract_pragmatic_features, PragmaticFeatures
from features.schema_features import extract_schema_features, SchemaFeatures


@dataclass
class AllFeatures:
    """Container for all 67 features."""
    surface: SurfaceFeatures
    semantic: SemanticFeatures
    pragmatic: PragmaticFeatures
    schema: SchemaFeatures

    def to_dict(self) -> Dict[str, Any]:
        """Flatten all features to a single dictionary."""
        result = {}

        # Add prefix to each feature category
        for key, val in self.surface.to_dict().items():
            result[f'surface_{key}'] = val

        for key, val in self.semantic.to_dict().items():
            result[f'semantic_{key}'] = val

        for key, val in self.pragmatic.to_dict().items():
            result[f'pragmatic_{key}'] = val

        for key, val in self.schema.to_dict().items():
            result[f'schema_{key}'] = val

        return result

    def to_vector(self) -> List[float]:
        """Convert all features to a flat vector for ML models."""
        d = self.to_dict()
        return [float(v) for v in d.values()]

    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names."""
        # Create dummy instances to get keys
        surface = SurfaceFeatures()
        semantic = SemanticFeatures()
        pragmatic = PragmaticFeatures()
        schema = SchemaFeatures()

        names = []
        for key in surface.to_dict().keys():
            names.append(f'surface_{key}')
        for key in semantic.to_dict().keys():
            names.append(f'semantic_{key}')
        for key in pragmatic.to_dict().keys():
            names.append(f'pragmatic_{key}')
        for key in schema.to_dict().keys():
            names.append(f'schema_{key}')

        return names


def extract_all_features(
    prompt: str,
    tools: Optional[List[Dict]] = None
) -> AllFeatures:
    """
    Extract all 67 features from a prompt and its tool definitions.

    Args:
        prompt: The user prompt/instruction text
        tools: Optional list of tool definitions (JSON schema format)

    Returns:
        AllFeatures dataclass containing all 67 features
    """
    surface = extract_surface_features(prompt)
    semantic = extract_semantic_features(prompt)
    pragmatic = extract_pragmatic_features(prompt)
    schema = extract_schema_features(tools or [])

    return AllFeatures(
        surface=surface,
        semantic=semantic,
        pragmatic=pragmatic,
        schema=schema
    )


def extract_features_from_dataset(
    data: List[Dict[str, Any]],
    prompt_key: str = 'prompt',
    tools_key: str = 'tools',
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract features from a dataset of prompts.

    Args:
        data: List of dictionaries with prompt and optional tools
        prompt_key: Key for prompt text in each dict
        tools_key: Key for tools list in each dict
        show_progress: Whether to show progress

    Returns:
        DataFrame with one row per sample, 67 feature columns
    """
    all_features = []
    n = len(data)

    for i, item in enumerate(data):
        if show_progress and (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{n}...")

        prompt = item.get(prompt_key, '')
        tools = item.get(tools_key, [])

        features = extract_all_features(prompt, tools)
        feature_dict = features.to_dict()

        # Add any metadata
        if 'id' in item:
            feature_dict['id'] = item['id']
        if 'sample_id' in item:
            feature_dict['sample_id'] = item['sample_id']

        all_features.append(feature_dict)

    df = pd.DataFrame(all_features)
    return df


def print_feature_summary():
    """Print summary of all 67 features."""
    print("=" * 70)
    print("COLM 2026 Consistency Predictor - 67 Feature Summary")
    print("=" * 70)

    names = AllFeatures.feature_names()

    # Group by category
    categories = {
        'surface': [],
        'semantic': [],
        'pragmatic': [],
        'schema': []
    }

    for name in names:
        for cat in categories:
            if name.startswith(cat):
                categories[cat].append(name)
                break

    for cat, features in categories.items():
        print(f"\n{cat.upper()} FEATURES ({len(features)}):")
        print("-" * 40)
        for f in features:
            # Remove prefix for cleaner display
            short_name = f.replace(f'{cat}_', '')
            print(f"  - {short_name}")

    print(f"\nTOTAL: {len(names)} features")


# Test
if __name__ == '__main__':
    print_feature_summary()

    print("\n" + "=" * 70)
    print("Feature Extraction Test")
    print("=" * 70)

    # Test prompt with tools
    test_prompt = """Please search for all JSON files that contain 'user' in the reports folder.
    If there are more than 10 results, return only the first 5.
    The files should be sorted by modification date."""

    test_tools = [
        {
            "name": "search_files",
            "description": "Search for files matching criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "extension": {"type": "string"},
                            "max_results": {"type": "integer"}
                        }
                    }
                }
            }
        }
    ]

    print(f"\nPrompt: {test_prompt[:80]}...")
    print(f"Tools: {len(test_tools)} tool(s)")

    # Extract features
    features = extract_all_features(test_prompt, test_tools)
    feature_dict = features.to_dict()

    print(f"\nExtracted {len(feature_dict)} features:")

    # Show by category
    for cat in ['surface', 'semantic', 'pragmatic', 'schema']:
        print(f"\n{cat.upper()}:")
        cat_features = {k: v for k, v in feature_dict.items() if k.startswith(cat)}
        for k, v in cat_features.items():
            if v != 0 and v != 0.5:  # Show non-default values
                short_k = k.replace(f'{cat}_', '')
                if isinstance(v, float):
                    print(f"  {short_k}: {v:.3f}")
                else:
                    print(f"  {short_k}: {v}")

    # Test vector conversion
    vector = features.to_vector()
    print(f"\nFeature vector length: {len(vector)}")

    # Test batch processing
    print("\n" + "=" * 70)
    print("Batch Processing Test")
    print("=" * 70)

    test_data = [
        {"id": 1, "prompt": "List all files.", "tools": []},
        {"id": 2, "prompt": "Could you please help me find documents?", "tools": test_tools},
        {"id": 3, "prompt": "You must return exactly 5 JSON results.", "tools": test_tools},
    ]

    df = extract_features_from_dataset(test_data, show_progress=False)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nSample of extracted features:")
    print(df[['id', 'surface_word_count', 'semantic_ambiguity_score',
              'pragmatic_task_clarity_score', 'schema_num_tools']].to_string())
