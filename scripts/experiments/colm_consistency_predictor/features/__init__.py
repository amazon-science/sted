"""
COLM 2026 Consistency Predictor - Feature Extraction Module

67 features across 4 categories:
- Surface Linguistic (25): modal verbs, hedging, politeness, speech acts
- Semantic (19): ambiguity, underspecification, complexity
- Pragmatic (17): task clarity, presuppositions, context dependency
- Schema (6): tool/parameter complexity
"""

from .surface_features import extract_surface_features, SurfaceFeatures
from .semantic_features import extract_semantic_features, SemanticFeatures
from .pragmatic_features import extract_pragmatic_features, PragmaticFeatures
from .schema_features import extract_schema_features, SchemaFeatures
from .extract_all_features import (
    extract_all_features,
    extract_features_from_dataset,
    AllFeatures
)

__all__ = [
    'extract_surface_features', 'SurfaceFeatures',
    'extract_semantic_features', 'SemanticFeatures',
    'extract_pragmatic_features', 'PragmaticFeatures',
    'extract_schema_features', 'SchemaFeatures',
    'extract_all_features', 'extract_features_from_dataset', 'AllFeatures',
]
