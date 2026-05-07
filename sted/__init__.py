"""
Semantic JSON Tree Consistency Evaluation Framework

This package provides tools for evaluating structural consistency of JSON outputs
using tree edit distance algorithms enhanced with semantic similarity.
"""

# Core lightweight imports (no bert_score dependency)
from .json_tree_node import JsonNode
from .similarity_cache import StringSimilarityCache
from .utils import collect_all_values, count_json_elements, parse_structured_outputs
from .format_converters import (
    detect_format,
    xml_to_dict,
    yaml_to_dict,
    html_to_dict,
    parse_structured_string,
)

# Optional imports - these require bert_score and torch
# Use lazy imports to allow package to work without these heavy dependencies
try:
    from .semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
    from .structural_consistency_analyzer import StructuralConsistencyAnalyzer
    # Short alias for main class (STED = Semantic Tree Edit Distance)
    STED = SemanticJsonTreeConsistencyEvaluator
except ImportError:
    SemanticJsonTreeConsistencyEvaluator = None
    StructuralConsistencyAnalyzer = None
    STED = None

try:
    from .llm_judge import LLMJudge, create_llm_judge
except ImportError:
    LLMJudge = None
    create_llm_judge = None

from .consistency_risk_assessor import ConsistencyRiskAssessor, RiskReport
from .consistency_predictor import ConsistencyPredictor, PredictionResult

__version__ = "0.1.0"
__author__ = "AWS Generative AI Innovation Center"

__all__ = [
    "STED",
    "SemanticJsonTreeConsistencyEvaluator",
    "StructuralConsistencyAnalyzer",
    "JsonNode",
    "StringSimilarityCache",
    "collect_all_values",
    "count_json_elements",
    "parse_structured_outputs",
    "detect_format",
    "xml_to_dict",
    "yaml_to_dict",
    "html_to_dict",
    "parse_structured_string",
    "LLMJudge",
    "create_llm_judge",
    "ConsistencyRiskAssessor",
    "RiskReport",
    "ConsistencyPredictor",
    "PredictionResult",
]
