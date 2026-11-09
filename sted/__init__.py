"""STED: compare structured LLM outputs field by field and rate their consistency.

Stable API::

    sted.compare(a, b, ignore=..., unordered=..., tolerance=...)   -> Comparison
    sted.consistency(outputs, ...)                                 -> Consistency
    sted.FieldRules(ignore=..., unordered=..., tolerance=...)
    sted.AgentConsistencyEvaluator          batch evaluation of agents / production logs
    sted.SemanticJsonTreeConsistencyEvaluator (alias sted.STED)   low-level evaluator, profiles
    sted.parse_structured_string, sted.detect_format              JSON / YAML / XML / HTML text

Everything else is importable from its submodule (``sted.otel_adapter``,
``sted.format_converters``, ...). Research utilities that used to be re-exported here
(LLM judge, consistency predictor / risk assessor, JSON helpers) still resolve with a
DeprecationWarning; import them from their submodules instead.
"""
import importlib
import warnings

from ._logging import get_logger, set_log_level
from .rules import FieldRules
from .api import compare, consistency, Comparison, Consistency, FieldDifference, UnstableField
from .format_converters import detect_format, parse_structured_string

__version__ = "0.2.0"
__author__ = "AWS Generative AI Innovation Center"

# Stable names whose modules pull in scipy and friends: loaded on first access.
_LAZY = {
    "SemanticJsonTreeConsistencyEvaluator": ("semantic_json_tree_consistency", "SemanticJsonTreeConsistencyEvaluator"),
    "STED": ("semantic_json_tree_consistency", "SemanticJsonTreeConsistencyEvaluator"),
    "AgentConsistencyEvaluator": ("agent_consistency_evaluator", "AgentConsistencyEvaluator"),
    "ConsistencyReport": ("agent_consistency_evaluator", "ConsistencyReport"),
    "PromptResult": ("agent_consistency_evaluator", "PromptResult"),
}

# No longer part of the top-level API: still resolvable, with a DeprecationWarning.
_DEPRECATED = {
    "StructuralConsistencyAnalyzer": ("structural_consistency_analyzer", "StructuralConsistencyAnalyzer"),
    "JsonNode": ("json_tree_node", "JsonNode"),
    "StringSimilarityCache": ("similarity_cache", "StringSimilarityCache"),
    "collect_all_values": ("utils", "collect_all_values"),
    "count_json_elements": ("utils", "count_json_elements"),
    "parse_structured_outputs": ("utils", "parse_structured_outputs"),
    "xml_to_dict": ("format_converters", "xml_to_dict"),
    "yaml_to_dict": ("format_converters", "yaml_to_dict"),
    "html_to_dict": ("format_converters", "html_to_dict"),
    "LLMJudge": ("llm_judge", "LLMJudge"),
    "create_llm_judge": ("llm_judge", "create_llm_judge"),
    "ConsistencyRiskAssessor": ("consistency_risk_assessor", "ConsistencyRiskAssessor"),
    "RiskReport": ("consistency_risk_assessor", "RiskReport"),
    "ConsistencyPredictor": ("consistency_predictor", "ConsistencyPredictor"),
    "PredictionResult": ("consistency_predictor", "PredictionResult"),
}


def __getattr__(name):
    if name in _LAZY:
        module, attr = _LAZY[name]
    elif name in _DEPRECATED:
        module, attr = _DEPRECATED[name]
        warnings.warn(f"sted.{name} is no longer part of the top-level API; "
                      f"use `from sted.{module} import {attr}`", DeprecationWarning, stacklevel=2)
    else:
        raise AttributeError(f"module 'sted' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), attr)
    if name in _LAZY:
        globals()[name] = value          # cache: later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "compare",
    "consistency",
    "FieldRules",
    "Comparison",
    "Consistency",
    "FieldDifference",
    "UnstableField",
    "AgentConsistencyEvaluator",
    "ConsistencyReport",
    "PromptResult",
    "SemanticJsonTreeConsistencyEvaluator",
    "STED",
    "parse_structured_string",
    "detect_format",
    "get_logger",
    "set_log_level",
]
