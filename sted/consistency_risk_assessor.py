"""
Consistency Risk Assessor for Tool-Calling Prompts

Pre-generation assessment tool that analyzes prompt + tool definitions
to identify factors likely to cause inconsistent LLM outputs.

Risk factors and thresholds are derived from empirical analysis of
225K+ samples across 19 LLMs (see docs/analysis/CONSISTENCY_FACTORS_ANALYSIS.md).

Usage:
    from sted import ConsistencyRiskAssessor

    assessor = ConsistencyRiskAssessor()
    report = assessor.assess(prompt, tools)
    report.print_report()
"""

from __future__ import annotations

import re
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def __lt__(self, other):
        order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}
        return order[self] < order[other]


@dataclass
class Mitigation:
    """A concrete action to reduce consistency risk."""
    description: str
    example_before: Optional[str] = None
    example_after: Optional[str] = None

    def __str__(self):
        s = self.description
        if self.example_before and self.example_after:
            s += f"\n      Before: {self.example_before}\n      After:  {self.example_after}"
        return s


@dataclass
class RiskFactor:
    """A single identified risk factor."""
    name: str
    severity: Severity
    value: float
    threshold: float
    description: str
    mitigation: Mitigation

    def __str__(self):
        icon = {"low": "~", "medium": "!", "high": "!!"}[self.severity.value]
        return f"[{icon}] {self.name} ({self.severity.value}): {self.description}"


@dataclass
class RiskReport:
    """Assessment report for a prompt + tool definition."""
    overall_risk: float  # 0.0 (safe) to 1.0 (high risk)
    risk_level: str  # "low", "medium", "high"
    risk_factors: list[RiskFactor] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def high_risk_factors(self) -> list[RiskFactor]:
        return [f for f in self.risk_factors if f.severity == Severity.HIGH]

    @property
    def medium_risk_factors(self) -> list[RiskFactor]:
        return [f for f in self.risk_factors if f.severity == Severity.MEDIUM]

    @property
    def low_risk_factors(self) -> list[RiskFactor]:
        return [f for f in self.risk_factors if f.severity == Severity.LOW]

    def print_report(self):
        """Print a formatted risk report to stdout."""
        print("=" * 70)
        print("CONSISTENCY RISK ASSESSMENT")
        print("=" * 70)
        print(f"Overall Risk: {self.overall_risk:.2f} ({self.risk_level.upper()})")
        print(f"Risk Factors: {len(self.high_risk_factors)} high, "
              f"{len(self.medium_risk_factors)} medium, "
              f"{len(self.low_risk_factors)} low")
        print()

        if self.metrics:
            print("--- Metrics ---")
            for k, v in self.metrics.items():
                print(f"  {k}: {v}")
            print()

        for severity_label, factors in [
            ("HIGH RISK", self.high_risk_factors),
            ("MEDIUM RISK", self.medium_risk_factors),
            ("LOW RISK", self.low_risk_factors),
        ]:
            if factors:
                print(f"--- {severity_label} ---")
                for f in factors:
                    print(f"  {f}")
                    print(f"    Mitigation: {f.mitigation}")
                    print()

        if not self.risk_factors:
            print("No significant risk factors identified.")

        print("=" * 70)

    def to_dict(self) -> dict:
        """Serialize report to dictionary."""
        return {
            "overall_risk": self.overall_risk,
            "risk_level": self.risk_level,
            "metrics": self.metrics,
            "risk_factors": [
                {
                    "name": f.name,
                    "severity": f.severity.value,
                    "value": f.value,
                    "threshold": f.threshold,
                    "description": f.description,
                    "mitigation": {
                        "description": f.mitigation.description,
                        "example_before": f.mitigation.example_before,
                        "example_after": f.mitigation.example_after,
                    }
                }
                for f in self.risk_factors
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Schema Analysis Helpers
# ============================================================================

def _compute_schema_depth(schema: dict, current_depth: int = 0) -> int:
    """Compute max nesting depth of a JSON schema."""
    if not isinstance(schema, dict):
        return current_depth

    max_depth = current_depth
    properties = schema.get("properties", {})
    for prop_schema in properties.values():
        if isinstance(prop_schema, dict):
            if prop_schema.get("type") == "object":
                max_depth = max(max_depth, _compute_schema_depth(prop_schema, current_depth + 1))
            elif prop_schema.get("type") == "array":
                items = prop_schema.get("items", {})
                if isinstance(items, dict) and items.get("type") == "object":
                    max_depth = max(max_depth, _compute_schema_depth(items, current_depth + 1))
                else:
                    max_depth = max(max_depth, current_depth + 1)

    return max_depth


def _count_params(schema: dict) -> int:
    """Count total parameters in a schema (including nested)."""
    if not isinstance(schema, dict):
        return 0
    properties = schema.get("properties", {})
    count = len(properties)
    for prop_schema in properties.values():
        if isinstance(prop_schema, dict):
            if prop_schema.get("type") == "object":
                count += _count_params(prop_schema)
            elif prop_schema.get("type") == "array":
                items = prop_schema.get("items", {})
                if isinstance(items, dict) and items.get("type") == "object":
                    count += _count_params(items)
    return count


def _get_optional_params(tool: dict) -> list[str]:
    """Get list of optional parameter names from a tool definition."""
    func = tool.get("function", tool)
    params = func.get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))
    return [name for name in properties if name not in required]


def _get_param_types(schema: dict) -> set[str]:
    """Get set of parameter types in a schema."""
    types = set()
    properties = schema.get("properties", {})
    for prop_schema in properties.values():
        if isinstance(prop_schema, dict):
            t = prop_schema.get("type", "unknown")
            types.add(t)
    return types


def _compute_tool_name_ambiguity(tool_names: list[str]) -> float:
    """
    Compute ambiguity score between tool names (0=distinct, 1=identical).
    Uses longest common prefix ratio and word overlap.
    """
    if len(tool_names) <= 1:
        return 0.0

    # Normalize names
    def normalize(name):
        # Split camelCase, snake_case, kebab-case
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return set(re.split(r'[_\-\s.]+', name.lower()))

    word_sets = [normalize(n) for n in tool_names]

    max_overlap = 0.0
    comparisons = 0
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = word_sets[i] & word_sets[j]
            union = word_sets[i] | word_sets[j]
            if union:
                overlap = len(intersection) / len(union)
                max_overlap = max(max_overlap, overlap)
            comparisons += 1

    return max_overlap


def _has_enum_params(schema: dict) -> bool:
    """Check if schema has enum-constrained parameters."""
    properties = schema.get("properties", {})
    for prop_schema in properties.values():
        if isinstance(prop_schema, dict) and "enum" in prop_schema:
            return True
    return False


def _compute_enum_coverage(tool_funcs: list[dict]) -> float:
    """Compute fraction of string params that lack enum constraints.

    Returns 0.0 (all string params have enums) to 1.0 (none do).
    Higher values = higher risk of inconsistency.
    """
    total_string_params = 0
    unconstrained_string_params = 0

    for func in tool_funcs:
        params_schema = func.get("parameters", {})
        properties = params_schema.get("properties", {})
        for prop_schema in properties.values():
            if isinstance(prop_schema, dict) and prop_schema.get("type") == "string":
                total_string_params += 1
                if "enum" not in prop_schema:
                    unconstrained_string_params += 1

    if total_string_params == 0:
        return 0.0
    return unconstrained_string_params / total_string_params


def _compute_description_quality(tool_funcs: list[dict]) -> float:
    """Compute average description quality score across all params.

    Returns 0.0 (no descriptions) to 1.0 (all params well-described).
    Based on: presence + length of param descriptions.
    """
    total_params = 0
    quality_sum = 0.0

    for func in tool_funcs:
        params_schema = func.get("parameters", {})
        properties = params_schema.get("properties", {})
        for prop_schema in properties.values():
            if not isinstance(prop_schema, dict):
                continue
            total_params += 1
            desc = prop_schema.get("description", "")
            if not desc:
                quality_sum += 0.0
            elif len(desc) < 10:
                quality_sum += 0.3
            elif len(desc) < 30:
                quality_sum += 0.6
            else:
                quality_sum += 1.0

    if total_params == 0:
        return 1.0  # No params = no description risk
    return quality_sum / total_params


# ============================================================================
# Prompt Analysis Helpers
# ============================================================================

VAGUE_TERMS = [
    "appropriate", "relevant", "suitable", "proper", "reasonable",
    "good", "best", "nice", "correct", "right",
    "some", "few", "several", "many", "various",
    "etc", "and so on", "and more", "similar",
    "if needed", "if necessary", "as needed", "if applicable",
    "maybe", "perhaps", "possibly", "probably",
]


def _count_vague_terms(text: str) -> list[str]:
    """Find vague/ambiguous terms in prompt text."""
    text_lower = text.lower()
    found = []
    for term in VAGUE_TERMS:
        if term in text_lower:
            found.append(term)
    return found


def _count_constraints(text: str) -> int:
    """Count explicit constraints in prompt (numbers, specific values, etc.)."""
    patterns = [
        r'\b\d+\b',              # specific numbers
        r'"[^"]+?"',             # quoted strings (specific values)
        r"'[^']+?'",             # single-quoted strings
        r'\bmust\b',             # mandatory constraints
        r'\bexactly\b',
        r'\bonly\b',
        r'\bno more than\b',
        r'\bat least\b',
        r'\bat most\b',
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


# ============================================================================
# Main Assessor
# ============================================================================

class ConsistencyRiskAssessor:
    """
    Assesses prompt + tool definitions for consistency risk factors.

    Analyzes schema complexity, parameter structure, prompt clarity,
    and tool disambiguation to identify factors empirically associated
    with inconsistent LLM outputs.

    Thresholds are derived from analysis of 225K+ samples across 19 LLMs.
    See: Wang et al., "STED and Consistency Scoring" (NeurIPS 2025 Workshop)

    Example:
        assessor = ConsistencyRiskAssessor()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search for products",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "category": {"type": "string"},
                            "min_price": {"type": "number"},
                            "max_price": {"type": "number"},
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        report = assessor.assess("Find me some good laptops", tools)
        report.print_report()
    """

    # Empirical thresholds from factor analysis (225K+ samples, 19 models)
    # Source: results/factor_analysis/correlations.csv, feature_importance.csv
    THRESHOLDS = {
        "total_params": {"medium": 7, "high": 10},
        "schema_complexity": {"medium": 12, "high": 18},
        "schema_depth": {"medium": 3, "high": 5},
        "num_tools": {"medium": 5, "high": 8},
        "tool_name_ambiguity": {"medium": 0.5, "high": 0.7},
        "query_length": {"medium": 400, "high": 600},
        "optional_param_ratio": {"medium": 0.3, "high": 0.5},
        "param_type_diversity": {"medium": 3, "high": 5},
        "enum_gap": {"medium": 0.5, "high": 0.8},
        # description_quality is inverted: lower = worse
        "description_quality": {"medium": 0.6, "high": 0.3},
    }

    # Weights for overall risk score (based on feature importance ranking)
    # Validated against EMNLP intervention experiments (enum p=0.042, desc p=0.016)
    WEIGHTS = {
        "total_params": 0.17,
        "schema_complexity": 0.17,
        "schema_depth": 0.08,
        "num_tools": 0.08,
        "tool_name_ambiguity": 0.08,
        "optional_param_ratio": 0.10,
        "query_length": 0.05,
        "param_type_diversity": 0.04,
        "vague_terms": 0.04,
        "constraint_score": 0.04,
        "enum_gap": 0.08,
        "description_quality": 0.07,
    }

    # Model-specific configurations derived from per-model Random Forest
    # feature importance analysis (225K+ samples, 19 models).
    # Models with rank correlation < 0.7 get their own profile.
    MODEL_CONFIGS = {
        "GPT-4.1-Mini": {
            "weights": {
                "total_params": 0.04,
                "schema_complexity": 0.04,
                "schema_depth": 0.28,
                "num_tools": 0.07,
                "tool_name_ambiguity": 0.06,
                "query_length": 0.14,
                "param_type_diversity": 0.15,
                "vague_terms": 0.03,
                "constraint_score": 0.03,
                "optional_param_ratio": 0.03,
                "enum_gap": 0.07,
                "description_quality": 0.06,
            },
            "thresholds": {
                "total_params": {"medium": 7, "high": 21},
                "schema_complexity": {"medium": 13, "high": 24},
                "schema_depth": {"medium": 3, "high": 5},
                "num_tools": {"medium": 6, "high": 11},
                "tool_name_ambiguity": {"medium": 0.63, "high": 0.94},
                "query_length": {"medium": 710, "high": 868},
                "optional_param_ratio": {"medium": 0.3, "high": 0.5},
                "param_type_diversity": {"medium": 2, "high": 4},
                "enum_gap": {"medium": 0.5, "high": 0.8},
                "description_quality": {"medium": 0.6, "high": 0.3},
            },
        },
    }

    # Map model IDs/names to display names for MODEL_CONFIGS lookup
    _MODEL_ALIASES = {
        # OpenRouter format
        "openai/gpt-4.1-mini": "GPT-4.1-Mini",
        # Bedrock format (unlikely but cover it)
        "openai.gpt-4.1-mini": "GPT-4.1-Mini",
        # Display name variations
        "gpt-4.1-mini": "GPT-4.1-Mini",
        "gpt4.1mini": "GPT-4.1-Mini",
    }

    def __init__(self, model_id: str = None, thresholds: dict = None,
                 weights: dict = None):
        """
        Initialize the assessor.

        Args:
            model_id: Optional model identifier. If provided, uses model-specific
                      thresholds and weights when available. Supports display names
                      (e.g., "GPT-4.1-Mini") and API model IDs
                      (e.g., "openai/gpt-4.1-mini").
            thresholds: Optional custom thresholds dict to override defaults.
                        Format: {"factor_name": {"medium": val, "high": val}}
            weights: Optional custom weights dict to override defaults.
                     Format: {"factor_name": weight_float}
        """
        self.model_id = model_id
        resolved_name = self._resolve_model_name(model_id) if model_id else None
        model_config = self.MODEL_CONFIGS.get(resolved_name, {}) if resolved_name else {}

        if resolved_name and model_id and resolved_name not in self.MODEL_CONFIGS:
            # model_id was given but no specific config found — use universal defaults
            pass

        # Thresholds: explicit > model-specific > universal
        self.thresholds = {**self.THRESHOLDS}
        if model_config.get("thresholds"):
            self.thresholds.update(model_config["thresholds"])
        if thresholds:
            self.thresholds.update(thresholds)

        # Weights: explicit > model-specific > universal
        self.weights = {**self.WEIGHTS}
        if model_config.get("weights"):
            self.weights.update(model_config["weights"])
        if weights:
            self.weights.update(weights)

    @classmethod
    def _resolve_model_name(cls, model_id: str) -> Optional[str]:
        """Resolve a model ID to a display name for MODEL_CONFIGS lookup."""
        if not model_id:
            return None
        # Direct match in MODEL_CONFIGS
        if model_id in cls.MODEL_CONFIGS:
            return model_id
        # Check aliases
        normalized = model_id.lower().strip()
        if normalized in {k.lower(): k for k in cls._MODEL_ALIASES}:
            for alias, display in cls._MODEL_ALIASES.items():
                if alias.lower() == normalized:
                    return display
        # Try MODEL_REGISTRY if available
        try:
            from .model_config import MODEL_REGISTRY
            if model_id in MODEL_REGISTRY:
                display = MODEL_REGISTRY[model_id][1]
                if display in cls.MODEL_CONFIGS:
                    return display
        except ImportError:
            pass
        # Case-insensitive match against MODEL_CONFIGS keys
        for key in cls.MODEL_CONFIGS:
            if key.lower() == normalized:
                return key
        return model_id  # Return as-is (will not match any config)

    def assess(self, prompt: str, tools: list[dict]) -> RiskReport:
        """
        Assess consistency risk for a prompt + tool definitions.

        Args:
            prompt: The user prompt / query text.
            tools: List of tool definitions in OpenAI function-calling format.
                   Each tool should have "type": "function" and a "function" key
                   with "name", "description", and "parameters".

        Returns:
            RiskReport with overall risk score, identified factors, and mitigations.
        """
        metrics = self._compute_metrics(prompt, tools)
        risk_factors = self._identify_risk_factors(prompt, tools, metrics)

        # Compute overall risk score (weighted sum of normalized factor severities)
        overall_risk = self._compute_overall_risk(metrics)
        risk_level = "high" if overall_risk >= 0.6 else "medium" if overall_risk >= 0.3 else "low"

        # Sort factors by severity (high first)
        risk_factors.sort(key=lambda f: f.severity, reverse=True)

        return RiskReport(
            overall_risk=round(overall_risk, 3),
            risk_level=risk_level,
            risk_factors=risk_factors,
            metrics=metrics,
        )

    def assess_tools(self, tools: list[dict]) -> RiskReport:
        """Assess only tool definitions (no prompt analysis)."""
        return self.assess("", tools)

    def assess_prompt(self, prompt: str) -> RiskReport:
        """Assess only the prompt text (no tool analysis)."""
        return self.assess(prompt, [])

    # ========================================================================
    # Metric Computation
    # ========================================================================

    def _compute_metrics(self, prompt: str, tools: list[dict]) -> dict:
        """Compute all assessment metrics from prompt and tools."""
        metrics = {}

        # --- Tool / Schema Metrics ---
        tool_funcs = []
        for tool in tools:
            if isinstance(tool, dict):
                func = tool.get("function", tool)
                tool_funcs.append(func)

        num_tools = len(tool_funcs)
        metrics["num_tools"] = num_tools

        tool_names = [f.get("name", "") for f in tool_funcs]
        metrics["tool_names"] = tool_names

        # Per-tool parameter analysis
        total_params = 0
        max_params = 0
        max_depth = 0
        all_types = set()
        total_optional = 0
        total_required = 0
        has_array_params = False
        has_object_params = False
        tools_with_enums = 0

        for func in tool_funcs:
            params_schema = func.get("parameters", {})
            properties = params_schema.get("properties", {})
            required = set(params_schema.get("required", []))

            n_params = _count_params(params_schema)
            total_params += n_params
            max_params = max(max_params, len(properties))

            depth = _compute_schema_depth(params_schema)
            max_depth = max(max_depth, depth)

            types = _get_param_types(params_schema)
            all_types.update(types)

            total_optional += len(properties) - len(required)
            total_required += len(required)

            for prop_schema in properties.values():
                if isinstance(prop_schema, dict):
                    if prop_schema.get("type") == "array":
                        has_array_params = True
                    if prop_schema.get("type") == "object":
                        has_object_params = True

            if _has_enum_params(params_schema):
                tools_with_enums += 1

        metrics["total_params"] = total_params
        metrics["max_params_per_tool"] = max_params
        metrics["avg_params_per_tool"] = round(total_params / max(num_tools, 1), 1)
        metrics["schema_depth"] = max_depth
        metrics["param_type_diversity"] = len(all_types)
        metrics["param_types"] = sorted(all_types)
        metrics["has_array_params"] = has_array_params
        metrics["has_object_params"] = has_object_params
        metrics["total_optional_params"] = total_optional
        metrics["total_required_params"] = total_required
        metrics["optional_param_ratio"] = round(
            total_optional / max(total_optional + total_required, 1), 2
        )
        metrics["tools_with_enums"] = tools_with_enums

        # Enum coverage: fraction of string params without enums (0=good, 1=bad)
        metrics["enum_gap"] = round(_compute_enum_coverage(tool_funcs), 3)

        # Description quality: average quality across params (0=bad, 1=good)
        metrics["description_quality"] = round(
            _compute_description_quality(tool_funcs), 3
        )

        # Schema complexity (composite score matching factor analysis)
        metrics["schema_complexity"] = round(
            max_depth * (total_params ** 0.5) * (1 + 0.2 * len(all_types)), 2
        )

        # Tool name ambiguity
        metrics["tool_name_ambiguity"] = round(
            _compute_tool_name_ambiguity(tool_names), 3
        )

        # --- Prompt Metrics ---
        metrics["query_length"] = len(prompt)
        metrics["word_count"] = len(prompt.split())

        vague = _count_vague_terms(prompt)
        metrics["vague_terms"] = vague
        metrics["vague_term_count"] = len(vague)

        metrics["constraint_count"] = _count_constraints(prompt)

        # Constraint density: constraints per word (higher = more specific)
        word_count = max(metrics["word_count"], 1)
        metrics["constraint_density"] = round(metrics["constraint_count"] / word_count, 3)

        return metrics

    # ========================================================================
    # Risk Factor Identification
    # ========================================================================

    def _identify_risk_factors(self, prompt: str, tools: list[dict],
                                metrics: dict) -> list[RiskFactor]:
        """Identify all risk factors from computed metrics."""
        factors = []

        # --- HIGH RISK: Total parameters ---
        total_params = metrics["total_params"]
        t = self.thresholds["total_params"]
        if total_params >= t["high"]:
            factors.append(RiskFactor(
                name="high_total_params",
                severity=Severity.HIGH,
                value=total_params,
                threshold=t["high"],
                description=(
                    f"{total_params} total parameters across {metrics['num_tools']} tools. "
                    f"Empirical data shows 71% consistency drop from 0-2 to 10+ params (r=-0.124)."
                ),
                mitigation=Mitigation(
                    description="Split tools with many parameters into focused sub-tools with fewer params each.",
                    example_before='create_event(title, date, time, location, description, attendees, reminder, recurrence, color, notes)',
                    example_after='create_event(title, date, time) + set_event_details(event_id, location, attendees) + set_event_options(event_id, reminder, recurrence)',
                ),
            ))
        elif total_params >= t["medium"]:
            factors.append(RiskFactor(
                name="moderate_total_params",
                severity=Severity.MEDIUM,
                value=total_params,
                threshold=t["medium"],
                description=f"{total_params} total parameters. Consider reducing if consistency is critical.",
                mitigation=Mitigation(
                    description="Remove rarely-used parameters or provide strong defaults.",
                ),
            ))

        # --- HIGH RISK: Schema complexity ---
        schema_complexity = metrics["schema_complexity"]
        t = self.thresholds["schema_complexity"]
        if schema_complexity >= t["high"]:
            factors.append(RiskFactor(
                name="high_schema_complexity",
                severity=Severity.HIGH,
                value=schema_complexity,
                threshold=t["high"],
                description=(
                    f"Schema complexity score {schema_complexity:.1f} "
                    f"(depth={metrics['schema_depth']}, params={metrics['total_params']}, "
                    f"types={metrics['param_type_diversity']}). "
                    f"Top correlated factor with inconsistency (r=-0.152)."
                ),
                mitigation=Mitigation(
                    description="Flatten nested schemas and reduce parameter count. Use string params with clear format descriptions instead of deeply nested objects.",
                    example_before='{"address": {"type": "object", "properties": {"street": {"type": "object", "properties": {"number": ..., "name": ...}}}}}',
                    example_after='{"street_number": {"type": "string"}, "street_name": {"type": "string"}, "city": {"type": "string"}}',
                ),
            ))
        elif schema_complexity >= t["medium"]:
            factors.append(RiskFactor(
                name="moderate_schema_complexity",
                severity=Severity.MEDIUM,
                value=schema_complexity,
                threshold=t["medium"],
                description=f"Schema complexity score {schema_complexity:.1f}. Moderate risk of inconsistency.",
                mitigation=Mitigation(
                    description="Consider flattening nested structures where possible.",
                ),
            ))

        # --- MEDIUM RISK: Schema depth ---
        schema_depth = metrics["schema_depth"]
        t = self.thresholds["schema_depth"]
        if schema_depth >= t["high"]:
            factors.append(RiskFactor(
                name="deep_nesting",
                severity=Severity.HIGH,
                value=schema_depth,
                threshold=t["high"],
                description=f"Schema nesting depth of {schema_depth}. Deep nesting increases structural variation (r=-0.138).",
                mitigation=Mitigation(
                    description="Flatten nested objects to max depth 2-3. Use dot-notation or underscore-separated flat params.",
                    example_before='user.address.street.number',
                    example_after='user_street_number',
                ),
            ))
        elif schema_depth >= t["medium"]:
            factors.append(RiskFactor(
                name="moderate_nesting",
                severity=Severity.MEDIUM,
                value=schema_depth,
                threshold=t["medium"],
                description=f"Schema nesting depth of {schema_depth}.",
                mitigation=Mitigation(
                    description="Consider flattening objects beyond depth 3.",
                ),
            ))

        # --- MEDIUM RISK: Optional parameters ---
        opt_ratio = metrics["optional_param_ratio"]
        t = self.thresholds["optional_param_ratio"]
        if opt_ratio >= t["high"] and metrics["total_optional_params"] > 2:
            factors.append(RiskFactor(
                name="many_optional_params",
                severity=Severity.HIGH,
                value=opt_ratio,
                threshold=t["high"],
                description=(
                    f"{metrics['total_optional_params']} optional params out of "
                    f"{metrics['total_optional_params'] + metrics['total_required_params']} total "
                    f"({opt_ratio:.0%}). Models randomly include/exclude optional params across runs."
                ),
                mitigation=Mitigation(
                    description="Make important params required. Remove rarely-needed optional params or move them to a separate tool.",
                    example_before='search(query, limit?, offset?, sort?, filter?, fields?)',
                    example_after='search(query, limit=10, sort="relevance") — keep only params that affect results',
                ),
            ))
        elif opt_ratio >= t["medium"] and metrics["total_optional_params"] > 1:
            factors.append(RiskFactor(
                name="some_optional_params",
                severity=Severity.MEDIUM,
                value=opt_ratio,
                threshold=t["medium"],
                description=f"{metrics['total_optional_params']} optional parameters ({opt_ratio:.0%}).",
                mitigation=Mitigation(
                    description="Mark the most important parameters as required and provide defaults in descriptions.",
                ),
            ))

        # --- MEDIUM RISK: Number of tools ---
        num_tools = metrics["num_tools"]
        t = self.thresholds["num_tools"]
        if num_tools >= t["high"]:
            factors.append(RiskFactor(
                name="many_tools",
                severity=Severity.HIGH,
                value=num_tools,
                threshold=t["high"],
                description=f"{num_tools} tools available. Tool selection ambiguity increases with more options (r=-0.113).",
                mitigation=Mitigation(
                    description="Group related tools or reduce the set to only relevant tools per context. Add 'Use ONLY when...' to each tool description.",
                    example_before='8 tools: get_user, find_user, search_user, list_users, ...',
                    example_after='3 tools: get_user_by_id (exact lookup), search_users (query-based), list_all_users (no filter)',
                ),
            ))
        elif num_tools >= t["medium"]:
            factors.append(RiskFactor(
                name="moderate_tool_count",
                severity=Severity.MEDIUM,
                value=num_tools,
                threshold=t["medium"],
                description=f"{num_tools} tools. Consider if all are needed for this prompt.",
                mitigation=Mitigation(
                    description="Only expose tools relevant to the current task context.",
                ),
            ))

        # --- MEDIUM RISK: Tool name ambiguity ---
        ambiguity = metrics["tool_name_ambiguity"]
        t = self.thresholds["tool_name_ambiguity"]
        if ambiguity >= t["high"]:
            factors.append(RiskFactor(
                name="ambiguous_tool_names",
                severity=Severity.HIGH,
                value=ambiguity,
                threshold=t["high"],
                description=(
                    f"Tool name ambiguity score {ambiguity:.2f}. "
                    f"Similar names cause tool selection confusion. "
                    f"Names: {metrics['tool_names']}"
                ),
                mitigation=Mitigation(
                    description="Use distinct, specific tool names with clear differentiating prefixes or verbs.",
                    example_before='get_data, fetch_data, retrieve_data',
                    example_after='get_user_profile, fetch_sales_report, retrieve_inventory_count',
                ),
            ))
        elif ambiguity >= t["medium"]:
            factors.append(RiskFactor(
                name="moderate_name_ambiguity",
                severity=Severity.MEDIUM,
                value=ambiguity,
                threshold=t["medium"],
                description=f"Tool name ambiguity score {ambiguity:.2f}. Some name overlap detected.",
                mitigation=Mitigation(
                    description="Add distinguishing prefixes or more specific verb+noun naming.",
                ),
            ))

        # --- LOW RISK: Long prompt ---
        query_length = metrics["query_length"]
        t = self.thresholds["query_length"]
        if query_length >= t["high"]:
            factors.append(RiskFactor(
                name="long_prompt",
                severity=Severity.MEDIUM,
                value=query_length,
                threshold=t["high"],
                description=f"Prompt length {query_length} chars. Longer prompts create more interpretation paths (r=-0.235).",
                mitigation=Mitigation(
                    description="Extract structured constraints into tool parameter descriptions. Keep prompt focused on intent.",
                ),
            ))
        elif query_length >= t["medium"]:
            factors.append(RiskFactor(
                name="moderate_prompt_length",
                severity=Severity.LOW,
                value=query_length,
                threshold=t["medium"],
                description=f"Prompt length {query_length} chars.",
                mitigation=Mitigation(
                    description="Consider moving detailed constraints into tool parameter descriptions.",
                ),
            ))

        # --- LOW RISK: Vague terms ---
        vague_count = metrics["vague_term_count"]
        if vague_count >= 3:
            factors.append(RiskFactor(
                name="vague_language",
                severity=Severity.MEDIUM,
                value=vague_count,
                threshold=3,
                description=f"{vague_count} vague terms found: {metrics['vague_terms'][:5]}. Ambiguous language invites varied interpretations.",
                mitigation=Mitigation(
                    description="Replace vague terms with specific, measurable criteria.",
                    example_before='"Find some good restaurants nearby"',
                    example_after='"Find restaurants within 2km with rating >= 4.0"',
                ),
            ))
        elif vague_count >= 1:
            factors.append(RiskFactor(
                name="minor_vagueness",
                severity=Severity.LOW,
                value=vague_count,
                threshold=1,
                description=f"{vague_count} vague term(s): {metrics['vague_terms']}.",
                mitigation=Mitigation(
                    description="Consider replacing vague terms with specific values where possible.",
                ),
            ))

        # --- LOW RISK: Parameter type diversity ---
        type_div = metrics["param_type_diversity"]
        t = self.thresholds["param_type_diversity"]
        if type_div >= t["high"]:
            factors.append(RiskFactor(
                name="high_type_diversity",
                severity=Severity.MEDIUM,
                value=type_div,
                threshold=t["high"],
                description=f"{type_div} different parameter types: {metrics['param_types']}. Mixed types increase schema complexity.",
                mitigation=Mitigation(
                    description="Standardize parameter types where possible. Use string with format constraints instead of mixed types.",
                ),
            ))
        elif type_div >= t["medium"]:
            factors.append(RiskFactor(
                name="moderate_type_diversity",
                severity=Severity.LOW,
                value=type_div,
                threshold=t["medium"],
                description=f"{type_div} parameter types used.",
                mitigation=Mitigation(
                    description="Consider if all type variations are necessary.",
                ),
            ))

        # --- POSITIVE: Enum constraints reduce risk ---
        if metrics["tools_with_enums"] > 0 and not any(
            f.name.startswith("high_") or f.name.startswith("many_") for f in factors
        ):
            # Don't add positive notes if there are high-severity issues
            pass  # Enums are good but we only flag risks, not positives

        # --- LOW RISK: No constraints in prompt ---
        if metrics["constraint_count"] == 0 and metrics["word_count"] > 20:
            factors.append(RiskFactor(
                name="no_explicit_constraints",
                severity=Severity.LOW,
                value=0,
                threshold=1,
                description="No explicit constraints (numbers, quoted values, 'must'/'exactly') found in prompt.",
                mitigation=Mitigation(
                    description="Add specific constraints to reduce ambiguity in expected output.",
                    example_before='"Get me some flight options"',
                    example_after='"Find flights from SFO to JFK on 2024-03-15, max 2 stops, under $500"',
                ),
            ))

        # --- MEDIUM RISK: String params without enum constraints ---
        enum_gap = metrics.get("enum_gap", 0)
        t = self.thresholds.get("enum_gap", {"medium": 0.5, "high": 0.8})
        total_string = 0
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function", tool)
            if not isinstance(func, dict):
                continue
            for p in func.get("parameters", {}).get("properties", {}).values():
                if isinstance(p, dict) and p.get("type") == "string":
                    total_string += 1
        if enum_gap >= t["high"] and total_string >= 2:
            factors.append(RiskFactor(
                name="low_enum_coverage",
                severity=Severity.MEDIUM,
                value=enum_gap,
                threshold=t["high"],
                description=(
                    f"{enum_gap:.0%} of string parameters lack enum constraints. "
                    f"Free-text strings allow varied formatting across runs."
                ),
                mitigation=Mitigation(
                    description="Add enum values to string parameters with a known set of valid values.",
                    example_before='{"status": {"type": "string"}}',
                    example_after='{"status": {"type": "string", "enum": ["active", "inactive", "pending"]}}',
                ),
            ))
        elif enum_gap >= t["medium"] and total_string >= 2:
            factors.append(RiskFactor(
                name="moderate_enum_coverage",
                severity=Severity.LOW,
                value=enum_gap,
                threshold=t["medium"],
                description=f"{enum_gap:.0%} of string params lack enums.",
                mitigation=Mitigation(
                    description="Consider adding enum constraints to string parameters with finite valid values.",
                ),
            ))

        # --- MEDIUM RISK: Poor parameter descriptions ---
        desc_quality = metrics.get("description_quality", 1.0)
        t = self.thresholds.get("description_quality", {"medium": 0.6, "high": 0.3})
        # Note: description_quality is inverted — lower = worse
        if metrics["total_params"] >= 2:
            if desc_quality <= t["high"]:
                factors.append(RiskFactor(
                    name="poor_descriptions",
                    severity=Severity.MEDIUM,
                    value=desc_quality,
                    threshold=t["high"],
                    description=(
                        f"Parameter description quality score {desc_quality:.2f}/1.00. "
                        f"Missing or short descriptions force the model to guess parameter semantics."
                    ),
                    mitigation=Mitigation(
                        description="Add detailed descriptions to each parameter explaining expected format and meaning.",
                        example_before='{"date": {"type": "string"}}',
                        example_after='{"date": {"type": "string", "description": "Date in ISO 8601 format (YYYY-MM-DD)"}}',
                    ),
                ))
            elif desc_quality <= t["medium"]:
                factors.append(RiskFactor(
                    name="sparse_descriptions",
                    severity=Severity.LOW,
                    value=desc_quality,
                    threshold=t["medium"],
                    description=f"Description quality {desc_quality:.2f}/1.00. Some params have short or missing descriptions.",
                    mitigation=Mitigation(
                        description="Expand parameter descriptions to clarify expected format and valid values.",
                    ),
                ))

        return factors

    # ========================================================================
    # Overall Risk Computation
    # ========================================================================

    def _compute_overall_risk(self, metrics: dict) -> float:
        """Compute weighted overall risk score from metrics."""
        risk = 0.0

        def normalize(value, medium_thresh, high_thresh):
            """Map value to 0-1 risk based on thresholds."""
            if value <= 0:
                return 0.0
            elif value < medium_thresh:
                return 0.1 * (value / medium_thresh)
            elif value < high_thresh:
                return 0.1 + 0.4 * ((value - medium_thresh) / (high_thresh - medium_thresh))
            else:
                return min(0.5 + 0.5 * ((value - high_thresh) / max(high_thresh, 1)), 1.0)

        for factor, weight in self.weights.items():
            if factor == "description_quality":
                # Inverted: lower quality = higher risk. Only matters with 2+ params.
                if metrics.get("total_params", 0) < 2:
                    continue
                quality = metrics.get("description_quality", 1.0)
                t = self.thresholds.get("description_quality", {"medium": 0.6, "high": 0.3})
                if quality >= t["medium"]:
                    risk += weight * 0.1 * (1.0 - quality) / max(1.0 - t["medium"], 0.01)
                elif quality >= t["high"]:
                    risk += weight * (0.1 + 0.4 * (t["medium"] - quality) / max(t["medium"] - t["high"], 0.01))
                else:
                    risk += weight * min(0.5 + 0.5 * (t["high"] - quality) / max(t["high"], 0.01), 1.0)
            elif factor == "enum_gap":
                # Only matters with 2+ string params
                if metrics.get("total_params", 0) < 2:
                    continue
                if factor in self.thresholds and factor in metrics:
                    t = self.thresholds[factor]
                    risk += weight * normalize(metrics[factor], t["medium"], t["high"])
            elif factor in self.thresholds and factor in metrics:
                t = self.thresholds[factor]
                risk += weight * normalize(metrics[factor], t["medium"], t["high"])
            elif factor == "vague_terms":
                risk += weight * min(metrics.get("vague_term_count", 0) / 5.0, 1.0)
            elif factor == "constraint_score":
                # Inverse: fewer constraints = higher risk
                density = metrics.get("constraint_density", 0)
                risk += weight * max(1.0 - density * 10, 0.0)

        return min(risk, 1.0)
