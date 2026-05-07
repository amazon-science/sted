"""Tests for ConsistencyRiskAssessor."""

import json
import pytest
from sted.consistency_risk_assessor import (
    ConsistencyRiskAssessor,
    RiskReport,
    RiskFactor,
    Severity,
    _compute_schema_depth,
    _count_params,
    _get_optional_params,
    _compute_tool_name_ambiguity,
    _count_vague_terms,
    _count_constraints,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def assessor():
    return ConsistencyRiskAssessor()


def _make_tool(name, properties, required=None, description=""):
    """Helper to build a tool definition."""
    params = {"type": "object", "properties": properties}
    if required:
        params["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        }
    }


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestSchemaDepth:
    def test_flat_schema(self):
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "number"}}}
        assert _compute_schema_depth(schema) == 0

    def test_one_level_nested(self):
        schema = {
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    }
                }
            }
        }
        assert _compute_schema_depth(schema) == 1

    def test_deeply_nested(self):
        schema = {
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        assert _compute_schema_depth(schema) == 2

    def test_array_with_nested_objects(self):
        schema = {
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        }
                    }
                }
            }
        }
        assert _compute_schema_depth(schema) == 1

    def test_empty_schema(self):
        assert _compute_schema_depth({}) == 0
        assert _compute_schema_depth({"properties": {}}) == 0

    def test_non_dict_input(self):
        assert _compute_schema_depth("not a dict") == 0


class TestCountParams:
    def test_flat_params(self):
        schema = {"properties": {"a": {}, "b": {}, "c": {}}}
        assert _count_params(schema) == 3

    def test_nested_params(self):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    }
                }
            }
        }
        assert _count_params(schema) == 4  # name + address + city + zip

    def test_empty(self):
        assert _count_params({}) == 0
        assert _count_params({"properties": {}}) == 0


class TestGetOptionalParams:
    def test_all_required(self):
        tool = {
            "function": {
                "parameters": {
                    "properties": {"a": {}, "b": {}},
                    "required": ["a", "b"],
                }
            }
        }
        assert _get_optional_params(tool) == []

    def test_some_optional(self):
        tool = {
            "function": {
                "parameters": {
                    "properties": {"a": {}, "b": {}, "c": {}},
                    "required": ["a"],
                }
            }
        }
        optional = _get_optional_params(tool)
        assert set(optional) == {"b", "c"}

    def test_no_required_field(self):
        tool = {
            "function": {
                "parameters": {
                    "properties": {"a": {}, "b": {}},
                }
            }
        }
        optional = _get_optional_params(tool)
        assert set(optional) == {"a", "b"}


class TestToolNameAmbiguity:
    def test_identical_names(self):
        score = _compute_tool_name_ambiguity(["get_data", "get_data"])
        assert score == 1.0

    def test_distinct_names(self):
        score = _compute_tool_name_ambiguity(["get_weather", "send_email"])
        assert score < 0.5

    def test_similar_names(self):
        score = _compute_tool_name_ambiguity(["get_data", "fetch_data", "retrieve_data"])
        assert score > 0.3  # "data" overlaps

    def test_single_tool(self):
        assert _compute_tool_name_ambiguity(["only_one"]) == 0.0

    def test_empty(self):
        assert _compute_tool_name_ambiguity([]) == 0.0


class TestVagueTerms:
    def test_finds_vague_terms(self):
        found = _count_vague_terms("Find some good items that are suitable")
        assert "some" in found
        assert "good" in found
        assert "suitable" in found

    def test_no_vague_terms(self):
        found = _count_vague_terms("Get user with id 123")
        assert len(found) == 0

    def test_case_insensitive(self):
        found = _count_vague_terms("Find APPROPRIATE items")
        assert "appropriate" in found


class TestCountConstraints:
    def test_numbers(self):
        assert _count_constraints("Find 5 items under 100 dollars") >= 2

    def test_quoted_values(self):
        assert _count_constraints('Set category to "electronics"') >= 1

    def test_must_keyword(self):
        assert _count_constraints("Results must include price") >= 1

    def test_no_constraints(self):
        assert _count_constraints("Find things") == 0


# ============================================================================
# Assessor Integration Tests
# ============================================================================

class TestAssessLowRisk:
    def test_simple_tool_specific_prompt(self, assessor):
        tools = [_make_tool(
            "get_weather",
            {"city": {"type": "string"}, "date": {"type": "string"}},
            required=["city", "date"],
        )]
        report = assessor.assess("What is the weather in Tokyo on 2024-01-15?", tools)

        assert report.overall_risk < 0.2
        assert report.risk_level == "low"
        assert len(report.high_risk_factors) == 0

    def test_single_required_param(self, assessor):
        tools = [_make_tool(
            "get_user",
            {"user_id": {"type": "string"}},
            required=["user_id"],
        )]
        report = assessor.assess("Get user 12345", tools)
        assert report.overall_risk < 0.1

    def test_empty_tools(self, assessor):
        report = assessor.assess("Hello world", [])
        assert report.risk_level == "low"


class TestAssessHighRisk:
    def test_many_params(self, assessor):
        props = {f"param_{i}": {"type": "string"} for i in range(12)}
        tools = [_make_tool("big_tool", props, required=["param_0"])]
        report = assessor.assess("Do something", tools)

        assert any(f.name == "high_total_params" for f in report.risk_factors)
        assert any(f.name == "many_optional_params" for f in report.risk_factors)

    def test_many_tools_with_similar_names(self, assessor):
        tools = [
            _make_tool(f"search_{target}",
                      {"query": {"type": "string"}},
                      required=["query"])
            for target in ["users", "products", "orders", "reviews",
                          "inventory", "reports", "analytics", "logs", "events"]
        ]
        report = assessor.assess("Search for recent activity", tools)

        assert any(f.name == "many_tools" for f in report.risk_factors)

    def test_vague_prompt(self, assessor):
        tools = [_make_tool(
            "search",
            {"query": {"type": "string"}},
            required=["query"],
        )]
        report = assessor.assess(
            "Find me some good and suitable items, maybe relevant ones, perhaps several nice options etc",
            tools,
        )

        assert any(f.name == "vague_language" for f in report.risk_factors)

    def test_deep_nesting(self, assessor):
        tools = [{
            "type": "function",
            "function": {
                "name": "create_order",
                "description": "Create order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer": {
                            "type": "object",
                            "properties": {
                                "address": {
                                    "type": "object",
                                    "properties": {
                                        "street": {
                                            "type": "object",
                                            "properties": {
                                                "number": {"type": "string"},
                                                "name": {"type": "string"},
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "required": ["customer"]
                }
            }
        }]
        report = assessor.assess("Create an order", tools)
        depth_factors = [f for f in report.risk_factors if "nesting" in f.name]
        assert len(depth_factors) > 0


class TestAssessMediumRisk:
    def test_moderate_params_and_optional(self, assessor):
        props = {f"field_{i}": {"type": "string"} for i in range(8)}
        tools = [_make_tool("update_record", props, required=["field_0", "field_1"])]
        report = assessor.assess("Update the record", tools)

        # 8 params with 6 optional should flag both param count and optional ratio
        assert any("param" in f.name for f in report.risk_factors)
        assert any("optional" in f.name for f in report.risk_factors)

    def test_name_ambiguity(self, assessor):
        tools = [
            _make_tool("get_user_data", {"id": {"type": "string"}}, required=["id"]),
            _make_tool("fetch_user_data", {"id": {"type": "string"}}, required=["id"]),
        ]
        report = assessor.assess("Get user info", tools)

        ambiguity_factors = [f for f in report.risk_factors if "ambig" in f.name]
        assert len(ambiguity_factors) > 0


class TestAssessToolsOnly:
    def test_assess_tools_no_prompt(self, assessor):
        tools = [_make_tool(
            "search",
            {f"p{i}": {"type": "string"} for i in range(15)},
            required=["p0"],
        )]
        report = assessor.assess_tools(tools)

        assert report.overall_risk > 0
        assert any(f.name == "high_total_params" for f in report.risk_factors)
        # Should not have prompt-related factors
        assert report.metrics["query_length"] == 0


class TestAssessPromptOnly:
    def test_assess_prompt_no_tools(self, assessor):
        report = assessor.assess_prompt(
            "Find some appropriate relevant items, maybe good suitable ones"
        )
        assert any(f.name == "vague_language" for f in report.risk_factors)
        assert report.metrics["num_tools"] == 0


# ============================================================================
# Risk Factor Properties
# ============================================================================

class TestRiskFactorContent:
    def test_all_factors_have_mitigations(self, assessor):
        """Every identified risk factor must have a mitigation."""
        props = {f"p{i}": {"type": "string"} for i in range(15)}
        tools = [
            _make_tool("search_data", props, required=["p0"]),
            _make_tool("fetch_data", {"q": {"type": "string"}}, required=["q"]),
            _make_tool("get_data", {"q": {"type": "string"}}, required=["q"]),
            _make_tool("retrieve_data", {"q": {"type": "string"}}, required=["q"]),
            _make_tool("find_data", {"q": {"type": "string"}}, required=["q"]),
            _make_tool("query_data", {"q": {"type": "string"}}, required=["q"]),
        ]
        report = assessor.assess(
            "Find some good suitable appropriate relevant nice items, maybe several various etc",
            tools,
        )

        for factor in report.risk_factors:
            assert factor.mitigation is not None
            assert factor.mitigation.description
            assert len(factor.mitigation.description) > 10

    def test_severity_ordering(self, assessor):
        """Risk factors should be sorted high → medium → low."""
        props = {f"p{i}": {"type": "string"} for i in range(12)}
        tools = [_make_tool("tool", props, required=["p0"])]
        report = assessor.assess("Find some good stuff maybe", tools)

        severities = [f.severity for f in report.risk_factors]
        severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        ordered = sorted(severities, key=lambda s: severity_order[s])
        assert severities == ordered


# ============================================================================
# Report Serialization
# ============================================================================

class TestReportSerialization:
    def test_to_dict(self, assessor):
        tools = [_make_tool("test", {"a": {"type": "string"}}, required=["a"])]
        report = assessor.assess("Test prompt", tools)

        d = report.to_dict()
        assert "overall_risk" in d
        assert "risk_level" in d
        assert "metrics" in d
        assert "risk_factors" in d
        assert isinstance(d["risk_factors"], list)

    def test_to_json(self, assessor):
        tools = [_make_tool("test", {"a": {"type": "string"}}, required=["a"])]
        report = assessor.assess("Test prompt", tools)

        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["overall_risk"] == report.overall_risk

    def test_to_dict_with_risk_factors(self, assessor):
        props = {f"p{i}": {"type": "string"} for i in range(12)}
        tools = [_make_tool("tool", props, required=["p0"])]
        report = assessor.assess("Do something", tools)

        d = report.to_dict()
        for rf in d["risk_factors"]:
            assert "name" in rf
            assert "severity" in rf
            assert "mitigation" in rf
            assert rf["severity"] in ("low", "medium", "high")


# ============================================================================
# Custom Thresholds
# ============================================================================

class TestCustomThresholds:
    def test_stricter_thresholds(self):
        strict = ConsistencyRiskAssessor(thresholds={
            "total_params": {"medium": 3, "high": 5},
        })
        tools = [_make_tool(
            "tool",
            {f"p{i}": {"type": "string"} for i in range(6)},
            required=["p0"],
        )]
        report = strict.assess("Do something", tools)

        # With strict thresholds, 6 params should be HIGH
        assert any(f.name == "high_total_params" for f in report.risk_factors)

    def test_lenient_thresholds(self):
        lenient = ConsistencyRiskAssessor(thresholds={
            "total_params": {"medium": 20, "high": 50},
        })
        tools = [_make_tool(
            "tool",
            {f"p{i}": {"type": "string"} for i in range(12)},
            required=["p0"],
        )]
        report = lenient.assess("Do something", tools)

        # With lenient thresholds, 12 params should NOT trigger high
        assert not any(f.name == "high_total_params" for f in report.risk_factors)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_everything(self, assessor):
        report = assessor.assess("", [])
        assert report.risk_level == "low"
        assert isinstance(report.risk_factors, list)

    def test_tool_without_function_wrapper(self, assessor):
        """Tool defined without the 'function' wrapper (bare format)."""
        tool = {
            "name": "simple_tool",
            "description": "A tool",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
        }
        report = assessor.assess("Test", [tool])
        assert report.metrics["num_tools"] == 1
        assert report.metrics["total_params"] == 1

    def test_tool_with_no_parameters(self, assessor):
        tools = [_make_tool("ping", {}, required=[])]
        report = assessor.assess("Ping the server", tools)
        assert report.metrics["total_params"] == 0
        assert report.risk_level == "low"

    def test_very_long_prompt(self, assessor):
        prompt = "Find items. " * 200  # ~2400 chars
        tools = [_make_tool("search", {"q": {"type": "string"}}, required=["q"])]
        report = assessor.assess(prompt, tools)
        assert any("prompt" in f.name.lower() or "length" in f.name.lower()
                   for f in report.risk_factors)

    def test_unicode_prompt(self, assessor):
        tools = [_make_tool("search", {"q": {"type": "string"}}, required=["q"])]
        report = assessor.assess("搜索一些好的产品 🔍", tools)
        assert isinstance(report.overall_risk, float)

    def test_risk_score_bounded(self, assessor):
        """Overall risk should always be between 0 and 1."""
        # Worst case scenario
        props = {f"p{i}": {"type": t} for i, t in
                 enumerate(["string"] * 10 + ["number"] * 5 + ["boolean"] * 3 +
                           ["array"] * 2 + ["object"] * 2)}
        tools = [
            _make_tool(f"search_{x}", props, required=["p0"])
            for x in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        ]
        prompt = "Find some good suitable appropriate relevant nice " * 20

        report = assessor.assess(prompt, tools)
        assert 0.0 <= report.overall_risk <= 1.0


class TestFilterProperties:
    def test_high_risk_factors_filter(self, assessor):
        props = {f"p{i}": {"type": "string"} for i in range(15)}
        tools = [_make_tool("tool", props, required=["p0"])]
        report = assessor.assess("Do something", tools)

        for f in report.high_risk_factors:
            assert f.severity == Severity.HIGH

    def test_medium_risk_factors_filter(self, assessor):
        props = {f"p{i}": {"type": "string"} for i in range(8)}
        tools = [_make_tool("tool", props, required=["p0", "p1", "p2", "p3"])]
        report = assessor.assess("Find some good items", tools)

        for f in report.medium_risk_factors:
            assert f.severity == Severity.MEDIUM

    def test_low_risk_factors_filter(self, assessor):
        tools = [_make_tool("search", {"q": {"type": "string"}}, required=["q"])]
        report = assessor.assess("Find some items", tools)

        for f in report.low_risk_factors:
            assert f.severity == Severity.LOW


# ====================================================================
# Model-Specific Thresholds Tests
# ====================================================================

class TestModelSpecificWeights:
    """Test that model_id selects appropriate weights and thresholds."""

    def test_gpt41mini_uses_specific_config(self):
        """GPT-4.1-Mini should use its own weights (schema_depth dominant)."""
        assessor = ConsistencyRiskAssessor(model_id="GPT-4.1-Mini")
        assert assessor.weights["schema_depth"] > 0.25  # 0.28 in config
        assert assessor.weights["param_type_diversity"] > 0.10  # 0.15

    def test_gpt41mini_different_from_default(self):
        """GPT-4.1-Mini risk scores should differ from default for same input."""
        default_assessor = ConsistencyRiskAssessor()
        gpt_assessor = ConsistencyRiskAssessor(model_id="GPT-4.1-Mini")

        # Deep nested schema — GPT-4.1-Mini weights schema_depth much higher
        nested = {
            "a": {
                "type": "object",
                "properties": {
                    "b": {
                        "type": "object",
                        "properties": {
                            "c": {"type": "string"},
                            "d": {"type": "number"},
                        },
                    }
                },
            }
        }
        tools = [_make_tool("nested_tool", nested, required=["a"])]
        prompt = "Process data"

        default_report = default_assessor.assess(prompt, tools)
        gpt_report = gpt_assessor.assess(prompt, tools)

        # Both should produce valid reports
        assert isinstance(default_report.overall_risk, float)
        assert isinstance(gpt_report.overall_risk, float)
        # Scores should differ (GPT-4.1-Mini weighs schema_depth much higher)
        assert default_report.overall_risk != gpt_report.overall_risk

    def test_model_id_case_insensitive(self):
        """Model ID lookup should be case-insensitive."""
        assessor = ConsistencyRiskAssessor(model_id="gpt-4.1-mini")
        assert assessor.weights["schema_depth"] > 0.25

    def test_model_id_via_api_format(self):
        """Should resolve OpenRouter-style model IDs."""
        assessor = ConsistencyRiskAssessor(model_id="openai/gpt-4.1-mini")
        assert assessor.weights["schema_depth"] > 0.25


class TestModelFallback:
    """Test fallback behavior for unknown models."""

    def test_unknown_model_uses_universal(self):
        """Unknown model_id should fall back to universal defaults."""
        assessor = ConsistencyRiskAssessor(model_id="some-unknown-model")
        assert assessor.weights == ConsistencyRiskAssessor.WEIGHTS
        assert assessor.thresholds == ConsistencyRiskAssessor.THRESHOLDS

    def test_no_model_id_uses_universal(self):
        """No model_id should use universal defaults."""
        assessor = ConsistencyRiskAssessor()
        assert assessor.weights == ConsistencyRiskAssessor.WEIGHTS
        assert assessor.thresholds == ConsistencyRiskAssessor.THRESHOLDS


class TestExplicitOverrides:
    """Test that explicit weights/thresholds override model-specific ones."""

    def test_explicit_weights_override_model(self):
        """Explicit weights should override model-specific config."""
        custom_weights = {"schema_depth": 0.01}
        assessor = ConsistencyRiskAssessor(
            model_id="GPT-4.1-Mini",
            weights=custom_weights,
        )
        # schema_depth should be overridden, but other model weights preserved
        assert assessor.weights["schema_depth"] == 0.01
        assert assessor.weights["param_type_diversity"] > 0.10  # model config

    def test_explicit_thresholds_override_model(self):
        """Explicit thresholds should override model-specific config."""
        custom_thresholds = {"schema_depth": {"medium": 1, "high": 2}}
        assessor = ConsistencyRiskAssessor(
            model_id="GPT-4.1-Mini",
            thresholds=custom_thresholds,
        )
        assert assessor.thresholds["schema_depth"] == {"medium": 1, "high": 2}
        # Other thresholds should still be model-specific
        assert assessor.thresholds["num_tools"]["high"] == 11  # GPT config

    def test_explicit_weights_without_model(self):
        """Explicit weights without model_id should override universal."""
        custom_weights = {"total_params": 0.50}
        assessor = ConsistencyRiskAssessor(weights=custom_weights)
        assert assessor.weights["total_params"] == 0.50
        # Other weights should be universal defaults
        assert assessor.weights["schema_complexity"] == 0.17


class TestBackwardCompatibility:
    """Ensure existing usage patterns still work identically."""

    def test_thresholds_only_init(self):
        """Old-style init with just thresholds should still work."""
        custom = {"total_params": {"medium": 5, "high": 8}}
        assessor = ConsistencyRiskAssessor(thresholds=custom)
        assert assessor.thresholds["total_params"] == {"medium": 5, "high": 8}
        assert assessor.weights == ConsistencyRiskAssessor.WEIGHTS

    def test_default_init_unchanged(self):
        """Default init should produce identical results as before."""
        assessor = ConsistencyRiskAssessor()
        tools = [_make_tool("search", {"q": {"type": "string"}}, required=["q"])]
        report = assessor.assess("Find items", tools)
        assert isinstance(report.overall_risk, float)
        assert report.risk_level in ("low", "medium", "high")
