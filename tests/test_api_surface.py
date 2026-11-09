"""The stable top-level API, and the names that moved out of it."""
import subprocess
import sys
import warnings

import pytest

import sted


def test_stable_names_resolve_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for name in sted.__all__:
            assert getattr(sted, name) is not None, name
    assert sted.STED is sted.SemanticJsonTreeConsistencyEvaluator


@pytest.mark.parametrize("name,module", [
    ("ConsistencyPredictor", "consistency_predictor"),
    ("ConsistencyRiskAssessor", "consistency_risk_assessor"),
    ("LLMJudge", "llm_judge"),
    ("JsonNode", "json_tree_node"),
    ("StructuralConsistencyAnalyzer", "structural_consistency_analyzer"),
])
def test_moved_names_warn_and_point_to_their_module(name, module):
    with pytest.warns(DeprecationWarning, match=f"from sted.{module} import {name}"):
        value = getattr(sted, name)
    assert value is getattr(__import__(f"sted.{module}", fromlist=[name]), name)


def test_unknown_attribute():
    with pytest.raises(AttributeError):
        sted.does_not_exist


def test_import_is_light():
    """`import sted` must not load the evaluator stack (scipy, deepdiff, torch)."""
    code = ("import sys, sted; heavy = [m for m in ('scipy', 'deepdiff', 'torch', 'sentence_transformers', "
            "'sted.semantic_json_tree_consistency') if m in sys.modules]; print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out
