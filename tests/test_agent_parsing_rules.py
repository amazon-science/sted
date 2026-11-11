"""AgentConsistencyEvaluator: JSON text is parsed, non-JSON text is invalid, field rules
apply, and the text mode follows sted.compare (works without torch)."""
import importlib.util

import pytest

from sted import AgentConsistencyEvaluator
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator


@pytest.fixture(autouse=True)
def _character_mode(monkeypatch):
    monkeypatch.setenv("STED_EMBEDDING_MODEL", "none")


def score(outputs, **kw):
    r = AgentConsistencyEvaluator(**kw).evaluate_outputs({"p": outputs}).per_prompt[0]
    return round(r.c_mean, 6), round(r.r_v, 6)


def test_json_text_is_parsed_before_comparison():
    assert score(['{"name": "f", "arguments": {"a": 1}}', {"name": "f", "arguments": {"a": 1}}]) == (1.0, 1.0)
    assert score(['{"a": 1}', '{"a": 2}']) == (0.0, 1.0)          # a changed value, not 0.935 string similarity
    assert score(['{"a": 1}', '{"a": 2}'], parse_json=False)[0] > 0.9   # the old string comparison


def test_text_that_is_not_json_is_invalid_by_default():
    outs = [{"a": 1}, {"a": 1}, "Sorry, I can't."]
    assert score(outs) == (1.0, pytest.approx(2 / 3, abs=1e-6))
    c_mean, r_v = score(outs, strict_json=False)                    # previous behaviour
    assert r_v == 1.0 and c_mean < 1.0


def test_field_rules():
    call = lambda rid, tags: {"name": "book", "arguments": {"request_id": rid, "tags": tags}}  # noqa: E731
    outs = [call("r1", ["a", "b"]), call("r2", ["b", "a"])]
    assert score(outs)[0] == 0.0
    assert score(outs, ignore=["request_id"], unordered=["tags"])[0] == 1.0
    with pytest.raises(ValueError, match="pre-built evaluator"):
        AgentConsistencyEvaluator(evaluator=SemanticJsonTreeConsistencyEvaluator(model_id=None), ignore=["x"])


def test_evaluate_pair_uses_the_same_rules():
    ev = AgentConsistencyEvaluator(ignore=["id"])
    assert ev.evaluate_pair('{"a": 1, "id": 1}', {"a": 1, "id": 2}) == 1.0
    assert ev.evaluate_pair("not json", {"a": 1}) == 0.0


def test_default_text_mode_follows_sted_compare(monkeypatch):
    monkeypatch.delenv("STED_EMBEDDING_MODEL")
    ev = AgentConsistencyEvaluator()
    if importlib.util.find_spec("sentence_transformers") is None:
        assert ev._evaluator.model_id is None                        # core install: no torch needed
    else:
        assert ev._evaluator.model_id == "all-MiniLM-L6-v2"
    monkeypatch.setenv("STED_EMBEDDING_MODEL", "none")
    assert AgentConsistencyEvaluator()._evaluator.model_id is None


def test_sted_consistency_parses_json_scalars_once():
    """sted.consistency parses once. A second parse would make the string "hello" invalid
    (not JSON) and turn the string "true" into the boolean true."""
    import sted
    assert sted.consistency(['"hello"', '"hello"']).n_valid == 2
    r = sted.consistency(['"true"', "true"])
    assert r.n_valid == 2 and r.share_same == 0.0          # a string is not a boolean
