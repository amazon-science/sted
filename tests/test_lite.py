"""Character-similarity mode (no embedding model): what `pip install sted` without the
`embeddings` extra runs. These tests need neither torch nor sentence-transformers."""
import random

import pytest

import sted
from sted.rules import FieldRules
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from tests.test_field_differences import _gen, _mutate

REF = {"name": "get_weather", "arguments": {"city": "Seattle", "days": 3, "units": "metric"}}
REORDERED = {"arguments": {"units": "metric", "days": "3", "city": "Seattle"}, "name": "get_weather"}
CHANGED = {"name": "get_weather", "arguments": {"city": "Seattle", "days": 5, "units": "metric"}}


@pytest.fixture
def no_embeddings(monkeypatch):
    """Fail loudly if anything tries to compute an embedding."""
    def boom(self, text):
        raise AssertionError(f"embedding requested for {text!r} in character mode")
    monkeypatch.setattr(SemanticJsonTreeConsistencyEvaluator, "_get_embedding", boom)


def test_evaluator_without_model(no_embeddings):
    ev = SemanticJsonTreeConsistencyEvaluator(model_id=None)
    assert ev.embedding_model is None and ev.leaf_literalism_tau == 1.0
    assert ev.calculate_tree_edit_distance_fast(REF, REORDERED) == 1.0
    assert ev.calculate_tree_edit_distance_fast(REF, CHANGED) == 0.0
    s = ev.calculate_tree_edit_distance_fast({"note": "window seats please"}, {"note": "window seat please"})
    assert 0.9 < s < 1.0          # one character edit in 19


def test_api_character_mode(no_embeddings):
    r = sted.compare(REF, CHANGED, embedding_model="none")
    assert r.text_similarity == "character"
    assert not r.same and [d.path for d in r.differences] == ["arguments.days"]
    assert sted.compare(REF, REORDERED, embedding_model="none").same
    c = sted.consistency([REF, REORDERED, CHANGED, "not json"], embedding_model="none")
    assert c.text_similarity == "character" and c.n_valid == 3
    assert [u.path for u in c.unstable_fields] == ["arguments.days"]


def test_rules_in_character_mode(no_embeddings):
    a = {"id": "r-1", "tags": ["x", "y"], "price": 10.0}
    b = {"id": "r-2", "tags": ["y", "x"], "price": 10.004}
    assert not sted.compare(a, b, embedding_model="none").same
    assert sted.compare(a, b, embedding_model="none", ignore=["id"], unordered=["tags"],
                        tolerance={"price": 0.01}).same


def test_min_field_score_equals_score_in_character_mode(no_embeddings):
    rng = random.Random(5)
    plain = SemanticJsonTreeConsistencyEvaluator(model_id=None)
    ruled = SemanticJsonTreeConsistencyEvaluator(model_id=None, field_rules=FieldRules(
        ignore=["id"], unordered=["**"], tolerance={"days": 1}))
    for _ in range(150):
        a = _gen(rng)
        b = _mutate(rng, a) if rng.random() < .8 else _gen(rng)
        ev = rng.choice([plain, ruled])
        diffs = ev.calculate_field_differences(a, b)
        assert min((d["score"] for d in diffs), default=1.0) == pytest.approx(
            ev.calculate_tree_edit_distance_fast(a, b), abs=1e-6), (a, b)


def test_env_var_selects_mode(monkeypatch, no_embeddings):
    monkeypatch.setenv("STED_EMBEDDING_MODEL", "none")
    assert sted.compare(REF, REF).text_similarity == "character"
