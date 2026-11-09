"""sted.compare / sted.consistency, including field rules."""
import json
import random

import pytest

import sted
from sted.rules import FieldRules
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from tests.test_field_differences import _gen, _mutate

A = {"name": "book_flight", "arguments": {"from": "SEA", "to": "JFK", "date": "2026-12-01",
                                          "passengers": 2, "request_id": "r-1", "seats": ["12A", "12B"],
                                          "price": 410.00}}
B = {"name": "book_flight", "arguments": {"to": "JFK", "from": "SEA", "date": "2026-12-01",
                                          "passengers": "2", "request_id": "r-9", "seats": ["12B", "12A"],
                                          "price": 410.004}}


def test_compare_without_rules_lists_every_difference():
    r = sted.compare(A, B)
    assert not r.same and r.min_readout == 0.0
    paths = {d.path for d in r.differences}
    assert paths == {"arguments.request_id", "arguments.seats[0]", "arguments.seats[1]", "arguments.price"}
    assert r.min_readout == min(d.score for d in r.differences)


def test_compare_with_rules_is_same():
    r = sted.compare(A, B, ignore=["request_id"], unordered=["seats"], tolerance={"price": 0.01})
    assert r.same and r.min_readout == 1.0 and r.differences == []


def test_each_rule_is_needed():
    full = dict(ignore=["request_id"], unordered=["seats"], tolerance={"price": 0.01})
    for drop in full:
        kw = {k: v for k, v in full.items() if k != drop}
        assert not sted.compare(A, B, **kw).same, drop


def test_unordered_array_reported_whole():
    b = json.loads(json.dumps(B))
    b["arguments"]["seats"] = ["12B", "14C"]
    r = sted.compare(A, b, ignore=["request_id"], unordered=["seats"], tolerance={"price": 0.01})
    assert [(d.path, d.a, d.b) for d in r.differences] == [("arguments.seats", ["12A", "12B"], ["12B", "14C"])]


def test_rules_object_and_kwargs_are_exclusive():
    rules = FieldRules(ignore=["request_id"])
    assert sted.compare(A, A, rules=rules).same
    with pytest.raises(ValueError):
        sted.compare(A, B, rules=rules, ignore=["x"])


def test_compare_str_and_to_dict():
    r = sted.compare(A, B)
    text = str(r)
    assert text.startswith("DIFFERENT") and "arguments.price" in text
    d = r.to_dict()
    assert json.loads(json.dumps(d))["differences"][0].keys() == {"path", "status", "a", "b", "score"}


def test_consistency_unstable_fields_and_summary():
    other = json.loads(json.dumps(A))
    other["arguments"]["date"] = "2026-12-02"
    other["arguments"]["request_id"] = "r-3"
    runs = [A] * 5 + [other] * 2 + ["I can't book that."]
    r = sted.consistency(runs, ignore=["request_id"])
    assert (r.n_outputs, r.n_valid, r.n_pairs) == (8, 7, 21)
    assert [u.path for u in r.unstable_fields] == ["arguments.date"]
    assert r.unstable_fields[0].share_of_pairs == pytest.approx(10 / 21, abs=1e-6)
    assert r.share_same == pytest.approx(11 / 21, abs=1e-6)
    assert "arguments.date" in str(r) and "validity 0.88" in str(r)


def test_consistency_values_for_unordered_array_are_whole():
    runs = [{"tags": ["a", "b"]}, {"tags": ["b", "a"]}, {"tags": ["a", "c"]}]
    r = sted.consistency(runs, unordered=["tags"])
    u = r.unstable_fields[0]
    assert u.path == "tags" and u.share_of_pairs == pytest.approx(2 / 3, abs=1e-6)
    assert {"value": ["a", "b"], "count": 1} in u.values


def test_min_field_score_equals_score_under_random_rules():
    rng = random.Random(11)
    keys = ["name", "city", "days", "q", "arguments", "id"]
    evaluators = []
    for _ in range(3):
        rules = FieldRules(ignore=rng.sample(keys, 1), unordered=rng.sample(keys, 2) + ["**"] * (rng.random() < .4),
                           tolerance={rng.choice(keys): rng.choice([0.5, 2, {"rel": 0.2}])})
        evaluators.append(SemanticJsonTreeConsistencyEvaluator(profile="typed", model_id="all-MiniLM-L6-v2",
                                                               field_rules=rules))
    for _ in range(120):
        a = _gen(rng)
        b = _mutate(rng, a) if rng.random() < .8 else _gen(rng)
        ev = rng.choice(evaluators)
        diffs = ev.calculate_field_differences(a, b)
        assert min((d["score"] for d in diffs), default=1.0) == pytest.approx(
            ev.calculate_tree_edit_distance_fast(a, b), abs=1e-6), (a, b, ev.field_rules)
