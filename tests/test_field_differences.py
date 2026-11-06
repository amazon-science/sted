"""calculate_field_differences explains the typed (min) readout: the tree score equals
the lowest per-field score, with siblings, paths and callee identity in context."""
import random

import pytest

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator, format_json_path


@pytest.fixture(scope="module")
def ev():
    return SemanticJsonTreeConsistencyEvaluator(profile="typed", model_id="all-MiniLM-L6-v2")


WORDS = ["Seattle", "Seattle, WA", "Paris", "hotels near the Louvre", "cheap hotels near Louvre",
         "metric", "2026-11-06", "11/06/2026", "abc-123", "ABC-123"]


def _leaf(rng):
    r = rng.random()
    if r < .3:
        return rng.choice([3, 5, 3.0, 10, 0.5])
    if r < .45:
        return rng.choice(["3", "5", "10"])
    if r < .55:
        return rng.choice([True, False, None])
    return rng.choice(WORDS)


def _gen(rng, d=0):
    r = rng.random()
    if d > 2 or r < .35:
        return _leaf(rng)
    if r < .7:
        return {rng.choice(["name", "city", "days", "q", "arguments", "id"]): _gen(rng, d + 1)
                for _ in range(rng.randint(0, 3))}
    return [_gen(rng, d + 1) for _ in range(rng.randint(0, 3))]


def _mutate(rng, x):
    if isinstance(x, dict) and x and rng.random() < .7:
        y = dict(x)
        k = rng.choice(list(y))
        r = rng.random()
        if r < .5:
            y[k] = _mutate(rng, y[k])
        elif r < .7:
            del y[k]
        else:
            y["extra"] = _leaf(rng)
        return dict(rng.sample(list(y.items()), len(y)))
    if isinstance(x, list) and x and rng.random() < .7:
        y = list(x)
        i = rng.randrange(len(y))
        y[i] = _mutate(rng, y[i])
        return y
    return _leaf(rng) if rng.random() < .6 else x


CALLS = [
    ({"name": "get_weather", "arguments": {"city": "Seattle"}}, {"name": "get_weathers", "arguments": {"city": "Seattle"}}),
    ([{"name": "a_tool", "arguments": {}}], [{"name": "b_tool", "arguments": {}}]),
    ({"user": {"name": "John Smith"}}, {"user": {"name": "Jon Smith"}}),
]


def test_min_field_score_equals_tree_score(ev):
    rng = random.Random(7)
    pairs = list(CALLS)
    for _ in range(150):
        a = _gen(rng)
        pairs.append((a, _mutate(rng, a) if rng.random() < .8 else _gen(rng)))
    for a, b in pairs:
        diffs = ev.calculate_field_differences(a, b)
        assert min((d["score"] for d in diffs), default=1.0) == pytest.approx(
            ev.calculate_tree_edit_distance_fast(a, b), abs=1e-6), (a, b, diffs)


def test_callee_name_is_exact_in_context(ev):
    d = ev.calculate_field_differences(*CALLS[0])
    assert d == [{"path": "name", "status": "changed", "a": "get_weather", "b": "get_weathers", "score": 0.0}]
    person = ev.calculate_field_differences(*CALLS[2])
    assert person[0]["path"] == "user.name" and person[0]["score"] > 0.5   # a person's name is text


def test_format_json_path():
    assert format_json_path(()) == "$"
    assert format_json_path((("key", "arguments"), ("key", "days"))) == "arguments.days"
    assert format_json_path((("index", 0), ("key", "name"))) == "[0].name"
    assert format_json_path((("key", "a.b"),)) == '["a.b"]'
