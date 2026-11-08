"""Path patterns and FieldRules (no embedding model needed)."""
import pytest

from sted.rules import FieldRules, PathPattern, numbers_within


@pytest.mark.parametrize("pattern,path,expected", [
    ("timestamp", "timestamp", True),
    ("timestamp", "meta.timestamp", True),          # bare key: any depth
    ("timestamp", "items[3].timestamp", True),
    ("timestamp", "timestamps", False),
    ("arguments.request_id", "arguments.request_id", True),
    ("arguments.request_id", "x.arguments.request_id", False),   # dotted: anchored
    ("items[*].id", "items[0].id", True),
    ("items[*].id", "items.id", False),
    ("items[2].id", "items[2].id", True),
    ("items[2].id", "items[1].id", False),
    ("*.id", "a.id", True),
    ("*.id", "a.b.id", False),
    ("**.id", "a.b.id", True),
    ("**.id", "id", True),
    ("arguments.ids[*]", "arguments.ids[2]", True),
    ('["a.b"]', '["a.b"]', True),
])
def test_pattern_matching(pattern, path, expected):
    assert PathPattern(pattern).matches(path) is expected


def test_pattern_matches_typed_paths():
    assert PathPattern("timestamp").matches((("key", "meta"), ("key", "timestamp")))
    assert PathPattern("items[*]").matches((("key", "items"), ("index", 4)))


def test_bad_patterns_rejected():
    for bad in ("", "  ", "a..b", "a[x]"):
        with pytest.raises(ValueError):
            PathPattern(bad)


def test_strip_removes_ignored_fields_everywhere():
    r = FieldRules(ignore=["timestamp", "items[*].debug"])
    doc = {"timestamp": 1, "name": "x", "meta": {"timestamp": 2, "k": 1},
           "items": [{"id": 1, "debug": "z"}, {"id": 2}]}
    assert r.strip(doc) == {"name": "x", "meta": {"k": 1}, "items": [{"id": 1}, {"id": 2}]}
    assert doc["timestamp"] == 1                      # input untouched
    assert FieldRules().strip(doc) is doc             # no rules: no copy


def test_tolerance_specs():
    r = FieldRules(tolerance={"price": 0.01, "weight": {"rel": 0.1}, "x": {"abs": 1, "rel": 0.5}})
    assert r.tolerance_for("arguments.price") == (0.01, 0.0)
    assert r.tolerance_for("weight") == (0.0, 0.1)
    assert r.tolerance_for("other") is None
    assert numbers_within(10.0, 10.009, (0.01, 0.0)) and not numbers_within(10.0, 10.02, (0.01, 0.0))
    assert numbers_within(100.0, 109.0, (0.0, 0.1)) and not numbers_within(100.0, 112.0, (0.0, 0.1))
    for bad in ({"p": -1}, {"p": {"abs": float("inf")}}, {"p": {"pct": 1}}):
        with pytest.raises(ValueError):
            FieldRules(tolerance=bad)


def test_coerce_and_key():
    r = FieldRules.coerce(ignore=["a"], unordered=["b"])
    assert r.key() == FieldRules(["a"], ["b"]).key()
    assert not FieldRules() and r
    with pytest.raises(ValueError):
        FieldRules.coerce(r, ignore=["c"])
