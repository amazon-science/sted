"""sted.testing: assert_same / assert_consistent / assert_snapshot (character mode)."""
import json

import pytest

from sted.testing import assert_consistent, assert_same, assert_snapshot

NONE = {"embedding_model": "none"}
CALL = {"name": "book_flight", "arguments": {"to": "JFK", "passengers": 2, "date": "2026-12-01", "request_id": "r-1"}}


def variant(**kw):
    return {"name": "book_flight", "arguments": {**CALL["arguments"], **kw}}


def test_assert_same_passes_and_returns_comparison():
    r = assert_same(variant(passengers="2"), CALL, ignore=["request_id"], **NONE)
    assert r.same


def test_assert_same_failure_names_the_field():
    with pytest.raises(AssertionError) as e:
        assert_same(variant(date="2026-12-02"), CALL, ignore=["request_id"], msg="booking", **NONE)
    text = str(e.value)
    assert text.startswith("booking\noutputs differ")
    assert '"2026-12-01" -> "2026-12-02"' in text and "arguments.date" in text


def test_assert_consistent():
    assert_consistent([CALL, variant(request_id="r-2")], ignore=["request_id"], **NONE)
    with pytest.raises(AssertionError) as e:
        assert_consistent([CALL, CALL, variant(date="2026-12-02")], ignore=["request_id"], **NONE)
    assert "same pairs 33% < 100%" in str(e.value) and "arguments.date" in str(e.value)
    assert_consistent([CALL, CALL, variant(date="2026-12-02")], ignore=["request_id"], min_same=0.3, **NONE)
    with pytest.raises(AssertionError, match="validity 0.67 < 1.00"):
        assert_consistent([CALL, CALL, "not json"], **NONE)


def test_snapshot_lifecycle(tmp_path, monkeypatch):
    snap = tmp_path / "snaps" / "booking.json"
    assert assert_snapshot(CALL, snap, **NONE) is None                       # written
    assert json.loads(snap.read_text()) == CALL
    assert assert_snapshot(variant(request_id="r-9"), snap, ignore=["request_id"], **NONE).same
    with pytest.raises(AssertionError) as e:
        assert_snapshot(variant(date="2026-12-02"), snap, ignore=["request_id"], **NONE)
    assert "STED_UPDATE_SNAPSHOTS=1" in str(e.value) and "arguments.date" in str(e.value)
    monkeypatch.setenv("STED_UPDATE_SNAPSHOTS", "1")
    assert assert_snapshot(variant(date="2026-12-02"), snap, **NONE) is None  # rewritten
    monkeypatch.delenv("STED_UPDATE_SNAPSHOTS")
    assert_snapshot(json.dumps(variant(date="2026-12-02")), snap, **NONE)    # JSON text accepted


def test_snapshot_rejects_non_json(tmp_path):
    with pytest.raises(AssertionError, match="not valid JSON"):
        assert_snapshot("Sorry, I can't.", tmp_path / "x.json", **NONE)
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    with pytest.raises(AssertionError, match="snapshot .* is not valid JSON"):
        assert_snapshot(CALL, bad, **NONE)
