"""Assertions for tests of LLM structured outputs (pytest, unittest, or plain asserts).

    from sted.testing import assert_same, assert_consistent, assert_snapshot

    def test_booking_call():
        call = agent("Book SEA to JFK on Dec 1 for 2")
        assert_snapshot(call, "snapshots/booking.json", ignore=["request_id"])

    def test_booking_is_stable():
        runs = [agent("Book SEA to JFK on Dec 1 for 2") for _ in range(5)]
        assert_consistent(runs, ignore=["request_id"], min_same=1.0)

Failures raise ``AssertionError`` whose message lists the fields that differ (or are
unstable), so the test report says what changed, not just that something did.

Snapshots: the first run writes the file and passes; later runs compare against it. Set
``STED_UPDATE_SNAPSHOTS=1`` to rewrite snapshots after an intended change. Every helper
accepts the field rules of :func:`sted.compare` (``ignore``, ``unordered``, ``tolerance``,
``rules``) and ``threshold``.
"""
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from . import api

__all__ = ["assert_same", "assert_consistent", "assert_snapshot", "UPDATE_ENV"]

UPDATE_ENV = "STED_UPDATE_SNAPSHOTS"


def assert_same(actual: Any, expected: Any, *, msg: Optional[str] = None, **kwargs) -> api.Comparison:
    """Assert two outputs are the same under the min readout (every field >= threshold).

    Returns the :class:`~sted.api.Comparison` so a test can inspect it further."""
    r = api.compare(expected, actual, **kwargs)
    if not r.same:
        raise AssertionError(_message(msg, "outputs differ (expected -> actual)", str(r)))
    return r


def assert_consistent(outputs: Sequence[Any], *, min_same: float = 1.0, min_validity: float = 1.0,
                      msg: Optional[str] = None, **kwargs) -> api.Consistency:
    """Assert repeated outputs are consistent.

    ``min_same``: required share of output pairs that are the same (default 1.0: all).
    ``min_validity``: required share of outputs that parse (default 1.0: all)."""
    r = api.consistency(outputs, **kwargs)
    problems = []
    if r.validity < min_validity:
        problems.append(f"validity {r.validity:.2f} < {min_validity:.2f}")
    same = r.share_same if r.share_same is not None else 0.0
    if same < min_same:
        problems.append(f"same pairs {same:.0%} < {min_same:.0%}")
    if problems:
        raise AssertionError(_message(msg, "outputs are not consistent: " + "; ".join(problems), str(r)))
    return r


def assert_snapshot(actual: Any, path: Union[str, Path], *, update: Optional[bool] = None,
                    msg: Optional[str] = None, **kwargs) -> Optional[api.Comparison]:
    """Compare ``actual`` with the JSON snapshot at ``path``.

    Missing snapshot: written, and the assertion passes (returns None). Existing
    snapshot: compared with :func:`assert_same`. ``update=True`` (or the environment
    variable ``STED_UPDATE_SNAPSHOTS=1``) rewrites the snapshot instead of comparing."""
    path = Path(path)
    value = api._parse(actual)
    if value is None:
        raise AssertionError(_message(msg, f"output is not valid JSON: {str(actual)[:120]!r}", ""))
    if update is None:
        update = os.environ.get(UPDATE_ENV, "").lower() in ("1", "true", "yes")
    if update or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        return None
    try:
        expected = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise AssertionError(_message(msg, f"snapshot {path} is not valid JSON ({e})", "")) from e
    r = api.compare(expected, value, **kwargs)
    if not r.same:
        detail = f"{r}\n\nIf the change is intended, rerun with {UPDATE_ENV}=1 to update {path}."
        raise AssertionError(_message(msg, f"output differs from snapshot {path} (snapshot -> actual)", detail))
    return r


def _message(user: Optional[str], head: str, body: str) -> str:
    parts = [p for p in (user, head, body) if p]
    return "\n".join(parts)
