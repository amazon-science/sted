"""Tests for sted.otel_adapter — OTel log records → STED trajectories.

Synthetic records modeled after the Claude Code monitoring docs schema
(claude_code.tool_result events with tool_name / tool_input / prompt.id /
event.sequence). Also tests OTLP-style nested-attribute records that arrive
when using the OTLP HTTP/JSON exporter.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sted import AgentConsistencyEvaluator
from sted.otel_adapter import (
    event_to_step,
    is_tool_event,
    iter_records_from_jsonl,
    load_trajectories_from_console_log,
    trajectories_from_records,
)


def _cc_tool_event(
    tool: str,
    args: dict,
    *,
    prompt_id: str = "p1",
    seq: int = 0,
    success: bool = True,
) -> dict:
    """Mimic the Claude Code console-exporter tool_result record shape."""
    return {
        "body": {
            "event.name": "claude_code.tool_result",
            "tool_name": tool,
            "tool_input": json.dumps(args),
            "success": str(success).lower(),
            "duration_ms": 12,
            "tool_result_size_bytes": 256,
            "prompt.id": prompt_id,
            "session.id": "s1",
            "event.sequence": seq,
            "event.timestamp": f"2026-05-14T00:00:0{seq}Z",
            "tool_use_id": f"tu_{seq}",
        }
    }


def _otlp_tool_event(tool: str, args: dict, prompt_id: str, seq: int) -> dict:
    """Mimic the OTLP JSON exporter shape: attributes as list of {key,value}."""
    return {
        "name": "claude_code.tool_result",
        "attributes": [
            {"key": "tool_name", "value": {"stringValue": tool}},
            {"key": "tool_input", "value": {"stringValue": json.dumps(args)}},
            {"key": "prompt.id", "value": {"stringValue": prompt_id}},
            {"key": "event.sequence", "value": {"intValue": seq}},
        ],
    }


def test_is_tool_event_true_for_tool_result():
    assert is_tool_event(_cc_tool_event("Read", {"file_path": "/a.py"}))


def test_is_tool_event_false_for_user_prompt():
    rec = {"body": {"event.name": "claude_code.user_prompt", "prompt": "hi"}}
    assert not is_tool_event(rec)


def test_is_tool_event_false_for_api_request():
    rec = {"body": {"event.name": "claude_code.api_request", "model": "opus"}}
    assert not is_tool_event(rec)


def test_event_to_step_extracts_name_and_args():
    rec = _cc_tool_event("Read", {"file_path": "/a.py"})
    step = event_to_step(rec)
    assert step is not None
    assert step["name"] == "Read"
    assert step["args"] == {"file_path": "/a.py"}
    assert step["success"] == "true"
    assert step["duration_ms"] == 12


def test_event_to_step_returns_none_for_non_tool():
    rec = {"body": {"event.name": "claude_code.api_error", "error": "boom"}}
    assert event_to_step(rec) is None


def test_event_to_step_handles_otlp_listattribute_shape():
    rec = _otlp_tool_event("Edit", {"file_path": "/x.py"}, prompt_id="pX", seq=3)
    step = event_to_step(rec)
    assert step is not None
    assert step["name"] == "Edit"
    assert step["args"] == {"file_path": "/x.py"}


def test_event_to_step_args_passthrough_when_not_json():
    rec = _cc_tool_event("Bash", {})
    rec["body"]["tool_input"] = "not-valid-json"
    step = event_to_step(rec)
    assert step["args"] == "not-valid-json"


def test_event_to_step_skipped_without_OTEL_LOG_TOOL_DETAILS():
    """When OTEL_LOG_TOOL_DETAILS is unset, tool_input is absent. We should
    still emit a step with the tool name, just no args."""
    rec = _cc_tool_event("Bash", {"command": "ls"})
    del rec["body"]["tool_input"]
    # Falls through is_tool_event via the event.name path.
    step = event_to_step(rec)
    assert step["name"] == "Bash"
    assert "args" not in step


def test_trajectories_grouped_by_prompt_id():
    records = [
        _cc_tool_event("Read", {"file": "a"}, prompt_id="P1", seq=0),
        _cc_tool_event("Edit", {"file": "a"}, prompt_id="P1", seq=1),
        _cc_tool_event("Read", {"file": "b"}, prompt_id="P2", seq=0),
    ]
    grouped = trajectories_from_records(records, group_by="prompt")
    assert set(grouped.keys()) == {"P1", "P2"}
    assert [s["name"] for s in grouped["P1"]] == ["Read", "Edit"]
    assert [s["name"] for s in grouped["P2"]] == ["Read"]


def test_sequence_order_preserved_when_records_arrive_out_of_order():
    records = [
        _cc_tool_event("C", {"i": 2}, prompt_id="P", seq=2),
        _cc_tool_event("A", {"i": 0}, prompt_id="P", seq=0),
        _cc_tool_event("B", {"i": 1}, prompt_id="P", seq=1),
    ]
    grouped = trajectories_from_records(records, group_by="prompt")
    assert [s["name"] for s in grouped["P"]] == ["A", "B", "C"]


def test_non_tool_events_filtered_out():
    records = [
        {"body": {"event.name": "claude_code.user_prompt", "prompt": "hi"}},
        _cc_tool_event("Read", {"file": "a"}, prompt_id="P", seq=0),
        {"body": {"event.name": "claude_code.api_request"}},
        {"body": {"event.name": "claude_code.hook_execution_start"}},
        _cc_tool_event("Edit", {"file": "a"}, prompt_id="P", seq=1),
    ]
    grouped = trajectories_from_records(records, group_by="prompt")
    assert [s["name"] for s in grouped["P"]] == ["Read", "Edit"]


def test_load_trajectories_from_console_log(tmp_path: Path):
    log = tmp_path / "session.jsonl"
    records = [
        _cc_tool_event("Read", {"file": "a"}, prompt_id="P1", seq=0),
        _cc_tool_event("Edit", {"file": "a"}, prompt_id="P1", seq=1),
        _cc_tool_event("Read", {"file": "b"}, prompt_id="P2", seq=0),
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    trajectories = load_trajectories_from_console_log(log)
    assert len(trajectories) == 2
    names = sorted(tuple(s["name"] for s in t) for t in trajectories)
    assert names == [("Read",), ("Read", "Edit")]


def test_iter_records_skips_bad_lines(tmp_path: Path):
    log = tmp_path / "noisy.jsonl"
    log.write_text(
        "this is plaintext startup banner\n"
        + json.dumps(_cc_tool_event("Read", {"f": "a"})) + "\n"
        + "\n"  # blank
        + "{not-json\n"
        + json.dumps(_cc_tool_event("Edit", {"f": "a"})) + "\n"
    )
    records = list(iter_records_from_jsonl(log))
    assert len(records) == 2


def test_invalid_group_by_raises():
    with pytest.raises(ValueError, match="group_by must be"):
        trajectories_from_records([], group_by="bogus")


# ---------- End-to-end: feed the adapter output into AgentConsistencyEvaluator
# ---------- to confirm the trajectory shape is consumable.


def test_end_to_end_two_identical_trajectories_score_perfectly():
    rec_a = [
        _cc_tool_event("Read", {"file_path": "/a.py"}, prompt_id="run1", seq=0),
        _cc_tool_event("Edit", {"file_path": "/a.py", "new": "x"},
                       prompt_id="run1", seq=1),
    ]
    rec_b = [
        _cc_tool_event("Read", {"file_path": "/a.py"}, prompt_id="run2", seq=0),
        _cc_tool_event("Edit", {"file_path": "/a.py", "new": "x"},
                       prompt_id="run2", seq=1),
    ]
    grouped = trajectories_from_records(rec_a + rec_b, group_by="prompt")
    trajectories = list(grouped.values())
    assert len(trajectories) == 2

    ev = AgentConsistencyEvaluator.for_trajectory()
    score = ev.evaluate_pair(trajectories[0], trajectories[1])
    assert score == pytest.approx(1.0, abs=1e-4)


def test_end_to_end_tool_name_drift_penalized():
    """Same parameter names, same args, but second run uses Bash where the
    first used Read — exact-match on tool names should make this < 1.0."""
    rec_a = [_cc_tool_event("Read", {"file_path": "/a.py"},
                            prompt_id="A", seq=0)]
    rec_b = [_cc_tool_event("Bash", {"file_path": "/a.py"},
                            prompt_id="B", seq=0)]
    grouped = trajectories_from_records(rec_a + rec_b, group_by="prompt")
    trajs = list(grouped.values())

    ev = AgentConsistencyEvaluator.for_trajectory()
    score = ev.evaluate_pair(trajs[0], trajs[1])
    assert score < 0.99
