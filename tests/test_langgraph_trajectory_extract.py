"""Test trajectory extraction from a synthetic LangGraph final state.

We don't depend on `langchain_core` being installed at import time of the
production codebase, but the extractor uses it. If langchain_core is missing,
these tests are skipped — there's no regression risk for users without it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from scripts.experiments.langgraph_tau_bench.trajectory_extract import (  # noqa: E402
    extract_trajectory,
)


def _ai_with_calls(calls):
    """Build an AIMessage with given tool_calls."""
    return AIMessage(content="", tool_calls=calls)


def test_extract_trajectory_returns_tool_calls_in_order():
    state = {
        "messages": [
            SystemMessage(content="system"),
            HumanMessage(content="A customer says: hi"),
            _ai_with_calls([
                {"name": "find_user_id_by_name_zip",
                 "args": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
                 "id": "tu_1"},
            ]),
            ToolMessage(content="result1", tool_call_id="tu_1"),
            _ai_with_calls([
                {"name": "get_order_details",
                 "args": {"order_id": "#W2378156"},
                 "id": "tu_2"},
            ]),
            ToolMessage(content="result2", tool_call_id="tu_2"),
            AIMessage(content="Done."),
        ]
    }
    traj = extract_trajectory(state)
    assert [s["name"] for s in traj] == [
        "find_user_id_by_name_zip", "get_order_details",
    ]
    assert traj[0]["args"]["zip"] == "19122"
    assert traj[1]["args"]["order_id"] == "#W2378156"


def test_extract_trajectory_handles_multiple_tool_calls_per_aimessage():
    """A single AIMessage may emit several tool_calls in parallel."""
    state = {
        "messages": [
            _ai_with_calls([
                {"name": "get_user_details", "args": {"user_id": "u1"}, "id": "a"},
                {"name": "get_order_details", "args": {"order_id": "o1"}, "id": "b"},
            ]),
        ]
    }
    traj = extract_trajectory(state)
    assert [s["name"] for s in traj] == ["get_user_details", "get_order_details"]


def test_extract_trajectory_skips_non_aimessages():
    state = {
        "messages": [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            ToolMessage(content="result", tool_call_id="x"),
        ]
    }
    assert extract_trajectory(state) == []


def test_extract_trajectory_skips_aimessages_without_tool_calls():
    state = {
        "messages": [
            AIMessage(content="just a text response"),
            AIMessage(content="another"),
        ]
    }
    assert extract_trajectory(state) == []


def test_extract_trajectory_handles_object_state():
    """LangGraph sometimes returns a state object with .messages instead of dict."""
    class _StateObj:
        messages = [_ai_with_calls([
            {"name": "Read", "args": {"file_path": "/a"}, "id": "z"},
        ])]
    traj = extract_trajectory(_StateObj())
    assert traj == [{"name": "Read", "args": {"file_path": "/a"}}]


