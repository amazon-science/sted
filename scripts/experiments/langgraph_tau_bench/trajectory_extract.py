"""Extract a trajectory (list of {name, args} steps) from a LangGraph state."""
from __future__ import annotations

from typing import Any, List


def extract_trajectory(final_state: Any) -> List[dict]:
    """Walk the message history and pull out every AIMessage's tool_calls.

    LangGraph stores tool calls as AIMessage.tool_calls
    (``[{"name", "args", "id"}, ...]``). We keep only ``name`` and ``args``
    — that's the trajectory shape ``AgentConsistencyEvaluator.for_trajectory``
    consumes.

    Skips ToolMessage (results) and HumanMessage / SystemMessage entirely.
    """
    # Late import so this module doesn't drag in langchain_core for the
    # scoring side of the pipeline.
    from langchain_core.messages import AIMessage  # type: ignore

    messages = final_state.get("messages", []) if isinstance(final_state, dict) \
        else getattr(final_state, "messages", [])

    steps: List[dict] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            if name is None:
                continue
            steps.append({"name": str(name), "args": args or {}})
    return steps
