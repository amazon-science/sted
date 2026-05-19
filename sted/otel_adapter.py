"""Adapter from OpenTelemetry log records to STED trajectory format.

Targets Claude Code's GenAI OTel events (`claude_code.tool_result`,
`claude_code.tool_decision`) but tolerates the slightly looser shapes that
LangChain / openllmetry / Traceloop emit. The output is a list of
``{"name", "args", "result_size_bytes", "duration_ms", "success"}`` dicts —
i.e. what ``AgentConsistencyEvaluator.for_trajectory()`` consumes.

Capture telemetry locally with::

    export CLAUDE_CODE_ENABLE_TELEMETRY=1
    export OTEL_LOGS_EXPORTER=console
    export OTEL_LOG_TOOL_DETAILS=1
    claude -p "your prompt" 2>&1 | tee session.jsonl

then::

    from sted.otel_adapter import load_trajectories_from_console_log
    trajectories = load_trajectories_from_console_log("session.jsonl")
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Union


TOOL_RESULT_EVENTS = frozenset({
    "claude_code.tool_result",
    "tool_result",
    "gen_ai.tool.message",
})

# Claude Code event names sometimes appear under "event.name" attribute,
# sometimes as the LogRecord body. Both are normalized.
_NAME_KEYS = ("event.name", "event_name", "name")
_TOOL_NAME_KEYS = ("tool_name", "gen_ai.tool.name", "tool.name")
_TOOL_INPUT_KEYS = ("tool_input", "gen_ai.tool.call.arguments", "tool_parameters")
_GROUP_KEYS = ("prompt.id", "session.id", "trace_id", "trace.id")
_SEQUENCE_KEYS = ("event.sequence", "timestamp", "event.timestamp")


def _flatten_attrs(record: dict) -> dict:
    """Pull attributes out of common OTel JSON shapes into a flat dict.

    The OTel SDK's console exporter emits LogRecords with attributes nested
    under ``attributes`` (a dict or a list of {key, value} pairs); some
    exporters flatten them at the top level; others (Claude Code) put them
    in ``body``. This handles all three.
    """
    attrs: dict = {}

    body = record.get("body")
    if isinstance(body, dict):
        attrs.update(body)

    raw = record.get("attributes")
    if isinstance(raw, dict):
        attrs.update(raw)
    elif isinstance(raw, list):
        for kv in raw:
            if isinstance(kv, dict) and "key" in kv:
                v = kv.get("value")
                if isinstance(v, dict):
                    # OTLP "AnyValue" — pick the first set field.
                    for k in ("stringValue", "intValue", "boolValue",
                              "doubleValue", "arrayValue", "kvlistValue"):
                        if k in v:
                            v = v[k]
                            break
                attrs[kv["key"]] = v

    # Top-level keys (some exporters flatten).
    for k, v in record.items():
        if k not in ("body", "attributes", "resource", "scope"):
            attrs.setdefault(k, v)

    return attrs


def _first(d: dict, keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _parse_args(value: Any) -> Any:
    """Tool inputs are JSON-serialized strings on the wire; parse if so."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def is_tool_event(record: dict) -> bool:
    """Return True if the record looks like a tool-result/tool-call event."""
    attrs = _flatten_attrs(record)
    name = _first(attrs, _NAME_KEYS) or record.get("name")
    if isinstance(name, str) and name in TOOL_RESULT_EVENTS:
        return True
    # Fall back: any record carrying a tool_name + tool_input/parameters.
    if _first(attrs, _TOOL_NAME_KEYS) and _first(attrs, _TOOL_INPUT_KEYS):
        return True
    return False


def event_to_step(record: dict) -> Optional[dict]:
    """Convert a single OTel log record into one trajectory step.

    Returns None if the record is not a tool event.
    """
    if not is_tool_event(record):
        return None
    attrs = _flatten_attrs(record)

    tool_name = _first(attrs, _TOOL_NAME_KEYS)
    if tool_name is None:
        return None

    step: dict = {"name": str(tool_name)}

    args = _first(attrs, _TOOL_INPUT_KEYS)
    if args is not None:
        step["args"] = _parse_args(args)

    # Optional metadata; included so trajectories are richer but exact-match
    # on tool name / parameter names still drives the score.
    for src, dst in (
        ("success", "success"),
        ("duration_ms", "duration_ms"),
        ("tool_result_size_bytes", "result_size_bytes"),
    ):
        if src in attrs:
            step[dst] = attrs[src]

    return step


def _group_key(record: dict) -> Optional[str]:
    attrs = _flatten_attrs(record)
    return _first(attrs, _GROUP_KEYS)


def _sequence_key(record: dict) -> Any:
    attrs = _flatten_attrs(record)
    return _first(attrs, _SEQUENCE_KEYS) or 0


def iter_records_from_jsonl(path: Union[str, Path]) -> Iterator[dict]:
    """Yield JSON objects from a newline-delimited file. Skips bad lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue


def trajectories_from_records(
    records: Iterable[dict],
    group_by: str = "prompt",
) -> dict:
    """Group OTel records into trajectories.

    Args:
        records: Iterable of decoded JSON log records.
        group_by: "prompt" groups by ``prompt.id``; "session" groups by
            ``session.id``; "single" treats all records as one trajectory.

    Returns:
        Dict keyed by group id (or "all" for single-group mode), each value
        a list of trajectory steps in sequence order.
    """
    if group_by not in ("prompt", "session", "single"):
        raise ValueError(f"group_by must be prompt/session/single, got {group_by!r}")

    by_group: dict = defaultdict(list)

    for rec in records:
        if not is_tool_event(rec):
            continue
        step = event_to_step(rec)
        if step is None:
            continue

        if group_by == "single":
            gid = "all"
        else:
            attrs = _flatten_attrs(rec)
            if group_by == "prompt":
                gid = _first(attrs, ("prompt.id",)) or _first(attrs, _GROUP_KEYS)
            else:  # session
                gid = _first(attrs, ("session.id",)) or _first(attrs, _GROUP_KEYS)
            if gid is None:
                gid = "ungrouped"

        by_group[gid].append((_sequence_key(rec), step))

    return {
        gid: [step for _, step in sorted(items, key=lambda p: p[0])]
        for gid, items in by_group.items()
    }


def load_trajectories_from_console_log(
    path: Union[str, Path],
    group_by: str = "prompt",
) -> List[List[dict]]:
    """Load and group trajectories from a Claude-Code console-exporter JSONL file.

    Returns a list of trajectories (each a list of step dicts). Group ids are
    discarded — call ``trajectories_from_records`` directly if you need them.
    """
    grouped = trajectories_from_records(iter_records_from_jsonl(path), group_by)
    return list(grouped.values())


# ---------------------------------------------------------------------------
# Claude Code session-transcript adapter
#
# Independent of OTel: Claude Code writes a per-session JSONL transcript at
# ``~/.claude/projects/<encoded-cwd>/<sessionId>.jsonl`` containing assistant
# messages with Anthropic-API-style ``tool_use`` content blocks. This is
# usually more reliable for offline trajectory analysis than OTel because it
# requires no telemetry setup.
# ---------------------------------------------------------------------------


def trajectory_from_session_transcript(path: Union[str, Path]) -> List[dict]:
    """Extract a single tool-call trajectory from a Claude Code session JSONL.

    Returns a list of ``{"name", "args"}`` dicts in the order tool_use blocks
    appear in the assistant turns.
    """
    steps: List[dict] = []
    for rec in iter_records_from_jsonl(path):
        if rec.get("type") != "assistant":
            continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if not name:
                continue
            step: dict = {"name": str(name)}
            tool_input = block.get("input")
            if tool_input is not None:
                step["args"] = tool_input
            steps.append(step)
    return steps


def load_trajectories_from_session_dir(
    dir_path: Union[str, Path],
    pattern: str = "*.jsonl",
) -> List[List[dict]]:
    """Load every session transcript under ``dir_path`` as a trajectory.

    Empty trajectories (sessions with no tool calls) are dropped.
    """
    p = Path(dir_path)
    out: List[List[dict]] = []
    for f in sorted(p.glob(pattern)):
        traj = trajectory_from_session_transcript(f)
        if traj:
            out.append(traj)
    return out
