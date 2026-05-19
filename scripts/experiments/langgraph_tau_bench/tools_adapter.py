"""Wrap tau-bench retail tools as LangChain StructuredTools.

Each tau-bench tool is a class with a static ``invoke(data, **kwargs)`` method
and a ``get_info()`` method returning an OpenAI-style function schema. This
module turns each into a LangChain ``StructuredTool`` whose function closes
over a per-task copy of ``env.data`` so tool calls don't leak across runs.

The agent only uses these tools to capture the *trajectory* (sequence of
tool_use blocks). Whether the tool calls were correct (i.e. matched the
gold ``Task.actions``) is out of scope for consistency evaluation.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, List, Type

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model


_JSON_TYPE_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _py_type_for(prop_schema: dict) -> type:
    """Best-effort JSON-schema → Python type. Falls back to ``Any``."""
    t = prop_schema.get("type")
    if isinstance(t, list):
        # Union types (e.g. ["string", "null"]) — collapse to the first
        # non-null type. Pydantic handles None via Optional anyway.
        for entry in t:
            if entry != "null":
                return _JSON_TYPE_TO_PY.get(entry, Any)
        return Any
    return _JSON_TYPE_TO_PY.get(t, Any)


def _make_args_model(name: str, schema: dict) -> Type:
    """Build a Pydantic model from a JSON-schema ``object`` definition."""
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict = {}
    for prop, sub in properties.items():
        py_type = _py_type_for(sub)
        description = sub.get("description", "")
        if prop in required:
            fields[prop] = (py_type, Field(..., description=description))
        else:
            fields[prop] = (py_type, Field(default=None, description=description))
    return create_model(f"{name}_Args", **fields)  # type: ignore[call-overload]


def wrap_tau_tool(tool_cls: Type, data_ref: dict) -> StructuredTool:
    """Wrap one tau-bench tool class as a LangChain StructuredTool.

    Args:
        tool_cls: Class with classmethod ``invoke(data, **kwargs) -> str``
            and classmethod ``get_info() -> dict`` (OpenAI schema).
        data_ref: A mutable dict — the per-task copy of ``env.data``. The
            tool will mutate this in place.

    Returns:
        StructuredTool that the LangGraph agent can call.
    """
    info = tool_cls.get_info()["function"]
    name = info["name"]
    description = info.get("description", "") or f"Tool: {name}"
    schema = info.get("parameters", {}) or {}
    args_model = _make_args_model(name, schema)

    def _run(**kwargs) -> str:
        # tau-bench tool invocations may raise on bad args; let them propagate
        # — LangGraph turns them into ToolMessage.error and the agent retries.
        return tool_cls.invoke(data_ref, **kwargs)

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=description,
        args_schema=args_model,
    )


def build_retail_tools(data_ref: dict) -> List[StructuredTool]:
    """Build all retail-domain StructuredTools bound to ``data_ref``.

    Imports tau-bench at call time so the rest of the codebase doesn't need
    it as a hard dependency.
    """
    from tau_bench.envs.retail.tools import ALL_TOOLS  # type: ignore

    return [wrap_tau_tool(cls, data_ref) for cls in ALL_TOOLS]


def fresh_retail_data() -> dict:
    """Return a deep copy of tau-bench's retail seed DB.

    Used per (task, run) to avoid state leakage from prior tool calls.
    """
    from tau_bench.envs.retail.data import load_data  # type: ignore

    return copy.deepcopy(load_data())
