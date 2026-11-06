"""Tests for the STED MCP server: the tool functions directly, and the server end to
end over stdio through the official MCP client (the transport MCP hosts launch)."""
from __future__ import annotations

import asyncio
import json
import sys

import pytest

pytest.importorskip("mcp")

from sted import mcp_server as ms  # noqa: E402

REF = {"name": "get_weather", "arguments": {"city": "Seattle", "days": 3, "units": "metric"}}
REORDERED = {"arguments": {"units": "metric", "days": "3", "city": "Seattle"}, "name": "get_weather"}
CHANGED = {"name": "get_weather", "arguments": {"city": "Seattle", "days": 5, "units": "metric"}}


def test_compare_identical_up_to_key_order_and_numeric_string():
    r = ms.compare_outputs(REF, REORDERED)
    assert r["min_readout"] == 1.0 and r["mean_readout"] == 1.0 and r["same"] is True


def test_compare_changed_argument_min_rejects_mean_dips():
    r = ms.compare_outputs(REF, CHANGED)
    assert r["min_readout"] == 0.0
    assert 0.5 < r["mean_readout"] < 1.0
    assert r["same"] is False


def test_compare_accepts_json_text():
    assert ms.compare_outputs(json.dumps(REF), json.dumps(REORDERED))["same"] is True


def test_compare_rejects_non_json_text_and_bad_threshold():
    with pytest.raises(ValueError, match="not valid JSON"):
        ms.compare_outputs("not json {", REF)
    with pytest.raises(ValueError, match="threshold"):
        ms.compare_outputs(REF, REF, threshold=1.5)


def test_evaluate_consistency_counts_validity_and_finds_worst_pair():
    outputs = [REF, REORDERED, "garbage {", CHANGED]
    r = ms.evaluate_consistency(outputs)
    assert r["n_outputs"] == 4 and r["n_valid"] == 3
    assert r["validity"] == 0.75
    assert r["n_pairs"] == 3
    # REF~REORDERED same; both differ from CHANGED in one argument
    assert r["share_same"] == pytest.approx(1 / 3, abs=1e-6)
    assert r["agreement_min"] == pytest.approx(1 / 3, abs=1e-6)
    assert r["agreement_min"] < r["agreement_mean"] < 1.0
    worst = r["least_agreeing_pair"]
    assert worst["min_readout"] == 0.0
    assert 3 in (worst["i"], worst["j"])        # indices refer to the input list
    assert 2 not in (worst["i"], worst["j"])    # the invalid output is never paired


def test_evaluate_consistency_too_few_valid():
    r = ms.evaluate_consistency([REF, "nope"])
    assert r["n_valid"] == 1 and r["agreement_min"] is None


def test_evaluate_consistency_input_limits():
    with pytest.raises(ValueError):
        ms.evaluate_consistency([REF])
    with pytest.raises(ValueError):
        ms.evaluate_consistency([REF] * (ms.MAX_OUTPUTS + 1))


def test_stdio_server_end_to_end():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "sted.mcp_server"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                assert tools == {"compare_outputs", "evaluate_consistency"}
                res = await session.call_tool("compare_outputs", {"output_a": REF, "output_b": CHANGED})
                assert not res.isError
                payload = json.loads(res.content[0].text)
                assert payload["same"] is False and payload["min_readout"] == 0.0
                res = await session.call_tool("evaluate_consistency", {"outputs": [REF, REORDERED, CHANGED]})
                assert not res.isError
                assert json.loads(res.content[0].text)["validity"] == 1.0

    asyncio.run(asyncio.wait_for(run(), timeout=240))
