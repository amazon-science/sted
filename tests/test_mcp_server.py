"""Tests for the STED MCP server: the tool functions directly, and the server end to
end over stdio through the official MCP client (the transport MCP hosts launch)."""
from __future__ import annotations

import asyncio
import os
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


def test_compare_non_object_values():
    assert ms.compare_outputs({}, {})["same"] is True          # identical empties
    assert ms.compare_outputs([], [])["same"] is True
    assert ms.compare_outputs({}, {"a": 1})["min_readout"] == 0.0
    assert ms.compare_outputs([1, 2, 3], [1, 2, 3])["same"] is True
    assert ms.compare_outputs([1, 2, 3], [3, 2, 1])["same"] is False   # arrays are positional
    assert ms.compare_outputs(42, 42)["same"] is True


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


def test_stdio_server_end_to_end(tmp_path):
    """Launch `python -m sted.mcp_server` over stdio and call both tools through the
    official client. Results are collected inside the session and asserted after it
    closes, so a failing check is reported as itself rather than as the client's
    BrokenResourceError on teardown; the server's stderr is attached on failure."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    calls = [
        ("compare_outputs", {"output_a": REF, "output_b": CHANGED}),
        ("evaluate_consistency", {"outputs": [REF, REORDERED, CHANGED]}),
        ("compare_outputs", {"output_a": {**REF, "request_id": "a", "tags": [1, 2]},
                             "output_b": {**REORDERED, "request_id": "b", "tags": [2, 1]},
                             "ignore": ["request_id"], "unordered": ["tags"]}),
    ]
    got = {}

    async def run(errlog):
        # The protocol path is what this test covers; embedding scoring is covered in-process
        # above. Character mode keeps the server start independent of torch load time.
        params = StdioServerParameters(command=sys.executable, args=["-m", "sted.mcp_server"],
                                       env={**os.environ, "STED_EMBEDDING_MODEL": "none"})
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                got["tools"] = {t.name for t in (await session.list_tools()).tools}
                for k, (name, args) in enumerate(calls):
                    res = await session.call_tool(name, args)
                    is_error = getattr(res, "isError", None)              # mcp 1.x
                    if is_error is None:
                        is_error = getattr(res, "is_error", False)        # mcp 2.x
                    got[k] = (bool(is_error), res.content[0].text if res.content else None)

    log_path = tmp_path / "server_stderr.log"
    error = None
    with open(log_path, "w") as errlog:
        try:
            asyncio.run(asyncio.wait_for(run(errlog), timeout=240))
        except BaseException as e:  # noqa: BLE001 - reported below with the server log
            error = e
    server_log = log_path.read_text()
    context = f"collected={got!r}\nserver stderr:\n{server_log[-4000:]}"
    if error is not None and len(got) < len(calls) + 1:
        raise AssertionError(f"MCP session failed: {error!r}\n{context}") from error
    assert got["tools"] == {"compare_outputs", "evaluate_consistency"}, context
    for k in range(len(calls)):
        assert got[k][0] is False, context
    first = json.loads(got[0][1])
    assert first["same"] is False and first["min_readout"] == 0.0, context
    assert json.loads(got[1][1])["validity"] == 1.0, context
    assert json.loads(got[2][1])["same"] is True, context


def test_compare_lists_field_differences():
    r = ms.compare_outputs(REF, {"name": "get_weather", "arguments": {"city": "Seattle, WA", "days": "5"}})
    by_path = {d["path"]: d for d in r["differences"]}
    assert by_path["arguments.days"] == {"path": "arguments.days", "status": "changed", "a": 3, "b": "5", "score": 0.0}
    assert by_path["arguments.units"]["status"] == "missing"
    assert 0.5 < by_path["arguments.city"]["score"] < 1.0
    assert r["min_readout"] == min(d["score"] for d in r["differences"])
    assert ms.compare_outputs(REF, REORDERED)["differences"] == []   # 3 vs "3" is not a difference


def test_compare_flags_changed_callee_name():
    r = ms.compare_outputs(REF, {**REF, "name": "get_weathers"})
    assert r["differences"][0]["path"] == "name" and r["differences"][0]["score"] == 0.0


def test_evaluate_consistency_unstable_fields():
    a = {"name": "search", "arguments": {"q": "hotels in Paris", "limit": 5}}
    b = {"name": "search", "arguments": {"q": "hotels in Paris", "limit": 10}}
    c = {"name": "search", "arguments": {"q": "hotels in Paris", "limit": 5, "sort": "price"}}
    r = ms.evaluate_consistency([a] * 6 + [b] * 3 + [c])          # 10 valid -> 45 pairs
    fields = {u["path"]: u for u in r["unstable_fields"]}
    assert fields["arguments.limit"]["share_of_pairs"] == pytest.approx(21 / 45, abs=1e-6)   # 7 x 3
    assert fields["arguments.sort"]["share_of_pairs"] == pytest.approx(9 / 45, abs=1e-6)     # 9 x 1
    assert {"value": 5, "count": 7} in fields["arguments.limit"]["values"]
    assert {"value": "<absent>", "count": 9} in fields["arguments.sort"]["values"]
    assert r["least_agreeing_pair"]["differences"]


def test_evaluate_consistency_unstable_fields_sampled_share_is_scaled():
    outs = [{"name": "search", "arguments": {"q": "hotels", "limit": 5 + i % 2}} for i in range(50)]
    r = ms.evaluate_consistency(outs)                              # 625 of 1,225 pairs disagree
    assert r["unstable_fields"][0]["share_of_pairs"] == pytest.approx(625 / 1225, abs=1e-6)
    assert "estimated" in r["note"]


def test_stdio_server_raw_jsonrpc():
    """The server speaks MCP over stdio, checked without the client library: newline-
    delimited JSON-RPC written to stdin, responses read from stdout. This is what any
    MCP host does, and it isolates the server from client-side transport issues."""
    import queue
    import subprocess
    import threading

    proc = subprocess.Popen([sys.executable, "-m", "sted.mcp_server"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ, "STED_EMBEDDING_MODEL": "none", "PYTHONUNBUFFERED": "1"})
    lines: "queue.Queue[bytes]" = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in iter(proc.stdout.readline, b"")], daemon=True).start()

    def send(msg):
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        proc.stdin.flush()

    def reply(msg_id, timeout=120):
        while True:
            try:
                line = lines.get(timeout=timeout)
            except queue.Empty:
                proc.kill()
                raise AssertionError(f"no reply to id {msg_id}; stderr:\n{proc.stderr.read().decode()[-3000:]}")
            msg = json.loads(line)              # every stdout line must be a JSON-RPC message
            if msg.get("id") == msg_id:
                return msg

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "raw-test", "version": "0"}}})
        assert reply(1)["result"]["serverInfo"]["name"] == "sted"
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert {t["name"] for t in reply(2)["result"]["tools"]} == {"compare_outputs", "evaluate_consistency"}
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "compare_outputs", "arguments": {"output_a": REF, "output_b": CHANGED}}})
        result = reply(3)["result"]
        assert result.get("isError") is not True, result
        payload = json.loads(result["content"][0]["text"])
        assert payload["same"] is False and payload["differences"][0]["path"] == "arguments.days"
        send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
              "params": {"name": "evaluate_consistency", "arguments": {"outputs": [REF, REORDERED, CHANGED]}}})
        assert json.loads(reply(4)["result"]["content"][0]["text"])["n_valid"] == 3
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
