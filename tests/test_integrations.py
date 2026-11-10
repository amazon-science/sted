"""LangSmith and Inspect AI adapters, end to end and offline (character mode).

LangSmith: `evaluate(..., upload_results=False)` on local examples with num_repetitions.
Inspect: `eval()` with the mockllm provider and Epochs. Each block skips if the framework
is not installed."""
import collections
import datetime
import itertools
import json
import os
import uuid

import pytest

os.environ.setdefault("STED_EMBEDDING_MODEL", "none")


def call(date, rid):
    return {"name": "book_flight", "arguments": {"to": "JFK", "date": date, "request_id": rid}}


WEATHER = {"name": "get_weather", "arguments": {"city": "Seattle"}}


# ------------------------------------------------------------------ LangSmith

def test_langsmith_row_evaluator_directly():
    from sted.integrations.langsmith import sted_evaluator
    ev = sted_evaluator(ignore=["request_id"])
    out = ev({"call": call("2026-12-02", "x")}, {"call": call("2026-12-01", "y")})
    res = {r["key"]: r for r in out["results"]}
    assert res["sted_same"]["score"] == 0 and "arguments.date" in res["sted_same"]["comment"]
    assert ev({"call": call("2026-12-01", "x")}, {"call": call("2026-12-01", "y")})["results"][0]["score"] == 1
    assert ev({"call": "not json"}, {"call": call("2026-12-01", "y")})["results"][0]["score"] == 0
    assert ev({"call": WEATHER}, {})["results"][0]["score"] is None


def test_langsmith_evaluate_offline_with_repetitions():
    ls = pytest.importorskip("langsmith")
    from langsmith import Client, evaluate
    from langsmith.schemas import Example
    from sted.integrations.langsmith import sted_consistency_evaluator, sted_evaluator

    os.environ["LANGSMITH_TRACING"] = "false"
    ds, now = uuid.uuid4(), datetime.datetime.now(datetime.timezone.utc)
    examples = [Example(id=uuid.uuid4(), dataset_id=ds, created_at=now, inputs={"q": q}, outputs={"call": ref})
                for q, ref in (("book dec 1", call("2026-12-01", "ref")), ("weather", WEATHER))]
    counter = itertools.count()

    def target(inputs):
        k = next(counter)
        if inputs["q"] == "weather":
            return {"call": WEATHER}
        return {"call": call("2026-12-01" if k % 3 else "2026-12-02", f"r-{k}")}

    res = evaluate(target, data=examples, evaluators=[sted_evaluator(ignore=["request_id"])],
                   summary_evaluators=[sted_consistency_evaluator(ignore=["request_id"])],
                   num_repetitions=3, upload_results=False, client=Client(api_key="offline-test"),
                   max_concurrency=0)
    rows = list(res)
    assert len(rows) == 6
    same = [next(e.score for e in r["evaluation_results"]["results"] if e.key == "sted_same") for r in rows]
    assert sorted(same) == [0, 1, 1, 1, 1, 1]
    summary = {s.key: s for s in res._summary_results["results"]}
    assert summary["sted_share_same"].score == pytest.approx((1 / 3 + 1.0) / 2, abs=1e-6)
    assert summary["sted_consistent_examples"].score == 0.5
    assert "arguments.date: unstable in 1 of 2" in summary["sted_share_same"].comment
    assert '"book dec 1"' in summary["sted_share_same"].comment


def test_langsmith_consistency_needs_repetitions():
    from types import SimpleNamespace as NS
    from sted.integrations.langsmith import sted_consistency_evaluator
    ev = sted_consistency_evaluator()
    out = ev([NS(outputs={"call": WEATHER})], [NS(id="e1", inputs={"q": "w"})])
    assert out["results"][0]["score"] is None and "num_repetitions" in out["results"][0]["comment"]


# ------------------------------------------------------------------ Inspect AI

def _mock_model(outputs):
    """mockllm with callable outputs (inspect_ai >= ~0.3.110); skip on older releases."""
    from inspect_ai.model import get_model
    try:
        return get_model("mockllm/model", custom_outputs=outputs)
    except ValueError as e:
        pytest.skip(f"this inspect_ai's mockllm does not take callable outputs: {e}")


def test_inspect_adapter_api_on_any_supported_version(tmp_path):
    """Scorer + reducer against the Inspect API itself, with generator outputs that every
    supported inspect_ai (>= 0.3.100) accepts."""
    pytest.importorskip("inspect_ai")
    try:
        _check_acp()
        from inspect_ai import Epochs, Task, eval as inspect_eval
        from inspect_ai.dataset import Sample
        from inspect_ai.model import ModelOutput, get_model
        from inspect_ai.solver import generate
    except ImportError as e:
        pytest.skip(f"inspect_ai not importable in this environment: {e}")
    from sted.integrations.inspect_ai import sted_agreement, sted_match
    gen = (ModelOutput.from_content(model="mockllm", content=json.dumps(WEATHER)) for _ in itertools.count())
    task = Task(dataset=[Sample(input="w", target=json.dumps(WEATHER))], solver=generate(), scorer=sted_match(),
                epochs=Epochs(3, [sted_agreement(), "mean"]))
    log = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=gen), display="none",
                       log_dir=str(tmp_path))[0]
    assert log.status == "success"
    assert {r.reducer: [float(x.value) for x in r.samples] for r in log.reductions} == \
        {"sted_agreement": [1.0], "mean": [1.0]}


def _check_acp():
    """Recent inspect_ai imports agent-client-protocol at eval time and breaks on its 1.x;
    older inspect_ai has no ACP module at all, which is fine."""
    import importlib.util
    if importlib.util.find_spec("inspect_ai.agent._acp") is not None:
        from inspect_ai.agent._acp import server  # noqa: F401

def _inspect_task(epochs=3):
    from inspect_ai import Epochs, Task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate
    from sted.integrations.inspect_ai import sted_agreement, sted_match
    return Task(dataset=[Sample(input="book SEA-JFK dec 1", target=json.dumps(call("2026-12-01", "ref"))),
                         Sample(input="weather in Seattle", target=json.dumps(WEATHER))],
                solver=generate(), scorer=sted_match(ignore=["request_id"]),
                epochs=Epochs(epochs, [sted_agreement(ignore=["request_id"]), "mean"]))


def test_inspect_eval_with_epoch_reducer(tmp_path):
    pytest.importorskip("inspect_ai")
    try:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model
        _check_acp()
    except ImportError as e:
        pytest.skip(f"inspect_ai not importable in this environment: {e}")

    seen = collections.Counter()

    def outputs(messages, tools, tool_choice, config):
        text = messages[-1].text
        seen[text] += 1
        body = WEATHER if "weather" in text else call("2026-12-02" if seen[text] == 2 else "2026-12-01", f"r-{seen[text]}")
        return ModelOutput.from_content(model="mockllm", content="```json\n" + json.dumps(body) + "\n```")

    log = inspect_eval(_inspect_task(), model=_mock_model(outputs),
                       display="none", log_dir=str(tmp_path))[0]
    assert log.status == "success"
    by_reducer = {r.reducer: {s.sample_id: float(s.value) for s in r.samples} for r in log.reductions}
    assert by_reducer["mean"] == {1: pytest.approx(2 / 3), 2: 1.0}            # 2 of 3 epochs correct
    assert by_reducer["sted_agreement"] == {1: pytest.approx(1 / 3), 2: 1.0}  # 1 of 3 epoch pairs the same
    s0 = next(s for s in log.samples if s.id == 1 and s.epoch == 2)
    assert s0.scores["sted_match"].value == "I" and "arguments.date" in s0.scores["sted_match"].explanation


def test_inspect_completion_parsing():
    pytest.importorskip("inspect_ai")
    from sted.integrations.inspect_ai import _from_completion
    assert _from_completion('```json\n{"a": 1}\n```') == {"a": 1}
    assert _from_completion('{"a": [1, 2]}') == {"a": [1, 2]}
    assert _from_completion("Sorry, no.") is None


def test_inspect_tool_calls_with_default_tool_loop(tmp_path):
    """answer_from="tool_calls" under generate()'s default loop: the model calls a real tool,
    gets the result, then answers in text, so the call is not in state.output."""
    pytest.importorskip("inspect_ai")
    try:
        from inspect_ai import Epochs, Task, eval as inspect_eval
        _check_acp()
        from inspect_ai.dataset import Sample
        from inspect_ai.model import ModelOutput, get_model
        from inspect_ai.solver import generate, use_tools
        from inspect_ai.tool import tool
    except ImportError as e:
        pytest.skip(f"inspect_ai not importable in this environment: {e}")
    from sted.integrations.inspect_ai import sted_agreement, sted_match

    @tool
    def book_flight():
        async def execute(to: str, date: str, request_id: str):
            """Book a flight.

            Args:
                to: destination airport
                date: travel date
                request_id: idempotency key
            """
            return "booked"
        return execute

    n = collections.Counter()

    def outputs(messages, tools, tool_choice, config):
        if messages[-1].role == "tool":                      # after the tool ran: final reply
            return ModelOutput.from_content(model="mockllm", content="Booked.")
        n["calls"] += 1
        date = "2026-12-02" if n["calls"] == 2 else "2026-12-01"
        return ModelOutput.for_tool_call(model="mockllm", tool_name="book_flight",
                                         tool_arguments={"to": "JFK", "date": date, "request_id": f"r{n['calls']}"})

    target = json.dumps([{"name": "book_flight", "arguments": {"to": "JFK", "date": "2026-12-01", "request_id": "x"}}])
    task = Task(dataset=[Sample(input="book SEA-JFK dec 1", target=target)],
                solver=[use_tools(book_flight()), generate()],               # default tool_calls="loop"
                scorer=sted_match(answer_from="tool_calls", ignore=["request_id"]),
                epochs=Epochs(3, [sted_agreement(ignore=["request_id"]), "mean"]))
    log = inspect_eval(task, model=_mock_model(outputs),
                       display="none", log_dir=str(tmp_path))[0]
    assert log.status == "success"
    assert all(s.output.completion == "Booked." for s in log.samples)        # final text, no tool call
    by_reducer = {r.reducer: [float(x.value) for x in r.samples] for r in log.reductions}
    assert by_reducer["mean"] == [pytest.approx(2 / 3)]
    assert by_reducer["sted_agreement"] == [pytest.approx(1 / 3)]
