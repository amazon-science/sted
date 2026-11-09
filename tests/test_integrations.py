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
        import inspect_ai._eval.eval  # noqa: F401
        from inspect_ai.agent._acp import server  # noqa: F401  (fails on agent-client-protocol 1.x)
    except ImportError as e:
        pytest.skip(f"inspect_ai not importable in this environment: {e}")

    seen = collections.Counter()

    def outputs(messages, tools, tool_choice, config):
        text = messages[-1].text
        seen[text] += 1
        body = WEATHER if "weather" in text else call("2026-12-02" if seen[text] == 2 else "2026-12-01", f"r-{seen[text]}")
        return ModelOutput.from_content(model="mockllm", content="```json\n" + json.dumps(body) + "\n```")

    log = inspect_eval(_inspect_task(), model=get_model("mockllm/model", custom_outputs=outputs),
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
