"""Inspect AI scorer and epoch reducer backed by STED.

    from inspect_ai import Task, Epochs
    from sted.integrations.inspect_ai import sted_match, sted_agreement

    Task(
        dataset=...,                      # targets: the expected JSON / tool call as text
        solver=...,
        scorer=sted_match(ignore=["request_id"]),
        epochs=Epochs(5, [sted_agreement(ignore=["request_id"]), "mean"]),
    )

``sted_match`` scores a sample's output against its target: CORRECT when the outputs are
the same under STED's min readout, with the differing fields in the explanation and
``min_readout`` / ``mean_readout`` / ``differences`` in the metadata. The structured
output is read from the completion text (``answer_from="completion"``, JSON, optionally
inside a ```json fence) or from the model's tool calls (``answer_from="tool_calls"``:
``[{"name": ..., "arguments": ...}]``). The parsed output is stored in ``Score.answer``.

``sted_agreement`` is an epoch reducer. Inspect's built-in reducers aggregate the epochs'
*scores*; this compares the epochs' *outputs* (``Score.answer``) with each other and
returns the share of same pairs as the value, with validity, agreement and the unstable
fields in the metadata and explanation. It works with any scorer that sets
``Score.answer`` to the output (``sted_match`` does).
"""
import json
import re
from typing import Any, List, Optional

from .. import api
from ._common import compare_kwargs, diff_comment

__all__ = ["sted_match", "sted_agreement"]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _from_completion(text: str) -> Any:
    m = _FENCE.search(text or "")
    body = m.group(1) if m else (text or "")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None


def _from_tool_calls(state) -> Any:
    calls = getattr(getattr(state.output, "message", None), "tool_calls", None) or []
    return [{"name": c.function, "arguments": c.arguments} for c in calls] or None


def sted_match(*, answer_from: str = "completion", threshold: float = api.DEFAULT_THRESHOLD,
               ignore=None, unordered=None, tolerance=None, rules=None, embedding_model: Optional[str] = None):
    """Inspect scorer: the sample's structured output vs. its target (JSON text)."""
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer, stderr

    if answer_from not in ("completion", "tool_calls"):
        raise ValueError(f"answer_from must be 'completion' or 'tool_calls', got {answer_from!r}")
    kw = compare_kwargs(threshold, rules, ignore, unordered, tolerance, embedding_model)

    @scorer(metrics=[accuracy(), stderr()], name="sted_match")
    def _make():
        async def score(state, target) -> Score:
            actual = _from_tool_calls(state) if answer_from == "tool_calls" else _from_completion(state.output.completion)
            answer = json.dumps(actual, ensure_ascii=False, sort_keys=True) if actual is not None else (state.output.completion or "")
            if actual is None:
                return Score(value=INCORRECT, answer=answer, explanation=f"output is not valid JSON ({answer_from})")
            try:
                expected = json.loads(target.text)
            except (json.JSONDecodeError, ValueError):
                return Score(value=INCORRECT, answer=answer, explanation="target is not valid JSON")
            r = api.compare(expected, actual, **kw)
            return Score(value=CORRECT if r.same else INCORRECT, answer=answer, explanation=diff_comment(r),
                         metadata={"min_readout": r.min_readout, "mean_readout": r.mean_readout,
                                   "differences": r.to_dict()["differences"]})
        return score

    return _make()


def sted_agreement(*, threshold: float = api.DEFAULT_THRESHOLD, ignore=None, unordered=None, tolerance=None,
                   rules=None, embedding_model: Optional[str] = None):
    """Inspect epoch reducer: consistency of the epochs' outputs with each other."""
    from inspect_ai.scorer import Score, score_reducer

    kw = compare_kwargs(threshold, rules, ignore, unordered, tolerance, embedding_model)

    @score_reducer(name="sted_agreement")
    def _make():
        def reduce(scores: List[Score]) -> Score:
            answers = [s.answer for s in scores]
            if len(answers) < 2:
                return Score(value=float("nan"), explanation="needs at least 2 epochs")
            r = api.consistency(answers, **kw)
            value = r.share_same if r.share_same is not None else 0.0
            return Score(value=value, answer=answers[0], explanation=str(r),
                         metadata={"share_same": r.share_same, "agreement_min": r.agreement_min,
                                   "validity": r.validity,
                                   "unstable_fields": [u.path for u in r.unstable_fields]})
        return reduce

    return _make()
