"""LangSmith evaluators backed by STED.

    from langsmith import evaluate
    from sted.integrations.langsmith import sted_evaluator, sted_consistency_evaluator

    evaluate(
        my_agent, data="booking-dataset",
        evaluators=[sted_evaluator(ignore=["request_id"])],              # vs. reference outputs
        summary_evaluators=[sted_consistency_evaluator(ignore=["request_id"])],
        num_repetitions=5,                                               # runs per example
    )

``sted_evaluator`` compares each run's output with the example's reference output and
returns ``sted_same`` (1/0), ``sted_min_readout`` and ``sted_mean_readout``; the comment on
``sted_same`` lists the fields that differ.

``sted_consistency_evaluator`` is a summary evaluator. It groups the experiment's runs by
example (the repetitions of ``num_repetitions``), compares each example's outputs with
each other (no reference needed) and returns ``sted_share_same`` (mean share of same
pairs), ``sted_agreement`` (mean pairwise min readout), ``sted_consistent_examples``
(share of examples whose repetitions are all the same) and, in the comment, the fields
that are unstable in how many examples. LangSmith's own repetition view averages scores;
this compares the outputs themselves.

Outputs and references are LangSmith dicts. ``output_key`` / ``reference_key`` pick the
structured value from them; by default a single-key dict yields its value and anything
else is compared whole. Field rules (``ignore``, ``unordered``, ``tolerance``, ``rules``)
and ``threshold`` are those of :func:`sted.compare`.
"""
import json
import statistics
from typing import Any, Dict, List, Optional

from .. import api
from ._common import compare_kwargs, diff_comment, pick, unstable_summary

__all__ = ["sted_evaluator", "sted_consistency_evaluator"]


def sted_evaluator(*, output_key: Optional[str] = None, reference_key: Optional[str] = None,
                   threshold: float = api.DEFAULT_THRESHOLD, ignore=None, unordered=None, tolerance=None,
                   rules=None, embedding_model: Optional[str] = None):
    """A LangSmith row evaluator ``(outputs, reference_outputs) -> results``."""
    kw = compare_kwargs(threshold, rules, ignore, unordered, tolerance, embedding_model)
    ref_key = reference_key if reference_key is not None else output_key

    def sted_match(outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> Dict[str, List[dict]]:
        if not reference_outputs:
            return {"results": [{"key": "sted_same", "score": None,
                                 "comment": "example has no reference outputs"}]}
        actual, expected = pick(outputs or {}, output_key), pick(reference_outputs, ref_key)
        try:
            r = api.compare(expected, actual, **kw)
        except ValueError as e:                       # e.g. output text that is not JSON
            return {"results": [{"key": "sted_same", "score": 0, "comment": str(e)}]}
        return {"results": [
            {"key": "sted_same", "score": int(r.same), "comment": diff_comment(r)},
            {"key": "sted_min_readout", "score": r.min_readout},
            {"key": "sted_mean_readout", "score": r.mean_readout},
        ]}

    return sted_match


def sted_consistency_evaluator(*, output_key: Optional[str] = None, threshold: float = api.DEFAULT_THRESHOLD,
                               ignore=None, unordered=None, tolerance=None, rules=None,
                               embedding_model: Optional[str] = None):
    """A LangSmith summary evaluator ``(runs, examples) -> results`` over repetitions."""
    kw = compare_kwargs(threshold, rules, ignore, unordered, tolerance, embedding_model)

    def sted_consistency(runs, examples) -> Dict[str, List[dict]]:
        groups: Dict[str, List[Any]] = {}
        for run, example in zip(runs, examples):
            key = _label(example, run)
            try:
                groups.setdefault(key, []).append(pick(run.outputs or {}, output_key))
            except KeyError:
                groups.setdefault(key, []).append(None)          # counts as an invalid output
        results = [(gid, api.consistency(outs, **kw)) for gid, outs in groups.items() if len(outs) >= 2]
        if not results:
            return {"results": [{"key": "sted_share_same", "score": None,
                                 "comment": "needs at least 2 runs per example (num_repetitions >= 2)"}]}
        scored = [r for _, r in results if r.share_same is not None]
        _, comment = unstable_summary(results)
        mean = lambda xs: statistics.mean(xs) if xs else None  # noqa: E731
        return {"results": [
            {"key": "sted_share_same", "score": mean([r.share_same for r in scored]), "comment": comment},
            {"key": "sted_agreement", "score": mean([r.agreement_min for r in scored])},
            {"key": "sted_consistent_examples",
             "score": sum(r.share_same == 1.0 for r in scored) / len(results)},
            {"key": "sted_validity", "score": mean([r.validity for _, r in results])},
        ]}

    return sted_consistency


def _label(example, run) -> str:
    """Group key shown in comments: the example's inputs, shortened, plus a short id."""
    eid = str(getattr(example, "id", None) or getattr(run, "reference_example_id", None) or "")
    inputs = getattr(example, "inputs", None)
    text = json.dumps(inputs, default=str, ensure_ascii=False) if inputs else ""
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{text} [{eid[:8]}]" if text else eid
