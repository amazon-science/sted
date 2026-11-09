"""W&B Weave: a STED scorer per row and consistency across `trials`.

    pip install sted weave
    python weave_example.py             # offline: WEAVE_DISABLED=true, no W&B login

Call weave.init("your-project") instead to log to W&B.
"""
import asyncio
import itertools
import json
import os

os.environ.setdefault("WEAVE_DISABLED", "true")

import weave  # noqa: E402
from weave import Scorer  # noqa: E402

import sted  # noqa: E402
from _fake_agent import REFERENCE, book_flight  # noqa: E402

RULES = {"ignore": ["request_id"]}
runs = itertools.count()


@weave.op
def agent(question: str) -> dict:
    return book_flight(next(runs))


class StedMatch(Scorer):
    """Each row vs. its expected call."""

    @weave.op
    def score(self, *, output, expected: dict) -> dict:
        r = sted.compare(expected, output, **RULES)
        return {"same": r.same, "min_readout": r.min_readout,
                "differences": [d.path for d in r.differences]}


class StedConsistency(Scorer):
    """Rows of the same question across trials, compared with each other in summarize()."""

    @weave.op
    def score(self, *, output, question: str) -> dict:
        return {"question": question, "output": json.dumps(output, sort_keys=True)}

    def summarize(self, score_rows: list) -> dict:
        groups = {}
        for row in score_rows:
            groups.setdefault(row["question"], []).append(row["output"])
        per = [sted.consistency(outs, **RULES) for outs in groups.values() if len(outs) > 1]
        return {"share_same": sum(c.share_same for c in per) / len(per),
                "unstable_fields": sorted({u.path for c in per for u in c.unstable_fields})}


if __name__ == "__main__":
    evaluation = weave.Evaluation(dataset=[{"question": "Book SEA to JFK on Dec 1 for 2", "expected": REFERENCE}],
                                  scorers=[StedMatch(), StedConsistency()], trials=3)
    summary = asyncio.run(evaluation.evaluate(agent))
    print(json.dumps({k: summary[k] for k in ("StedMatch", "StedConsistency")}, indent=2, default=str))
    assert abs(summary["StedConsistency"]["share_same"] - 1 / 3) < 1e-6
    assert summary["StedConsistency"]["unstable_fields"] == ["arguments.date"]
    print("ok")
