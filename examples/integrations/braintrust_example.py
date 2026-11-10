"""Braintrust: a STED code scorer, plus consistency across trials.

    pip install sted braintrust
    python braintrust_example.py        # offline: no_send_logs=True, no API key needed

Remove no_send_logs to log the experiment to Braintrust.
"""
import itertools
import json

from braintrust import Eval

import sted
from _fake_agent import REAL, REFERENCE, book_flight

RULES = {"ignore": ["request_id"]}


def sted_same(input, output, expected):
    """Braintrust scorer: 1 if output and expected are the same, differences in metadata."""
    try:
        r = sted.compare(expected, output, **RULES)
    except ValueError as e:              # the output is not JSON, e.g. a text reply
        return {"name": "sted_same", "score": 0.0, "metadata": {"error": str(e)}}
    return {"name": "sted_same", "score": float(r.same),
            "metadata": {"min_readout": r.min_readout, "differences": r.to_dict()["differences"]}}


def trial_consistency(result) -> dict:
    """Compare each example's trial outputs with each other (Braintrust averages scores)."""
    groups = {}
    for row in result.results:
        groups.setdefault(json.dumps(row.input, sort_keys=True, default=str), []).append(row.output)
    return {inp: sted.consistency(outs, **RULES) for inp, outs in groups.items() if len(outs) > 1}


if __name__ == "__main__":
    runs = itertools.count()
    result = Eval("sted-example", data=[{"input": "Book SEA to JFK on Dec 1 for 2", "expected": REFERENCE}],
                  task=lambda input: book_flight(next(runs)), scores=[sted_same],
                  trial_count=3, no_send_logs=True)
    for inp, c in trial_consistency(result).items():
        print(inp, "\n", c)
        if not REAL:
            assert abs(c.share_same - 1 / 3) < 1e-6 and [u.path for u in c.unstable_fields] == ["arguments.date"]
    print("sted_same per trial:", [r.scores["sted_same"] for r in result.results])
    if not REAL:
        assert sorted(r.scores["sted_same"] for r in result.results) == [0.0, 1.0, 1.0]
    assert len(result.results) == 3 and all(r.scores["sted_same"] in (0.0, 1.0) for r in result.results)
    print("ok")
