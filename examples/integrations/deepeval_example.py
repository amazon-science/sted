"""DeepEval: STED as a custom metric.

    pip install sted deepeval
    DEEPEVAL_TELEMETRY_OPT_OUT=YES python deepeval_example.py      # or: deepeval test run

Note: DeepEval copies metrics by calling __init__ with the attributes that share a
parameter's name (deepeval.metrics.utils.copy_metrics). Keep every parameter explicit
and stored under its own name, or it is silently dropped in the copy.
"""
import json
import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import evaluate  # noqa: E402
from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

import sted  # noqa: E402
from _fake_agent import REFERENCE, book_flight  # noqa: E402


class StedMatch(BaseMetric):
    def __init__(self, threshold: float = 0.95, ignore=None, unordered=None, tolerance=None):
        self.threshold, self.ignore, self.unordered, self.tolerance = threshold, ignore, unordered, tolerance

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        r = sted.compare(test_case.expected_output, test_case.actual_output, threshold=self.threshold,
                         ignore=self.ignore, unordered=self.unordered, tolerance=self.tolerance)
        self.score, self.success, self.reason = r.min_readout, r.same, str(r)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "STED match"


if __name__ == "__main__":
    cases = [LLMTestCase(input="Book SEA to JFK on Dec 1 for 2", actual_output=json.dumps(book_flight(run)),
                         expected_output=json.dumps(REFERENCE)) for run in (0, 1)]
    result = evaluate(test_cases=cases, metrics=[StedMatch(ignore=["request_id"])])
    verdicts = [t.success for t in result.test_results]
    print(verdicts, result.test_results[1].metrics_data[0].reason)
    assert sorted(verdicts) == [False, True]
    # DeepEval reruns tests (`deepeval test run -r N`) but has no cross-run metric:
    print(sted.consistency([book_flight(run) for run in range(3)], ignore=["request_id"]))
    print("ok")
