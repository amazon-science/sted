"""MLflow 3 / Databricks: STED as a custom GenAI scorer.

    pip install sted mlflow
    python mlflow_example.py           # local sqlite tracking store, no server

On Databricks, drop set_tracking_uri; the same scorer runs in mlflow.genai.evaluate.
"""
import os
import tempfile

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

import sted
from _fake_agent import REAL, REFERENCE, book_flight


@scorer
def sted_same(outputs, expectations) -> Feedback:
    try:
        r = sted.compare(expectations["expected_call"], outputs, ignore=["request_id"])
    except ValueError as e:              # the output is not JSON, e.g. a text reply
        return Feedback(value=False, rationale=str(e))
    return Feedback(value=r.same, rationale=str(r), metadata={"min_readout": r.min_readout})


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(tempfile.mkdtemp(), "mlflow.db"))
    data = [{"inputs": {"q": "Book SEA to JFK on Dec 1 for 2"}, "outputs": book_flight(run),
             "expectations": {"expected_call": REFERENCE}} for run in (0, 1)]
    result = mlflow.genai.evaluate(data=data, scorers=[sted_same])
    print(result.metrics)
    for _, row in result.result_df.iterrows():
        print(row.get("sted_same/value"), str(row.get("sted_same/rationale", "")).splitlines()[:2])
    if not REAL:
        assert result.metrics["sted_same/mean"] == 0.5
    assert 0.0 <= result.metrics["sted_same/mean"] <= 1.0
    print("ok")
