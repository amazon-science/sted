"""Ragas (0.4): STED as a custom numeric metric.

    pip install sted ragas
    python ragas_example.py

Ragas 0.4.3 imports `langchain_community.chat_models.vertexai`, removed in
langchain-community 0.4; if `import ragas` fails, `pip install "langchain-community<0.4"`.
"""
import json

try:
    from ragas.metrics import MetricResult, numeric_metric
except ImportError as e:      # Ragas < 0.4 has a different custom-metric API
    raise SystemExit(f"this example needs ragas>=0.4 ({e}); pip install -U ragas") from e

import sted
from _fake_agent import REFERENCE, book_flight

RULES = {"ignore": ["request_id"]}


@numeric_metric(name="sted_min_readout", allowed_values=(0.0, 1.0))
def sted_min_readout(response: str, reference: str) -> MetricResult:
    """Lowest field score of the response vs. the reference; the differing fields are the reason."""
    r = sted.compare(reference, response, **RULES)
    return MetricResult(value=r.min_readout, reason=str(r))


if __name__ == "__main__":
    scores = [sted_min_readout.score(response=json.dumps(book_flight(run)), reference=json.dumps(REFERENCE))
              for run in (0, 1)]
    for s in scores:
        print(s.value, "|", s.reason.splitlines()[0])
    assert [s.value for s in scores] == [1.0, 0.0] and "arguments.date" in scores[1].reason
    print("ok")
