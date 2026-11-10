"""LangSmith adapters against a real model on Bedrock (no LangSmith account needed).

    python scripts/integration_checks/langsmith_real_llm.py [--reps 5]

Runs langsmith.evaluate(upload_results=False) with sted_evaluator (vs. reference calls)
and sted_consistency_evaluator (across num_repetitions), in whatever text mode sted
picks (embeddings when installed). Prints per-run and summary results.
"""
import argparse
import datetime
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("LANGSMITH_TRACING", "false")

from langsmith import Client, evaluate  # noqa: E402
from langsmith.schemas import Example  # noqa: E402

from _bedrock_agent import MODEL_ID, PROMPTS, call  # noqa: E402
from sted.integrations.langsmith import sted_consistency_evaluator, sted_evaluator  # noqa: E402

REFERENCES = {
    "flight": [{"name": "book_flight", "arguments": {"origin": "Seattle", "destination": "New York",
                                                     "date": "2026-11-13", "passengers": 2, "seat_preference": "window"}}],
    "weather": [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}],
    "hotel": [{"name": "search_hotels", "arguments": {"query": "quiet hotel near the Louvre", "city": "Paris",
                                                      "max_price": 300}}],
    "flight_exact": [{"name": "book_flight", "arguments": {"origin": "SEA", "destination": "JFK",
                                                           "date": "2026-12-01", "passengers": 1, "seat_preference": "aisle"}}],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    ds, now = uuid.uuid4(), datetime.datetime.now(datetime.timezone.utc)
    examples = [Example(id=uuid.uuid4(), dataset_id=ds, created_at=now, inputs={"case": k, "prompt": p},
                        outputs={"calls": REFERENCES[k]}) for k, p in PROMPTS.items()]
    t = time.time()
    res = evaluate(lambda inputs: {"calls": call(inputs["prompt"])}, data=examples,
                   evaluators=[sted_evaluator()], summary_evaluators=[sted_consistency_evaluator()],
                   num_repetitions=args.reps, upload_results=False, client=Client(api_key="offline"),
                   max_concurrency=4)
    rows = list(res)
    print(f"model {MODEL_ID}, {len(rows)} runs in {time.time() - t:.0f}s")
    for r in rows:
        ev = {e.key: e for e in r["evaluation_results"]["results"]}
        same = ev["sted_same"]
        print(f"  {r['example'].inputs['case']:<13} same={same.score} min={ev['sted_min_readout'].score:.3f} "
              f"mean={ev['sted_mean_readout'].score:.3f}  {(same.comment or '').splitlines()[1:2]}")
    print("summary:")
    for s in res._summary_results["results"]:
        print(f"  {s.key:<26} {s.score!r}")
        if s.comment:
            print("    " + s.comment.replace("\n", "\n    "))
    return rows, res


if __name__ == "__main__":
    main()
