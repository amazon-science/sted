"""Throughput of consistency scoring at evaluation-suite scale.

    python scripts/integration_checks/scale_benchmark.py [--examples 300] [--reps 5] [--mode none|auto]

Synthetic tool calls shaped like the real-model runs (a booking with a drifting date, a
weather call with a drifting `days` and optional `units`, a hotel search with a reworded
free-text query). Times sted.consistency per example and the LangSmith summary evaluator
over all runs.
"""
import argparse
import os
import random
import statistics
import sys
import time
from types import SimpleNamespace as NS


def outputs_for(kind: int, rng: random.Random):
    if kind == 0:
        return [{"name": "book_flight", "arguments": {"origin": "Seattle", "destination": "New York",
                 "date": rng.choice(["2026-11-13"] * 3 + ["2026-11-17", "2026-11-20"]), "passengers": 2,
                 "seat_preference": "window"}}]
    if kind == 1:
        args = {"city": "Paris", "days": rng.choice([3, 5, 7])}
        if rng.random() < .4:
            args["units"] = "metric"
        return [{"name": "get_weather", "arguments": args}]
    return [{"name": "search_hotels", "arguments": {"city": "Paris", "max_price": 300,
             "query": rng.choice(["quiet hotel near the Louvre", "quiet hotel near Louvre",
                                  "quiet hotels close to the Louvre museum"])}}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=300)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mode", default="none", help='"none" (character) or "auto" (embeddings if installed)')
    args = ap.parse_args()
    if args.mode == "none":
        os.environ["STED_EMBEDDING_MODEL"] = "none"
    import sted
    from sted.integrations.langsmith import sted_consistency_evaluator

    rng = random.Random(0)
    groups = [[outputs_for(i % 3, rng) for _ in range(args.reps)] for i in range(args.examples)]
    sted.consistency(groups[0])                                     # load the stack once

    t = time.perf_counter()
    times = []
    for g in groups:
        t0 = time.perf_counter()
        sted.consistency(g)
        times.append(time.perf_counter() - t0)
    total = time.perf_counter() - t
    pairs = args.examples * args.reps * (args.reps - 1) // 2
    mode = sted.consistency(groups[0]).text_similarity
    print(f"[{mode}] sted.consistency: {args.examples} examples x {args.reps} runs ({pairs} pairs) in {total:.1f}s; "
          f"per example median {statistics.median(times)*1000:.0f} ms, p95 {sorted(times)[int(.95*len(times))]*1000:.0f} ms")

    runs = [NS(outputs={"calls": o}) for g in groups for o in g]
    examples = [NS(id=f"e{i}", inputs={"case": i}) for i, g in enumerate(groups) for _ in g]
    t = time.perf_counter()
    out = sted_consistency_evaluator()(runs, examples)
    print(f"[{mode}] LangSmith summary evaluator over {len(runs)} runs: {time.perf_counter() - t:.1f}s; "
          f"share_same {out['results'][0]['score']:.3f}")

    big = [outputs_for(1, rng) for _ in range(50)]
    t = time.perf_counter()
    r = sted.consistency(big)
    print(f"[{mode}] one example with 50 outputs (1225 pairs): {time.perf_counter() - t:.1f}s; "
          f"unstable {[u.path for u in r.unstable_fields]}")


if __name__ == "__main__":
    sys.exit(main())
