"""Inspect AI adapters against a real model on Bedrock.

    python scripts/integration_checks/inspect_real_llm.py [--epochs 5]

Real tools with generate()'s default tool loop (the model calls a tool, gets a result,
then answers in text), sted_match(answer_from="tool_calls") vs. the reference call, and
the sted_agreement epoch reducer next to Inspect's mean reducer.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from inspect_ai import Epochs, Task, eval as inspect_eval  # noqa: E402
from inspect_ai.dataset import Sample  # noqa: E402
from inspect_ai.model import GenerateConfig  # noqa: E402
from inspect_ai.solver import generate, use_tools  # noqa: E402
from inspect_ai.tool import tool  # noqa: E402

from _bedrock_agent import MODEL_ID, PROMPTS, REGION  # noqa: E402
from langsmith_real_llm import REFERENCES  # noqa: E402
from sted.integrations.inspect_ai import sted_agreement, sted_match  # noqa: E402


@tool
def book_flight():
    async def execute(origin: str, destination: str, date: str, passengers: int, seat_preference: str = "any"):
        """Book a flight.

        Args:
            origin: departure city or airport
            destination: arrival city or airport
            date: travel date, YYYY-MM-DD
            passengers: number of passengers
            seat_preference: window, aisle or any
        """
        return f"Booked {passengers} on {date}, confirmation ABC123."
    return execute


@tool
def get_weather():
    async def execute(city: str, days: int = 1, units: str = "metric"):
        """Weather forecast for a city.

        Args:
            city: city name
            days: number of days
            units: metric or imperial
        """
        return f"{city}: mild, 15C, light rain on day 2."
    return execute


@tool
def search_hotels():
    async def execute(query: str, city: str, max_price: float = 0):
        """Search hotels with a free-text query.

        Args:
            query: free-text search query
            city: city
            max_price: maximum price per night
        """
        return "Hotel Lumiere (240 EUR), Le Petit Louvre (280 EUR)."
    return execute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault("AWS_REGION", REGION)
    task = Task(dataset=[Sample(id=k, input=p, target=json.dumps(REFERENCES[k])) for k, p in PROMPTS.items()],
                solver=[use_tools(book_flight(), get_weather(), search_hotels()), generate()],
                scorer=sted_match(answer_from="tool_calls"),
                epochs=Epochs(args.epochs, [sted_agreement(), "mean"]), config=GenerateConfig(temperature=1.0))
    log = inspect_eval(task, model=f"bedrock/{MODEL_ID}", display="none", log_dir="/tmp/inspect_real_logs")[0]
    print("status:", log.status, "| samples:", len(log.samples))
    by = {r.reducer: {s.sample_id: float(s.value) for s in r.samples} for r in log.reductions}
    print(f"  {'case':<13} {'mean (correct)':>15} {'sted_agreement':>15}")
    for k in PROMPTS:
        print(f"  {k:<13} {by['mean'][k]:>15.2f} {by['sted_agreement'][k]:>15.2f}")
    s = next(x for x in log.samples if x.id == "flight" and x.scores["sted_match"].value == "I") \
        if any(x.id == "flight" and x.scores["sted_match"].value == "I" for x in log.samples) else log.samples[0]
    print("example:", s.id, "epoch", s.epoch, "final reply:", repr(s.output.completion[:60]))
    print("  answer:", s.scores["sted_match"].answer[:150])
    print("  explanation:", s.scores["sted_match"].explanation.replace("\n", " / ")[:200])
    for r in log.reductions:
        if r.reducer == "sted_agreement":
            for x in r.samples:
                if x.value < 1:
                    print("  unstable:", x.sample_id, (x.metadata or {}).get("unstable_fields"))
    return log


if __name__ == "__main__":
    main()
