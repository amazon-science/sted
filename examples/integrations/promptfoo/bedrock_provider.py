"""promptfoo Python provider: the prompt goes to the Bedrock agent, its tool call comes back
as JSON text. Used by promptfooconfig.real.yaml."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["STED_EXAMPLE_AGENT"] = "bedrock"

from _fake_agent import book_flight  # noqa: E402


def call_api(prompt, options, context):
    out = book_flight(0, prompt=prompt)
    return {"output": out if isinstance(out, str) else json.dumps(out)}
