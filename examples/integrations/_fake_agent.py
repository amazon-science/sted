"""A stand-in for your agent, so every example runs offline.

`book_flight(run)` returns a tool call; run 1 drifts to another date, the others agree.
Replace it with your own agent or model call.

Set STED_EXAMPLE_AGENT=bedrock to call a real model instead (Claude Haiku 4.5 on AWS
Bedrock by default, STED_EXAMPLE_MODEL to change; needs AWS credentials). The real
model's answers vary, so the examples then print their results instead of checking
exact numbers.
"""
import json
import os

REFERENCE = {"name": "book_flight", "arguments": {"to": "JFK", "date": "2026-12-01",
                                                  "passengers": 2, "request_id": "ref"}}
REAL = os.environ.get("STED_EXAMPLE_AGENT", "").lower() == "bedrock"
PROMPT = os.environ.get("STED_EXAMPLE_PROMPT") or (
    "Book a flight to New York JFK for two people on the first Tuesday of December 2026. "
    "Today is Tuesday 2026-11-10.")
_TOOL = {"tools": [{"toolSpec": {"name": "book_flight", "description": "Book a flight.", "inputSchema": {"json": {
    "type": "object", "properties": {"to": {"type": "string", "description": "IATA airport code"},
                                     "date": {"type": "string", "description": "YYYY-MM-DD"},
                                     "passengers": {"type": "integer"}},
    "required": ["to", "date", "passengers"]}}}}]}
_client = None


def book_flight(run: int, prompt: str = None):
    if REAL:
        return _bedrock_book_flight(prompt or PROMPT)
    return {"name": "book_flight",
            "arguments": {"to": "JFK", "date": "2026-12-02" if run == 1 else "2026-12-01",
                          "passengers": "2", "request_id": f"r-{run}"}}


def _bedrock_book_flight(prompt: str):
    """One real tool call; a text-only reply (no tool call) comes back as that text, which
    sted counts as an invalid output."""
    global _client
    import boto3
    _client = _client or boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    r = _client.converse(modelId=os.environ.get("STED_EXAMPLE_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
                         messages=[{"role": "user", "content": [{"text": prompt}]}], toolConfig=_TOOL,
                         inferenceConfig={"temperature": 1.0, "maxTokens": 300})
    for block in r["output"]["message"]["content"]:
        if "toolUse" in block:
            return {"name": block["toolUse"]["name"], "arguments": block["toolUse"]["input"]}
    return " ".join(b.get("text", "") for b in r["output"]["message"]["content"])


if __name__ == "__main__":
    print(json.dumps(book_flight(0)))
