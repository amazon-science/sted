"""A real tool-calling agent on Bedrock (Converse API) for the integration checks."""
import os

import boto3

MODEL_ID = os.environ.get("STED_CHECK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.environ.get("AWS_REGION", "us-west-2")
TODAY = "Today is Tuesday 2026-11-10."

TOOLS = [
    {"name": "book_flight", "description": "Book a flight.", "properties": {
        "origin": {"type": "string"}, "destination": {"type": "string"},
        "date": {"type": "string", "description": "YYYY-MM-DD"}, "passengers": {"type": "integer"},
        "seat_preference": {"type": "string", "enum": ["window", "aisle", "any"]}},
     "required": ["origin", "destination", "date", "passengers"]},
    {"name": "get_weather", "description": "Weather forecast for a city.", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "units": {"type": "string", "enum": ["metric", "imperial"]}},
     "required": ["city"]},
    {"name": "search_hotels", "description": "Search hotels with a free-text query.", "properties": {
        "query": {"type": "string", "description": "free-text search query"}, "city": {"type": "string"},
        "max_price": {"type": "number"}},
     "required": ["query", "city"]},
]

PROMPTS = {
    "flight": "Book me a flight from Seattle to New York next Friday for two people, I like window seats. " + TODAY,
    "weather": "What's the weather going to be like in Paris over the next few days?",
    "hotel": "Find me a quiet hotel near the Louvre in Paris, under 300 euros a night.",
    "flight_exact": "Book 1 passenger SEA to JFK on 2026-12-01, aisle seat.",
}


def tool_config():
    return {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"], "inputSchema": {"json": {
        "type": "object", "properties": t["properties"], "required": t["required"]}}}} for t in TOOLS]}


_client = None


def call(prompt: str, temperature: float = 1.0) -> list:
    """The model's tool calls for one prompt: [{"name": ..., "arguments": {...}}]."""
    global _client
    _client = _client or boto3.client("bedrock-runtime", region_name=REGION)
    r = _client.converse(modelId=MODEL_ID, messages=[{"role": "user", "content": [{"text": prompt}]}],
                         toolConfig=tool_config(), inferenceConfig={"temperature": temperature, "maxTokens": 500})
    return [{"name": b["toolUse"]["name"], "arguments": b["toolUse"]["input"]}
            for b in r["output"]["message"]["content"] if "toolUse" in b]
