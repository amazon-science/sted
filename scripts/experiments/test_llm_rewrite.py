#!/usr/bin/env python3
"""
Test LLM-based rewriting for feature intervention.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.eval.generate_tool_calls import get_bedrock_client

def llm_rewrite(client, query: str, instruction: str, model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0") -> str:
    """Use LLM to rewrite a query according to instruction."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"""Rewrite the following query according to the instruction.
Keep the semantic meaning EXACTLY the same - only change the format/style as specified.
Output ONLY the rewritten query, nothing else.

INSTRUCTION: {instruction}

ORIGINAL QUERY:
{query}

REWRITTEN QUERY:"""
                }
            ]
        }
    ]

    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={"maxTokens": 2000, "temperature": 0.0}
    )

    return response['output']['message']['content'][0]['text'].strip()


def test_numbered_list_rewrite():
    """Test LLM rewriting for numbered list removal."""

    client = get_bedrock_client()

    # Test cases with numbered lists
    test_queries = [
        """Our company is sending a senior manager to Beijing for a three-day market-insight workshop in the week of 12-18 May 2025. The manager will stay at the Four Seasons Hotel. Please provide a weather briefing that:
1. Summarises the expected daily weather conditions for each of the three days
2. Provides specific clothing recommendations based on the forecast
3. Includes any weather alerts that might affect travel plans""",

        """I need help with my research project. Please:
1. Search for papers on transformer architectures
2. Summarize the key findings
3. Create a bibliography in APA format""",

        """Plan my trip to Tokyo:
1. Find flights from San Francisco
2. Book a hotel near Shibuya station
3. Create an itinerary for 5 days"""
    ]

    print("=" * 70)
    print("LLM-BASED REWRITING TEST: Remove Numbered List")
    print("=" * 70)

    instruction = "Convert the numbered list into natural prose/paragraph format. Keep ALL information and context intact."

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}")
        print(f"{'='*70}")
        print(f"\nORIGINAL:\n{query}")

        rewritten = llm_rewrite(client, query, instruction)

        print(f"\nREWRITTEN:\n{rewritten}")
        print(f"\nOriginal length: {len(query)} chars, {len(query.split())} words")
        print(f"Rewritten length: {len(rewritten)} chars, {len(rewritten.split())} words")


def test_add_numbered_list():
    """Test LLM rewriting for adding numbered list."""

    client = get_bedrock_client()

    test_queries = [
        "I need you to search for recent papers on large language models, summarize the key findings, and create a reading list for my team.",

        "Please help me plan a birthday party by finding a venue, ordering a cake, and sending invitations to the guest list.",

        "Get the weather forecast for New York, check if there are any travel advisories, and book a flight for next Tuesday."
    ]

    print("\n" + "=" * 70)
    print("LLM-BASED REWRITING TEST: Add Numbered List")
    print("=" * 70)

    instruction = "Convert the request into a numbered list format, clearly separating each distinct task or requirement."

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}")
        print(f"{'='*70}")
        print(f"\nORIGINAL:\n{query}")

        rewritten = llm_rewrite(client, query, instruction)

        print(f"\nREWRITTEN:\n{rewritten}")


if __name__ == "__main__":
    test_numbered_list_rewrite()
    test_add_numbered_list()
