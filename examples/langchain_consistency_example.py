"""LangChain integration: measure consistency of a LangChain chat model.

Run with:
    pip install sted langchain langchain-openai && python examples/langchain_consistency_example.py

Or with Bedrock instead of OpenAI:
    pip install sted langchain langchain-aws

If neither is installed, this script falls back to a mock chat model so you
can see the evaluator workflow without external deps.
"""
from __future__ import annotations

import json
import random

from sted import AgentConsistencyEvaluator


def _make_agent():
    """Return a callable: prompt -> JSON-like dict.

    Tries (in order): ChatOpenAI -> ChatBedrock -> mock.
    """
    # 1) LangChain + OpenAI (via OpenRouter or OpenAI directly)
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        def agent(prompt: str):
            response = llm.invoke(
                f"Return ONLY a JSON object with keys 'tool' and 'args' for: {prompt}"
            )
            try:
                return json.loads(response.content)
            except Exception:
                return response.content  # fall back to raw string
        return agent, "ChatOpenAI"
    except Exception:
        pass

    # 2) LangChain + Bedrock
    try:
        from langchain_aws import ChatBedrock

        llm = ChatBedrock(
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            model_kwargs={"temperature": 0.7},
        )

        def agent(prompt: str):
            response = llm.invoke(
                f"Return ONLY a JSON object with keys 'tool' and 'args' for: {prompt}"
            )
            try:
                return json.loads(response.content)
            except Exception:
                return response.content
        return agent, "ChatBedrock"
    except Exception:
        pass

    # 3) Mock fallback — simulates a stochastic agent
    def mock_agent(prompt: str):
        tools = ["search", "lookup", "search"]  # weighted toward "search"
        return {"tool": random.choice(tools), "args": {"q": prompt[:30]}}
    return mock_agent, "mock"


def main():
    agent, backend = _make_agent()
    print(f"Using backend: {backend}")

    evaluator = AgentConsistencyEvaluator(
        structural_weight=0.5,
        max_parallel_runs=4,
    )
    prompts = [
        "What's the weather in Seattle?",
        "Book me a flight to NYC tomorrow",
        "Find recent papers on tree edit distance",
    ]
    report = evaluator.evaluate(agent, prompts, n_runs=5, progress=True)
    print(report.summary())


if __name__ == "__main__":
    main()
