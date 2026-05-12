"""Strands SDK integration: measure consistency of a Strands agent.

Run with:
    pip install sted strands-agents && python examples/strands_consistency_example.py

If strands_agents is not installed, this script falls back to a mock agent so
you can see the evaluator workflow without external deps.
"""
from __future__ import annotations

import json
import random

from sted import AgentConsistencyEvaluator


def _make_agent():
    """Return a callable: prompt -> JSON-like dict.

    Tries strands_agents first, falls back to a mock agent.
    """
    try:
        # Strands SDK exposes an Agent abstraction with `__call__(prompt) -> str`
        from strands_agents import Agent  # type: ignore

        agent_obj = Agent(
            model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            system_prompt=(
                "Return ONLY a single-line JSON object with keys 'tool' and "
                "'args' — no commentary."
            ),
        )

        def agent(prompt: str):
            response = agent_obj(prompt)
            text = response if isinstance(response, str) else str(response)
            try:
                return json.loads(text)
            except Exception:
                return text
        return agent, "strands_agents.Agent"
    except Exception:
        pass

    # Mock fallback — simulates a slightly stochastic tool-calling agent
    def mock_agent(prompt: str):
        tools = ["search_web", "search_web", "lookup_kb"]
        return {"tool": random.choice(tools), "args": {"q": prompt[:30]}}
    return mock_agent, "mock"


def main():
    agent, backend = _make_agent()
    print(f"Using backend: {backend}")

    evaluator = AgentConsistencyEvaluator(
        structural_weight=0.5,
        max_parallel_runs=4,
        timeout_seconds=30.0,  # safety net for slow scoring on large outputs
    )
    prompts = [
        "Look up today's S&P 500 closing price.",
        "Find the most recent NeurIPS paper on tree edit distance.",
        "What time is the next train from London to Paris?",
    ]
    report = evaluator.evaluate(agent, prompts, n_runs=5, progress=True)
    print(report.summary())

    # Demonstrate cache_stats() — useful for monitoring long-running batches.
    print("\nCache stats after evaluation:", evaluator.cache_stats())


if __name__ == "__main__":
    main()
