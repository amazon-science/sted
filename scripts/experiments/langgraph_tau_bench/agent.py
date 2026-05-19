"""Build a LangGraph ReAct agent with Bedrock Claude + tau-bench retail tools."""
from __future__ import annotations

from typing import List, Optional

from langchain_aws import ChatBedrockConverse
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent


# tau-bench's retail wiki/policy is the canonical system prompt for the domain.
def load_retail_system_prompt() -> str:
    """Return the retail domain policy/wiki as a system prompt.

    The wiki tells the agent what it can and cannot do (e.g. only modify
    pending orders, never disclose payment IDs). We rewrite the standard
    "you are a customer service agent" preamble to discourage the agent
    from role-playing as the *customer* (the task instruction is written
    in second person to the user simulator, not the agent).
    """
    from tau_bench.envs.retail.wiki import WIKI  # type: ignore

    preamble = (
        "You are a customer-service agent for a retail store. A customer "
        "will describe their goal in the next message; help them by calling "
        "the appropriate tools. Do not roleplay as the customer. When the "
        "customer's goal is fully resolved, respond with a brief confirmation "
        "and stop. Follow the policies below strictly.\n\n"
    )
    return preamble + WIKI


def build_agent(
    tools: List[StructuredTool],
    system_prompt: Optional[str] = None,
    model_id: str = "us.anthropic.claude-sonnet-4-6",
    region_name: str = "us-east-1",
    temperature: float = 1.0,
    max_tokens: int = 4096,
):
    """Build a single-node LangGraph ReAct agent.

    Returns the compiled graph ready for ``.invoke({"messages": [...]})``.
    """
    if system_prompt is None:
        system_prompt = load_retail_system_prompt()

    llm = ChatBedrockConverse(
        model_id=model_id,
        region_name=region_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return create_react_agent(llm, tools=tools, prompt=system_prompt)
