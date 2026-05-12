"""Async helpers for AgentConsistencyEvaluator.

Useful when your agent_fn is itself an async coroutine (e.g., async LLM SDK
calls). The synchronous AgentConsistencyEvaluator is preferred for CPU-bound
STED evaluation; this module is for the I/O-bound generation step.

Example:
    import asyncio
    from sted import AgentConsistencyEvaluator
    from sted.agent_consistency_async import evaluate_async

    evaluator = AgentConsistencyEvaluator(n_workers=4)

    async def my_async_agent(prompt: str) -> dict:
        return await async_llm_call(prompt)

    report = asyncio.run(evaluate_async(
        evaluator, my_async_agent, prompts, n_runs=10
    ))
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable, Optional

from .agent_consistency_evaluator import (
    AgentConsistencyEvaluator,
    ConsistencyReport,
    PromptResult,
    _is_valid,
)
from ._logging import get_logger

logger = get_logger("async")

AsyncAgentFn = Callable[[str], Awaitable[object]]


async def evaluate_async(
    evaluator: AgentConsistencyEvaluator,
    agent_fn: AsyncAgentFn,
    prompts: Iterable[str],
    n_runs: int = 10,
    bottom_k: int = 5,
    max_concurrent: int = 16,
    progress: bool = False,
) -> ConsistencyReport:
    """Async-compatible evaluate(). Invokes async agent_fn n_runs times per
    prompt with a bounded semaphore, then runs synchronous STED scoring.

    Args:
        evaluator: AgentConsistencyEvaluator instance (sync; used for STED).
        agent_fn: async callable returning a JSON-like value.
        prompts: iterable of prompt strings.
        n_runs: repeats per prompt.
        bottom_k: how many least-consistent prompts to surface.
        max_concurrent: max in-flight agent_fn calls (across all prompts).
        progress: if True, log per-prompt progress.

    Returns:
        ConsistencyReport.
    """
    prompts = list(prompts)
    sem = asyncio.Semaphore(max_concurrent)

    async def _one_call(prompt: str):
        async with sem:
            try:
                return await agent_fn(prompt), None
            except Exception as e:
                logger.warning("agent_fn raised on prompt=%r: %s", prompt[:60], e)
                return None, f"{type(e).__name__}: {e}"

    # Schedule all (prompt, run) tasks
    tasks = []
    for prompt in prompts:
        for _ in range(n_runs):
            tasks.append(_one_call(prompt))

    if progress:
        logger.info("async evaluate: %d prompts × %d runs = %d tasks "
                    "(max_concurrent=%d)",
                    len(prompts), n_runs, len(tasks), max_concurrent)

    results = await asyncio.gather(*tasks)

    # Group results back by prompt
    outputs_per_prompt: dict[str, list] = {}
    errors_per_prompt: dict[str, list] = {}
    for i, prompt in enumerate(prompts):
        start = i * n_runs
        end = start + n_runs
        prompt_results = results[start:end]
        outputs_per_prompt[prompt] = [r[0] for r in prompt_results]
        errors_per_prompt[prompt] = [r[1] for r in prompt_results if r[1]]

    # Hand off to sync scorer (STED is CPU-bound; threading there is fine).
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _score_with_errors(
            evaluator, outputs_per_prompt, errors_per_prompt, bottom_k
        ),
    )


def _score_with_errors(
    evaluator: AgentConsistencyEvaluator,
    outputs_per_prompt: dict,
    errors_per_prompt: dict,
    bottom_k: int,
) -> ConsistencyReport:
    """Score collected outputs and propagate any agent errors."""
    report = evaluator.evaluate_outputs(outputs_per_prompt, bottom_k=bottom_k)
    # Inject async-collected agent errors into the report's PromptResults.
    for r in report.per_prompt:
        errs = errors_per_prompt.get(r.prompt, [])
        if errs and not r.error:
            unique = list(dict.fromkeys(errs))
            r.error = (
                f"{len(errs)} agent invocations raised; first: {unique[0]}"
                if len(unique) > 1
                else f"{len(errs)} agent invocations raised: {unique[0]}"
            )
    return report
