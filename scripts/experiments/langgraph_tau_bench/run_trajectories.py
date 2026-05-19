"""Run a LangGraph agent on tau-bench retail tasks N times each, capture trajectories.

Output: JSONL with one record per (task_idx, run_idx). Resumable — re-running
skips completed (task_idx, run_idx) pairs already in the output file.

Usage:
    python run_trajectories.py \\
        --output results/full_n115_k5/trajectories.jsonl \\
        --n-runs 5 --max-tasks 115 --max-workers 6 \\
        --temperature 1.0 \\
        --model-id us.anthropic.claude-sonnet-4-6-v1:0
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import traceback
from pathlib import Path
from threading import Lock
from typing import Optional

# Add project root to path so `import sted` works without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from .agent import build_agent, load_retail_system_prompt  # noqa: E402
from .tools_adapter import build_retail_tools, fresh_retail_data  # noqa: E402
from .trajectory_extract import extract_trajectory  # noqa: E402


def load_retail_tasks(max_tasks: Optional[int] = None) -> list:
    """Return tau-bench retail TEST tasks (115 total)."""
    from tau_bench.envs.retail.tasks_test import TASKS_TEST  # type: ignore

    tasks = list(TASKS_TEST)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def already_done(output_path: Path) -> set:
    """Return the set of (task_idx, run_idx) already written."""
    done: set = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["task_idx"], rec["run_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_one(
    task_idx: int,
    run_idx: int,
    instruction: str,
    model_id: str,
    region_name: str,
    temperature: float,
    recursion_limit: int,
) -> dict:
    """Build a fresh env + agent, run the task once, return a result dict."""
    t_start = time.time()
    record = {
        "task_idx": task_idx,
        "run_idx": run_idx,
        "instruction": instruction,
        "trajectory": [],
        "n_messages": 0,
        "elapsed_ms": 0,
        "error": None,
        "stop_reason": None,
    }

    try:
        data_ref = fresh_retail_data()
        tools = build_retail_tools(data_ref)
        agent = build_agent(
            tools=tools,
            system_prompt=load_retail_system_prompt(),
            model_id=model_id,
            region_name=region_name,
            temperature=temperature,
        )

        # Frame the customer's goal as a third-person request — the raw task
        # instruction is written in second person to the user simulator and
        # would otherwise prompt the agent to roleplay as the customer.
        user_message = f"A customer says:\n\n{instruction}"

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": recursion_limit},
        )
        record["trajectory"] = extract_trajectory(result)
        record["n_messages"] = len(result.get("messages", []))
        # Stop reason is best-effort; create_react_agent doesn't surface a
        # canonical field so we infer it from the last message.
        last = (result.get("messages") or [{}])[-1]
        record["stop_reason"] = getattr(last, "type", None) or "unknown"
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        # Keep traceback in stderr for debugging but not in the record.
        traceback.print_exc(file=sys.stderr)

    record["elapsed_ms"] = int((time.time() - t_start) * 1000)
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True,
                   help="JSONL output path (resumable).")
    p.add_argument("--n-runs", type=int, default=5,
                   help="Number of repeats per task (default 5).")
    p.add_argument("--max-tasks", type=int, default=None,
                   help="Optional cap on number of tasks (for pilot).")
    p.add_argument("--task-indices", type=str, default=None,
                   help="Comma-separated task indices to run (e.g. '0,3,7,12').")
    p.add_argument("--max-workers", type=int, default=4,
                   help="ThreadPoolExecutor workers (default 4).")
    p.add_argument("--model-id", type=str,
                   default="us.anthropic.claude-sonnet-4-6")
    p.add_argument("--region", type=str, default="us-east-1")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--recursion-limit", type=int, default=50)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load tasks
    if args.task_indices:
        all_tasks = load_retail_tasks()
        indices = [int(x) for x in args.task_indices.split(",")]
        tasks = [(i, all_tasks[i]) for i in indices]
    else:
        all_tasks = load_retail_tasks(args.max_tasks)
        tasks = list(enumerate(all_tasks))

    print(f"Loaded {len(tasks)} tasks; running {args.n_runs} runs each "
          f"({len(tasks) * args.n_runs} total LLM invocations)")

    # Build the (task_idx, run_idx) work list, skipping already-done pairs.
    done = already_done(args.output)
    if done:
        print(f"Resuming: {len(done)} pairs already done, skipping.")
    work: list = []
    for task_idx, task in tasks:
        for run_idx in range(args.n_runs):
            if (task_idx, run_idx) in done:
                continue
            work.append((task_idx, run_idx, task.instruction))

    if not work:
        print("Nothing to do.")
        return 0

    print(f"Pending: {len(work)} pairs to run.")

    # Stream-write JSONL. Use a lock so workers can't interleave lines.
    write_lock = Lock()

    def _do(item):
        task_idx, run_idx, instruction = item
        rec = run_one(
            task_idx=task_idx,
            run_idx=run_idx,
            instruction=instruction,
            model_id=args.model_id,
            region_name=args.region,
            temperature=args.temperature,
            recursion_limit=args.recursion_limit,
        )
        with write_lock, open(args.output, "a") as f:
            f.write(json.dumps(rec) + "\n")
        ok = "OK " if rec["error"] is None else "ERR"
        print(f"[{ok}] task={task_idx:3d} run={run_idx} steps={len(rec['trajectory']):2d} "
              f"elapsed_ms={rec['elapsed_ms']:6d} "
              f"error={rec['error'] or ''}", flush=True)
        return rec

    t_total = time.time()
    n_ok = n_err = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for rec in pool.map(_do, work):
            if rec["error"] is None:
                n_ok += 1
            else:
                n_err += 1

    elapsed = time.time() - t_total
    print(f"\nDone. {n_ok} OK, {n_err} errors in {elapsed:.1f}s "
          f"({elapsed / max(1, n_ok + n_err):.1f}s/run).")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
