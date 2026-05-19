# LangGraph × tau-bench — Trajectory Consistency Eval

Run a LangGraph ReAct agent on tau-bench retail tasks N times each, capture the tool-call trajectory, and score run-to-run consistency with STED.

## What this measures

For each retail task (115 in the test split), how *reproducible* is the agent's plan when invoked multiple times at the same temperature? Output: a per-task `c_mean` (mean pairwise STED similarity across runs) and `c_std`.

**Important:** this is *consistency*, not correctness. We do **not** drive tau-bench's user simulator and we do not check tool-call outcomes against `Task.actions`. See "Methodological choices" below.

## Files

| File | Role |
|---|---|
| `tools_adapter.py` | Wraps tau-bench retail tool classes (`ALL_TOOLS`) as LangChain `StructuredTool`s, each closed over a per-task copy of `env.data`. |
| `agent.py` | Builds a `create_react_agent` with `ChatBedrockConverse` + the wrapped tools + the retail `WIKI` system prompt. |
| `trajectory_extract.py` | Walks `final_state["messages"]`, pulls `AIMessage.tool_calls`, returns `[{"name", "args"}, ...]`. |
| `run_trajectories.py` | Main runner. Loads tasks, runs N times each, streams JSONL to disk. Resumable. |
| `score_trajectories.py` | Loads the JSONL, runs `AgentConsistencyEvaluator.for_trajectory()`, writes a report. |

## Quick start

```bash
# One-time setup (in the per-experiment .venv):
pip install langgraph "langchain-aws>=0.2" \
    "tau-bench @ git+https://github.com/sierra-research/tau-bench"
pip install -e /path/to/sted-internal

# Pilot (20 tasks × 5 runs)
python -m scripts.experiments.langgraph_tau_bench.run_trajectories \
    --output results/pilot_n20_k5/trajectories.jsonl \
    --max-tasks 20 --n-runs 5 --max-workers 4

python -m scripts.experiments.langgraph_tau_bench.score_trajectories \
    --input  results/pilot_n20_k5/trajectories.jsonl \
    --output results/pilot_n20_k5/report.json
```

The runner writes one JSONL line per `(task_idx, run_idx)` pair as it completes, so you can ctrl-C and re-run; already-done pairs are skipped.

## Methodological choices

1. **No user-simulator loop.** tau-bench retail is designed to be evaluated through a user-simulator LLM (`LLMUserSimulationEnv`); each task's `instruction` is a role-play brief written *to the simulator* in second person ("You are Yusuf Rossi..."). Driving the simulator would inject a *second* LLM consistency variable, which would confound the trajectory-consistency signal we want. Instead, we frame the instruction as `"A customer says: <instruction>"` and let the agent serve the request in one turn.
2. **No env-mutation across runs.** Each (task, run) gets a fresh `copy.deepcopy(load_data())`. tau-bench's mutating tools (`cancel_pending_order`, `modify_pending_order_*`) won't leak.
3. **No correctness check.** We capture `AIMessage.tool_calls` only; we do not call `env.calculate_reward(...)`. Future extension: pair this with a separate correctness pass that drives the simulator and computes the standard tau-bench reward.
4. **Recursion limit.** Set to 50 by default — runs that exceed it are recorded with `error="GraphRecursionError"` and excluded from scoring (validity rate `r_v` reflects this).

## Output schema

`trajectories.jsonl`:

```json
{"task_idx": 0, "run_idx": 3, "instruction": "...",
 "trajectory": [{"name": "find_user_id_by_name_zip",
                 "args": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}, ...],
 "n_messages": 9, "elapsed_ms": 14200, "error": null, "stop_reason": "ai"}
```

`report.json`:

```json
{
  "summary": {"n_tasks_scored": 113, "n_errors": 4, ...},
  "report": {
    "mean_consistency": 0.78, "mean_validity": 0.96, "mean_c_adj": 0.75,
    "per_prompt": [...], "worst_prompts": [...]
  }
}
```

## EC2 reproduction

Use the experiment-management flow (see `CLAUDE.md` "Experiment Management Rules"):

```bash
python experiments/scripts/manage_experiment.py create \
    --name langgraph_tau_bench_retail \
    --description "Trajectory consistency, 115 retail tasks × 5 runs, Sonnet 4.6" \
    --script scripts/experiments/langgraph_tau_bench/run_trajectories.py \
    --paper INTERNAL

python experiments/scripts/manage_experiment.py setup-ec2 --experiment <id>
```

A `t3.xlarge` is plenty (this workload is LLM-API-bound, not compute-bound).
