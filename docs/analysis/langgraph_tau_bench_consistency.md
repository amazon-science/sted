# Trajectory Consistency of LangGraph + tau-bench Retail

**Status:** Both runs complete (Sonnet 4.6 + Haiku 4.5), 1,150 LLM calls total.

**TL;DR.** A LangGraph ReAct agent on the 115 tau-bench retail tasks (5 runs each, T = 1.0) gives:

- **Sonnet 4.6**: `c_mean = 0.855` (per-scored-task), 12 % of tasks `< 0.50` — bimodal failure (4-of-5 give up, 1-of-5 commits to a 12-step plan).
- **Haiku 4.5**: `c_mean = 0.901` (per-scored-task), 2 % of tasks `< 0.50` — uniform failure (all-of-5 give up; prompt is dropped before scoring).

**The "better" model depends on the metric:** Haiku scores higher on per-scored-task consistency, but Sonnet has a higher non-empty rate (88 % vs 79 %). Combining the two gives the deployment-relevant composite `c_mean × non_empty_rate = 0.752` (Sonnet) vs `0.711` (Haiku) — Sonnet wins end-to-end despite losing per-task. This kind of tradeoff is exactly what a single accuracy or single consistency number obscures, and what STED-on-trajectories surfaces. The two models fail on *different* prompts (only 3 / 12 of each model's worst-12 overlap), suggesting prompt-level non-determinism is model-specific rather than universal.

---

## 1. Setup

| Component | Value |
|---|---|
| Agent framework | LangGraph 0.x `create_react_agent` |
| LLM | Claude Sonnet 4.6 via Bedrock (`us.anthropic.claude-sonnet-4-6`) |
| Domain | tau-bench retail (115 test-split tasks, 16 tools) |
| Runs per task | 5 |
| Total LLM calls | 575 |
| Temperature | 1.0 |
| Recursion limit | 50 |
| Wall time | ~10 minutes (8 workers, m5.16xlarge via SSM) |
| Metric | `AgentConsistencyEvaluator.for_trajectory()` (STED w/ exact-match on tool/parameter names, positional matching on step lists, embedding similarity on parameter values) |

The agent saw each task as a *single user turn* — we did **not** drive tau-bench's `LLMUserSimulationEnv`. This was a deliberate choice: the user simulator is itself an LLM, which would inject a second source of run-to-run variance and confound the agent-side consistency signal we want to measure. Trade-off: tasks designed around a multi-turn clarification dialogue terminate after one tool burst.

Trajectories captured by walking `final_state["messages"]` and pulling every `AIMessage.tool_calls` entry, dropping `ToolMessage` results.

---

## 2. Headline numbers

| Metric | Value |
|---|---|
| Tasks scored | 106 / 115 |
| Tasks dropped (<2 valid runs) | 9 |
| **Mean trajectory consistency `c_mean`** | **0.855** |
| Mean validity rate `r_v` | 1.000 |
| Mean `c_adj` (= `r_v · c_mean`) | 0.855 |

`r_v = 1.0` because every captured trajectory parses cleanly — there are no malformed `tool_use` blocks. This is a property of LangGraph + Bedrock's structured output, not a property of the agent's decision-making.

---

## 3. Distribution across 106 scored tasks

| `c_mean` range | Tasks | Share | Mean trajectory length | Interpretation |
|---|---|---|---|---|
| [0.95, 1.00] | 57 | **54 %** | 7.0 steps | Plan fully reproducible |
| [0.85, 0.95) | 22 | 21 % | 6.9 steps | One step of drift across runs |
| [0.70, 0.85) | 8 | 8 % | 4.8 steps | Moderate drift |
| [0.50, 0.70) | 7 | 7 % | 3.0 steps | Notable inconsistency |
| **[0.00, 0.50)** | **12** | **11 %** | **3.5 steps** | **Severe — deployment red flag** |

**Three things stand out.** (1) The bottom-12 cluster is real and reproducible. (2) Stable tasks are *longer* (mean 7.0 steps) than unstable tasks (mean 3.5 steps) — a length-stability correlation suggesting that once the agent commits to a plan, it follows it through; the inconsistency is concentrated in the *commit* step. (3) Validity is uniform across all buckets; this is a *trajectory*-level inconsistency, not a parsing issue.

---

## 4. Anatomy of an inconsistent task

The bottom-12 pattern is consistent: trajectories within one prompt are **bimodal** — each run is either a long, planful trajectory (≈8–12 steps) or a short give-up (0–1 steps). Two illustrative cases:

### Task 53 — Sofia Li bicycle refund (`c_mean = 0.000`)

```
"You are Sofia Li, residing in San Antonio, 78260. The bicycle you received was
damaged during delivery, and you want to get a refund..."

run 0: []                                                  ← gave up immediately
run 1: ['think']                                           ← thought, then gave up
run 2: []                                                  ← gave up
run 3: []                                                  ← gave up
run 4: ['think', 'find_user_id_by_name_zip',               ← actually attempted
        'get_user_details', 'get_order_details ×4',
        'think']                                           (8 steps)
```

**4 of 5 runs** stalled. The single committed run found the order. From a customer's perspective: 80 % of the time, no progress.

### Task 87 — Yusuf Hernandez modify pending orders (`c_mean = 0.396`)

```
run 0: ['think']                                           ← gave up
run 1: ['think']                                           ← gave up
run 2: ['think', 'find_user_id_by_email', 'get_user_details',
        'get_order_details ×5', 'think']                   ← 9-step plan
run 3: identical to run 2                                  ← 9-step plan
run 4: identical to run 2                                  ← 9-step plan
```

3 of 5 runs are *bit-identical*; the other 2 stalled. This is the cleanest illustration of the bimodality — when the agent commits, it commits the same way every time; the variance lives entirely in whether it commits at all.

### Why it happens

These prompts share a property: the customer's goal is stated up front but **requires a clarifying question** that the (absent) user simulator would have answered. With no simulator, the agent samples between "ask for clarification" (which terminates with no tool call, since there's nobody to answer) and "infer + proceed" (which produces the planful 8–12-step trajectory). At T = 1.0 the sample comes out roughly 50/50, hence the bimodal scores.

This is informative on its own: **the prompts that triggered ambiguity-driven bimodality are exactly the ones an agent operator would want to flag for prompt engineering or human-in-the-loop**. The metric ranks them at the bottom by construction, without needing oracle correctness labels.

---

## 5. Tool-call vocabulary

Across all 575 runs, the agent used 9 distinct tools out of 16 available:

| Tool | Calls |
|---|---|
| `get_order_details` | 1145 |
| `think` | 909 |
| `get_product_details` | 418 |
| `get_user_details` | 351 |
| `find_user_id_by_name_zip` | 320 |
| `find_user_id_by_email` | 101 |
| `list_all_product_types` | 18 |
| `calculate` | 10 |
| `transfer_to_human_agents` | 1 |

The 7 unused tools (`exchange_delivered_order_items`, `cancel_pending_order`, `modify_pending_order_*`, `return_delivered_order_items`, `modify_user_address`) are all *mutating* tools — the agent never reached the "execute the action" stage in any run because the policy requires user confirmation first, which the (absent) user simulator never provided. This is a single-turn-mode artifact, documented as a known limitation in §1.

---

## 6. Cross-model baseline: Haiku 4.5

The same 115 × 5 batch was run on Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`), same temperature, same workers, same code path. The result reverses the naive expectation:

| Metric | Sonnet 4.6 | Haiku 4.5 | Δ |
|---|---|---|---|
| Total runs | 575 | 575 | — |
| Mean `c_mean` (per scored task) | 0.855 | **0.901** | **+0.046 (Haiku)** |
| Tasks scored (≥ 2 valid runs) | 106 / 115 | 97 / 115 | −9 (Haiku) |
| Empty trajectories | 71 (12 %) | **122 (21 %)** | +9 pp (Haiku) |
| % tasks `c_mean ≥ 0.95` | 54 % | 48 % | −6 pp |
| % tasks `c_mean < 0.50` | **11 %** | **2 %** | **−9 pp (Haiku)** |
| Mean trajectory length | 5.7 steps | 4.4 steps | −1.3 (Haiku) |
| Mean call latency | 20.5 s | **8.1 s** | −2.5× (Haiku) |

### Two different failure modes

**Sonnet stalls bimodally**, Haiku stalls uniformly. The same prompt that Sonnet handles 4-of-5 well and 1-of-5 with a give-up trajectory is the kind that Haiku either handles 5-of-5 well or gives up 5-of-5 (and gets dropped from the scored set). This shows up clearly in the bottom-12 overlap:

```
Sonnet bottom-12: {3, 12, 14, 18, 22, 53, 62, 63, 87, 92, 98, 99}
Haiku  bottom-12: {21, 32, 44, 46, 47, 62, 63, 72, 92, 105, 112, 113}
Overlap:          {62, 63, 92}  (only 3 of 12 — 25 %)
```

The two models fail on *different* prompts. Sonnet is hurt most by tasks that need a clarification dialogue (it inconsistently risks an inferred action); Haiku is hurt most by tasks that need many reasoning steps (it more readily gives up entirely, which removes the prompt from the scored set rather than dragging its `c_mean` down).

### Tasks where the two models differ most

| Task | Sonnet `c_mean` | Haiku `c_mean` | Δ |
|---|---|---|---|
| 53 (Sofia Li bicycle refund) | 0.000 | **1.000** | −1.000 |
| 14 (mia_garcia gaming cancel) | 0.200 | **1.000** | −0.800 |
| 22 (Ethan Garcia address change) | 0.325 | **1.000** | −0.675 |
| 3 (Yusuf Rossi tshirt count) | 0.369 | **1.000** | −0.631 |
| 47 (daiki_johnson) | **0.683** | 0.158 | **+0.525** |
| 84 | 0.591 | **1.000** | −0.409 |

Haiku scoring c=1.000 on Sonnet's worst tasks is **not** a Haiku competence win — those are the prompts where Haiku gave up *uniformly* on enough runs that the bottom-tail signal moved into the "empty trajectories" pile rather than the "low c_mean" pile. Validity rate `r_v` would distinguish them, but Haiku's `r_v` is 1.000 too because the agent's *captured* runs are all valid; the dropped uniformly-empty ones are filtered out before scoring.

### What the right composite is

The single number you'd report depends on your deployment regime:

| If you care about… | Use this number |
|---|---|
| "Will the agent do the same thing each time *when it acts*?" | `c_mean` per scored task — Haiku wins (0.901 vs 0.855). |
| "How often does the agent take *any* action vs stalling?" | non-empty rate — Sonnet wins (88 % vs 79 %). |
| "Will I get the same outcome at the customer level?" | `c_mean × non_empty_rate` — **Sonnet 0.752 vs Haiku 0.711** (Sonnet wins). |

The third metric is the deployment-honest composite. **Sonnet is more reliable end-to-end despite scoring lower on the per-scored-task `c_mean`** — because it engages with more prompts. This is exactly the kind of tradeoff a single accuracy or single consistency number would obscure.

### Cost / latency tradeoff

Haiku is 2.5× faster per call (8.1 s vs 20.5 s mean) and substantially cheaper. For a deployment that retries on stall (and has an acceptable retry budget), the Haiku-style failure mode (uniform stall → caller retries) may be more tolerable than the Sonnet-style failure mode (bimodal commit → caller gets a wrong-shape result they have to detect and reject). For a deployment without retry plumbing, Sonnet's higher engagement rate is the safer default.

---

## 7. What this validates about STED + `for_trajectory`

1. **The metric ranks usefully.** The bottom-12 cluster contains exactly the prompts where the agent's plan is non-deterministic in a way that a customer would notice.
2. **Exact-match on tool / parameter names is the right default for trajectories.** The c=1.000 top tasks have bit-identical `name` and `args`-key sets across runs; the embedding-similarity on `args` *values* lets paraphrase variation in `think` thoughts slide without dropping the score.
3. **`r_v = 1.000` everywhere.** Validity-rate dimension carries no signal here — `c_adj` reduces to `c_mean`. For agent trajectories from production-grade frameworks, `r_v` is mostly a guard against the framework breaking, not against the agent's decisions.
4. **Single-turn mode is workable.** Even without driving the user simulator, 106 / 115 prompts produced ≥ 2 valid runs and gave a meaningful score. The 9 dropped prompts plus the 12 worst-consistency prompts are the same population: those needing dialogue-style clarification.

---

## 8. Reproduction

```bash
# Code (already committed in this repo):
scripts/experiments/langgraph_tau_bench/{tools_adapter,agent,trajectory_extract,
                                          run_trajectories,score_trajectories}.py

# Deps (per-experiment venv):
pip install langgraph langchain-aws langchain-core boto3 numpy scipy \
            sentence-transformers tqdm pydantic
pip install "tau-bench @ git+https://github.com/sierra-research/tau-bench"
pip install -e .

# Run (575 calls, ~10 min on m5.xlarge or larger):
python -m scripts.experiments.langgraph_tau_bench.run_trajectories \
    --output results/full_n115_k5/trajectories.jsonl \
    --max-tasks 115 --n-runs 5 --max-workers 8 \
    --temperature 1.0 \
    --model-id us.anthropic.claude-sonnet-4-6

# Score:
python -m scripts.experiments.langgraph_tau_bench.score_trajectories \
    --input  results/full_n115_k5/trajectories.jsonl \
    --output results/full_n115_k5/report.json
```

Artifacts:
- Trajectories: `experiments_local/langgraph_tau_bench/full_n115_k5/trajectories.jsonl`
- Report: `experiments_local/langgraph_tau_bench/full_n115_k5/report.json`
- S3 mirror: `s3://guanghu-experiment-temp/sted_traj_eval/results/`

---

## 9. Limitations and next steps

- **Single-turn agent mode** confounds the bottom-tail. Tasks where the simulator would have unblocked the agent get scored as inconsistent. The cross-model result in §6 already shows that the failure mode is partly model-specific (Sonnet bimodal, Haiku uniform), but disentangling agent-prompt ambiguity from agent-decision-making would require re-running a subset with the user simulator wired in.
- **No correctness check.** This is consistency-only. The natural follow-up is to also score `Task.actions` matching for a cross-cut: do high-consistency prompts also produce correct actions, or does the agent confidently make the same wrong call every time?
- **`r_v` blind spot.** The validity rate is 1.000 for both models because *captured* runs all parse, but uniformly-empty trajectories are silently dropped at scoring time rather than flagged as `r_v < 1`. A deployment-honest composite (§6, third row) needs `c_mean × non_empty_rate` — the scorer should compute this directly in a future revision.
- **Cross-vendor not yet measured.** The Haiku-vs-Sonnet split shows the failure mode is model-dependent within the Anthropic family. Cross-vendor (GPT-4o, Llama 3.3 70B via OpenRouter) would close the loop on whether bottom-tail prompts are *agent-prompt* properties or *Anthropic-specific* properties.
