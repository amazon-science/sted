# Evaluating Claude Code Trajectory Consistency with STED

This guide shows how to score how *consistently* Claude Code (or any agent) executes a tool-call trajectory across repeated runs of the same prompt, using `AgentConsistencyEvaluator.for_trajectory()`.

A trajectory is the ordered sequence of `tool_use` calls the agent issues. STED compares trajectories with **exact match on tool names and parameter names** (so `Read` ≠ `Bash`, and `file_path` ≠ `path`) and **embedding similarity on parameter values and free-text fields**, with positional matching across step lists.

---

## 1. Where the data comes from

You don't need to enable OpenTelemetry. Claude Code automatically writes a JSONL transcript per session at:

```
~/.claude/projects/<encoded-cwd>/<sessionId>.jsonl
```

`<encoded-cwd>` is the working directory with `/` replaced by `-`. For example, sessions started in `/private/tmp/my_eval` land in `~/.claude/projects/-private-tmp-my_eval/`.

Each transcript contains every assistant message with full Anthropic-API content blocks — `tool_use` blocks have `name` and `input` fields, which is exactly what we need.

If you do want OTel (e.g. for non-Claude-Code agents), set:

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=console
export OTEL_LOG_TOOL_DETAILS=1
claude -p "..." 2>&1 | tee run.jsonl
```

The adapter handles both formats.

---

## 2. Capture a batch of runs

Run the same prompt N times, each in its own working directory, so each session gets its own transcript:

```bash
for i in $(seq 1 10); do
  mkdir -p /tmp/eval_run/run_$i && cd /tmp/eval_run/run_$i
  claude -p "List the files in the current directory using Bash, then read /etc/hostname using Read. Stop after that." \
    --permission-mode acceptEdits >/dev/null 2>&1
done
```

Or run them all in the same cwd — every session still gets a unique `<sessionId>.jsonl`, just under one project folder. Choose whichever is convenient.

---

## 3. Score the trajectories

### Programmatic

```python
from sted import AgentConsistencyEvaluator
from sted.otel_adapter import load_trajectories_from_session_dir

trajectories = load_trajectories_from_session_dir(
    "/Users/me/.claude/projects/-tmp-eval_run"
)
# trajectories is List[List[step]] where each step is {"name": ..., "args": ...}

evaluator = AgentConsistencyEvaluator.for_trajectory()
report = evaluator.evaluate_outputs({"my_prompt": trajectories})
print(report.summary())
```

### CLI

A ready-made runner is shipped at `scripts/eval/score_otel_trajectories.py`:

```bash
# Score one session-transcript folder
python scripts/eval/score_otel_trajectories.py \
    ~/.claude/projects/-tmp-eval_run/*.jsonl

# Or, if you used OTel console export and have one mixed file with prompt.id markers:
python scripts/eval/score_otel_trajectories.py session.jsonl --group-by prompt
```

---

## 4. What you get back

```
Agent Consistency Report
============================================================
Prompts evaluated:    1
Runs per prompt:      10

Mean consistency:     0.849  (high)
Mean validity:        1.000
Mean c_adj:           0.849  (deployment-aware)

Least consistent prompts:
  c_mean=0.849  r_v=1.00  | my_prompt
```

Key fields:

| Field | Meaning |
|---|---|
| `c_mean` | Mean pairwise STED similarity across the C(N,2) run pairs. 1.0 = identical trajectories; lower = the agent's plan diverged. |
| `c_std` | Standard deviation across pairs. High `c_std` with mid `c_mean` = some pairs match, others don't (e.g. 4 of 5 runs identical, 1 outlier). |
| `r_v` | Validity rate — fraction of runs that produced any tool calls. 0 means the agent went silent. |
| `c_adj` | `r_v × c_mean` — drops to 0 if any runs failed entirely, useful for deployment gates. |

---

## 5. Real-world example

Two prompts × 5 runs each, scored on actual Claude Code transcripts.

### 5a. Sample trajectory data

Each session JSONL records assistant messages with Anthropic-style content blocks. The `tool_use` blocks look like this on disk (one per assistant turn that issued a tool call):

```json
{
  "type": "tool_use",
  "name": "Bash",
  "input": {
    "command": "ls -1",
    "description": "List files in current directory"
  }
}
```

After the adapter strips everything except `tool_use` blocks, a full trajectory for the deterministic prompt looks like:

```json
[
  {
    "name": "Bash",
    "args": {
      "command": "ls -1",
      "description": "List files in current directory"
    }
  },
  {
    "name": "Read",
    "args": {
      "file_path": "/Users/me/Documents/genai/projects/sted-internal/LICENSE"
    }
  }
]
```

The open-ended prompt's outlier run (run 5 of 5) is one step longer — the agent ran an extra `Bash` after `Grep`, where the other 4 runs stopped at 2 steps:

```json
[
  {
    "name": "Grep",
    "args": {
      "pattern": "hungarian|linear_sum_assignment|Munkres|munkres",
      "path": "/Users/me/Documents/genai/projects/sted-internal/sted/",
      "glob": "*.py",
      "-i": true,
      "output_mode": "files_with_matches"
    }
  },
  {
    "name": "Bash",
    "args": {
      "command": "ls /.../sted/ 2>&1 | head -20",
      "description": "Check directory exists"
    }
  },
  {
    "name": "Bash",
    "args": {
      "command": "ls /.../sted/",
      "description": "Check directory exists"
    }
  }
]
```

### 5b. Scores

| Prompt | Trajectories observed | c_mean | Interpretation |
|---|---|---|---|
| Specific: `"ls then read LICENSE"` | `[Bash, Read]` × 5 | **1.000** | Plan is fully stable. |
| Open-ended: `"find Hungarian-using files. Use whatever tools you think best."` | `[Grep, Bash]` × 4, `[Grep, Bash, Bash]` × 1 | **0.849** | One run took an extra Bash; flag for prompt review. |

The metric flagged the open-ended phrasing as less stable. In practice this is the signal you want for "is my agent's plan reproducible enough to deploy?"

---

## 6. What the trajectory mode actually configures

`AgentConsistencyEvaluator.for_trajectory()` is shorthand for the underlying STED with:

| Setting | Value |
|---|---|
| `exact_match_all_keys` | `True` — every JSON key (parameter names) must match exactly. |
| `exact_match_fields` | `{"name", "tool", "tool_name", "function", "function_name"}` — values at these fields use string equality, not embeddings. So `search` vs `lookup` scores 0, not ~0.6. |
| `order_sensitive_fields` | `{"trace", "steps", "tool_calls", "messages", "actions", "history"}` — step lists are matched positionally rather than via Hungarian. |

Embedding similarity *still* applies to parameter values (e.g. `command="ls"` vs `command="ls -la"` will score high on the value, while `Bash` vs `Read` on the tool name scores 0).

---

## 7. Limitations to know

- **Step-alignment is purely positional.** Two correct runs that differ only by an extra retry step will be over-penalized at the position where they diverge. A future hybrid mode (Hungarian on tool-call set, positional within a tool) is not yet built.
- **Free-text observations dominate noise.** If you include `tool_result` content in the trajectory, paraphrase variation in tool outputs will lower scores even when the agent's plan was identical. The session adapter currently keeps only `tool_use` blocks (the agent's chosen actions), not results — if you need both, customize the adapter.
- **OTel console export is unreliable in some setups.** If `OTEL_LOGS_EXPORTER=console` produces no output for you, fall back to the session-transcript path — it works without any telemetry config.

---

## 8. End-to-end check

```bash
# 1. Capture
for i in 1 2 3 4 5; do
  cd /tmp/socb_demo
  claude -p "Use Bash to run 'echo hi'. Stop." --permission-mode acceptEdits >/dev/null
done

# 2. Score
python -c "
from sted.otel_adapter import load_trajectories_from_session_dir
from sted import AgentConsistencyEvaluator

trajs = load_trajectories_from_session_dir(
    '/Users/me/.claude/projects/-tmp-socb_demo'
)
print(f'{len(trajs)} trajectories, lengths: {[len(t) for t in trajs]}')
ev = AgentConsistencyEvaluator.for_trajectory()
print(ev.evaluate_outputs({'demo': trajs}).summary())
"
```

Expected output: 5 trajectories of length 1, `c_mean = 1.000`.

---

## See also

- `sted/otel_adapter.py` — input format adapters (Claude Code session JSONL, OTel console JSONL, OTLP JSON).
- `sted/agent_consistency_evaluator.py` — the evaluator class and its `for_trajectory()` constructor.
- `tests/test_otel_adapter.py` — 16 tests covering all input shapes plus end-to-end scoring.
- `scripts/eval/score_otel_trajectories.py` — CLI runner.
