# STED and Consistency Scoring: A Framework for Evaluating LLM Structured Output Reliability

A framework for measuring consistency in LLM-generated structured outputs. It combines the configurable STED JSON-similarity family with repeated-run aggregation and explicitly separates validity from consistency.

> 📄 **Paper**: [STED and Consistency Scoring: A Framework for Evaluating LLM Structured Output Reliability](docs/STED_and_Consistency_Scoring.pdf) - Accepted at NeurIPS 2025 Workshop on Structured Probabilistic Inference & Generative Modeling

## What's New

**November 2026 — v0.2.0**
- **Breaking: new default scoring.** `SemanticJsonTreeConsistencyEvaluator()` now uses
  `profile="typed"` (the *min readout*): a score of θ means every matched field scores at
  least θ, so a changed argument is no longer averaged away. Scores will differ from
  earlier releases; pass `profile="legacy"` to reproduce them exactly.
- **Two readouts.** `profile="typed"` decides whether two outputs are the same;
  `profile="typed_mean"` grades how far they drift. See the [profile table](#basic-similarity-calculation).
- **Type-aware leaves.** Literals (identifiers, dates, codes) compare exactly, numbers by
  value (`3` equals `"3"`), free text by embedding similarity.
- **20-model results** on Toucan tool calling across 11 temperatures, see
  [Results and Findings](#results-and-findings).
- **MCP server** `sted-mcp` with `compare_outputs` and `evaluate_consistency` tools that
  report which fields differ, see [MCP Server](#mcp-server).
- **`sted.compare(a, b)` and `sted.consistency(outputs)`**: one call each, readable
  output, every differing or unstable field listed, see [Quick Start](#quick-start).
- **Field rules**: `ignore`, `unordered` (arrays as sets) and numeric `tolerance` by path
  pattern, in the library, the CLI and the MCP tools.
- **`sted` command**: `compare`, `consistency` and `diff` subcommands with CI exit codes,
  see [Command line](#command-line).
- **Breaking: no torch by default.** The core install no longer pulls in
  sentence-transformers/torch and compares free-text fields by character similarity
  (with a one-time warning). To keep embedding-based scores, install
  `"sted[embeddings]"`; results report the mode in `text_similarity`.
- **Field-level differences**: `calculate_field_differences(a, b)` lists each differing
  field with its score; under `typed` the score is the lowest of them.
- Fixed a regular expression that could backtrack exponentially on long runs of spaces.

**May 2026**
- `AgentConsistencyEvaluator` with trajectory mode, an OpenTelemetry adapter and the
  `sted-eval` CLI.

**2025**
- STED and Consistency Scoring accepted at the NeurIPS 2025 Workshop on Structured
  Probabilistic Inference & Generative Modeling.

## Table of Contents

- [What's New](#whats-new)
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [STED Effectiveness Verification](#sted-effectiveness-verification)
- [LLM Consistency Benchmarking](#llm-consistency-benchmarking)
- [Key Components](#key-components)
- [Results and Findings](#results-and-findings)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

Large Language Models (LLMs) are increasingly deployed for structured data generation tasks, yet their output consistency remains a critical challenge for production applications. This framework addresses this challenge through two key contributions:

1. **STED (Semantic Tree Edit Distance)**: a JSON tree similarity that aligns two outputs field by field and scores each field by its type (literals exactly, numbers by value, text by embedding similarity). It has two readouts over the same field scores:
   - **min readout** (`profile="typed"`, the default): the lowest field score. A pair scoring 0.95 has *every* matched field at or above 0.95, so a threshold on it decides whether two outputs are the same.
   - **mean readout** (`profile="typed_mean"`): the average field score. It grades how far outputs drift and is suited to monitoring and ranking, but must not be thresholded: one changed field is diluted by the unchanged ones.
2. **Consistency Scoring Framework**: samples each prompt repeatedly and reports **validity** (share of outputs that parse) separately from **agreement** (mean pairwise STED over the valid outputs).

Validation artifacts cover human preference, generated perturbations, constructed triples, and deterministic sandbox outcomes. These instruments disagree in informative ways; no universal semantic-equivalence ranking is claimed.

## Key Features

- **Semantic Tree Edit Distance (STED)**: Advanced similarity calculation combining structural and semantic analysis
- **Multiple Variation Types**: Support for schema, expression, and semantic variations
- **LLM Benchmarking**: Comprehensive evaluation of different LLMs across temperature settings
- **MCP Server**: `sted-mcp` exposes STED to MCP clients (Claude Code, Claude Desktop, other agents) as two tools, see [MCP Server](#mcp-server)
- **Synthetic Dataset Generation**: Automated generation of variation datasets for evaluation
- **Visualization Tools**: Rich plotting and analysis capabilities for results interpretation

## Installation

`sted` is not published on PyPI; `pip install sted` will not install this library.
Install it from GitHub instead:

```bash
# Core: no torch (~170 MB environment); text fields compared by character similarity
pip install "git+https://github.com/amazon-science/sted.git"

# + semantic comparison of free-text fields with a local embedding model (pulls in torch)
pip install "sted[embeddings] @ git+https://github.com/amazon-science/sted.git"

# + MCP server
pip install "sted[embeddings,mcp] @ git+https://github.com/amazon-science/sted.git"
```

Which you need depends on your outputs. Tool calls, IDs, numbers, dates and enums are
compared exactly either way, so the core install is enough for most tool-calling checks.
Install `embeddings` when outputs carry free text (descriptions, queries, summaries) whose
wording may vary: `"hotels near the Louvre"` vs `"hotels close to the Louvre"` scores 0.98
with embeddings and 0.85 by characters. Every result reports which was used
(`text_similarity`); force one with `embedding_model="none"` or a model name, or set
`STED_EMBEDDING_MODEL`.

Or from a local clone:

```bash
git clone https://github.com/amazon-science/sted.git
cd sted
pip install -e .          # or: uv pip install -e .
pip install -e ".[embeddings,mcp]"   # with extras
```

For development:
```bash
pip install -e ".[dev]"
# Or: uv sync
```

### Credentials Setup

**AWS Credentials** (for Bedrock embedding models):
```bash
aws configure
# Or set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
```

**OpenAI API Key** (optional, for OpenAI model evaluation):
```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

### Troubleshooting

**NumPy/PyTorch Compatibility Error**

If you see `_ARRAY_API not found` or "module compiled using NumPy 1.x cannot be run in NumPy 2.x":

```bash
# Option 1: Downgrade NumPy (if using PyTorch < 2.4)
pip install "numpy<2"

# Option 2: Upgrade PyTorch (recommended)
pip install --upgrade "torch>=2.4"
```

## Quick Start

### Compare two outputs

```python
import sted

a = {"name": "book_flight", "arguments": {"from": "SEA", "to": "JFK", "passengers": 2,
                                          "request_id": "r-1", "seats": ["12A", "12B"], "price": 410.00}}
b = {"name": "book_flight", "arguments": {"to": "JFK", "from": "SEA", "passengers": "2",
                                          "request_id": "r-9", "seats": ["12B", "12A"], "price": 410.004}}

print(sted.compare(a, b))
```

```
DIFFERENT  min 0.000  mean 0.750  (threshold 0.95)
  arguments.price                410.0 -> 410.004   [changed, 0.000]
  arguments.request_id           "r-1" -> "r-9"   [changed, 0.000]
  arguments.seats[0]             "12A" -> "12B"   [changed, 0.000]
  arguments.seats[1]             "12B" -> "12A"   [changed, 0.000]
```

Key order and `2` vs `"2"` are not differences. The rest are, until you say otherwise:

```python
r = sted.compare(a, b,
                 ignore=["request_id"],          # drop this field (any depth)
                 unordered=["seats"],            # compare this array as a set
                 tolerance={"price": 0.01})      # numbers equal within 0.01
r.same            # True
r.differences     # []
```

`r.same` is `min_readout >= threshold` (default 0.95): the *lowest* field score, so one
wrong field is enough to make two outputs different. `r.mean_readout` averages the fields
instead, which grades drift but cannot decide sameness. `r.differences` lists every field
that differs with its score, and its lowest score is exactly `r.min_readout`.

### Rate the consistency of repeated runs

Ten runs of the same prompt, e.g. collected from your agent or from logs: six book Dec 1,
three book Dec 2, one refuses.

```python
def call(date, rid):
    return {"name": "book_flight", "arguments": {"from": "SEA", "to": "JFK", "date": date,
                                                 "passengers": 2, "request_id": rid}}

runs = ([call("2026-12-01", f"r-{i}") for i in range(6)]
        + [call("2026-12-02", f"r-{i}") for i in range(6, 9)]
        + ["Sorry, I can't book flights."])
print(sted.consistency(runs, ignore=["request_id"]))
```

```
validity 0.90 (9/10)   agreement min 0.500 / mean 0.938   same pairs 50% of 36
unstable fields:
  arguments.date                 differs in 50% of pairs   "2026-12-01" x6, "2026-12-02" x3
```

Outputs may be JSON values or JSON text; text that does not parse counts as invalid.
`.to_dict()` on either result gives plain JSON for logging or CI.

### Field rules

Patterns are matched against paths as STED prints them (`arguments.items[0].id`):

| Pattern | Matches |
|---|---|
| `request_id` | that key at any depth (a bare name is shorthand for `**.request_id`) |
| `arguments.request_id` | exactly that path |
| `items[*].id` | `id` in every element of `items` |
| `*.id` / `**.id` | `id` one level down / at any depth |

`ignore` drops matching fields from both outputs; `unordered` compares matching arrays as
multisets (other arrays stay positional, since the order of a list argument usually changes
the call); `tolerance` maps a pattern to an absolute tolerance, or to
`{"abs": a, "rel": r}`. The same rules are accepted by the MCP tools. For reuse, build them
once: `rules = sted.FieldRules(ignore=[...])`, then `sted.compare(a, b, rules=rules)`.

### Lower-level evaluator and profiles

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Default profile "typed" = min readout (decides "the same")
same = SemanticJsonTreeConsistencyEvaluator()
# "typed_mean" = mean readout (grades drift)
drift = SemanticJsonTreeConsistencyEvaluator(profile="typed_mean")

ref       = {"name": "get_weather", "arguments": {"city": "Seattle", "days": 3, "units": "metric"}}
reordered = {"arguments": {"units": "metric", "days": "3", "city": "Seattle"}, "name": "get_weather"}
changed   = {"name": "get_weather", "arguments": {"city": "Seattle", "days": 5, "units": "metric"}}

for other in (reordered, changed):
    print(same.calculate_tree_edit_distance_opt(ref, other, variation_type="combined"),
          drift.calculate_tree_edit_distance_opt(ref, other, variation_type="combined"))
# 1.0   1.0     key order and "3" vs 3 are harmless
# 0.0   0.833   a changed argument: min rejects, mean only dips
```

Other profiles are kept as ablations and for reproducing earlier results:

| Profile | What it is |
|---|---|
| `typed` (default) | Min readout: positional arrays, exact keys, typed leaves, bottleneck aggregation |
| `typed_mean` | Mean readout: same as `typed` with arithmetic-mean aggregation |
| `path_f1` (alias `ranking`) | Harmonic mean of path-anchored soft precision/recall; a ranker, not a threshold |
| `certificate` | Bottleneck with literal character leaves (earlier threshold profile) |
| `rank` | Order-adaptive arrays, bottleneck, legacy leaf kernel |
| `legacy` | Original embedding/additive/mean scoring from the NeurIPS 2025 workshop paper |

```python
legacy = SemanticJsonTreeConsistencyEvaluator(profile="legacy", model_id="all-MiniLM-L6-v2")
```

### Batch Consistency Evaluation

```python
from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer

evaluator = SemanticJsonTreeConsistencyEvaluator()
analyzer = StructuralConsistencyAnalyzer(evaluator)

# Multiple LLM outputs for the same prompt
json_outputs = [
    {'name': 'Alice', 'age': 25, 'city': 'New York'},
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Alice', 'age': 25, 'location': 'New York City'},
]

result = analyzer.evaluate_structural_consistency(
    json_outputs, method_name="ted", variation_type="combined"
)
print(f"Consistency: {result['consistency_metrics']['consistency_coefficient']:.4f}")
```

For more examples, see `examples/basic_usage.py` and [Library Usage Guide](./docs/guides/LIBRARY_USAGE.md).

### Agent Consistency Evaluation (high-level API)

`AgentConsistencyEvaluator` is a production-ready wrapper that evaluates an
LLM agent's run-to-run consistency on a set of prompts:

```python
from sted import AgentConsistencyEvaluator

# Either point at a live agent...
evaluator = AgentConsistencyEvaluator(n_workers=4)

def my_agent(prompt: str) -> dict:
    return llm_call(prompt)  # any JSON-returning LLM call

report = evaluator.evaluate(
    agent_fn=my_agent,
    prompts=["Get weather in Seattle", "Book a flight to NYC"],
    n_runs=10,
)

# ...or score pre-collected production logs without re-running the agent
report = evaluator.evaluate_outputs({
    "weather_query": [output_run_1, output_run_2, ...],
    "flight_query": [...],
})

print(report.summary())
# Agent Consistency Report
# ============================================================
# Prompts evaluated:    2
# Runs per prompt:      10
#
# Mean consistency:     0.939  (high)
# Mean validity:        1.000
# Mean c_adj:           0.939  (deployment-aware)
#
# Least consistent prompts:
#   c_mean=0.883  r_v=1.00  | Book a flight to NYC
#   c_mean=0.932  r_v=1.00  | Get weather in Seattle
```

The evaluator separates **validity** ($r_v$ — fraction of runs that produced
parseable output) from **consistency** ($c_\text{mean}$ — pairwise STED on the
parseable subset), matching the deployment-aware framing of the STED paper.

### Command line

```bash
sted compare a.json b.json --ignore request_id      # which fields differ; exit 1 if different
sted consistency runs.jsonl --min-same 0.9          # which fields are unstable; exit 1 below 90% same pairs
sted diff before/ after/ --rules rules.json         # regression check by file; exit 1 if any differ
```

```
$ sted diff before after --rules rules.json
DIFFERENT  q2.json   min 0.000
  arguments.date                 "2026-12-01" -> "2026-12-02"   [changed, 0.000]
MISSING_AFTER q3.json

3 files: 1 different, 1 missing after, 1 same
```

`consistency` reads a JSON array or JSONL of outputs (a line that is not JSON counts as an
invalid output), or rows of `{"prompt": ..., "outputs": [...]}` and rates each prompt.
`diff` pairs files by relative path (`--glob`, default `*.json`). Rules come from
`--ignore` / `--unordered` / `--tolerance PATTERN=VALUE` or a JSON/YAML file with the
same three keys. `--json` prints machine-readable results. Exit codes: 0 same, 1 different,
2 usage or input error, so each command can gate a CI step.

### MCP Server

`sted-mcp` serves STED over the [Model Context Protocol](https://modelcontextprotocol.io)
(stdio), so an agent can check its own outputs or compare runs. Requires Python ≥ 3.10
and `mcp` 1.x (`sted[mcp]` pins `mcp>=1.2,<2`; mcp 2.x is not supported yet).

```bash
pip install "sted[mcp] @ git+https://github.com/amazon-science/sted.git"

# Claude Code
claude mcp add sted -- sted-mcp
```

Claude Desktop (`claude_desktop_config.json`; Desktop does not read your shell `PATH`,
so give the absolute path printed by `which sted-mcp`):

```json
{ "mcpServers": { "sted": { "command": "/absolute/path/to/sted-mcp" } } }
```

| Tool | Input | Returns |
|---|---|---|
| `compare_outputs` | `output_a`, `output_b`, `threshold` (0.95), optional `ignore` / `unordered` / `tolerance` | `min_readout`, `mean_readout`, `same` (= min readout ≥ threshold), `differences` (each differing field: path, changed / missing / extra, both values, field score) |
| `evaluate_consistency` | `outputs` (2–50 generations of one prompt), `threshold` (0.95), optional `ignore` / `unordered` / `tolerance` | `validity`, `agreement_min`, `agreement_mean`, `share_same`, `unstable_fields` (which fields disagree, in what share of pairs, with the values seen), `least_agreeing_pair` with its `differences` |

Outputs can be JSON values or JSON text; text that does not parse counts as invalid.
Free-text fields are embedded locally with `all-MiniLM-L6-v2` (override with
`STED_EMBEDDING_MODEL`), so no cloud credentials are needed. For example, four runs of a
search tool call, one of which is a refusal:

```json
{"n_outputs": 4, "n_valid": 3, "validity": 0.75,
 "agreement_min": 0.333, "agreement_mean": 0.833, "share_same": 0.333,
 "unstable_fields": [{"path": "arguments.limit", "share_of_pairs": 0.667,
                      "values": [{"value": 5, "count": 2}, {"value": 10, "count": 1}]}],
 "least_agreeing_pair": {"i": 0, "j": 2, "min_readout": 0.0,
   "differences": [{"path": "arguments.limit", "status": "changed", "a": 5, "b": 10, "score": 0.0}]}}
```

The refusal is invalid, the two runs that differ only in key order count as the same, and
`unstable_fields` names the one field that varies: `limit`. The minimum readout always
equals the lowest field score, so the listed differences fully explain it. The same
breakdown is available in Python as
`SemanticJsonTreeConsistencyEvaluator(profile="typed").calculate_field_differences(a, b)`.

### Performance

**Benchmark setup**:
- **Workload**: 200 prompts × 10 runs/prompt = **2,000 LLM outputs**, **9,000
  pairwise STED comparisons** (`200 × C(10, 2) = 200 × 45`)
- **Output shape**: synthetic tool-call payloads, 4–6 keys, depth 1–2,
  ~30% per-run drift on one field
- **Hardware**: MacBook M-series laptop (8 cores), single Python process,
  single `SemanticJsonTreeConsistencyEvaluator` instance shared across
  workers
- **Profile used by this historical throughput benchmark**: `legacy`
- **Embedding model for that profile**: `all-MiniLM-L6-v2` (CPU)
- **Reproduce**: `python -c "from sted import AgentConsistencyEvaluator; ..."`
  with the snippet shown above

| Configuration                                       | Time (200 prompts × 10 runs = 2,000 outputs) | Speedup |
|-----------------------------------------------------|---------------------------------------------:|--------:|
| Baseline (sequential, no batch encoding)            |                                     **70 s** |      1× |
| + cross-prompt batch precompute *(default in `evaluate_outputs`)* |                       **11 s** |    6.2× |
| + 4 worker threads (`n_workers=4`)                  |                                    **4.8 s** | **15×** |

**Per-prompt latency at 4 workers**: ~24 ms/prompt average (~0.5 ms per
pairwise STED comparison) including all embedding computation.

Both optimizations are on by default in `evaluate_outputs`; threading is
opt-in via `n_workers` in the constructor and is bit-identical to the
sequential path (regression-tested under `tests/test_agent_consistency_evaluator.py::test_n_workers_produces_same_results_as_sequential`).

Linear extrapolation to larger workloads (`n_workers=4`):
- 1,000 prompts × 10 runs ≈ **24 s**
- 10,000 prompts × 10 runs ≈ **4 min**

Further notes:
- Embedding cache is shared across prompts, so repeated string values
  (common in tool-call args) benefit from cross-prompt cache hits — actual
  speedup on logs with high string overlap can exceed the table above.
- Threads beyond ~4 saturate due to the Python GIL on cosine math; for
  workloads larger than ~10K prompts, prefer multiple processes over more
  threads. If you have a Bedrock embedding backend, use
  `precompute_embeddings()` manually to push the encode batch out to
  async API calls.
- Per-pair STED on small-to-medium JSON (≤50 keys, depth ≤4) is
  ~50–100 ms cold / ~10 ms warm; pathological inputs (deep recursion,
  200+ element arrays) can take seconds because of the inherent O(n³)
  cost of Hungarian assignment on the cost matrix.

### Production caveats (alpha-quality library)

`sted` is currently classified as **Development Status :: 3 - Alpha**.
Suitable for: internal pilots, batch evaluation of production logs,
research benchmarks. **Not yet recommended for**: latency-critical online
inference, mission-critical agent monitoring, regulated workloads.

Known limits to plan around:

| Concern | Status | Mitigation |
|---|---|---|
| Hungarian is O(n³) on array length | known | Set `use_greedy=True` for arrays >30 |
| No SLA / on-call story | not in scope | Use behind your own circuit breaker |
| Thread-safety verified empirically only (no architectural locks) | known | One evaluator per process if paranoid |
| `evaluate_outputs(timeout_seconds=...)` per-pair | available | Default disabled to preserve current semantics |
| Streaming / large-batch via `evaluate_outputs_streaming(...)` | available | Use for >10K prompts |
| Logging via `logging.getLogger("sted")` | available | `sted.set_log_level("DEBUG")` |
| No dashboards / async monitoring | not in scope | See LangSmith / Braintrust for that layer |
| Cosine similarity calibration drifts with embedder | known | Pin embedder version; validate on your domain |

Batch CLI for production logs (one JSON row per prompt with `outputs`):

```bash
sted-eval --input production_logs.jsonl --workers 4 --output report.json
sted-eval --input small.jsonl --summary-only --log-level INFO
```

Input JSONL format: one row per prompt with `{"prompt": "...", "outputs": [...]}`.

Async usage:

```python
import asyncio
from sted import AgentConsistencyEvaluator
from sted.agent_consistency_async import evaluate_async

async def my_async_agent(prompt):
    return await my_async_llm.ainvoke(prompt)

evaluator = AgentConsistencyEvaluator(n_workers=4)
report = asyncio.run(evaluate_async(evaluator, my_async_agent, prompts, n_runs=10))
```

## Dataset

The framework supports two datasets for evaluation:

### ShareGPT Dataset (Structured Output)

JSON structured output generation tasks:
- **sharegpt-structured-output-json**: 30 samples (report extraction, data parsing)
- **sharegpt-quizz-generation-json-output**: 50 samples (MCQ generation)
- **Total**: 80 samples (75 valid after parsing error exclusion)

```bash
# Download ShareGPT data
python scripts/data/download_sharegpt_data.py
```

### Toucan Dataset (Tool Calling)

Tool/function calling evaluation dataset:
- **Source**: [Nexusflow/Toucan](https://huggingface.co/datasets/Nexusflow/Toucan)
- **Content**: Multi-turn conversations with tool calls
- **Use case**: Evaluating LLM tool calling consistency

```bash
# Download Toucan data
python scripts/data/download_toucan_data.py
```

### Generate Synthetic Datasets

For STED effectiveness verification:

```bash
python scripts/data/generate_synthetic_datasets.py --base-dataset-dir sharegpt_data
```

Creates three variation types:
- **Schema Variation**: Field name changes, structure flattening/nesting
- **Expression Variation**: Different expressions with same semantic meaning
- **Semantic Variation**: Changes in semantic content

## STED Effectiveness Verification

### Similarity Progression Analysis

```bash
python scripts/dataset_analysis/analyze_semantic_expression_variation_progression.py \
  synthetic_dataset/expression_variation_dataset_*.json \
  synthetic_dataset/semantic_variation_dataset_*.json \
  --output-dir results/variation_progression
```

### Visualization

```bash
# Expression and Semantic Variation
python scripts/visualization/visualize_variation_progression.py

# Schema Variation
python scripts/visualization/visualize_schema_variation.py
```

## LLM Consistency Benchmarking

The framework supports two experiment modes for different datasets:

| Mode | Dataset | Use Case |
|------|---------|----------|
| `structured` | ShareGPT | JSON structured output generation (MCQ, reports, etc.) |
| `tool-calling` | Toucan | Tool/function calling evaluation |

### Model Setup

The framework supports models from multiple providers. Model configuration is centralized in `sted/model_config.py`:

```python
# sted/model_config.py
MODEL_REGISTRY = {
    # model_id -> (provider, display_name)
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": ("bedrock", "Claude-3.7-Sonnet"),
    "us.deepseek.v3-v1:0": ("bedrock", "DeepSeek-V3.1"),
    "openai/gpt-4o": ("openai", "GPT-4o"),
    "google/gemini-2.5-pro": ("openai", "Gemini-2.5-Pro"),
    # ... add more models here
}
```

**Provider Types:**

| Provider | Model ID Format | API Used | Credentials |
|----------|----------------|----------|-------------|
| `bedrock` | `us.<provider>.<model>-v1:0` | AWS Bedrock Converse API | AWS credentials |
| `openai` | `<provider>/<model>` | OpenAI-compatible API | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |

**To add a new model:**

1. Add entry to `MODEL_REGISTRY` in `sted/model_config.py`:
   ```python
   "us.meta.llama3-3-70b-instruct-v1:0": ("bedrock", "Llama-3.3-70B"),
   ```

2. **Configure credentials** in `.env`:
   ```bash
   # For Bedrock models - only AWS credentials needed (via aws configure)
   
   # For OpenAI-compatible APIs - set these environment variables
   OPENAI_API_KEY=<your-api-key>
   OPENAI_BASE_URL=https://openrouter.ai/api/v1  # Optional, for OpenRouter
   ```

**Note**: When using Bedrock models, you don't need to set `OPENAI_API_KEY`. The script only requires OpenAI credentials when using models with the `"openai"` provider type.

### Step 1: Generate LLM Outputs

#### ShareGPT Experiment (Structured Output)

For JSON structured output generation tasks (MCQ generation, report extraction, etc.):

```bash
# Download ShareGPT data first
python scripts/data/download_sharegpt_data.py

# Run temperature experiment
python scripts/eval/run_temperature_experiment.py \
  --mode structured \
  --data-dir sharegpt_data \
  --output-dir llm_gen_results/sharegpt \
  --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --max-tokens 8000
```

**Key parameters:**
- `--mode structured`: Use structured output mode (default)
- `--data-dir`: Directory containing ShareGPT JSON files
- `--run-num`: Number of inference runs per sample (default: 10)
- `--sample-limit`: Maximum samples to process (default: all samples)
- `--include-schema`: Include JSON schema in prompt for better formatting

#### Toucan Experiment (Tool Calling)

For tool/function calling evaluation:

```bash
# Download Toucan data first
python scripts/data/download_toucan_data.py

# Run temperature experiment
python scripts/eval/run_temperature_experiment.py \
  --mode tool-calling \
  --dataset-path toucan_data/toucan_tool_calls.json \
  --output-dir llm_gen_results/toucan \
  --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --max-tokens 1024
```

**Key parameters:**
- `--mode tool-calling`: Use tool calling mode
- `--dataset-path`: Path to Toucan dataset JSON file
- `--run-num`: Number of inference runs per sample (default: 10)
- `--sample-limit`: Maximum samples to process (default: all samples)
- `--max-workers`: Parallel workers for inference (default: 10)
- `--start-idx`: Starting index in dataset (default: 0)

#### Running Individual Temperatures

For finer control, you can run `generate_structured_outputs.py` or `generate_tool_calls.py` directly:

```bash
# ShareGPT - single temperature
python scripts/eval/generate_structured_outputs.py \
  --data-dir sharegpt_data \
  --output-dir llm_gen_results/sharegpt/generations-model-timestamp \
  --temperature 0.5 \
  --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0

# Toucan - single temperature
python scripts/eval/generate_tool_calls.py \
  --dataset-path toucan_data/toucan_tool_calls.json \
  --dataset-type toucan \
  --output-dir llm_gen_results/toucan/generations-model-timestamp \
  --temperature 0.5 \
  --model us.anthropic.claude-3-5-haiku-20241022-v1:0
```

### Step 2: Calculate Consistency Metrics

```bash
python scripts/eval/calculate_consistency_metrics.py
```

### Step 3: Visualize Results

```bash
python scripts/visualization/visualize_consistency_scores.py
```

![LLM Consistency Scores](images/consistency_score_by_consistency_type_with_errors.png)

For all scripts, see [Scripts Reference](./docs/guides/SCRIPTS_REFERENCE.md).

## Key Components

### STED Algorithm

STED aligns two JSON trees and scores matched leaves with a type-aware kernel:

- **Typed leaves**: booleans/null and literals (identifiers, dates, codes, numbers) compare exactly; a JSON number equals a numeric string of the same value; free text compares by embedding similarity
- **Readouts**: the min readout (`typed`, default) takes the worst matched field, so a threshold certifies every field; the mean readout (`typed_mean`) averages them to grade drift
- **Legacy profile**: embedding-aware labels/values, bag-style array matching, additive channels, and arithmetic-mean aggregation; retained for historical reproducibility
- **Hungarian matching**: computes the configured local assignment in O(n³) time
- **Multi-Level Similarity**: Integrates structural, key, value, and type similarities with configurable weights

The semantic update cost: `γ_upd(v1, v2) = w_s · γ_struct(v1, v2) + w_c · γ_content(v1, v2)`

### Variation Types

| Type | Description | Example |
|------|-------------|---------|
| Schema | Field name/structure changes | `"user_name"` → `"userName"` |
| Expression | Linguistic variations, same meaning | `"good book"` → `"nice book"` |
| Semantic | Content meaning changes | Should trigger alerts |

### Algorithm Complexity

- Tree construction: O(n)
- Embedding computation: O(k) with caching
- Optimized STED: O(n₁ × n₂ × (n₁ + n₂))
- Hungarian algorithm: O(max(n₁, n₂)³)

See [STED Complexity Analysis](./docs/api/sted_complexity_analysis.md) for details.

## Results and Findings

![Tool-call consistency of 20 LLMs](images/socb_toucan_consistency.png)

**20 LLMs** on Toucan tool calling: 1,006 prompts, 10 generations per prompt at each of
11 temperatures (0.0–1.0), scored with STED's min readout (`profile="typed"`).
Agreement $c_{mean}$ is the mean pairwise STED among a prompt's *valid* outputs; validity
$r_v$ is the share of outputs that parse. They are reported separately.

- **No model is fully consistent at T = 0.** Agreement ranges from 0.26 (Mistral-Large-3)
  to 0.98 (Claude-3.5-Haiku).
- **Temperature effects are model-specific.** Claude-Haiku-4.5 drops from 0.94 to 0.71
  between T = 0 and T = 1; GPT-4.1-Mini is unchanged (0.60); Llama-3.3-70B rises slightly.
- **Validity and agreement are different axes.** Llama-3.3-70B parses only 33% of the
  time; Grok-4.1-Fast parses 95% but agrees at 0.68.

| Provider | Models |
|----------|--------|
| Anthropic | Claude 3.5 Haiku, Claude 3.5 Sonnet, Claude 3.7 Sonnet, Claude Sonnet 4, Claude Opus 4, Claude Haiku 4.5, Claude Sonnet 4.5, Claude Opus 4.5 |
| Amazon | Nova 2 Lite |
| OpenAI | GPT-4.1 Mini, GPT-OSS-120B |
| Google | Gemini 2.5 Flash Lite |
| Meta | Llama 3.3 70B |
| Alibaba | Qwen3 32B, Qwen3 235B A22B |
| Mistral | Mistral Large 3 (675B) |
| NVIDIA | Nemotron 3 Nano 30B A3B |
| xAI | Grok 4.1 Fast |
| MiniMax | MiniMax M2 |
| Xiaomi | MiMo V2 Flash |

The NeurIPS 2025 workshop paper evaluated an earlier set of 19 models (Claude 3/3.5/3.7,
Nova, Llama 3.1/3.3/4, GPT-4o, Gemini 2.5, DeepSeek V3, Qwen3) with the `legacy` profile.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{wang2025sted,
  title={STED and Consistency Scoring: A Framework for Evaluating LLM Structured Output Reliability},
  author={Wang, Guanghui and Yu, Jinze and Zhang, Xing and Jiang, Dayuan and Deb, Tomal and Liu, Xuefeng and He, Peiyang and Song, Yin},
  booktitle={NeurIPS 2025 Workshop on Structured Probabilistic Inference \& Generative Modeling},
  year={2025}
}
```
