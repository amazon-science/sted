# Using STED with evaluation frameworks

Copy the file for your framework; each is a short, runnable script built on
`sted.compare` / `sted.consistency`. A stand-in agent (`_fake_agent.py`) makes every
example run offline; replace it with your own. CI runs all of them, so they track the
frameworks' current APIs.

LangSmith and Inspect AI have adapters in the package itself
(`sted.integrations.langsmith`, `sted.integrations.inspect_ai`, see the main README).

| Framework | File | What you get | Repeated runs |
|---|---|---|---|
| Braintrust | [`braintrust_example.py`](braintrust_example.py) | code scorer `sted_same` with the differing fields in metadata | `trial_count=N` + `trial_consistency(result)` compares the trials' outputs |
| W&B Weave | [`weave_example.py`](weave_example.py) | `StedMatch` scorer per row | `trials=N` + `StedConsistency.summarize` compares the trials' outputs |
| DeepEval / Confident AI | [`deepeval_example.py`](deepeval_example.py) | `StedMatch(BaseMetric)`, reason lists the differing fields | none built in; call `sted.consistency` on reruns |
| Ragas 0.4 | [`ragas_example.py`](ragas_example.py) | `@numeric_metric` `sted_min_readout` with a reason | none built in |
| MLflow 3 / Databricks | [`mlflow_example.py`](mlflow_example.py) | `@scorer` `sted_same` returning `Feedback` with a rationale | none built in |
| promptfoo | [`promptfoo/`](promptfoo/) | Python assertion `sted_assert.py`, rules from `vars.sted_rules` | `--repeat` reruns; compare outputs with `sted consistency` |

Run one: `cd examples/integrations && python braintrust_example.py` (prints `ok`).

With a real model: `STED_EXAMPLE_AGENT=bedrock python braintrust_example.py` calls Claude
Haiku 4.5 on AWS Bedrock (`STED_EXAMPLE_MODEL`, `STED_EXAMPLE_PROMPT` to change); for
promptfoo, `npx promptfoo@latest eval -c promptfooconfig.real.yaml --repeat 3`. The
examples then print their results instead of checking exact numbers.

Every scorer treats an output that is not JSON as a failed comparison (score 0, the error
as the reason) and `sted.consistency` counts it as invalid. That matters with real models:
on the 2026-11-10 run, a clear prompt ("the first Tuesday of December") gave the same
correct call every time in all six frameworks, while an ambiguous one ("next Friday or the
one after, whichever is cheaper") made the model answer in text instead of calling the
tool, which shows up as validity 0, not as a consistent run.

Framework quirks found while writing these (versions of Nov 2026):

- **DeepEval** copies a metric by calling `__init__` with the attributes that share a
  parameter's name, so keep every parameter explicit and stored under its own name.
- **Ragas 0.4.3** fails to import with `langchain-community` 0.4 (it imports
  `chat_models.vertexai`); pin `langchain-community<0.4`.
- **MLflow 3.18** no longer accepts a file tracking store; use `sqlite:///mlflow.db`.
- **Inspect AI 0.3.288** fails to start with `agent-client-protocol` 1.x; the
  `sted[inspect]` extra pins it below 1.0.

## Checks against a real model

`scripts/integration_checks/` runs the LangSmith and Inspect adapters against a real
tool-calling model on AWS Bedrock (Claude Haiku 4.5 by default, `STED_CHECK_MODEL` to
change it; needs AWS credentials, no LangSmith account) and benchmarks throughput:

```bash
python scripts/integration_checks/langsmith_real_llm.py --reps 5
python scripts/integration_checks/inspect_real_llm.py --epochs 5
python scripts/integration_checks/scale_benchmark.py --mode auto
```

Results on 2026-11-10 (temperature 1.0, 4 prompts × 5 runs): a booking for "next Friday"
gave the right date in 3 of 5 runs and 11-15 or 11-17 otherwise; a weather request varied
`days` (3/5/7) and sometimes added `units`; a hotel search reworded its free-text query
("near the Louvre" / "near Louvre"), which the embedding mode scores 0.997 and accepts; an
explicit booking was identical in all runs. Scoring 300 examples × 5 runs takes about 6 s
in either text mode.
