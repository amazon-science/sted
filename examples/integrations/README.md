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

Framework quirks found while writing these (versions of Nov 2026):

- **DeepEval** copies a metric by calling `__init__` with the attributes that share a
  parameter's name, so keep every parameter explicit and stored under its own name.
- **Ragas 0.4.3** fails to import with `langchain-community` 0.4 (it imports
  `chat_models.vertexai`); pin `langchain-community<0.4`.
- **MLflow 3.18** no longer accepts a file tracking store; use `sqlite:///mlflow.db`.
- **Inspect AI 0.3.288** fails to start with `agent-client-protocol` 1.x; the
  `sted[inspect]` extra pins it below 1.0.
