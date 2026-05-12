"""Tests for AgentConsistencyEvaluator."""
import json

import pytest

from sted import AgentConsistencyEvaluator, ConsistencyReport, PromptResult


# ---------- Test fixtures ----------

def deterministic_agent(prompt: str) -> dict:
    """Always returns the same output regardless of prompt content."""
    return {"tool_name": "search", "args": {"q": "fixed"}}


def per_prompt_deterministic_agent(prompt: str) -> dict:
    """Returns a prompt-determined but per-call-stable output."""
    return {"echo": prompt, "len": len(prompt)}


class StochasticAgent:
    """Cycles through a fixed sequence of outputs to simulate variance."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def __call__(self, prompt: str):
        out = self.outputs[self.calls % len(self.outputs)]
        self.calls += 1
        return out


@pytest.fixture
def evaluator():
    return AgentConsistencyEvaluator(max_parallel_runs=1)


# ---------- Basic happy paths ----------


def test_perfectly_consistent_agent(evaluator):
    report = evaluator.evaluate(deterministic_agent, ["q1", "q2"], n_runs=5)
    assert isinstance(report, ConsistencyReport)
    assert report.n_prompts == 2
    assert report.n_runs == 5
    assert report.mean_validity == pytest.approx(1.0)
    assert report.mean_consistency == pytest.approx(1.0, abs=1e-6)
    assert report.mean_c_adj == pytest.approx(1.0, abs=1e-6)
    for r in report.per_prompt:
        assert r.n_valid == 5
        assert r.r_v == 1.0
        assert r.c_mean == pytest.approx(1.0, abs=1e-6)
        assert r.c_std == pytest.approx(0.0, abs=1e-6)


def test_per_prompt_deterministic_perfectly_consistent(evaluator):
    """Different outputs per prompt, but each prompt repeats stably."""
    report = evaluator.evaluate(
        per_prompt_deterministic_agent, ["abc", "defgh"], n_runs=4
    )
    assert report.mean_consistency == pytest.approx(1.0, abs=1e-6)
    # Both prompts should hit c_mean=1
    for r in report.per_prompt:
        assert r.c_mean == pytest.approx(1.0, abs=1e-6)


def test_two_distinct_outputs_alternating():
    """Two outputs alternating produce intermediate consistency."""
    agent = StochasticAgent([
        {"tool": "A", "args": {"x": 1}},
        {"tool": "B", "args": {"x": 2}},
    ])
    evaluator = AgentConsistencyEvaluator(max_parallel_runs=1)
    report = evaluator.evaluate(agent, ["only-prompt"], n_runs=4)
    r = report.per_prompt[0]
    assert r.n_valid == 4
    # For 4 runs alternating ABAB: pairs are (A,B),(A,A),(A,B),(B,A),(B,B),(A,B)
    # i.e. C(4,2)=6 pairs; 2 same-pairs (A,A) and (B,B), 4 different.
    # c_mean should be strictly between 0 and 1.
    assert 0.0 < r.c_mean < 1.0
    assert r.r_v == 1.0


# ---------- Validity edge cases ----------


def test_invalid_outputs_lower_validity():
    """None / empty outputs reduce r_v but keep valid pairs scoring well."""
    agent = StochasticAgent([
        {"tool": "A"},
        None,
        {"tool": "A"},
        "",
        {"tool": "A"},
    ])
    evaluator = AgentConsistencyEvaluator(max_parallel_runs=1)
    report = evaluator.evaluate(agent, ["p"], n_runs=5)
    r = report.per_prompt[0]
    assert r.n_runs == 5
    assert r.n_valid == 3  # 3 dicts, 1 None, 1 empty string
    assert r.r_v == pytest.approx(0.6)
    assert r.c_mean == pytest.approx(1.0, abs=1e-6)  # all 3 valid are identical
    assert r.c_adj == pytest.approx(0.6, abs=1e-6)


def test_all_invalid_returns_zero():
    agent = StochasticAgent([None, "", [], {}])
    evaluator = AgentConsistencyEvaluator(max_parallel_runs=1)
    report = evaluator.evaluate(agent, ["p"], n_runs=4)
    r = report.per_prompt[0]
    assert r.n_valid == 0
    assert r.r_v == 0.0
    assert r.c_mean == 0.0
    assert r.c_adj == 0.0


def test_one_valid_output_insufficient_for_consistency():
    """With only 1 valid output, c_mean is undefined and reported as 0."""
    agent = StochasticAgent([{"tool": "A"}, None, None, None])
    evaluator = AgentConsistencyEvaluator(max_parallel_runs=1)
    report = evaluator.evaluate(agent, ["p"], n_runs=4)
    r = report.per_prompt[0]
    assert r.n_valid == 1
    assert r.r_v == 0.25
    assert r.c_mean == 0.0  # cannot compute from 1 sample
    assert "fewer than 2 valid" in (r.error or "")


# ---------- Agent error handling ----------


def test_agent_raising_exception_treated_as_invalid():
    calls = {"n": 0}

    def flaky(prompt: str):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("transient")
        return {"ok": True}

    evaluator = AgentConsistencyEvaluator(max_parallel_runs=1)
    report = evaluator.evaluate(flaky, ["p"], n_runs=4)
    r = report.per_prompt[0]
    assert r.n_valid == 2  # only odd-indexed calls succeed
    assert r.r_v == 0.5


# ---------- evaluate_outputs (no agent invocation) ----------


def test_evaluate_outputs_from_logs(evaluator):
    outputs = {
        "prompt1": [{"a": 1}, {"a": 1}, {"a": 1}],
        "prompt2": [{"x": 1}, {"x": 2}, {"x": 3}],
    }
    report = evaluator.evaluate_outputs(outputs)
    assert report.n_prompts == 2
    p1 = next(r for r in report.per_prompt if r.prompt == "prompt1")
    p2 = next(r for r in report.per_prompt if r.prompt == "prompt2")
    assert p1.c_mean == pytest.approx(1.0, abs=1e-6)
    assert p2.c_mean < 1.0  # 3 different outputs


# ---------- evaluate_pair ----------


def test_evaluate_pair_identical(evaluator):
    s = evaluator.evaluate_pair({"a": 1}, {"a": 1})
    assert s == pytest.approx(1.0, abs=1e-6)


def test_evaluate_pair_invalid(evaluator):
    assert evaluator.evaluate_pair(None, {"a": 1}) == 0.0
    assert evaluator.evaluate_pair("", {"a": 1}) == 0.0


# ---------- Report API ----------


def test_report_summary_runs_without_error(evaluator):
    report = evaluator.evaluate(
        deterministic_agent, ["q1", "q2", "q3"], n_runs=3
    )
    summary = report.summary()
    assert isinstance(summary, str)
    assert "Agent Consistency Report" in summary
    assert "Mean consistency" in summary


def test_report_to_dict_is_serializable(evaluator):
    report = evaluator.evaluate(
        deterministic_agent, ["q1", "q2"], n_runs=3
    )
    d = report.to_dict()
    # Should be JSON-serializable
    json_str = json.dumps(d)
    assert "n_prompts" in json_str
    assert "per_prompt" in json_str


def test_report_worst_prompts_ranked_by_c_mean():
    """Confirm worst_prompts is ascending by c_mean."""
    # Build agent that gives different consistency per prompt.
    counter = {"q1": 0, "q2": 0}

    def varied(prompt: str):
        counter[prompt] += 1
        if prompt == "q1":
            return {"x": 1}  # always same -> high consistency
        else:
            # Alternate between two outputs -> low consistency
            return {"x": counter[prompt] % 2}

    evaluator = AgentConsistencyEvaluator(max_parallel_runs=1)
    report = evaluator.evaluate(varied, ["q1", "q2"], n_runs=4, bottom_k=2)
    # q2 should be the lowest-consistency prompt
    assert report.worst_prompts[0].prompt == "q2"
    assert report.worst_prompts[0].c_mean < report.worst_prompts[1].c_mean


# ---------- JSON-string output handling ----------


def test_json_string_outputs_parsed(evaluator):
    """Agents returning JSON strings should be treated as their parsed form."""
    agent = StochasticAgent([
        '{"tool": "A"}',
        '{"tool": "A"}',
        '{"tool": "A"}',
    ])
    report = evaluator.evaluate(agent, ["p"], n_runs=3)
    r = report.per_prompt[0]
    assert r.n_valid == 3
    # Note: even though they're identical strings, _is_valid returns True for
    # non-empty strings; the underlying STED handles equal strings by scoring
    # them identically (>=0.95 on canonicalized form).
    assert r.c_mean >= 0.95


# ---------- Bug fixes regression tests ----------


def test_list_of_nones_rejected_as_invalid(evaluator):
    """List with all None elements should not be considered valid."""
    agent = StochasticAgent([
        [None, None, None],
        [None, None, None],
    ])
    report = evaluator.evaluate(agent, ["p"], n_runs=2)
    r = report.per_prompt[0]
    assert r.n_valid == 0  # both lists contain only invalid entries
    assert r.r_v == 0.0


def test_strict_json_mode_rejects_plain_strings():
    """With strict_json, 'I don't know' and similar non-JSON strings are invalid."""
    from sted.agent_consistency_evaluator import _is_valid

    assert _is_valid("I don't know") is True  # default lenient
    assert _is_valid("I don't know", strict_json=True) is False
    # JSON-shaped strings still parse in strict mode
    assert _is_valid('{"a": 1}', strict_json=True) is True
    assert _is_valid("[1, 2]", strict_json=True) is True


def test_agent_error_propagated_to_result():
    """When agent raises, error message is captured in PromptResult.error."""
    def always_raises(prompt):
        raise ValueError("simulated network error")

    evaluator = AgentConsistencyEvaluator(max_parallel_runs=1)
    report = evaluator.evaluate(always_raises, ["p"], n_runs=4)
    r = report.per_prompt[0]
    assert r.n_valid == 0
    assert r.error is not None
    assert "ValueError" in r.error
    assert "simulated network error" in r.error


def test_iterable_prompts_accepted(evaluator):
    """prompts can be a generator (Iterable, not just Sequence)."""
    report = evaluator.evaluate(
        deterministic_agent,
        (f"prompt-{i}" for i in range(3)),
        n_runs=2,
    )
    assert report.n_prompts == 3


def test_empty_prompts_list_no_crash(evaluator):
    """Edge case: zero prompts."""
    report = evaluator.evaluate(deterministic_agent, [], n_runs=3)
    assert report.n_prompts == 0
    # summary should not crash on empty input
    summary = report.summary()
    assert "Mean consistency" in summary


def test_bottom_k_larger_than_n_prompts(evaluator):
    """bottom_k=10 with 2 prompts should not error and should return ≤2."""
    report = evaluator.evaluate(
        deterministic_agent, ["p1", "p2"], n_runs=3, bottom_k=10
    )
    assert len(report.worst_prompts) == 2
    assert len(report.most_invalid) == 2


# ---------- Parallelism: workers + cross-prompt precompute ----------


def test_n_workers_produces_same_results_as_sequential():
    """Threading must not change scores."""
    outputs = {
        f"p_{i}": [{"f": v} for v in [i, i, i, i+1]]  # 4 outputs each
        for i in range(8)
    }

    seq = AgentConsistencyEvaluator(n_workers=1)
    par = AgentConsistencyEvaluator(n_workers=4)

    r_seq = seq.evaluate_outputs(outputs)
    r_par = par.evaluate_outputs(outputs)

    seq_dict = {r.prompt: r.c_mean for r in r_seq.per_prompt}
    par_dict = {r.prompt: r.c_mean for r in r_par.per_prompt}
    for k in seq_dict:
        assert abs(seq_dict[k] - par_dict[k]) < 1e-6, \
            f"{k}: seq={seq_dict[k]} par={par_dict[k]}"


def test_precompute_embeddings_does_not_change_results():
    """precompute_embeddings is a perf optimization — must not change scores."""
    outputs = {
        f"p_{i}": [{"tool": "search", "q": f"v_{i}_{j}"} for j in range(4)]
        for i in range(5)
    }

    ev = AgentConsistencyEvaluator(n_workers=1)
    r_with = ev.evaluate_outputs(outputs, precompute_embeddings=True)

    ev2 = AgentConsistencyEvaluator(n_workers=1)
    r_without = ev2.evaluate_outputs(outputs, precompute_embeddings=False)

    with_dict = {r.prompt: r.c_mean for r in r_with.per_prompt}
    without_dict = {r.prompt: r.c_mean for r in r_without.per_prompt}
    for k in with_dict:
        assert abs(with_dict[k] - without_dict[k]) < 1e-6, \
            f"{k}: with={with_dict[k]} without={without_dict[k]}"


def test_evaluate_with_workers_runs(evaluator):
    """Verify the threaded path of evaluate() runs without crashing."""
    par = AgentConsistencyEvaluator(n_workers=2)
    report = par.evaluate(deterministic_agent, ["p1", "p2", "p3"], n_runs=3)
    assert report.n_prompts == 3
    assert all(r.c_mean == pytest.approx(1.0, abs=1e-6) for r in report.per_prompt)
