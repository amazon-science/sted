"""MCP server exposing STED consistency scoring to agents.

Run it over stdio (the transport MCP clients such as Claude Code and Claude Desktop
launch)::

    pip install "sted[mcp] @ git+https://github.com/amazon-science/sted.git"
    sted-mcp

Tools
-----
``compare_outputs(output_a, output_b, threshold=0.95)``
    Both STED readouts for one pair. ``min_readout`` is the lowest matched-field
    score, so ``same`` (min_readout >= threshold) means every field agrees at that
    level. ``mean_readout`` averages the fields and grades how far apart they are.

``evaluate_consistency(outputs, threshold=0.95)``
    Repeated outputs of one prompt: validity (share that parse), agreement under
    both readouts over all pairs of valid outputs, the share of pairs that are the
    same, and the least-agreeing pair.

Outputs may be JSON values or JSON text. Text that does not parse as JSON counts as
invalid. Embeddings for free-text fields use a local Sentence-Transformers model
(``STED_EMBEDDING_MODEL``, default ``all-MiniLM-L6-v2``); no cloud credentials are
needed. Models load on the first tool call, not at server start.
"""
# No `from __future__ import annotations`: FastMCP inspects real annotation
# objects when it builds the tool schemas.
import contextlib
import functools
import json
import os
import statistics
import sys
import threading
from typing import Any, Dict, List, Optional

MAX_OUTPUTS = 50                 # 1,225 pairs; keeps one tool call bounded
MAX_DIFFS = 20                   # differing fields listed per pair
MAX_DIFF_PAIRS = 300             # pairs searched for unstable fields (evenly sampled beyond)
MAX_VALUE_CHARS = 200            # long string values are truncated in the response
PAIR_TIMEOUT_SECONDS = 30.0
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_lock = threading.Lock()
_evaluators: Dict[str, Any] = {}


def _quiet(fn):
    """Send library prints to stderr: over stdio, stdout carries the MCP protocol."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)
    return wrapper


def _embedding_model() -> str:
    return os.environ.get("STED_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _agent_evaluator(profile: str):
    """One AgentConsistencyEvaluator per profile, built on first use."""
    with _lock:
        if profile not in _evaluators:
            from .agent_consistency_evaluator import AgentConsistencyEvaluator
            from .semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

            sted = SemanticJsonTreeConsistencyEvaluator(profile=profile, model_id=_embedding_model())
            _evaluators[profile] = AgentConsistencyEvaluator(
                evaluator=sted, timeout_seconds=PAIR_TIMEOUT_SECONDS)
        return _evaluators[profile]


def _parse(output: Any) -> Any:
    """JSON text -> value; text that is not JSON -> None (invalid)."""
    if isinstance(output, str):
        try:
            return json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return None
    return output


def _short(v: Any) -> Any:
    if isinstance(v, str) and len(v) > MAX_VALUE_CHARS:
        return v[:MAX_VALUE_CHARS] + "..."
    return v


def _differences(a: Any, b: Any, limit: int = MAX_DIFFS) -> List[Dict[str, Any]]:
    """Fields that differ under the min readout, lowest score first."""
    diffs = _agent_evaluator("typed")._evaluator.calculate_field_differences(a, b)
    return [{**d, "a": _short(d["a"]), "b": _short(d["b"]), "score": round(d["score"], 6)}
            for d in diffs[:limit]]


def _check_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")


@_quiet
def compare_outputs(output_a: Any, output_b: Any, threshold: float = 0.95) -> Dict[str, Any]:
    """Compare two structured outputs (JSON objects, arrays, or JSON text) with STED.

    Returns min_readout (lowest matched-field score; use it to decide whether the
    outputs are the same), mean_readout (average field score; use it to grade how far
    apart they are), same = min_readout >= threshold, and differences: the fields that
    differ, lowest score first, each with path, status (changed / missing from b /
    extra in b), the two values and the field score. min_readout equals the lowest
    field score.
    """
    _check_threshold(threshold)
    a, b = _parse(output_a), _parse(output_b)
    for name, raw, val in (("output_a", output_a, a), ("output_b", output_b, b)):
        if val is None:
            raise ValueError(f"{name} is not valid JSON: {str(raw)[:80]!r}")
    # STED directly: AgentConsistencyEvaluator.evaluate_pair scores an empty object
    # as invalid (0.0), but an explicit pair of {} and {} is identical.
    lo = _agent_evaluator("typed")._evaluator.calculate_tree_edit_distance_fast(a, b)
    mean = _agent_evaluator("typed_mean")._evaluator.calculate_tree_edit_distance_fast(a, b)
    return {
        "min_readout": round(float(lo), 6),
        "mean_readout": round(float(mean), 6),
        "same": bool(lo >= threshold),
        "threshold": threshold,
        "differences": _differences(a, b),
    }


def _pairs(n: int) -> List[tuple]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


@_quiet
def evaluate_consistency(outputs: List[Any], threshold: float = 0.95) -> Dict[str, Any]:
    """Rate the consistency of repeated outputs of the same prompt.

    outputs: 2 to 50 generations (JSON values or JSON text). Returns validity (share
    that parse), agreement_min / agreement_mean (mean pairwise STED over valid outputs
    under each readout), share_same (pairs with min readout >= threshold),
    unstable_fields (fields scoring below threshold in some pair: the share of pairs
    in which each disagrees, and the values seen across outputs), and the least-agreeing pair by index into
    `outputs` with its differing fields.
    """
    _check_threshold(threshold)
    if not isinstance(outputs, list) or len(outputs) < 2:
        raise ValueError("outputs must be a list of at least 2 generations")
    if len(outputs) > MAX_OUTPUTS:
        raise ValueError(f"at most {MAX_OUTPUTS} outputs per call, got {len(outputs)}")

    parsed = [_parse(o) for o in outputs]
    results = {
        profile: _agent_evaluator(profile).evaluate_outputs({"prompt": parsed}).per_prompt[0]
        for profile in ("typed", "typed_mean")
    }
    lo, mean = results["typed"], results["typed_mean"]
    out: Dict[str, Any] = {
        "n_outputs": len(outputs),
        "n_valid": lo.n_valid,
        "validity": round(lo.r_v, 6),
    }
    if lo.n_valid < 2:
        out.update(agreement_min=None, agreement_mean=None, share_same=None,
                   least_agreeing_pair=None, note="fewer than 2 valid outputs")
        return out

    # Pairwise lists follow the order of the valid outputs; map back to input indices.
    from .agent_consistency_evaluator import _is_valid
    valid_idx = [i for i, o in enumerate(parsed) if _is_valid(o)]
    pairs = _pairs(len(valid_idx))
    lo_sims = lo.pairwise_similarities
    scored = [(s, p) for s, p in zip(lo_sims, pairs) if s is not None]
    worst_s, (wi, wj) = min(scored, key=lambda t: t[0])
    finite_mean = [s for s in mean.pairwise_similarities if s is not None]

    from ._paths import EmptyContainer, terminal_paths
    from .semantic_json_tree_consistency import format_json_path

    # A pair at or above threshold has every field at or above it, so only pairs
    # below threshold can hold an unstable field.
    below = [(s, p) for s, p in scored if s < threshold]
    n_below = len(below)
    if len(below) > MAX_DIFF_PAIRS:
        step = len(below) / MAX_DIFF_PAIRS
        below = [below[int(k * step)] for k in range(MAX_DIFF_PAIRS)]
    sted = _agent_evaluator("typed")._evaluator
    counts: Dict[str, int] = {}
    for _, (i, j) in below:
        for d in sted.calculate_field_differences(parsed[valid_idx[i]], parsed[valid_idx[j]]):
            if d["score"] < threshold:
                counts[d["path"]] = counts.get(d["path"], 0) + 1

    leaves = [{format_json_path(p): v for p, v in terminal_paths(parsed[k]).items()} for k in valid_idx]
    def values_at(path):
        seen: Dict[str, List[Any]] = {}
        for lv in leaves:
            v = lv.get(path, "<absent>")
            if isinstance(v, EmptyContainer):
                v = {} if v is EmptyContainer.OBJECT else []
            key = json.dumps(v, sort_keys=True, default=str)
            seen.setdefault(key, [_short(v), 0])[1] += 1
        ranked = sorted(seen.values(), key=lambda t: -t[1])[:5]
        return [{"value": v, "count": c} for v, c in ranked]

    # Share of all valid pairs in which the field disagrees (scaled up from the
    # sample when the disagreeing pairs were sampled).
    scale = (n_below / len(below)) / len(scored) if below else 0.0
    unstable = [{"path": path, "share_of_pairs": round(c * scale, 6), "values": values_at(path)}
                for path, c in sorted(counts.items(), key=lambda t: (-t[1], t[0]))[:MAX_DIFFS]]

    out.update(
        agreement_min=round(statistics.mean(s for s, _ in scored), 6),
        agreement_mean=round(statistics.mean(finite_mean), 6) if finite_mean else None,
        share_same=round(sum(s >= threshold for s, _ in scored) / len(scored), 6),
        n_pairs=len(pairs),
        threshold=threshold,
        unstable_fields=unstable,
        least_agreeing_pair={"i": valid_idx[wi], "j": valid_idx[wj], "min_readout": round(worst_s, 6),
                             "differences": _differences(parsed[valid_idx[wi]], parsed[valid_idx[wj]])},
    )
    if n_below > MAX_DIFF_PAIRS:
        out["unstable_fields_note"] = (f"share_of_pairs estimated from an even sample of "
                                       f"{MAX_DIFF_PAIRS} of the {n_below} disagreeing pairs")
    errors = {r.error for r in (lo, mean) if r.error}
    if errors:
        out["note"] = "; ".join(sorted(errors))
    return out


def build_server(name: str = "sted"):
    """The FastMCP server with both tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The STED MCP server needs mcp>=1.2,<2 (mcp 2.x renamed FastMCP). "
            'Install with: pip install "mcp>=1.2,<2"'
        ) from e
    server = FastMCP(name)
    server.tool()(compare_outputs)
    server.tool()(evaluate_consistency)
    return server


def main(argv: Optional[List[str]] = None) -> None:
    """Console entry point `sted-mcp`: serve over stdio."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
