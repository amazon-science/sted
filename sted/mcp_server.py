"""MCP server exposing STED consistency scoring to agents.

Run it over stdio (the transport MCP clients such as Claude Code and Claude Desktop
launch)::

    pip install "sted[mcp] @ git+https://github.com/amazon-science/sted.git"
    sted-mcp

Tools (thin wrappers over ``sted.compare`` / ``sted.consistency``)
-----
``compare_outputs(output_a, output_b, threshold=0.95, ignore, unordered, tolerance)``
    Both STED readouts for one pair and every differing field. ``min_readout`` is the
    lowest field score, so ``same`` (min_readout >= threshold) means every field agrees
    at that level. ``mean_readout`` averages the fields and grades how far apart they are.

``evaluate_consistency(outputs, threshold=0.95, ignore, unordered, tolerance)``
    Repeated outputs of one prompt: validity (share that parse), agreement under both
    readouts over all pairs of valid outputs, the share of pairs that are the same, the
    fields that disagree, and the least-agreeing pair.

Field rules use path patterns: ``request_id`` (that key at any depth),
``arguments.tags``, ``items[*].id``, ``**.timestamp``.

Outputs may be JSON values or JSON text. Text that does not parse as JSON counts as
invalid. Embeddings for free-text fields use a local Sentence-Transformers model
(``STED_EMBEDDING_MODEL``, default ``all-MiniLM-L6-v2``); no cloud credentials are
needed. Models load on the first tool call, not at server start.
"""
# No `from __future__ import annotations`: FastMCP inspects real annotation
# objects when it builds the tool schemas.
import contextlib
import functools
import sys
from typing import Any, Dict, List, Optional

from . import api

MAX_OUTPUTS = 50                 # 1,225 pairs; keeps one tool call bounded
MAX_DIFFS = 20                   # differing fields listed per pair
MAX_DIFF_PAIRS = 300             # pairs searched for unstable fields (evenly sampled beyond)
MAX_VALUE_CHARS = api.MAX_VALUE_CHARS


def _quiet(fn):
    """Send library prints to stderr: over stdio, stdout carries the MCP protocol."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)
    return wrapper


@_quiet
def compare_outputs(output_a: Any, output_b: Any, threshold: float = 0.95,
                    ignore: Optional[List[str]] = None, unordered: Optional[List[str]] = None,
                    tolerance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Compare two structured outputs (JSON objects, arrays, or JSON text) with STED.

    Returns min_readout (lowest field score; use it to decide whether the outputs are
    the same), mean_readout (average field score; use it to grade how far apart they
    are), same = min_readout >= threshold, and differences: the fields that differ,
    lowest score first, each with path, status (changed / missing from b / extra in b),
    the two values and the field score. min_readout equals the lowest field score.

    Optional field rules (path patterns such as "request_id", "arguments.tags",
    "items[*].id", "**.timestamp"): ignore = fields to drop before comparing;
    unordered = arrays to compare as sets; tolerance = {pattern: absolute tolerance}
    for numbers.
    """
    r = api.compare(output_a, output_b, threshold=threshold, ignore=ignore, unordered=unordered,
                    tolerance=tolerance, max_differences=MAX_DIFFS)
    return r.to_dict()


@_quiet
def evaluate_consistency(outputs: List[Any], threshold: float = 0.95,
                         ignore: Optional[List[str]] = None, unordered: Optional[List[str]] = None,
                         tolerance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Rate the consistency of repeated outputs of the same prompt.

    outputs: 2 to 50 generations (JSON values or JSON text). Returns validity (share
    that parse), agreement_min / agreement_mean (mean pairwise STED over valid outputs
    under each readout), share_same (pairs with min readout >= threshold),
    unstable_fields (fields scoring below threshold in some pair: the share of pairs
    in which each disagrees, and the values seen across outputs), and the
    least-agreeing pair by index into `outputs` with its differing fields.

    Optional field rules as in compare_outputs (ignore / unordered / tolerance).
    """
    if not isinstance(outputs, list) or len(outputs) < 2:
        raise ValueError("outputs must be a list of at least 2 generations")
    if len(outputs) > MAX_OUTPUTS:
        raise ValueError(f"at most {MAX_OUTPUTS} outputs per call, got {len(outputs)}")
    r = api.consistency(outputs, threshold=threshold, ignore=ignore, unordered=unordered,
                        tolerance=tolerance, max_fields=MAX_DIFFS, max_diff_pairs=MAX_DIFF_PAIRS)
    return r.to_dict()


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
