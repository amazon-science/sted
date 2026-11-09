"""Helpers shared by the framework adapters."""
import json
from typing import Any, Dict, List, Optional, Tuple

from .. import api
from ..rules import FieldRules


def pick(value: Any, key: Optional[str]) -> Any:
    """The structured value to score from a framework's output/reference object.

    With ``key``, take that key of a dict. Without it, a dict with exactly one key
    yields that key's value (e.g. {"output": {...}}), anything else is used as is."""
    if key is not None:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"key {key!r} not found in {str(value)[:120]!r}")
        return value[key]
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.values()))
    return value


def compare_kwargs(threshold: float, rules: Optional[FieldRules], ignore, unordered, tolerance,
                   embedding_model: Optional[str]) -> Dict[str, Any]:
    return {"threshold": threshold, "rules": FieldRules.coerce(rules, ignore=ignore, unordered=unordered,
                                                                tolerance=tolerance),
            "embedding_model": embedding_model}


def diff_comment(r: "api.Comparison", limit: int = 10) -> str:
    """One line per differing field, for evaluator comments / explanations."""
    if r.same and not r.differences:
        return "same"
    lines = [f"{'same' if r.same else 'different'} (min {r.min_readout:.3f}, mean {r.mean_readout:.3f})"]
    for d in r.differences[:limit]:
        a = "(absent)" if d.status == "extra" else json.dumps(d.a, default=str)
        b = "(absent)" if d.status == "missing" else json.dumps(d.b, default=str)
        lines.append(f"{d.path}: {a} -> {b} [{d.status}, {d.score:.3f}]")
    if len(r.differences) > limit:
        lines.append(f"... {len(r.differences) - limit} more")
    return "\n".join(lines)


def unstable_summary(results: List[Tuple[str, "api.Consistency"]], limit: int = 10) -> Tuple[List[Tuple[str, int]], str]:
    """Fields unstable in how many groups, and a short text report."""
    counts: Dict[str, int] = {}
    for _, r in results:
        for u in r.unstable_fields:
            counts[u.path] = counts.get(u.path, 0) + 1
    ranked = sorted(counts.items(), key=lambda t: (-t[1], t[0]))
    n = len(results)
    lines = [f"{path}: unstable in {c} of {n}" for path, c in ranked[:limit]]
    worst = sorted(((gid, r) for gid, r in results if r.share_same is not None), key=lambda t: t[1].share_same)[:3]
    for gid, r in worst:
        if r.share_same < 1.0:
            lines.append(f"least consistent: {gid} (same pairs {r.share_same:.0%}, validity {r.validity:.0%})")
    return ranked, "\n".join(lines) or "all groups consistent"
