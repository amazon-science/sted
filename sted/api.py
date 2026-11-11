"""The two calls most users need: ``sted.compare`` and ``sted.consistency``.

    >>> import sted
    >>> r = sted.compare({"name": "get_weather", "arguments": {"city": "Seattle", "days": 3}},
    ...                  {"arguments": {"days": "3", "city": "Seattle"}, "name": "get_weather"})
    >>> r.same
    True

Both use STED's ``typed`` profile. ``min_readout`` is the lowest field score, so
``same`` (min_readout >= threshold) means every field agrees at that level, and the
``differences`` list explains it exactly: its lowest score *is* the min readout.
``mean_readout`` averages the field scores and grades how far apart two outputs are.

Text fields are compared by a local Sentence-Transformers model when the ``embeddings``
extra is installed (``pip install "sted[embeddings]"``), otherwise by character
similarity; ``text_similarity`` in every result says which. Force either with
``embedding_model="all-MiniLM-L6-v2"`` or ``embedding_model="none"``, or set
``STED_EMBEDDING_MODEL``.

Field rules (path patterns, see ``sted.rules``)::

    sted.compare(a, b,
                 ignore=["request_id", "**.timestamp"],    # drop these fields
                 unordered=["arguments.tags"],             # compare as a set
                 tolerance={"price": 0.01})                # numeric tolerance
"""
import dataclasses
import json
import os
import importlib.util
import statistics
import threading
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .rules import FieldRules

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.95
MAX_VALUE_CHARS = 200
PAIR_TIMEOUT_SECONDS = 30.0

_lock = threading.Lock()
_cache: Dict[tuple, Any] = {}


# ---------------------------------------------------------------- results

@dataclasses.dataclass
class FieldDifference:
    """One field that differs: ``status`` is changed / missing (from b) / extra (in b)."""
    path: str
    status: str
    a: Any
    b: Any
    score: float


@dataclasses.dataclass
class Comparison:
    same: bool
    min_readout: float
    mean_readout: float
    threshold: float
    differences: List[FieldDifference]
    text_similarity: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        head = (f"{'SAME' if self.same else 'DIFFERENT'}  min {self.min_readout:.3f}  "
                f"mean {self.mean_readout:.3f}  (threshold {self.threshold})")
        return "\n".join([head] + [_format_diff(d) for d in self.differences])


@dataclasses.dataclass
class UnstableField:
    """A field scoring below threshold in some pairs: in what share, with which values."""
    path: str
    share_of_pairs: float
    values: List[Dict[str, Any]]          # [{"value": v, "count": n}], most common first


@dataclasses.dataclass
class Consistency:
    n_outputs: int
    n_valid: int
    validity: float
    agreement_min: Optional[float]
    agreement_mean: Optional[float]
    share_same: Optional[float]
    n_pairs: int
    threshold: float
    unstable_fields: List[UnstableField]
    least_agreeing_pair: Optional[Dict[str, Any]]    # {"i", "j", "min_readout", "differences"}
    note: Optional[str] = None
    text_similarity: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        if d["note"] is None:
            d.pop("note")
        return d

    def __str__(self) -> str:
        if self.agreement_min is None:
            return (f"{self.n_valid}/{self.n_outputs} valid outputs: "
                    f"{self.note or 'not enough to compare'}")
        lines = [f"validity {self.validity:.2f} ({self.n_valid}/{self.n_outputs})   "
                 f"agreement min {self.agreement_min:.3f} / mean {self.agreement_mean:.3f}   "
                 f"same pairs {self.share_same:.0%} of {self.n_pairs}"]
        if self.unstable_fields:
            lines.append("unstable fields:")
            for u in self.unstable_fields:
                vals = ", ".join(f"{json.dumps(v['value'], default=str)} x{v['count']}" for v in u.values)
                lines.append(f"  {u.path:<30} differs in {u.share_of_pairs:.0%} of pairs   {vals}")
        if self.note:
            lines.append(f"note: {self.note}")
        return "\n".join(lines)


def _format_diff(d: FieldDifference) -> str:
    a = json.dumps(d.a, default=str) if d.status != "extra" else "(absent)"
    b = json.dumps(d.b, default=str) if d.status != "missing" else "(absent)"
    return f"  {d.path:<30} {a} -> {b}   [{d.status}, {d.score:.3f}]"


# ---------------------------------------------------------------- evaluators

_warned_no_embeddings = False


def _embedding_model(explicit: Optional[str]) -> Optional[str]:
    """Model id to use, or None for character similarity on text fields.

    Explicit argument, then STED_EMBEDDING_MODEL, then the default model if
    sentence-transformers is installed, else None (with a one-time warning)."""
    global _warned_no_embeddings
    choice = explicit or os.environ.get("STED_EMBEDDING_MODEL")
    if choice:
        return None if choice.lower() in ("none", "char", "character") else choice
    if importlib.util.find_spec("sentence_transformers") is not None:
        return DEFAULT_EMBEDDING_MODEL
    if not _warned_no_embeddings:
        warnings.warn('sentence-transformers is not installed: text fields are compared by '
                      'character similarity. For semantic comparison: pip install "sted[embeddings]"',
                      stacklevel=3)
        _warned_no_embeddings = True
    return None


def _text_similarity_label(model: Optional[str]) -> str:
    return f"embedding:{model}" if model else "character"


def _evaluators(rules: FieldRules, model: Optional[str]):
    """(min-readout evaluator, mean-readout evaluator), each wrapped for consistency runs."""
    key = (rules.key(), model)
    with _lock:
        if key not in _cache:
            from .agent_consistency_evaluator import AgentConsistencyEvaluator
            from .semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

            pair = []
            for profile in ("typed", "typed_mean"):
                sted = SemanticJsonTreeConsistencyEvaluator(profile=profile, model_id=model, field_rules=rules)
                # Outputs arrive parsed (_parse); do not parse JSON scalars a second time.
                pair.append(AgentConsistencyEvaluator(evaluator=sted, timeout_seconds=PAIR_TIMEOUT_SECONDS,
                                                      parse_json=False, strict_json=False))
            _cache[key] = tuple(pair)
        return _cache[key]


def _parse(value: Any) -> Any:
    """JSON text -> value; text that is not JSON -> None (invalid)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return value


def _short(v: Any) -> Any:
    if isinstance(v, str) and len(v) > MAX_VALUE_CHARS:
        return v[:MAX_VALUE_CHARS] + "..."
    return v


def _check_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")


def _differences(sted, a: Any, b: Any, limit: Optional[int]) -> List[FieldDifference]:
    diffs = sted.calculate_field_differences(a, b)
    if limit is not None:
        diffs = diffs[:limit]
    return [FieldDifference(d["path"], d["status"], _short(d["a"]), _short(d["b"]), round(d["score"], 6))
            for d in diffs]


# ---------------------------------------------------------------- public API

def compare(a: Any, b: Any, *, threshold: float = DEFAULT_THRESHOLD,
            ignore: Optional[Sequence[str]] = None, unordered: Optional[Sequence[str]] = None,
            tolerance: Optional[Mapping[str, Any]] = None, rules: Optional[FieldRules] = None,
            max_differences: Optional[int] = None, embedding_model: Optional[str] = None) -> Comparison:
    """Compare two structured outputs (JSON values or JSON text).

    Returns a :class:`Comparison`: ``same`` (min readout >= threshold), both readouts,
    and every differing field, lowest score first.
    """
    _check_threshold(threshold)
    rules = FieldRules.coerce(rules, ignore=ignore, unordered=unordered, tolerance=tolerance)
    pa, pb = _parse(a), _parse(b)
    for name, raw, val in (("a", a, pa), ("b", b, pb)):
        if val is None:
            raise ValueError(f"{name} is not valid JSON: {str(raw)[:80]!r}")
    model = _embedding_model(embedding_model)
    lo_eval, mean_eval = _evaluators(rules, model)
    # STED directly: AgentConsistencyEvaluator.evaluate_pair scores an empty object as
    # invalid (0.0), but an explicit pair of {} and {} is identical.
    lo = float(lo_eval._evaluator.calculate_tree_edit_distance_fast(pa, pb))
    mean = float(mean_eval._evaluator.calculate_tree_edit_distance_fast(pa, pb))
    return Comparison(same=lo >= threshold, min_readout=round(lo, 6), mean_readout=round(mean, 6),
                      threshold=threshold,
                      differences=_differences(lo_eval._evaluator, pa, pb, max_differences),
                      text_similarity=_text_similarity_label(model))


def consistency(outputs: Sequence[Any], *, threshold: float = DEFAULT_THRESHOLD,
                ignore: Optional[Sequence[str]] = None, unordered: Optional[Sequence[str]] = None,
                tolerance: Optional[Mapping[str, Any]] = None, rules: Optional[FieldRules] = None,
                max_fields: Optional[int] = 20, max_diff_pairs: int = 300,
                embedding_model: Optional[str] = None) -> Consistency:
    """Rate the consistency of repeated outputs of the same prompt.

    ``validity`` is the share of outputs that parse; agreement is the mean pairwise STED
    over valid outputs under each readout; ``share_same`` the share of pairs with min
    readout >= threshold; ``unstable_fields`` names the fields that disagree and how
    often. Pairs at or above threshold cannot hold an unstable field, so only the
    others are searched, evenly sampled down to ``max_diff_pairs``.
    """
    _check_threshold(threshold)
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence) or len(outputs) < 2:
        raise ValueError("outputs must be a list of at least 2 generations")
    rules = FieldRules.coerce(rules, ignore=ignore, unordered=unordered, tolerance=tolerance)
    model = _embedding_model(embedding_model)
    lo_eval, mean_eval = _evaluators(rules, model)
    parsed = [_parse(o) for o in outputs]
    lo = lo_eval.evaluate_outputs({"prompt": parsed}).per_prompt[0]
    mean = mean_eval.evaluate_outputs({"prompt": parsed}).per_prompt[0]

    from .agent_consistency_evaluator import _is_valid
    valid_idx = [i for i, o in enumerate(parsed) if _is_valid(o)]
    base = dict(n_outputs=len(outputs), n_valid=lo.n_valid, validity=round(lo.r_v, 6), threshold=threshold,
                text_similarity=_text_similarity_label(model))
    if lo.n_valid < 2:
        return Consistency(**base, agreement_min=None, agreement_mean=None, share_same=None, n_pairs=0,
                           unstable_fields=[], least_agreeing_pair=None, note="fewer than 2 valid outputs")

    pairs = [(i, j) for i in range(len(valid_idx)) for j in range(i + 1, len(valid_idx))]
    scored = [(s, p) for s, p in zip(lo.pairwise_similarities, pairs) if s is not None]
    finite_mean = [s for s in mean.pairwise_similarities if s is not None]
    worst_s, (wi, wj) = min(scored, key=lambda t: t[0])
    sted = lo_eval._evaluator

    below = [(s, p) for s, p in scored if s < threshold]
    n_below = len(below)
    if n_below > max_diff_pairs:
        step = n_below / max_diff_pairs
        below = [below[int(k * step)] for k in range(max_diff_pairs)]
    counts: Dict[str, int] = {}
    for _, (i, j) in below:
        for d in sted.calculate_field_differences(parsed[valid_idx[i]], parsed[valid_idx[j]]):
            if d["score"] < threshold:
                counts[d["path"]] = counts.get(d["path"], 0) + 1
    scale = (n_below / len(below)) / len(scored) if below else 0.0
    ranked = sorted(counts.items(), key=lambda t: (-t[1], t[0]))
    if max_fields is not None:
        ranked = ranked[:max_fields]
    stripped = [rules.strip(parsed[k]) for k in valid_idx]
    unstable = [UnstableField(path, round(c * scale, 6), _values_at(sted, stripped, path)) for path, c in ranked]

    notes = sorted({r.error for r in (lo, mean) if r.error})
    if n_below > max_diff_pairs:
        notes.append(f"share_of_pairs estimated from an even sample of {max_diff_pairs} "
                     f"of the {n_below} disagreeing pairs")
    return Consistency(
        **base,
        agreement_min=round(statistics.mean(s for s, _ in scored), 6),
        agreement_mean=round(statistics.mean(finite_mean), 6) if finite_mean else None,
        share_same=round(sum(s >= threshold for s, _ in scored) / len(scored), 6),
        n_pairs=len(pairs),
        unstable_fields=unstable,
        least_agreeing_pair={"i": valid_idx[wi], "j": valid_idx[wj], "min_readout": round(worst_s, 6),
                             "differences": [dataclasses.asdict(d) for d in _differences(
                                 sted, parsed[valid_idx[wi]], parsed[valid_idx[wj]], 20)]},
        note="; ".join(notes) or None,
    )


def _values_at(sted, docs: List[Any], path: str) -> List[Dict[str, Any]]:
    """Distinct values at ``path`` across outputs, most common first (top 5)."""
    from ._paths import EmptyContainer, terminal_paths
    from .semantic_json_tree_consistency import format_json_path

    seen: Dict[str, List[Any]] = {}
    for doc in docs:
        leaves = terminal_paths(doc)
        if sted.field_rules.unordered:
            leaves = sted._collapse_unordered(doc, leaves)
        flat = {format_json_path(p): v for p, v in leaves.items()}
        v = flat.get(path, "<absent>")
        if isinstance(v, EmptyContainer):
            v = {} if v is EmptyContainer.OBJECT else []
        v = getattr(v, "value", v)          # an unordered array reported whole
        key = json.dumps(v, sort_keys=True, default=str)
        seen.setdefault(key, [_short(v), 0])[1] += 1
    ranked = sorted(seen.values(), key=lambda t: -t[1])[:5]
    return [{"value": v, "count": c} for v, c in ranked]
