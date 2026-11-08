"""Per-field comparison rules: which fields to ignore, which arrays are sets, and which
numbers may differ by a tolerance.

Path patterns
-------------
A pattern is matched against a field's full path, written the way STED reports paths
(``arguments.items[0].id``):

* ``arguments.request_id``   exact path
* ``items[*].id``            ``[*]`` matches any array index
* ``*.id``                   ``*`` matches exactly one object key
* ``**.timestamp``           ``**`` matches any number of steps (including none)
* ``timestamp``              a bare key with no ``.``/``[`` matches that key at any depth,
                             i.e. it is shorthand for ``**.timestamp``

Rules
-----
``ignore``      fields (or whole subtrees) removed from both outputs before comparing.
``unordered``   arrays compared as multisets: element order does not matter. Other arrays
                stay positional, since the order of a list argument usually changes the call.
``tolerance``   ``{pattern: abs}`` or ``{pattern: {"abs": a, "rel": r}}``: numbers at matching
                paths are equal when ``|x - y| <= max(abs, rel * max(|x|, |y|))``.
"""
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

Token = Tuple[str, Any]          # ("key", name) | ("index", i)
_ANY_KEY = ("key", "*")
_ANY_INDEX = ("index", "*")
_DEEP = ("deep", None)

_PATH_TOKEN = re.compile(r'\[(\d+|\*|"(?:[^"\\]|\\.)*")\]|\.?([^.\[\]]+)')


def _tokenize(text: str, *, pattern: bool) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    text = text.strip()
    if text in ("", "$"):
        return tokens
    while pos < len(text):
        m = _PATH_TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot parse path {'pattern ' if pattern else ''}{text!r} at {pos}")
        bracket, key = m.group(1), m.group(2)
        if bracket is not None:
            if bracket == "*":
                if not pattern:
                    raise ValueError(f"'[*]' is only valid in patterns: {text!r}")
                tokens.append(_ANY_INDEX)
            elif bracket.startswith('"'):
                tokens.append(("key", json.loads(bracket)))
            else:
                tokens.append(("index", int(bracket)))
        elif pattern and key == "**":
            tokens.append(_DEEP)
        elif pattern and key == "*":
            tokens.append(_ANY_KEY)
        else:
            tokens.append(("key", key))
        pos = m.end()
    return tokens


def path_tokens(path: Union[str, Sequence[Token]]) -> List[Token]:
    """A path as tokens; accepts a typed path tuple or a path string."""
    if isinstance(path, str):
        if path == "root":            # JsonNode's wrapper around an object document
            return []
        if path.startswith("root."):
            path = path[len("root."):]
        return _tokenize(path, pattern=False)
    return [tuple(t) for t in path]


class PathPattern:
    """A compiled path pattern; ``matches(path)`` takes a path string or typed path."""

    def __init__(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty path pattern: {text!r}")
        self.text = text
        tokens = _tokenize(text, pattern=True)
        bare = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", text.strip())
        self.tokens = [_DEEP] + tokens if bare else tokens

    def matches(self, path: Union[str, Sequence[Token]]) -> bool:
        return _match(self.tokens, path_tokens(path))

    def __repr__(self) -> str:
        return f"PathPattern({self.text!r})"


def _match(pat: List[Token], path: List[Token]) -> bool:
    if not pat:
        return not path
    head = pat[0]
    if head == _DEEP:
        return any(_match(pat[1:], path[i:]) for i in range(len(path) + 1))
    if not path:
        return False
    kind, value = path[0]
    if head == _ANY_KEY:
        ok = kind == "key"
    elif head == _ANY_INDEX:
        ok = kind == "index"
    else:
        ok = head == (kind, value)
    return ok and _match(pat[1:], path[1:])


ToleranceSpec = Union[float, int, Mapping[str, float]]


class FieldRules:
    """ignore / unordered / tolerance rules, compiled once and hashable for caching."""

    def __init__(self, ignore: Optional[Iterable[str]] = None,
                 unordered: Optional[Iterable[str]] = None,
                 tolerance: Optional[Mapping[str, ToleranceSpec]] = None):
        self.ignore = [PathPattern(p) for p in (ignore or [])]
        self.unordered = [PathPattern(p) for p in (unordered or [])]
        self.tolerance: List[Tuple[PathPattern, float, float]] = []
        for p, spec in (tolerance or {}).items():
            if isinstance(spec, Mapping):
                unknown = set(spec) - {"abs", "rel"}
                if unknown:
                    raise ValueError(f"tolerance for {p!r}: unknown keys {sorted(unknown)}")
                a, r = float(spec.get("abs", 0.0)), float(spec.get("rel", 0.0))
            else:
                a, r = float(spec), 0.0
            if a < 0 or r < 0 or not (math.isfinite(a) and math.isfinite(r)):
                raise ValueError(f"tolerance for {p!r} must be finite and non-negative")
            self.tolerance.append((PathPattern(p), a, r))

    @classmethod
    def coerce(cls, rules: Optional["FieldRules"] = None, *, ignore=None, unordered=None,
               tolerance=None) -> "FieldRules":
        if rules is not None:
            if ignore or unordered or tolerance:
                raise ValueError("pass either rules= or ignore/unordered/tolerance, not both")
            return rules
        return cls(ignore, unordered, tolerance)

    def __bool__(self) -> bool:
        return bool(self.ignore or self.unordered or self.tolerance)

    def key(self) -> tuple:
        return (tuple(p.text for p in self.ignore), tuple(p.text for p in self.unordered),
                tuple((p.text, a, r) for p, a, r in self.tolerance))

    def is_ignored(self, path) -> bool:
        return any(p.matches(path) for p in self.ignore)

    def is_unordered(self, path) -> bool:
        return any(p.matches(path) for p in self.unordered)

    def tolerance_for(self, path) -> Optional[Tuple[float, float]]:
        for p, a, r in self.tolerance:
            if p.matches(path):
                return a, r
        return None

    def strip(self, doc: Any) -> Any:
        """A copy of ``doc`` without the ignored fields (no-op without ignore rules)."""
        if not self.ignore:
            return doc
        return self._strip(doc, [])

    def _strip(self, node: Any, path: List[Token]) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                p = path + [("key", k)]
                if not self.is_ignored(p):
                    out[k] = self._strip(v, p)
            return out
        if isinstance(node, list):
            # Ignoring an array element removes it, so later indices shift; patterns that
            # address elements are meant for [*]-style rules (e.g. items[*].debug).
            out_list = []
            for i, v in enumerate(node):
                p = path + [("index", i)]
                if not self.is_ignored(p):
                    out_list.append(self._strip(v, p))
            return out_list
        return node

    def __repr__(self) -> str:
        return (f"FieldRules(ignore={[p.text for p in self.ignore]}, "
                f"unordered={[p.text for p in self.unordered]}, "
                f"tolerance={[(p.text, a, r) for p, a, r in self.tolerance]})")


def numbers_within(x: float, y: float, tol: Tuple[float, float]) -> bool:
    a, r = tol
    return abs(x - y) <= max(a, r * max(abs(x), abs(y)))
