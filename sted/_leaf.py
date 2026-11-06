"""The fourth interpretive axis: is a leaf value a token, or a form?

Why this module exists
----------------------
``_order`` (λ), ``_key_identity`` (μ) and ``_aggregation`` (p) each bracket a question about
*structure*. This one is about the leaf, and it exists because §12.6's soundness theorem left
the whole burden of certification there and Instrument J then showed the burden cannot be
carried at the default configuration.

``_aggregation``'s theorem says: at ``p = -inf``, ``s_min(a, b) >= theta`` implies every matched
pair at every level scores at least ``theta``, with no dependence on arity. That is a real
guarantee about the *correspondence*, and it is worth nothing unless a leaf score of ``theta``
means something about the two values. It does not, today:

**Measured (Instrument J, 7,076 executed single-leaf substitutions on the 208 BFCL executable
calls, 444 execution-equivalent).** ``sup`` over the non-equivalent pairs of the reported
similarity is exactly **1.0000** on every channel, so no ``theta < 1`` is sound. The pairs that
reach 1.0 are all casefold-equal code strings at ``estimate_derivative.arguments.function``,
e.g. ``'lambda x: 3*x**2 + 2*x + 1'`` against ``'LAMBDA X: 3*X**2 + 2*X + 1'``, which the callee
rejects with a ``NameError``. The cause is mechanical rather than statistical: the sentence
encoder is uncased, so the two strings have the same token sequence and the cosine is 1 **by
construction**. This is a violation of identity of indiscernibles -- ``s(A, B) = 1 ==> A = B`` --
which the axiom battery in §6 does not test, because it tests only the forward direction
``s(A, A) = 1``.

**The two readings.**

* *Form.* A leaf value is a form whose surface details -- case, whitespace, unicode
  normalisation -- are semantically inert, so ``'male'`` and ``'MALE'`` denote the same thing
  and a metric should be blind to the difference. This is the reading behind every
  embedding-based similarity, and it is the reading STED takes for strings of four characters
  or more.
* *Token.* A leaf value is a token some consumer will interpret literally. ``'LAMBDA X: ...'``
  is not a paraphrase of ``'lambda x: ...'``; it is a different program, and one that fails.
  This is the reading STED takes for strings *shorter* than four characters, and for numbers
  and booleans, where the kernel is exact.

Neither reading is right. The oracle says so directly: of 24 executed case-only substitutions,
**14 are equivalent and 10 are not**, and the split is by callee -- case is inert at
``calculate_nutritional_needs.gender``/``goal`` and load-bearing at
``estimate_derivative.function``. Making the string kernel case-sensitive would fix the 10 false
accepts and break the 14 correct ones. (Whitespace is different: all 38 executed
strip-equal substitutions are equivalent, so that invariance is safe on this corpus.)

So this is the same epistemic situation as λ and μ -- two defensible readings, no static signal
that resolves them, STED silently committing to one -- and it gets the same treatment.

    k_tau(u, v) = (1 - tau) * k_embed(u, v) + tau * k_char(u, v),      tau in [0, 1]

    k_embed(u, v) = max(0, cos(E(u), E(v)))                 the form reading
    k_char(u, v)  = max(0, 1 - d_lev(u, v) / max(|u|, |v|))  the token reading

``tau = None`` (the default) is published STED, which is **not** either endpoint: it is
``tau = 0`` for strings of length >= ``min_length_for_embeddings`` and ``tau = 1`` below that.
That constant is therefore a hard-coded answer to a question Instrument J shows is
slot-dependent, and making it an axis is the point of this module.

What is provable here, and what is not
--------------------------------------
**There is no bracketing lemma.** λ, μ and p all have one, and each time it came from a
pointwise inequality between the two readings' kernels. Here no such inequality holds in either
direction: on the 360 string records of Instrument J the character kernel *exceeds* the
embedding kernel on **28** of them (largest excess 0.2928), and the embedding exceeds the
character kernel on most of the rest. ``k_tau`` is affine in ``tau`` with slope
``k_char - k_embed``, whose sign varies pair by pair, so the document score is monotone in
``tau`` for neither direction and the interval endpoints are not ordered a priori. Callers must
take ``min``/``max`` of the two ends rather than assume which is which; that is what
``compute_leaf_literalism_interval`` documents and what the two-sided bound below states.

**Lemma (two-sided leaf bound).** ``min(k_embed, k_char) <= k_tau <= max(k_embed, k_char)``
pointwise, with equality at the endpoints. *Proof.* ``k_tau`` is a convex combination. QED
This is weaker than λ/μ/p's bracket -- it bounds the leaf kernel, not the tree score, because
without a sign for the slope the tree-level induction has no direction to run in.

What the token end buys is not accuracy. It is that a threshold on it *means something*:

**Theorem (edit-radius certificate).** Fix ``tau in (0, 1]`` and let ``L = max(|u|, |v|)``. If
``k_tau(u, v) >= theta`` and ``theta > 1 - tau``, then

    d_lev(u, v) <= (1 - (theta - (1 - tau)) / tau) * L.

*Proof.* ``k_embed <= 1``, so ``theta <= k_tau <= (1 - tau) + tau * k_char``, giving
``k_char >= (theta - (1 - tau)) / tau =: theta_c``, which is positive by assumption. On
``k_char > 0`` the truncation is inactive and ``k_char = 1 - d/L``, so ``d <= (1 - theta_c) * L``.
QED

At ``tau = 1`` this is ``d <= (1 - theta) * L``: the score bounds the number of character edits
between the two values, so ``theta = 1`` forces ``u = v`` and identity of indiscernibles holds by
construction rather than by luck. At ``tau = 0`` the hypothesis ``theta > 1`` is unsatisfiable and
the statement is empty -- correctly, since the measurement above exhibits pairs at
``k_embed = 1`` with ``d = 26``.

**Corollary (an end-to-end document certificate).** Configure ``aggregation_p = -inf``,
``cost_combination = 'product'`` and ``tau = 1``. For a matched leaf pair agreeing on key, path
and type, ``structural_update_cost`` returns exactly 0, so the t-norm gives
``s_upd = k_char^(1 - w)``; combined with the soundness theorem, ``s_min(a, b) >= theta`` implies
every such matched leaf pair satisfies

    d_lev(u, v) <= (1 - theta^(1 / (1 - w))) * max(|u|, |v|),

i.e. ``d <= (1 - theta^2) * L`` at the default ``w = 0.5``. This is the first statement in this
line of work that turns a reported number into a checkable claim about the documents, and every
one of its three ingredients is load-bearing: ``p = -inf`` removes the arity dependence,
``product`` removes the ``gamma_upd <= 1 - w`` floor that makes every ``theta <= 0.5`` vacuous
under Eq. (2), and ``tau = 1`` is what makes a leaf threshold say anything at all.

Scope and limits
----------------
* **A certificate is not an improvement, and this one costs accuracy.** On Instrument J's 360
  string records the embedding kernel is much the better classifier (AUC 0.9823 against 0.7936)
  and admits no sound threshold; the character kernel admits one (``sup`` over non-equivalent
  pairs 0.8000) and rejects 62.5% of the non-identical equivalent pairs at it. Neither end
  dominates, which is precisely why both are offered.
* **``tau`` cannot be calibrated from values.** Instrument J's held-out ceiling for the class of
  all value-pair kernels is 0.5068 AUC by callee (0.5044 on numbers), so no setting of ``tau``,
  and no replacement kernel at either end, generalises above chance on that corpus.
  Interchangeability is a property of the pair *and the slot*.
* **Numbers are untouched, deliberately.** ``_compare_numbers`` is exact, i.e. already at
  ``tau = 1``, and Instrument J says it should stay there: the equivalence rate *rises* with
  relative difference (0.0276 in [0.001, 0.1) against 0.0884 in [1, 10)), so numeric proximity
  is evidence of non-equivalence and ``1/(1 + reldiff)`` scores 0.4044, below chance. ``tau``
  is a string axis; extending it to numbers would import a signal measured to be inverted.
* ``tau`` applies to leaf **values** only, never to field names -- those are μ's axis. The two
  act on disjoint terms of Eq. (1), so they compose, but they are not the same question: a key
  is a name, a value is a datum.
* ``tau != None`` forces the embedding branch for short strings, which published STED avoids on
  the grounds that embeddings are unreliable below four characters. That claim is not
  contradicted here; ``tau = 0`` is the *explicit* form reading, and if it is worse on short
  strings that is a fact about the axis, not a bug in it.
"""

import math
import re
from typing import Any, Callable, Optional

# ``_levenshtein`` lives in ``semantic_json_tree_consistency``, which imports *this* module, and
# it is defined below that import. So the reference has to be resolved on first use rather than
# at module load, and cached rather than re-resolved on every leaf comparison.
_LEV: Optional[Callable[[str, str], int]] = None


def _lev(a: str, b: str) -> int:
    global _LEV
    if _LEV is None:
        from .semantic_json_tree_consistency import _levenshtein
        _LEV = _levenshtein
    return _LEV(a, b)


def char_kernel(u: str, v: str) -> float:
    """The token reading: length-normalised Levenshtein similarity, truncated at zero.

    Identical to the short-string branch of ``_calculate_semantic_similarity``, extracted so
    both ends of the axis are expressed in one place. ``k_char(u, v) == 1`` iff ``u == v``,
    which is the property the whole module is built on.
    """
    if u == v:
        return 1.0
    max_len = max(len(u), len(v))
    if max_len == 0:                                    # pragma: no cover - u == v catches this
        return 1.0
    return max(0.0, 1.0 - _lev(u, v) / max_len)


def blend(embed_sim: float, char_sim: float, tau: float) -> float:
    """Convex combination of the form and token readings.

    Unlike ``_key_identity.field_similarity`` this is **not** monotone in one direction: the
    slope ``char_sim - embed_sim`` has no fixed sign (28/360 string records in Instrument J have
    it positive). Callers that want an interval must take ``min``/``max``, not the endpoints in
    a fixed order.
    """
    if tau <= 0.0:
        return embed_sim
    if tau >= 1.0:
        return char_sim
    return (1.0 - tau) * embed_sim + tau * char_sim


def resolve_tau(spec: Any) -> Optional[float]:
    """Normalise ``leaf_literalism_tau`` to a float in [0, 1], or ``None``.

    ``None`` is preserved rather than mapped to a default, because published STED is not at
    either endpoint: it is the length switch on ``min_length_for_embeddings``. Mapping ``None``
    to 0.0 would silently change short-string scores, so the call sites branch on ``None``.
    There is no ``'auto'``: whether a value's surface form is load-bearing is a fact about the
    consumer, and Instrument J's 14-vs-10 case split shows it varies by callee.
    """
    if spec is None:
        return None
    if isinstance(spec, bool) or not isinstance(spec, (int, float)):
        raise ValueError(
            f"leaf_literalism_tau must be None or a number in [0, 1], got {spec!r}")
    tau = float(spec)
    if tau != tau:
        raise ValueError('leaf_literalism_tau must not be NaN')
    return min(1.0, max(0.0, tau))


def is_active(tau: Optional[float]) -> bool:
    """True when ``tau`` replaces the published length switch. ``None`` means it does not."""
    return tau is not None


def edit_radius(theta: float, tau: Optional[float], max_len: int) -> Optional[float]:
    """The edit-radius certificate of the theorem above, or ``None`` when it is vacuous.

    Returns the largest Levenshtein distance compatible with ``k_tau >= theta`` at a leaf whose
    longer value has length ``max_len``. ``None`` means the hypothesis ``theta > 1 - tau`` fails,
    so the threshold certifies nothing -- which is the case for **every** ``theta`` at
    ``tau = 0`` and for ``tau = None`` on any string the length switch sends to the embedding.
    """
    if tau is None or tau <= 0.0:
        return None
    if theta <= 1.0 - tau:
        return None
    theta_c = (theta - (1.0 - tau)) / tau
    return max(0.0, (1.0 - theta_c) * max_len)


def document_edit_radius(theta: float, tau: Optional[float], structural_weight: float,
                         max_len: int) -> Optional[float]:
    """The corollary: the same bound from a ``p = -inf`` + ``product`` *document* score.

    ``s_min >= theta`` charges every matched leaf pair at least ``theta``, and for a pair
    agreeing on key, path and type the t-norm makes that charge ``k_tau^(1 - w)``. So the leaf
    threshold implied by a document threshold is ``theta^(1 / (1 - w))``. ``w = 1`` is rejected
    rather than special-cased: it prices the content channel at zero, so no document score says
    anything about a leaf value.
    """
    w = float(structural_weight)
    if w >= 1.0:
        return None
    return edit_radius(theta ** (1.0 / (1.0 - w)), tau, max_len)


# --------------------------------------------------------------------------- literals
# Strings that a consumer will parse, not read: numbers, coordinates, dates, times,
# amounts, identifiers. For these the token reading is the only sane one, and the
# length-normalised Levenshtein is the wrong ruler -- one edit in "40.3772,-105.5217"
# is a different place, but costs 1/17. `leaf_literal_exact=True` routes any pair of
# such strings to an exact kernel: parsed numeric equality where the string is a
# number or a list of numbers, verbatim equality (whitespace-trimmed) otherwise.
# Off by default so every published number is reproduced bit for bit.

_LIT_NUMBER = re.compile(r"^\s*[-+]?(\d[\d,]*)?(\.\d+)?([eE][-+]?\d+)?\s*$")
_LIT_NUMLIST = re.compile(r"^\s*[-+]?\d+(\.\d+)?(\s*[,;/ ]\s*[-+]?\d+(\.\d+)?)+\s*$")
_LIT_DATE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[-+]\d{2}:?\d{2})?)?"
                       r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}(:\d{2})?)\s*$")
_LIT_AMOUNT = re.compile(r"^\s*[$€£¥]?\s*[-+]?\d[\d,]*(\.\d+)?\s*(%|[A-Za-z]{1,3})?\s*$")
_LIT_IDENT = re.compile(r"^[A-Za-z]{0,3}[-_#]?\d[\dA-Za-z_.:-]*$")
_LIT_WORD = re.compile(r"[A-Za-z]{4,}")
# Code: a consumer executes it, so it is a token however long it is. Instrument J's
# failure and the sweep's only false accepts were all here -- different lambda
# polynomials that an uncased encoder scores 0.96+.
_LIT_CODE = re.compile(r"(\blambda\s+\w*\s*:|\bdef\s+\w+\s*\(|\breturn\s+[\w(\[]|\bfunction\s*\w*\s*\(|=>|"
                       r"\w\*\*\w|==|!=|<=|>=|\+=|-=|::|^\s*(import|from)\s+\w|"
                       r"^\s*(SELECT|INSERT|UPDATE|DELETE)\b|[)\]}];\s*($|\n))")
# Prose that merely mentions code: several words, sentence punctuation, and no
# line that is code-shaped end to end. "return on investment" is not a program.
_PROSE = re.compile(r"[.!?]\s+[A-Z]|[.!?]$")
_REL_TOL = 1e-9
_ZERO_ABS_TOL = 1e-12   # absolute tolerance only against an exact zero (oracle audit, 2026-09-18)


_CODE_LINE = re.compile(
    r"^\s*(import|from|def|class|return|lambda|if|elif|else|for|while|try|except|with)\b.*$"   # statement
    r"|^\s*[\w.\[\]'\"]+\s*(=|\+=|-=|\*=|/=)\s*.+$"                                   # assignment
    r"|^\s*[\w.]+\(.*\)\s*;?\s*$"                                                        # bare call
    r"|^.*[;{}]\s*$"                                                                          # C-style line end
    r"|^\s*(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\b.*$", re.I)


def _looks_like_code(t: str) -> bool:
    """Code-shaped overall: most non-empty lines are statements, assignments or calls.
    A prose sentence that mentions ``groupby()`` or ``return on investment`` is not."""
    lines = [l for l in t.splitlines() if l.strip()]
    if not lines:
        return False
    return sum(1 for l in lines if _CODE_LINE.match(l)) / len(lines) >= 0.6


def literal_class(s: Any) -> Optional[str]:
    """Which literal class a string belongs to, or ``None`` for ordinary text."""
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if _LIT_CODE.search(t) and not (len(t.split()) >= 8 and _PROSE.search(t) and not _looks_like_code(t)):
        return "code"
    if len(t) > 64 or _LIT_WORD.search(t):
        return None
    if _LIT_NUMBER.match(t) and any(ch.isdigit() for ch in t):
        return "number"
    # Dates before number lists: "01/15/2024" is digits joined by "/" and would
    # otherwise parse as three numbers, making 01/15/2024 == 1/15/2024.0.
    if _LIT_DATE.match(t):
        return "datetime"
    if _LIT_NUMLIST.match(t):
        return "number_list"
    if _LIT_AMOUNT.match(t):
        return "amount"
    if _LIT_IDENT.match(t):
        return "identifier"
    return None


def numbers_close(a: float, b: float) -> bool:
    """Numeric equality up to float noise: relative 1e-9, and an absolute 1e-12 only against
    an exact zero (a subtraction that should have given 0 but gave 1e-15).

    The absolute tolerance used to apply everywhere, so every number below about 1e-3 in
    magnitude compared equal to any other such number: the oracle audit found
    ``charge: 1.6e-19`` scored identical to ``1.76e-19`` and to ``-1.6e-19``."""
    if a == 0.0 or b == 0.0:
        return abs(a - b) <= _ZERO_ABS_TOL
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=0.0)


def _parse_numbers(t: str) -> Optional[list]:
    try:
        return [float(x.replace(",", "")) for x in re.split(r"\s*[;/ ]\s*|\s*,\s*(?=[-+]?\d)", t.strip()) if x]
    except ValueError:
        return None


def numeric_values(s: Any) -> Optional[list]:
    """Every number a numeric literal denotes: one for a number or amount, several for
    a number list such as ``"32.0835,-81.0998"``; ``None`` for anything else."""
    if isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        return [float(s)]
    if not isinstance(s, str):
        return None
    cls = literal_class(s)
    if cls == "number_list":
        return _parse_numbers(s)
    if cls in ("number", "amount"):
        v = numeric_value(s) if cls == "number" else _parse_numbers(re.sub(r"[^\d.,+\- ]", "", s).strip())
        if cls == "amount":
            return v if v and len(v) == 1 else None
        return None if v is None else [v]
    return None


NEAR_MISS_SLOPE = 10.0   # relative error at which a numeric near miss reaches 0
NEAR_MISS_CAP = 0.9      # no near miss scores as equality


def numeric_near_miss(u: list, v: list) -> float:
    """Graded kernel for two unequal numeric literals, for the ranking profile only.

    Per element, relative error r = |a-b| / max(|a|, |b|) scores max(0, 1 - 10 r):
    a last-digit change scores about 0.9, a 5% change 0.45, 10% or more 0. A list
    literal is one value, so the *minimum* over its elements is taken (a coordinate
    pair 7 degrees off in longitude is another place, whatever the latitude does),
    scaled by 0.9 so a near miss never reaches equality; a different element count
    is 0. The slope is deliberately steep: annotator A
    preferred the numerically nearer candidate on 9 of 11 numeric-only pairs,
    but a 10% difference in an amount, a count or a coordinate is another value,
    and an id differing in one digit is another id -- which this kernel still
    scores 0.9 and is the reason it stays out of the certificate profiles.
    """
    if not u or not v or len(u) != len(v):
        return 0.0
    sims = []
    for a, b in zip(u, v):
        scale = max(abs(a), abs(b))
        if scale == 0:
            sims.append(1.0)
            continue
        sims.append(max(0.0, 1.0 - NEAR_MISS_SLOPE * abs(a - b) / scale))
    return NEAR_MISS_CAP * min(sims)


def numeric_value(s: Any) -> Optional[float]:
    """The number a string denotes, or ``None``. Used to compare a JSON number with a
    numeric string by value: the sandbox executes ``20`` and ``"20"`` identically for
    every one of the 200 such pairs in the oracle corpus, and the certificate scored
    them 0.949 -- a type penalty of 0.05 landing a hair under the 0.95 threshold."""
    if isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    t = s.strip().replace(",", "")
    if not t or not _LIT_NUMBER.match(t) or not any(ch.isdigit() for ch in t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def literal_equal(u: str, v: str) -> bool:
    """Exact kernel for two literal strings. Numbers compare as numbers (so ``"300"``
    equals ``"300.0"`` and ``"40.3772,-105.5217"`` equals ``"40.3772, -105.5217"``);
    everything else compares verbatim after trimming."""
    cu, cv = literal_class(u), literal_class(v)
    if cu is None or cv is None:
        return u == v
    if cu == cv == "number":
        try:
            return numbers_close(float(u.strip().replace(",", "")), float(v.strip().replace(",", "")))
        except ValueError:
            return u.strip() == v.strip()
    if cu == cv == "number_list":
        pu, pv = _parse_numbers(u), _parse_numbers(v)
        if pu is not None and pv is not None and len(pu) == len(pv):
            return all(numbers_close(a, b) for a, b in zip(pu, pv))
    return u.strip() == v.strip()
