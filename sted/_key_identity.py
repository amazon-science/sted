"""The second interpretive ambiguity in JSON: is a key a label or an identifier?

``_order`` handles the first one (is an array a sequence or a bag?). This module handles the
other one, and it has the same shape: two defensible readings of the same document, STED
silently committing to one of them, and no static signal that resolves it from the pair alone.

**The two readings.**

* *Label.* A key is a human-readable description of its value. ``{"n": 20}`` and
  ``{"n_value": 20}`` say the same thing, so a consistent renaming is a relabeling and a
  similarity metric should be invariant to it. This is the reading behind Object Aligner's
  Weisfeiler-Leman relabeling invariance, and behind this repo's own A2 probe.
* *Identifier.* A key is a name some consumer declared. ``{"n_value": 20}`` is not a
  paraphrase of ``{"n": 20}``; it is a different call, and one that fails.

STED assumes the label reading: ``_calculate_structural_similarity`` scores field names with
the embedding kernel, so ``n``/``n_value`` are near-synonyms and the rename is nearly free.
The existing ``exact_match_all_keys`` flag assumes the identifier reading. Neither is wrong;
what is wrong is choosing one silently, because the choice decides the answer.

**Which reading is right is measurable, and it is domain-dependent.** Renaming every
argument key of the 207 executable BFCL calls (``k -> k_value``) and invoking the declared
callee, all 207/207 renamed calls fail (operator I6 in
``results/metric_verification_v2/exec_oracle.json``). So for schema-bound tool calling the
identifier reading is correct, STED's mean deviation of 0.178 -- scoring a non-executable
call at about 0.82 -- is a false accept, and SLOT's 0.733 is the right answer. For free-form
JSON, where nothing downstream declares the keys, the label reading is correct and A2's
finding (STED max deviation 0.029 against SLOT's 0.199-0.249) is the right answer. The two
instruments do not disagree; they measure different domains, and they bracket the truth.

**So bracket it, as with array order.** Let ``e = embed_sim(k1, k2)`` be the label reading's
field-name score and ``x = 1[k1 == k2]`` the identifier reading's. Then

    field_sim(mu) = (1 - mu) * e + mu * x,    mu in [0, 1]

with mu=0 reproducing published STED exactly and mu=1 reproducing ``exact_match_all_keys``.

**Lemma (bracketing).** ``x <= e``, so ``field_sim`` is non-increasing in mu and therefore
``s_1 <= s_mu <= s_0``.

*Proof.* If ``k1 == k2`` then ``x = 1`` and ``_calculate_semantic_similarity`` returns
exactly 1.0 by its first branch, so ``x = e``. Otherwise ``x = 0`` and ``e >= 0``, because
both of that function's branches are truncated at zero (``max(0.0, 1 - d/L)`` for short
strings, ``max(0.0, cos)`` for embeddings). So ``x <= e`` pointwise. ``field_sim`` is affine
in mu with slope ``x - e <= 0``, hence non-increasing; ``_calculate_structural_similarity``
is increasing in ``field_sim``, ``structural_update_cost`` is decreasing in it, and the
tree-level score is decreasing in every node cost. The endpoints follow from mu=0 and mu=1
substituted into the definition. QED

The interval width ``s_0 - s_1`` is the part of the score that rests on reading keys as
labels. It is the same construction as the order interval and composes with it: both axes
are monotone and act on disjoint terms of Eq. (1) -- mu on the field-name half of the
structural channel, lambda on array child matching -- so the joint extremes over both
readings are the corners ``(lambda, mu) = (0, 0)`` and ``(1, 1)``.

Unlike lambda, mu is not derivable from the pair: whether keys are declared elsewhere is a
fact about the deployment, not about the two documents. There is deliberately no 'auto'
here. A caller who knows the schema should set mu=1; a caller who does not should read the
interval.
"""

from typing import Any


def field_similarity(embed_sim: float, exact_sim: float, mu: float) -> float:
    """Blend the label and identifier readings of a field name.

    ``exact_sim <= embed_sim`` is guaranteed by the caller's kernel (see the module
    docstring), so the result is non-increasing in ``mu``.
    """
    if mu <= 0.0:
        return embed_sim
    if mu >= 1.0:
        return exact_sim
    return (1.0 - mu) * embed_sim + mu * exact_sim


def resolve_mu(spec: Any, default: float = 0.0) -> float:
    """Normalise ``key_identity_mu`` to a float in [0, 1].

    ``None`` means "not configured" and yields ``default`` (0.0, i.e. published STED).
    Anything else must be a number; there is no 'auto', for the reason in the module
    docstring.
    """
    if spec is None:
        return float(default)
    if isinstance(spec, bool) or not isinstance(spec, (int, float)):
        raise ValueError(
            f"key_identity_mu must be None or a number in [0, 1], got {spec!r}")
    return min(1.0, max(0.0, float(spec)))
