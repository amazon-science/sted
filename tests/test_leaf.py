#!/usr/bin/env python3
"""Tests for the leaf-literalism axis (``sted._leaf``).

τ is the fourth interpretation axis and the only one *without* a bracketing lemma, so its
obligations differ from λ/μ/p's:

1. **τ = None reproduces the published metric exactly.** Bit-identical. τ = None is not an
   endpoint — published STED is τ=1 below ``min_length_for_embeddings`` and τ=0 at or above it —
   so the ``None`` branch has to delegate to the untouched function rather than resolve to a
   number. These tests pin that.
2. **The endpoints are what they claim.** τ=1 is the character kernel *and nothing else* (it must
   not consult the embedding at all, which is also where its speed comes from); τ=0 is the
   embedding cosine even for strings the length switch would have sent to characters.
3. **No bracketing lemma is asserted, and the crossing that forbids one is exhibited.** A test
   that only checked ``min <= k_tau <= max`` would pass under a false monotonicity claim too, so
   there is an explicit witness pair where the character kernel *exceeds* the embedding kernel.
4. **The edit-radius certificate holds, and is vacuous exactly where the theorem says.** This is
   the axis's actual contribution: at τ=1 a leaf threshold bounds the Levenshtein distance, at
   τ=0 no threshold bounds anything, and the identity of indiscernibles that Instrument J found
   violated at the default configuration is restored at τ=1.
5. **The two-cache hazard.** τ is absent from the subtree cache key like μ and p, *and* absent
   from ``StringSimilarityCache``'s ``(s1, s2)`` key while being the only axis that changes the
   number that cache stores. Both have to be invalidated, including through the batch
   precompute path that writes into the same cache.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sted._leaf import (                                                    # noqa: E402
    blend, char_kernel, document_edit_radius, edit_radius, is_active, resolve_tau)
from sted.semantic_json_tree_consistency import (                           # noqa: E402
    SemanticJsonTreeConsistencyEvaluator, _levenshtein)

MODEL = 'all-MiniLM-L6-v2'

_EVALUATORS = {}


def _ev(**kwargs):
    """One evaluator per distinct configuration — each construction loads the model."""
    kwargs.setdefault("profile", "legacy")
    key = tuple(sorted(kwargs.items()))
    if key not in _EVALUATORS:
        _EVALUATORS[key] = SemanticJsonTreeConsistencyEvaluator(model_id=MODEL, **kwargs)
    return _EVALUATORS[key]


# --------------------------------------------------------------------- the math in isolation

def test_resolve_tau_preserves_none_and_clamps_numbers():
    """``None`` must survive: mapping it to 0.0 would change every short-string score."""
    assert resolve_tau(None) is None
    assert resolve_tau(0) == 0.0
    assert resolve_tau(1) == 1.0
    assert resolve_tau(0.25) == 0.25
    assert resolve_tau(-3.0) == 0.0
    assert resolve_tau(7.0) == 1.0
    for bad in ('1.0', True, [0.5], object()):
        with pytest.raises(ValueError):
            resolve_tau(bad)
    with pytest.raises(ValueError):
        resolve_tau(float('nan'))


def test_is_active_is_exactly_not_none():
    assert not is_active(None)
    for tau in (0.0, 0.5, 1.0):
        assert is_active(tau)


def test_char_kernel_is_one_iff_equal():
    """Identity of indiscernibles, which is the whole point of the token end."""
    assert char_kernel('abc', 'abc') == 1.0
    assert char_kernel('', '') == 1.0
    assert char_kernel('lambda x: x', 'LAMBDA X: X') < 1.0
    assert char_kernel('male', 'MALE') < 1.0
    assert char_kernel('abc', 'abd') == pytest.approx(2.0 / 3.0)
    assert char_kernel('abc', 'xyz') == 0.0
    # Truncated at zero rather than going negative, matching the short-string branch.
    assert char_kernel('a', 'bcdefgh') >= 0.0


def test_char_kernel_matches_the_published_short_string_branch():
    """τ=1 must be the *same* expression the length switch already uses, not a variant."""
    for u, v in (('a', 'b'), ('abc', 'abd'), ('yes', 'no'), ('1011', '1101')):
        max_len = max(len(u), len(v))
        expected = max(0.0, 1.0 - _levenshtein(u, v) / max_len)
        assert char_kernel(u, v) == expected


def test_blend_is_a_convex_combination_with_exact_endpoints():
    for e, c in ((0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (0.0, 0.0), (1.0, 1.0)):
        assert blend(e, c, 0.0) == e
        assert blend(e, c, 1.0) == c
        for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
            k = blend(e, c, tau)
            assert min(e, c) - 1e-12 <= k <= max(e, c) + 1e-12
        assert blend(e, c, 0.5) == pytest.approx((e + c) / 2)


def test_blend_has_no_fixed_slope_sign():
    """The reason there is no bracketing lemma, stated as a test rather than a comment."""
    rising = [blend(0.2, 0.8, t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    falling = [blend(0.8, 0.2, t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert rising == sorted(rising)
    assert falling == sorted(falling, reverse=True)


# --------------------------------------------------------------------- the certificate

@pytest.mark.parametrize('tau', [1.0, 0.75, 0.5])
@pytest.mark.parametrize('theta', [0.99, 0.9, 0.8, 0.6])
def test_edit_radius_is_a_true_bound(tau, theta):
    """Brute force: no string pair can beat the certified radius while scoring >= theta.

    The bound is stated over ``k_tau``, and ``k_embed <= 1`` is the only fact about the
    embedding it uses, so the adversary is free to pick the most favourable ``k_embed``.
    """
    radius = edit_radius(theta, tau, 12)
    if radius is None:
        assert theta <= 1.0 - tau
        return
    for u, v in (('abcdefghijkl', 'abcdefghijkl'), ('abcdefghijkl', 'abcdefghijkX'),
                 ('abcdefghijkl', 'abcdefXXXXXX'), ('abcdefghijkl', 'XXXXXXXXXXXX'),
                 ('abcdefghijkl', 'ABCDEFGHIJKL')):
        k_tau = blend(1.0, char_kernel(u, v), tau)      # k_embed = 1 is the worst case
        if k_tau >= theta:
            assert _levenshtein(u, v) <= radius + 1e-9, (u, v, k_tau, radius)


def test_edit_radius_is_vacuous_at_the_form_end_and_below_one_minus_tau():
    """τ=0 certifies nothing, and that is the measured fact it encodes, not a limitation."""
    for theta in (0.0, 0.5, 0.9, 1.0):
        assert edit_radius(theta, 0.0, 10) is None
        assert edit_radius(theta, None, 10) is None
    assert edit_radius(0.4, 0.5, 10) is None            # theta <= 1 - tau
    assert edit_radius(0.6, 0.5, 10) is not None


def test_edit_radius_at_tau_one_forces_equality_at_theta_one():
    assert edit_radius(1.0, 1.0, 40) == pytest.approx(0.0)
    assert edit_radius(0.9, 1.0, 40) == pytest.approx(4.0)
    assert edit_radius(0.5, 1.0, 40) == pytest.approx(20.0)


def test_document_edit_radius_matches_the_t_norm_composition():
    """``s_min >= theta`` under 'product' implies ``k_tau >= theta ** (1 / (1 - w))``."""
    assert document_edit_radius(0.9, 1.0, 0.5, 100) == pytest.approx(
        edit_radius(0.9 ** 2, 1.0, 100))
    assert document_edit_radius(0.9, 1.0, 0.0, 100) == pytest.approx(
        edit_radius(0.9, 1.0, 100))
    # w = 1 prices the content channel at zero, so no document score bounds a value.
    assert document_edit_radius(0.99, 1.0, 1.0, 100) is None
    # And the document bound is never tighter than the leaf bound at the same theta.
    assert document_edit_radius(0.9, 1.0, 0.5, 100) >= edit_radius(0.9, 1.0, 100)


# --------------------------------------------------------------------- end-to-end behaviour

PAIRS = [
    ({'name': 'f', 'arguments': {'n': 20, 'mode': 'ascending'}},
     {'name': 'f', 'arguments': {'n': 20, 'mode': 'descending'}}),
    ({'name': 'f', 'arguments': {'gender': 'male'}},
     {'name': 'f', 'arguments': {'gender': 'MALE'}}),
    ({'name': 'f', 'arguments': {'function': 'lambda x: 3*x**2 + 2*x + 1'}},
     {'name': 'f', 'arguments': {'function': 'LAMBDA X: 3*X**2 + 2*X + 1'}}),
    ({'name': 'f', 'arguments': {'a': '1011', 'b': 'no'}},
     {'name': 'f', 'arguments': {'a': '1101', 'b': 'yes'}}),
    ({'a': {'b': [1, 2], 'c': 'hello world'}},
     {'a': {'b': [1, 2], 'c': 'goodbye world'}}),
    ({'name': 'f', 'arguments': {'xs': [1, 2, 3]}},
     {'name': 'g', 'arguments': {'ys': [3, 2, 1], 'extra': True}}),
]


@pytest.mark.parametrize('cost_combination', ['additive', 'product'])
@pytest.mark.parametrize('variation_type', ['combined', 'structural', 'content'])
def test_tau_none_reproduces_the_published_metric(cost_combination, variation_type):
    """Exact equality: the ``None`` branch delegates rather than resolving to a number."""
    published = _ev(cost_combination=cost_combination)
    explicit = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", cost_combination=cost_combination,
        leaf_literalism_tau=None)
    for a, b in PAIRS:
        assert (published.calculate_tree_edit_distance_fast(a, b, variation_type)
                == explicit.calculate_tree_edit_distance_fast(a, b, variation_type))


def test_tau_none_is_not_an_endpoint():
    """The published metric is a *length switch*, so it differs from both ends.

    ``'no'``/``'yes'`` is below ``min_length_for_embeddings`` and so published-scored by
    characters; ``'hello world'``/``'goodbye world'`` is above it and so published-scored by the
    embedding. No single τ can match both, which is exactly why ``None`` is kept distinct.
    """
    short = ({'v': 'no'}, {'v': 'yes'})
    long = ({'v': 'hello world'}, {'v': 'goodbye world'})
    pub, form, token = _ev(), _ev(leaf_literalism_tau=0.0), _ev(leaf_literalism_tau=1.0)
    s_short = pub.calculate_tree_edit_distance_fast(*short, 'content')
    s_long = pub.calculate_tree_edit_distance_fast(*long, 'content')
    assert s_short == pytest.approx(token.calculate_tree_edit_distance_fast(*short, 'content'))
    assert s_long == pytest.approx(form.calculate_tree_edit_distance_fast(*long, 'content'))
    assert s_short != pytest.approx(form.calculate_tree_edit_distance_fast(*short, 'content'))
    assert s_long != pytest.approx(token.calculate_tree_edit_distance_fast(*long, 'content'))


def test_the_uncased_encoder_scores_a_case_change_at_one_and_tau_one_does_not():
    """Instrument J's identity-of-indiscernibles violation, and its repair.

    The sentence encoder is uncased, so ``u`` and ``u.upper()`` share a token sequence and the
    cosine is 1.0 *by construction*. The callee disagrees (``NameError``). τ=1 restores the
    axiom that a score of 1 means the values are equal.
    """
    code = 'lambda x: 3*x**2 + 2*x + 1'
    form, token = _ev(leaf_literalism_tau=0.0), _ev(leaf_literalism_tau=1.0)
    assert form._leaf_value_similarity(code, code.upper()) == pytest.approx(1.0, abs=1e-6)
    k_token = token._leaf_value_similarity(code, code.upper())
    assert k_token < 1.0
    # And the certificate the score now carries is the true edit distance bound.
    radius = edit_radius(k_token, 1.0, max(len(code), len(code.upper())))
    assert _levenshtein(code, code.upper()) <= radius + 1e-9


def test_tau_one_never_consults_the_embedding_for_values():
    """A correctness claim: at the token end the embedding carries zero weight, so consulting
    it could only introduce a difference. Field names still embed — that is μ's axis, and the
    batch pre-warm in ``calculate_tree_edit_distance_fast`` still runs for them.
    """
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", leaf_literalism_tau=1.0)
    calls = []
    original = ev._embedding_cosine
    ev._embedding_cosine = lambda a, b: (calls.append((a, b)), original(a, b))[1]
    for a, b in PAIRS:
        ev.calculate_tree_edit_distance_fast(a, b, 'content')
    assert calls == []


def test_the_two_ends_cross_so_no_bracketing_lemma_is_available():
    """The witness that forbids the lemma λ, μ and p all have.

    Instrument J found the character kernel above the embedding kernel on 28 of 360 string
    records; one such pair is enough to make the point, and both directions are checked so the
    test fails if either kernel is silently replaced by something ordered.
    """
    form, token = _ev(leaf_literalism_tau=0.0), _ev(leaf_literalism_tau=1.0)
    # Same topic, different surface: the embedding is generous, characters are not.
    e_hi = form._leaf_value_similarity('ascending', 'descending')
    c_lo = token._leaf_value_similarity('ascending', 'descending')
    assert c_lo < e_hi
    # Unrelated topic, near-identical surface: the ordering flips.
    e_lo = form._leaf_value_similarity('cat food order', 'cat food odder')
    c_hi = token._leaf_value_similarity('cat food order', 'cat food odder')
    assert c_hi > e_lo


def test_interval_returns_both_ends_and_leaves_the_evaluator_untouched():
    a, b = ({'v': 'ascending'}, {'v': 'descending'})
    ev = _ev(leaf_literalism_tau=0.5)
    s_token, s_form = ev.compute_leaf_literalism_interval(a, b, 'content')
    assert s_token == pytest.approx(
        _ev(leaf_literalism_tau=1.0).calculate_tree_edit_distance_fast(a, b, 'content'))
    assert s_form == pytest.approx(
        _ev(leaf_literalism_tau=0.0).calculate_tree_edit_distance_fast(a, b, 'content'))
    # A mid-τ score lies inside the interval however the two ends happen to be ordered.
    mid = ev.calculate_tree_edit_distance_fast(a, b, 'content')
    assert min(s_token, s_form) - 1e-9 <= mid <= max(s_token, s_form) + 1e-9
    assert ev.leaf_literalism_tau == 0.5

    identical = ({'v': 'ascending'}, {'v': 'ascending'})
    lo, hi = ev.compute_leaf_literalism_interval(*identical, 'content')
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_assumption_interval_still_excludes_tau():
    """τ is not monotone, so adding it would turn a proved interval into an unproved one."""
    a = {'name': 'f', 'arguments': {'xs': [1, 2, 3, 4]}}
    b = {'name': 'f', 'arguments': {'xs_value': [4, 1, 2, 3]}}
    ev = _ev()
    before = ev.compute_assumption_interval(a, b)
    assert ev.leaf_literalism_tau is None
    assert ev.compute_assumption_interval(a, b) == before


def test_changing_tau_invalidates_both_caches():
    """τ is in neither cache key, and unlike μ and p it changes what ``_cache`` stores."""
    a, b = ({'v': 'hello world'}, {'v': 'goodbye world'})
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", subtree_cache_size=10000)
    ev.calculate_tree_edit_distance_fast(a, b, 'content')        # populate under the switch
    ev.leaf_literalism_tau = 1.0
    after = ev.calculate_tree_edit_distance_fast(a, b, 'content')
    fresh = _ev(leaf_literalism_tau=1.0).calculate_tree_edit_distance_fast(a, b, 'content')
    assert after == pytest.approx(fresh, abs=1e-12)

    # Re-assigning the same value must not clear anything, so only the *change* invalidates.
    ev.clear_subtree_cache()
    ev.calculate_tree_edit_distance_fast(a, b, 'content')
    before_hits = ev.get_cache_stats()['subtree_cache_hits']
    ev.leaf_literalism_tau = 1.0
    ev.calculate_tree_edit_distance_fast(a, b, 'content')
    assert ev.get_cache_stats()['subtree_cache_hits'] > before_hits


def test_the_precompute_path_respects_tau():
    """``_precompute_leaf_similarities`` writes into the cache ``_compare_strings`` reads."""
    from sted.json_tree_node import JsonNode
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL, profile="legacy", leaf_literalism_tau=1.0)
    tree1 = JsonNode.from_dict({'root': {'v': 'hello world'}})
    tree2 = JsonNode.from_dict({'root': {'v': 'goodbye world'}})
    ev._precompute_leaf_similarities(tree1, tree2)
    assert ev._cache.get('hello world', 'goodbye world') == pytest.approx(
        char_kernel('hello world', 'goodbye world'))


def test_invalid_tau_is_rejected_at_assignment_not_at_scoring():
    ev = _ev()
    with pytest.raises(ValueError):
        ev.leaf_literalism_tau = 'literal'
    assert ev.leaf_literalism_tau is None
