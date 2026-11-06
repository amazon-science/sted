"""Numeric tolerance and the optional exact kernel for literal string leaves.

Two changes, both to how STED prices leaf *values*:

* ``_compare_numbers`` equates numbers up to float noise (rel 1e-9, abs 1e-12).
  Before, ``0.1 + 0.2`` and ``0.3`` were "completely different" (leaf content
  cost 1, document 0.5). Nothing else about the kernel changes: it stays binary.
* ``leaf_literal_exact`` (default False) routes pairs of literal strings --
  numbers, coordinate lists, dates, amounts, identifiers -- to an exact kernel.
  With it off, every published number is reproduced; the tests pin the documented
  off-state values so a silent change to the default would be caught.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sted import _leaf                                                        # noqa: E402
from sted.semantic_json_tree_consistency import (                           # noqa: E402
    SemanticJsonTreeConsistencyEvaluator as Evaluator)


# --------------------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("s, cls", [
    ("40.3772", "number"), ("-105.5217", "number"), ("1,250", "number"), ("1e-9", "number"),
    ("300", "number"), ("40.3772,-105.5217", "number_list"), ("40.3772, -105.5217", "number_list"),
    ("1 2 3", "number_list"), ("2024-01-15", "datetime"), ("2024-01-15T10:30:00Z", "datetime"),
    ("14:05", "datetime"), ("01/15/2024", "datetime"), ("$1,250.00", "amount"), ("45%", "amount"),
    ("2GB", "amount"), ("INV-2024-001", "identifier"), ("id_1234", "identifier"),
    ("550e8400-e29b", "identifier"),
    ("lambda x: 3*x**2 + 2*x + 1", "code"), ("def f(x): return x", "code"), ("a == b", "code"),
    ("x => x * 2", "code"), ("SELECT id FROM t", "code"),
    ("import json\n\nfruits = ['a', 'b']\nprint(sorted(fruits))", "code"),
])
def test_literal_class_recognises_parsed_values(s, cls):
    assert _leaf.literal_class(s) == cls


@pytest.mark.parametrize("s", [
    "hello", "Boulder, Colorado", "male", "", "   ", "lambda expressions", "lambda",
    "Responsible for budget management, profitability, and return on investment. Success is measured by cost.",
    "Main data structure for sales data processing, used with operations like groupby(), merge(), and Series.",
    "A single six-week sprint can capture, critique, and synthesize **divergent** views. It ends with a decision.",
    "the 3 little pigs", "Room 101 is booked", 42, None, "x" * 70,
])
def test_literal_class_leaves_ordinary_text_alone(s):
    assert _leaf.literal_class(s) is None


def test_numeric_value_parses_numbers_and_numeric_strings_only():
    assert _leaf.numeric_value(20) == 20.0
    assert _leaf.numeric_value("20") == 20.0
    assert _leaf.numeric_value(" 1,250.5 ") == 1250.5
    assert _leaf.numeric_value("twenty") is None
    assert _leaf.numeric_value(True) is None
    assert _leaf.numeric_value("") is None


def test_numbers_close_absorbs_float_noise_only():
    assert _leaf.numbers_close(0.1 + 0.2, 0.3)
    assert _leaf.numbers_close(20, 20.0)
    assert _leaf.numbers_close(1e6, 1e6 * (1 + 1e-12))
    assert _leaf.numbers_close(0.0, 1e-15)
    assert not _leaf.numbers_close(0.6, 0.6000001)
    assert not _leaf.numbers_close(20, 21)
    assert not _leaf.numbers_close(1e6, 1e6 + 1)


def test_literal_equal_compares_numbers_as_numbers_and_the_rest_verbatim():
    assert _leaf.literal_equal("300", "300.0")
    assert _leaf.literal_equal("1,250", "1250")
    assert _leaf.literal_equal("40.3772,-105.5217", "40.3772, -105.5217")
    assert not _leaf.literal_equal("40.3772,-105.5217", "40.3772,105.5217")
    assert not _leaf.literal_equal("40.3772,-105.5217", "-105.5217,40.3772")
    assert not _leaf.literal_equal("40.3772,-105.5217", "40.3772,-105.5218")
    assert _leaf.literal_equal("2024-01-15", " 2024-01-15 ")
    assert not _leaf.literal_equal("2024-01-15", "2024-01-16")
    assert not _leaf.literal_equal("$1,250.00", "$1,250.50")
    # a literal against ordinary text falls back to plain equality
    assert not _leaf.literal_equal("300", "three hundred")


# --------------------------------------------------------------------------- through the evaluator
@pytest.fixture(scope="module")
def ev():
    """The certificate profile (literal character leaves)."""
    return Evaluator(profile="certificate")


@pytest.fixture(scope="module")
def ev_lit():
    return Evaluator(profile="certificate", leaf_literal_exact=True)


@pytest.fixture(scope="module")
def typed():
    return Evaluator()   # default profile


def doc(ev, a, b):
    return ev.calculate_similarity_with_deepdiff_opt({"v": a}, {"v": b})


def ted(ev, a, b):
    """The tree-edit path -- the one every instrument in the paper calls."""
    return ev.calculate_tree_edit_distance_opt({"v": a}, {"v": b}, variation_type="combined")


def test_numeric_leaves_are_binary_but_tolerate_float_noise(ev):
    assert doc(ev, 0.6, 0.60) == 1.0
    assert doc(ev, 20, 20.0) == 1.0
    assert doc(ev, 0.1 + 0.2, 0.3) == 1.0, "float noise used to cost a full leaf"
    assert doc(ev, 0.6, 0.6000001) == pytest.approx(0.5)
    assert doc(ev, 20, 21) == pytest.approx(0.5)
    assert doc(ev, 1_000_000, 1_000_001) == pytest.approx(0.5), "the kernel is binary, not graded"


def test_default_leaves_literal_strings_on_the_string_kernel(ev):
    """Pinned off-state values: a silent change to the default would be caught here."""
    assert not ev.leaf_literal_exact
    assert doc(ev, "40.3772,-105.5217", "40.3772,-105.5218") == pytest.approx(0.9706, abs=1e-3)
    assert doc(ev, "40.3772,-105.5217", "40.3772,105.5217") == pytest.approx(0.9706, abs=1e-3)
    assert doc(ev, "40.3772", "40.3773") == pytest.approx(0.9286, abs=1e-3)


def test_literal_exact_prices_a_changed_literal_like_a_changed_number(ev_lit):
    assert doc(ev_lit, "40.3772,-105.5217", "40.3772,-105.5218") == pytest.approx(0.5)
    assert doc(ev_lit, "40.3772,-105.5217", "40.3772,105.5217") == pytest.approx(0.5)
    assert doc(ev_lit, "40.3772", "40.3773") == pytest.approx(0.5)
    assert doc(ev_lit, "2024-01-15", "2024-01-16") == pytest.approx(0.5)
    # same value, different rendering: equal
    assert doc(ev_lit, "300", "300.0") == 1.0
    assert doc(ev_lit, "40.3772,-105.5217", "40.3772, -105.5217") == 1.0


def test_literal_exact_does_not_touch_ordinary_text(ev, ev_lit):
    for a, b in [("Boulder, Colorado", "Boulder Colorado"), ("male", "MALE"),
                 ("a quick brown fox", "a slow brown fox"), ("300", "three hundred")]:
        assert doc(ev_lit, a, b) == pytest.approx(doc(ev, a, b)), (a, b)


def test_literal_exact_applies_inside_arrays_too(ev_lit):
    s = ev_lit.calculate_similarity_with_deepdiff_opt(
        {"pts": ["40.3772,-105.5217", "51.5074,-0.1278"]},
        {"pts": ["40.3772,-105.5217", "51.5074,0.1278"]})
    assert s < 0.9, "a sign flip inside an array element must not be priced as a small edit"


# --------------------------------------------------------------------------- the typed profile
def test_default_profile_is_typed_and_certificate_is_still_available():
    assert Evaluator().profile == "typed"
    e = Evaluator(profile="certificate")
    assert not e.leaf_literal_exact and not e.numeric_string_equality


def test_typed_compares_number_and_numeric_string_by_value(ev, typed):
    # the I10 knife-edge: 0.949 under the certificate, a constant on the threshold
    assert ted(ev, 20, "20") == pytest.approx(0.9487, abs=1e-3)
    assert ted(typed, 20, "20") == 1.0
    assert ted(typed, 0.5, "0.5") == 1.0
    assert ted(typed, 20, "21") == 0.0
    assert ted(typed, 20, "twenty") == 0.0


def test_typed_applies_number_string_equality_inside_arrays(typed, ev):
    same = typed.calculate_tree_edit_distance_opt({"v": [0.5, 0.3]}, {"v": ["0.5", "0.3"]}, variation_type="combined")
    assert same == 1.0
    assert ev.calculate_tree_edit_distance_opt({"v": [0.5, 0.3]}, {"v": ["0.5", "0.3"]}, variation_type="combined") == pytest.approx(0.9487, abs=1e-3)
    assert typed.calculate_tree_edit_distance_opt({"v": [0.5, 0.3]}, {"v": ["0.5", "0.9"]}, variation_type="combined") == 0.0


def test_typed_reads_code_as_a_token_and_prose_as_a_form(typed, ev):
    # different programs: exact kernel -> the leaf is simply wrong
    assert ted(typed, "lambda x: 3*x**2 + 2*x + 1", "lambda x: x**2") == 0.0
    assert ted(ev, "lambda x: 3*x**2 + 2*x + 1", "lambda x: x**2") > 0.7, "the certificate's char kernel calls them 73% similar"
    # prose paraphrase keeps the form reading (embedding), far above the token kernel
    assert ted(typed, "Boulder, Colorado, United States", "Boulder Colorado USA") > 0.95
    assert ted(ev, "Boulder, Colorado, United States", "Boulder Colorado USA") < 0.8


def test_typed_keeps_the_literal_pricing(typed):
    assert ted(typed, "40.3772,-105.5217", "40.3772,105.5217") == 0.0
    assert ted(typed, "2024-01-15", "2024-01-16") == 0.0
    assert ted(typed, 0.1 + 0.2, 0.3) == 1.0
    assert ted(typed, "300", "300.0") == 1.0


# --------------------------------------------------------------------------- long strings
LONG_A = ("The user wants a comprehensive rollout strategy for the new billing system. " * 6).strip()
LONG_B = ("To create a risk-aware rollout strategy for the billing system, I will first list the stakeholders, "
          "then stage the deployment by region, and finally define rollback criteria. " * 4).strip()
LONG_C = ("Quarterly sales in the northeast region declined by four percent, driven mostly by weather. " * 6).strip()


def test_long_strings_are_no_longer_zeroed_by_array_semantics(typed, ev):
    """Two paraphrase-related texts with different chunk counts used to score exactly
    0 under any positional/bottleneck profile, because the chunk lists were compared
    as JSON arrays."""
    assert len(LONG_A) >= typed.chunk_size and len(LONG_B) >= typed.chunk_size
    assert len(typed._split_natural_text(LONG_A)) != len(typed._split_natural_text(LONG_B))
    s_typed = typed._compare_strings(LONG_A, LONG_B)
    assert 0.3 < s_typed < 1.0, s_typed
    assert typed._compare_strings(LONG_A, LONG_A) == 1.0
    assert typed._compare_strings(LONG_A, LONG_B) > typed._compare_strings(LONG_A, LONG_C), "related beats unrelated"
    # symmetric
    assert typed._compare_strings(LONG_A, LONG_B) == pytest.approx(typed._compare_strings(LONG_B, LONG_A))
    # the certificate reads chunks as tokens (char kernel) but must not zero them either
    assert ev._compare_strings(LONG_A, LONG_B) > 0.0


# --------------------------------------------------------------------------- the ranking profile
@pytest.fixture(scope="module")
def rank():
    return Evaluator(profile="ranking")


def test_ranking_profile_is_path_f1_and_not_the_default(rank):
    assert Evaluator().aggregation_mode == "tree" and Evaluator().profile == "typed"
    assert rank.aggregation_mode == "path_f1" and "name" in rank.exact_name_fields


def test_path_f1_basics(rank):
    ref = {"tool_calls": [{"name": "f", "arguments": {"a": 1, "b": "x"}}]}
    assert rank.calculate_tree_edit_distance_opt(ref, ref, variation_type="combined") == 1.0
    extra = {"tool_calls": [{"name": "f", "arguments": {"a": 1, "b": "x", "c": 5}}]}      # 4 cand paths, 3 hit
    missing = {"tool_calls": [{"name": "f", "arguments": {"a": 1}}]}                       # 2 cand paths, 3 ref paths
    s_extra = rank.calculate_tree_edit_distance_opt(ref, extra, variation_type="combined")
    s_missing = rank.calculate_tree_edit_distance_opt(ref, missing, variation_type="combined")
    assert s_extra == pytest.approx(2 * 0.75 * 1.0 / 1.75)            # P=3/4, R=1
    assert s_missing == pytest.approx(2 * 1.0 * (2 / 3) / (1 + 2 / 3))  # P=1, R=2/3
    assert 0 < s_extra < 1 and 0 < s_missing < 1, "no deviation is free and none is fatal"
    # symmetric
    assert rank.calculate_tree_edit_distance_opt(extra, ref, variation_type="combined") == pytest.approx(s_extra)


def test_path_f1_prices_a_wrong_callee_by_its_name_leaf_only(rank):
    ref = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Boulder"}}]}
    other_fn = {"tool_calls": [{"name": "get_forecast", "arguments": {"city": "Boulder"}}]}
    other_city = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Denver"}}]}
    s_fn = rank.calculate_tree_edit_distance_opt(ref, other_fn, variation_type="combined")
    assert s_fn == pytest.approx(0.5), "name leaf 0 (exact), argument leaf 1: F1 of (0.5, 0.5)"
    s_city = rank.calculate_tree_edit_distance_opt(ref, other_city, variation_type="combined")
    assert 0.5 < s_city < 1.0, "a different short string is priced by the leaf kernel, not zeroed"


def test_path_f1_missing_call_vs_fabricated_call_with_coinciding_keys(rank):
    """Documented behaviour, not a target: with no identity gate a fabricated call
    whose argument keys and values coincide with the reference's earns those
    leaves, so this pair ties. The gate that would break the tie was measured on
    the annotation pool and lost 16:3 against the annotator, so it is not applied."""
    ref = {"tool_calls": [{"name": "bmi", "arguments": {"h": 170, "w": 70}},
                          {"name": "bsa", "arguments": {"h": 170, "w": 70}}]}
    missing_call = {"tool_calls": [{"name": "bmi", "arguments": {"h": 170, "w": 70}}]}
    fabricated = {"tool_calls": [{"name": "bmi", "arguments": {"h": 170, "w": 70}},
                                 {"name": "crcl", "arguments": {"h": 170, "w": 70, "age": 75, "scr": 1.4, "sex": "f"}}]}
    a = rank.calculate_tree_edit_distance_opt(ref, missing_call, variation_type="combined")
    b = rank.calculate_tree_edit_distance_opt(ref, fabricated, variation_type="combined")
    assert a == pytest.approx(2 / 3) and b == pytest.approx(2 / 3)
    # but a fabricated call with *different* argument values is worse than a missing one
    fabricated2 = {"tool_calls": [{"name": "bmi", "arguments": {"h": 170, "w": 70}},
                                  {"name": "crcl", "arguments": {"h": 999, "w": 1, "age": 75}}]}
    assert rank.calculate_tree_edit_distance_opt(ref, fabricated2, variation_type="combined") < a


def test_path_f1_uses_the_typed_leaf_rules(rank):
    ref = {"v": 20, "d": "2024-01-15", "t": "The quick brown fox jumps over the lazy dog"}
    assert rank.calculate_tree_edit_distance_opt(ref, {"v": "20", "d": "2024-01-15", "t": ref["t"]}, variation_type="combined") == 1.0
    s_date = rank.calculate_tree_edit_distance_opt(ref, {"v": 20, "d": "2024-01-16", "t": ref["t"]}, variation_type="combined")
    assert s_date == pytest.approx(2 / 3)
    s_text = rank.calculate_tree_edit_distance_opt(ref, {"v": 20, "d": "2024-01-15", "t": "A quick brown fox leaps over a lazy dog"}, variation_type="combined")
    assert 2 / 3 < s_text < 1.0, "prose paraphrase keeps the embedding reading"


@pytest.fixture(scope="module")
def rank_aligned():
    return Evaluator(profile="ranking", align_calls_by_name=True)


def test_ranking_default_keeps_call_order_positional(rank):
    """Measured: the annotator reads call order as meaning (24 verdicts changed by
    alignment, 4 for / 20 against), so the ranking profile stays positional."""
    assert not rank.align_calls_by_name
    ref = {"tool_calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]}
    swapped = {"tool_calls": [ref["tool_calls"][1], ref["tool_calls"][0]]}
    assert rank.calculate_tree_edit_distance_opt(ref, swapped, variation_type="combined") < 1.0


def test_path_f1_aligns_calls_by_name_before_scoring(rank_aligned):
    """With the option on, a reordering of the same calls is the same set of calls."""
    rank = rank_aligned
    ref = {"tool_calls": [{"name": "convert", "arguments": {"amount": 100}},
                          {"name": "historical", "arguments": {"date": "2024-01-15"}},
                          {"name": "latest", "arguments": {"base": "USD"}}]}
    reordered = {"tool_calls": [ref["tool_calls"][0], ref["tool_calls"][2], ref["tool_calls"][1]]}
    assert rank.calculate_tree_edit_distance_opt(ref, reordered, variation_type="combined") == 1.0
    # reordered AND one value wrong: only that leaf is charged
    reordered_wrong = {"tool_calls": [ref["tool_calls"][0], ref["tool_calls"][2],
                                      {"name": "historical", "arguments": {"date": "2023-01-15"}}]}
    s = rank.calculate_tree_edit_distance_opt(ref, reordered_wrong, variation_type="combined")
    assert s == pytest.approx(5 / 6), s
    # a genuinely different third call: its name leaf is 0 and its argument is extra/missing
    other = {"tool_calls": [ref["tool_calls"][0], ref["tool_calls"][2], {"name": "delete", "arguments": {"id": 7}}]}
    assert rank.calculate_tree_edit_distance_opt(ref, other, variation_type="combined") < 5 / 6
    # a missing call is a hole: recall drops, precision does not
    fewer = {"tool_calls": [ref["tool_calls"][2], ref["tool_calls"][0]]}
    s_fewer = rank.calculate_tree_edit_distance_opt(ref, fewer, variation_type="combined")
    assert s_fewer == pytest.approx(2 * 1.0 * (4 / 6) / (1 + 4 / 6))


def test_path_f1_alignment_handles_repeated_names_and_plain_lists(rank_aligned):
    rank = rank_aligned
    ref = {"tool_calls": [{"name": "search", "arguments": {"q": "a"}}, {"name": "search", "arguments": {"q": "b"}}]}
    swapped = {"tool_calls": [{"name": "search", "arguments": {"q": "b"}}, {"name": "search", "arguments": {"q": "a"}}]}
    # same-name calls prefer their own index, so a swap of arguments is not hidden
    assert rank.calculate_tree_edit_distance_opt(ref, swapped, variation_type="combined") < 1.0
    # lists without a name field stay positional
    assert rank.calculate_tree_edit_distance_opt({"v": [1, 2, 3]}, {"v": [3, 2, 1]}, variation_type="combined") < 1.0
    assert rank.calculate_tree_edit_distance_opt({"v": [1, 2, 3]}, {"v": [1, 2, 3]}, variation_type="combined") == 1.0


def test_readouts_share_one_leaf_table(rank):
    ref = {"tool_calls": [{"name": "f", "arguments": {"a": 1, "b": "x", "c": "The quick brown fox jumps"}}]}
    same = ref
    one_wrong = {"tool_calls": [{"name": "f", "arguments": {"a": 2, "b": "x", "c": ref["tool_calls"][0]["arguments"]["c"]}}]}
    extra = {"tool_calls": [{"name": "f", "arguments": {"a": 1, "b": "x", "c": ref["tool_calls"][0]["arguments"]["c"], "d": 0}}]}
    paraphrase = {"tool_calls": [{"name": "f", "arguments": {"a": 1, "b": "x", "c": "A quick brown fox leaps"}}]}
    r = rank.calculate_readouts(ref, same);        assert r == {"score": 1.0, "certified": 1.0}
    r = rank.calculate_readouts(ref, one_wrong);   assert r["certified"] == 0.0 and 0.5 < r["score"] < 1.0
    r = rank.calculate_readouts(ref, extra);       assert r["certified"] == 0.0 and 0.5 < r["score"] < 1.0, "an extra path fails certification but barely dents the score"
    r = rank.calculate_readouts(ref, paraphrase);  assert 0.0 < r["certified"] < 1.0 and r["certified"] <= r["score"] < 1.0
    # score equals the path-F1 entry point
    assert rank.calculate_readouts(ref, paraphrase)["score"] == pytest.approx(rank.calculate_tree_edit_distance_opt(ref, paraphrase, variation_type="combined"))
    # certified is the minimum leaf: if it passes theta, every leaf did
    theta = r["certified"]
    lr = {rank._normalise_path(k): v for k, v in rank._leaf_paths(ref).items()}
    lc = {rank._normalise_path(k): v for k, v in rank._leaf_paths(paraphrase).items()}
    assert all(rank._path_leaf_similarity(k, lr[k], lc[k]) >= theta - 1e-12 for k in lr)


# --------------------------------------------------------------------------- analyzer shape bug
def test_analyzer_keeps_single_call_ground_truth_as_a_list():
    """An output identical to the ground truth must score 1.0 whatever the shape.
    The analyzer used to unwrap a one-element ground-truth list, comparing a dict
    against a list; legacy then scored the identical pair 0.75 and path-F1 0."""
    from sted.structural_consistency_analyzer import StructuralConsistencyAnalyzer
    gt = [{"name": "think-tool-get_thought_stats", "arguments": {}}]
    runs = [gt, gt, [{"name": "think-tool-get_thought_stats", "arguments": {"limit": 5}}]]
    for prof in ("legacy", "typed", "ranking"):
        an = StructuralConsistencyAnalyzer(Evaluator(profile=prof))
        rep = an.evaluate_structural_consistency(runs, gt, method_name="sted", variation_type="combined", validity_rate=1.0)
        sims = rep["raw_similarities"]
        assert sims[0] == 1.0 and sims[1] == 1.0, (prof, sims)
        assert sims[2] < 1.0, (prof, sims)


def test_ranking_treats_name_as_identity_only_on_calls(rank):
    """`name` beside `arguments` is a callee; `name` in a record is a value."""
    call_a = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Boulder"}}]}
    call_b = {"tool_calls": [{"name": "get_forecast", "arguments": {"city": "Boulder"}}]}
    assert rank.calculate_tree_edit_distance_opt(call_a, call_b, variation_type="combined") == pytest.approx(0.5)
    person_a = {"name": "John", "age": 30}
    person_b = {"name": "Jane", "age": 30}
    s = rank.calculate_tree_edit_distance_opt(person_a, person_b, variation_type="combined")
    assert 0.5 < s < 1.0, "a person's name is a string value: graded, not zeroed"


def test_ranking_grades_literal_near_misses_and_certificates_do_not(rank, typed):
    """A case-only identifier difference beats a different identifier for the ranker only."""
    a = {"tool_calls": [{"name": "list_exports", "arguments": {"botId": "B12345"}}]}
    case_only = {"tool_calls": [{"name": "list_exports", "arguments": {"botId": "b12345"}}]}
    other = {"tool_calls": [{"name": "list_exports", "arguments": {"botId": "C99999"}}]}
    r_case = rank.calculate_tree_edit_distance_opt(a, case_only, variation_type="combined")
    r_other = rank.calculate_tree_edit_distance_opt(a, other, variation_type="combined")
    assert r_other < r_case < 1.0
    t_case = typed.calculate_tree_edit_distance_opt(a, case_only, variation_type="combined")
    assert t_case == typed.calculate_tree_edit_distance_opt(a, other, variation_type="combined")
    assert rank._compare_strings("MID", "mid") == pytest.approx(0.95)
    assert rank._compare_strings("Q3CC-CRT3-SZ2G", "q3cc-crt3-sz2g") == pytest.approx(0.95)
    assert rank._compare_strings("B12345", "C99999") < 0.5
    assert typed._compare_strings("MID", "mid") == 0.0
    # dates are never near misses; numbers are graded by relative error (ranker only)
    assert rank._compare_strings("2024-01-15", "2024-01-16") == 0.0
    assert 0.85 < rank._compare_strings("300", "301") < 0.9
    assert rank._compare_strings("300", "400") == 0.0


def test_ranking_near_miss_is_for_token_sized_literals_only(rank):
    """Two long, different code literals are a miss, not an edit-ratio computation."""
    a = "def f(x):\n    return x + 1\n" * 40
    b = "def g(y):\n    return y - 1\n" * 40
    assert rank._compare_strings(a, b) == 0.0
    assert rank._compare_strings("get_user(1)", "get_user(2)") > 0.5


def test_ranking_numeric_near_miss_is_graded_and_capped(rank, typed):
    """A nearly-right number ranks above a wrong one under ranking; certificates stay binary."""
    gt = {"tool_calls": [{"name": "geo", "arguments": {"coords": "32.0835,-81.0998", "n": 999.0}}]}
    def with_(coords, n):
        return {"tool_calls": [{"name": "geo", "arguments": {"coords": coords, "n": n}}]}
    near = rank.calculate_tree_edit_distance_opt(gt, with_("32.0836,-81.0998", 1000.0), variation_type="combined")
    far = rank.calculate_tree_edit_distance_opt(gt, with_("40.7128,-74.0060", 59.99), variation_type="combined")
    assert far < near < 1.0
    assert rank._compare_strings("32.0835,-81.0998", "40.7128,-74.0060") == 0.0
    assert rank._compare_strings("32.0835,-81.0998", "32.0835") == 0.0, "different element counts"
    assert rank._compare_numbers(999.0, 1000.0) == pytest.approx(0.9 * (1 - 10 * 1 / 1000))
    assert rank._compare_numbers(999.0, 59.99) == 0
    assert rank._numeric_string_match(999, "1000") == pytest.approx(0.9 * (1 - 10 * 1 / 1000))
    assert typed._compare_numbers(999.0, 1000.0) == 0
    assert typed._compare_strings("32.0835,-81.0998", "32.0836,-81.0998") == 0.0
    assert typed._numeric_string_match(999, "1000") == 0.0


def test_positional_arrays_follow_the_profile_aggregation(typed):
    """Under a bottleneck profile a positional array is as bad as its worst element; legacy keeps the mean."""
    from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator as E
    legacy = E(profile="legacy")
    gt = {"tool_calls": [{"name": "f", "arguments": {"v": [1, 2, 3, 4, 5, 6, 7, 8]}}]}
    one_off = {"tool_calls": [{"name": "f", "arguments": {"v": [1, 2, 3, 4, 5, 6, 7, 9]}}]}
    rotated = {"tool_calls": [{"name": "f", "arguments": {"v": [2, 3, 4, 5, 6, 7, 8, 1]}}]}
    for cand in (one_off, rotated):
        assert typed.calculate_tree_edit_distance_opt(gt, cand, variation_type="combined") < 0.95, "a wrong element is not diluted"
    assert legacy._compare_arrays_ordered([1, 2, 3, 4], [1, 2, 3, 5], "a", "b") == pytest.approx(0.25)
    assert typed._compare_arrays_ordered([1, 2, 3, 4], [1, 2, 3, 5], "a", "b") == pytest.approx(1.0)
    assert typed._compare_arrays_ordered([1, 2, 3], [1, 2, 3, 4], "a", "b") == pytest.approx(1.0)
    assert typed._compare_arrays_ordered([1, 2, 3], [1, 2, 3], "a", "b") == 0


def test_typed_treats_callee_names_as_identities(typed):
    """A substituted function name is a full leaf miss under typed; a person's name is still text."""
    gt = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Boulder", "days": 3}}]}
    other_fn = {"tool_calls": [{"name": "get_forecast", "arguments": {"city": "Boulder", "days": 3}}]}
    s = typed.calculate_tree_edit_distance_opt(gt, other_fn, variation_type="combined")
    assert s < 0.95, s
    person = {"record": {"name": "John Smith", "age": 30}}
    person2 = {"record": {"name": "Jon Smith", "age": 30}}
    assert typed.calculate_tree_edit_distance_opt(person, person2, variation_type="combined") > 0.5


def test_identifier_keys_gate_object_valued_fields(typed):
    """Under mu = 1 a renamed object-valued key voids its subtree; legacy keeps its graded reading."""
    from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator as E
    legacy = E(profile="legacy"); cert = E(profile="certificate")
    a, b = {"a": {"x": 1}}, {"b": {"x": 1}}
    assert typed.calculate_tree_edit_distance_opt(a, b, variation_type="combined") == 0.0
    assert cert.calculate_tree_edit_distance_opt(a, b, variation_type="combined") == 0.0
    assert legacy.calculate_tree_edit_distance_opt(a, b, variation_type="combined") == pytest.approx(0.75)
    # BFCL shape: the tool name is an object-valued key
    assert typed.calculate_tree_edit_distance_opt({"get_weather": {"city": "Boulder"}}, {"get_forecast": {"city": "Boulder"}}, variation_type="combined") == 0.0
    # equal keys are untouched
    assert typed.calculate_tree_edit_distance_opt({"a": {"x": 1, "y": 2}}, {"a": {"x": 1, "y": 3}}, variation_type="combined") == \
        pytest.approx(E(profile="typed").calculate_tree_edit_distance_opt({"a": {"x": 1, "y": 2}}, {"a": {"x": 1, "y": 3}}, variation_type="combined"))


def test_numbers_close_is_relative_only():
    """Small-magnitude numbers are not all equal; float noise still is."""
    from sted._leaf import numbers_close
    assert not numbers_close(1.6e-19, 1.76e-19)
    assert not numbers_close(1.6e-19, -1.6e-19)
    assert numbers_close(0.1 + 0.2, 0.3)
    assert numbers_close(0.0, -0.0)
    assert numbers_close(0.0, 1e-15)          # float noise against an exact zero
    assert not numbers_close(1e-15, 2e-15)     # but two non-zero tiny numbers are compared relatively
    assert numbers_close(300.0, 300.0000000001)
