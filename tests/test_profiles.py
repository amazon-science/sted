from sted.semantic_json_tree_consistency import (
    STED_PROFILES,
    SemanticJsonTreeConsistencyEvaluator,
)


MODEL = "all-MiniLM-L6-v2"


def test_typed_is_the_public_default_and_certificate_is_selectable():
    ev = SemanticJsonTreeConsistencyEvaluator(model_id=MODEL)
    assert ev.profile == "typed"
    assert ev.leaf_literal_exact and ev.numeric_string_equality and ev.leaf_literalism_tau is None
    ev = SemanticJsonTreeConsistencyEvaluator(model_id=MODEL, profile="certificate")
    assert ev.profile == "certificate"
    assert ev.cost_combination == "product"
    assert ev.array_order_lambda == 1.0
    assert ev.key_identity_mu == 1.0
    assert ev.aggregation_p == float("-inf")
    assert ev.leaf_literalism_tau == 1.0
    assert ev.sort_arrays is False


def test_legacy_profile_reproduces_published_configuration():
    ev = SemanticJsonTreeConsistencyEvaluator(model_id=MODEL, profile="legacy")
    assert ev.cost_combination == "additive"
    assert ev.array_order_lambda is None
    assert ev.key_identity_mu is None
    assert ev.aggregation_p is None
    assert ev.leaf_literalism_tau is None
    assert ev.sort_arrays is True


def test_explicit_axes_override_the_selected_profile():
    ev = SemanticJsonTreeConsistencyEvaluator(
        model_id=MODEL,
        profile="certificate",
        cost_combination="additive",
        array_order_lambda=None,
        key_identity_mu=None,
        aggregation_p=None,
        leaf_literalism_tau=None,
    )
    assert ev.cost_combination == STED_PROFILES["legacy"]["cost_combination"]
    assert ev.array_order_lambda is None
    assert ev.key_identity_mu is None
    assert ev.aggregation_p is None
    assert ev.leaf_literalism_tau is None
