"""The README quick-start examples, as tests. Local embedding model, no cloud credentials."""
import pytest

from sted.semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

REF = {"name": "get_weather", "arguments": {"city": "Seattle", "days": 3, "units": "metric"}}
REORDERED = {"arguments": {"units": "metric", "days": "3", "city": "Seattle"}, "name": "get_weather"}
CHANGED = {"name": "get_weather", "arguments": {"city": "Seattle", "days": 5, "units": "metric"}}


@pytest.fixture(scope="module")
def same():
    return SemanticJsonTreeConsistencyEvaluator(model_id="all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def drift():
    return SemanticJsonTreeConsistencyEvaluator(profile="typed_mean", model_id="all-MiniLM-L6-v2")


def _score(ev, a, b):
    return ev.calculate_tree_edit_distance_opt(a, b, variation_type="combined")


def test_readme_example_harmless_differences(same, drift):
    assert _score(same, REF, REORDERED) == 1.0
    assert _score(drift, REF, REORDERED) == 1.0


def test_readme_example_changed_argument(same, drift):
    assert _score(same, REF, CHANGED) == 0.0
    assert _score(drift, REF, CHANGED) == pytest.approx(0.833, abs=1e-3)


def test_legacy_profile_original_example():
    legacy = SemanticJsonTreeConsistencyEvaluator(profile="legacy", model_id="all-MiniLM-L6-v2")
    s = _score(legacy, {"name": "John", "age": 30, "city": "New York"},
               {"name": "John", "age": 30, "location": "NYC"})
    assert 0.0 < s < 1.0
