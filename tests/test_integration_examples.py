"""Run each framework example in examples/integrations; skip when its framework is absent."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "integrations"
CASES = [("braintrust_example.py", "braintrust"), ("weave_example.py", "weave"),
         ("deepeval_example.py", "deepeval"), ("ragas_example.py", "ragas"), ("mlflow_example.py", "mlflow")]
MIN_VERSION = {"ragas": (0, 4), "mlflow": (3, 0)}     # the examples use these APIs


def _too_old(module):
    import importlib.metadata as md
    want = MIN_VERSION.get(module)
    if not want:
        return None
    have = md.version(module)
    parts = tuple(int(x) for x in have.split(".")[:2] if x.isdigit())
    return have if parts < want else None


def _env():
    return {**os.environ, "STED_EMBEDDING_MODEL": "none", "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
            "WEAVE_DISABLED": "true", "PROMPTFOO_DISABLE_TELEMETRY": "1"}


@pytest.mark.parametrize("script,module", CASES)
def test_example(script, module):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    old = _too_old(module)
    if old:
        pytest.skip(f"{module} {old} is older than the example needs ({MIN_VERSION[module]})")
    p = subprocess.run([sys.executable, script], cwd=EXAMPLES, env=_env(), capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), (p.stdout[-2000:] + p.stderr[-3000:])


def test_promptfoo_example(tmp_path):
    if os.environ.get("STED_RUN_PROMPTFOO") != "1" or shutil.which("npx") is None:
        pytest.skip("set STED_RUN_PROMPTFOO=1 with Node installed to run promptfoo")
    out = tmp_path / "out.json"
    p = subprocess.run(["npx", "-y", "promptfoo@latest", "eval", "-c", "promptfooconfig.yaml", "--no-cache",
                        "-o", str(out)], cwd=EXAMPLES / "promptfoo",
                       env={**_env(), "PROMPTFOO_PYTHON": sys.executable}, capture_output=True, text=True,
                       timeout=600)
    import json
    results = json.loads(out.read_text())["results"]["results"]
    verdicts = {r["testCase"]["description"]: r["success"] for r in results}
    assert verdicts == {'same call, reordered keys and "2" for 2': True, "drifted date (expected to fail)": False}, p.stdout[-2000:]


def test_promptfoo_assertion_function():
    sys.path.insert(0, str(EXAMPLES / "promptfoo"))
    os.environ.setdefault("STED_EMBEDDING_MODEL", "none")
    from sted_assert import get_assert
    ctx = {"vars": {"expected": '{"a": 1, "id": "x"}', "sted_rules": '{"ignore": ["id"]}'}}
    assert get_assert('{"id": "y", "a": 1}', ctx)["pass"] is True
    r = get_assert('{"a": 2, "id": "x"}', ctx)
    assert r["pass"] is False and "a" in r["reason"]
    assert get_assert("not json", ctx)["pass"] is False
