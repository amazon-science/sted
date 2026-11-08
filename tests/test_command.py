"""The `sted` command line (character mode, so it runs with or without torch)."""
import json

import pytest

from sted.command import main

NONE = ["--embedding-model", "none"]


def call(**kw):
    args = {"from": "SEA", "to": "JFK", "date": "2026-12-01", "passengers": 2, "request_id": "r-1"}
    args.update(kw)
    return {"name": "book_flight", "arguments": args}


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()
    w = lambda p, v: (tmp_path / p).write_text(json.dumps(v))
    w("before/q1.json", call()); w("after/q1.json", call(request_id="r-2", passengers="2"))
    w("before/q2.json", call()); w("after/q2.json", call(date="2026-12-02", request_id="r-3"))
    w("before/q3.json", call())
    w("rules.json", {"ignore": ["request_id"]})
    lines = [json.dumps(call(date=d, request_id="x")) for d in ["2026-12-01"] * 6 + ["2026-12-02"] * 3]
    (tmp_path / "runs.jsonl").write_text("\n".join(lines + ["Sorry, I can't book flights."]) + "\n")
    return tmp_path


def test_compare_exit_codes(tree, capsys):
    assert main(["compare", str(tree / "before/q2.json"), str(tree / "after/q2.json"), *NONE]) == 1
    out = capsys.readouterr().out
    assert out.startswith("DIFFERENT") and "arguments.date" in out and "arguments.request_id" in out
    assert main(["compare", str(tree / "before/q1.json"), str(tree / "after/q1.json"),
                 "--ignore", "request_id", *NONE]) == 0
    assert capsys.readouterr().out.startswith("SAME")


def test_compare_json_output(tree, capsys):
    main(["compare", str(tree / "before/q2.json"), str(tree / "after/q2.json"), "--json", *NONE])
    d = json.loads(capsys.readouterr().out)
    assert d["same"] is False and d["text_similarity"] == "character"
    assert {x["path"] for x in d["differences"]} == {"arguments.date", "arguments.request_id"}


def test_diff_directories(tree, capsys):
    rc = main(["diff", str(tree / "before"), str(tree / "after"), "--rules", str(tree / "rules.json"), *NONE])
    out = capsys.readouterr().out
    assert rc == 1
    assert "DIFFERENT  q2.json" in out and "MISSING_AFTER q3.json" in out and "q1.json" not in out
    assert "3 files: 1 different, 1 missing after, 1 same" in out
    main(["diff", str(tree / "before"), str(tree / "after"), "--ignore", "request_id", "--json", *NONE])
    d = json.loads(capsys.readouterr().out)
    assert d["summary"] == {"same": 1, "different": 1, "missing_after": 1}


def test_diff_all_same_exits_zero(tree, capsys):
    assert main(["diff", str(tree / "before"), str(tree / "before"), *NONE]) == 0
    assert "3 files: 3 same" in capsys.readouterr().out


def test_consistency_jsonl_and_gate(tree, capsys):
    f = str(tree / "runs.jsonl")
    assert main(["consistency", f, "--ignore", "request_id", *NONE]) == 0
    out = capsys.readouterr().out
    assert "validity 0.90 (9/10)" in out and "arguments.date" in out
    assert main(["consistency", f, "--ignore", "request_id", "--min-same", "0.9", *NONE]) == 1
    assert "FAIL" in capsys.readouterr().err
    assert main(["consistency", f, "--ignore", "request_id", "--min-same", "0.5", *NONE]) == 0


def test_consistency_prompt_rows(tmp_path, capsys):
    rows = [{"prompt": "p1", "outputs": [call(), call()]}, {"prompt": "p2", "outputs": [call(), call(to="LAX")]}]
    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert main(["consistency", str(f), "--json", *NONE]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["p1"]["share_same"] == 1.0 and d["p2"]["unstable_fields"][0]["path"] == "arguments.to"


def test_tolerance_and_unordered_flags(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"price": 10.0, "tags": ["x", "y"]}))
    b.write_text(json.dumps({"price": 10.004, "tags": ["y", "x"]}))
    assert main(["compare", str(a), str(b), *NONE]) == 1
    assert main(["compare", str(a), str(b), "--tolerance", "price=0.01", "--unordered", "tags", *NONE]) == 0


@pytest.mark.parametrize("argv", [
    ["compare", "missing.json", "missing2.json"],
    ["compare", "{a}", "{b}", "--tolerance", "price"],
    ["compare", "{a}", "{b}", "--threshold", "2"],
    ["consistency", "{a}"],
])
def test_usage_errors_exit_2(tmp_path, capsys, argv):
    a = tmp_path / "a.json"
    a.write_text("[1]")
    argv = [x.replace("{a}", str(a)).replace("{b}", str(a)) for x in argv]
    assert main(argv + NONE) == 2
    assert "sted:" in capsys.readouterr().err
