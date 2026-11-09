"""The ``sted`` command line.

    sted compare a.json b.json             which fields differ        (exit 1 if different)
    sted consistency runs.jsonl            which fields are unstable  (exit 1 below --min-same)
    sted diff before/ after/               regression check by file   (exit 1 if any differ)

Field rules on every subcommand: ``--ignore PATTERN``, ``--unordered PATTERN``,
``--tolerance PATTERN=VALUE`` (repeatable), or ``--rules rules.json`` with
``{"ignore": [...], "unordered": [...], "tolerance": {...}}`` (YAML too if PyYAML is
installed). ``--json`` prints machine-readable results.

Exit codes: 0 same / consistent, 1 different / inconsistent, 2 usage or input error.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXIT_OK, EXIT_DIFFERENT, EXIT_ERROR = 0, 1, 2


class InputError(Exception):
    pass


# ---------------------------------------------------------------- inputs

def _read_json(path: str) -> Any:
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise InputError(f"{path}: not valid JSON ({e})") from e


def _read_runs(path: str) -> Dict[str, List[Any]]:
    """Groups of outputs keyed by prompt.

    Accepted: a JSON array of outputs; JSONL with one output per line; or JSONL /
    JSON array of ``{"prompt": ..., "outputs": [...]}`` rows (the sted-eval format).
    A JSONL line that is not valid JSON is kept as text and counts as invalid."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    stripped = text.lstrip()
    rows: List[Any]
    if stripped.startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as e:
            raise InputError(f"{path}: not valid JSON ({e})") from e
    else:
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(line)          # an unparseable generation: invalid output
    if rows and all(isinstance(r, dict) and "outputs" in r for r in rows):
        groups: Dict[str, List[Any]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(str(r.get("prompt", f"row {i}")), []).extend(r["outputs"])
        return groups
    return {Path(path).name if path != "-" else "stdin": rows}


def _parse_tolerance(items: Sequence[str]) -> Dict[str, float]:
    out = {}
    for item in items or []:
        pattern, sep, value = item.rpartition("=")
        if not sep or not pattern:
            raise InputError(f"--tolerance expects PATTERN=VALUE, got {item!r}")
        try:
            out[pattern] = float(value)
        except ValueError:
            raise InputError(f"--tolerance {item!r}: {value!r} is not a number") from None
    return out


def _rules(args) -> "Any":
    from .rules import FieldRules

    spec: Dict[str, Any] = {"ignore": [], "unordered": [], "tolerance": {}}
    if args.rules:
        p = Path(args.rules)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise InputError("YAML rules need PyYAML: pip install pyyaml") from None
            loaded = yaml.safe_load(text) or {}
        else:
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as e:
                raise InputError(f"{args.rules}: not valid JSON ({e})") from e
        unknown = set(loaded) - set(spec)
        if unknown:
            raise InputError(f"{args.rules}: unknown keys {sorted(unknown)} "
                             "(expected ignore, unordered, tolerance)")
        for k in spec:
            if k in loaded:
                spec[k] = loaded[k]
    spec["ignore"] = list(spec["ignore"]) + list(args.ignore or [])
    spec["unordered"] = list(spec["unordered"]) + list(args.unordered or [])
    spec["tolerance"] = {**spec["tolerance"], **_parse_tolerance(args.tolerance)}
    try:
        return FieldRules(**spec)
    except ValueError as e:
        raise InputError(str(e)) from e


# ---------------------------------------------------------------- commands

def _cmd_compare(args) -> int:
    from . import api

    a, b = _read_json(args.a), _read_json(args.b)
    r = api.compare(a, b, threshold=args.threshold, rules=_rules(args),
                    embedding_model=args.embedding_model)
    print(json.dumps(r.to_dict(), indent=2, default=str) if args.json else r)
    return EXIT_OK if r.same else EXIT_DIFFERENT


def _cmd_consistency(args) -> int:
    from . import api

    groups = _read_runs(args.file)
    rules = _rules(args)
    results = {}
    worst = 1.0
    for prompt, outputs in groups.items():
        if len(outputs) < 2:
            raise InputError(f"{prompt!r}: needs at least 2 outputs, got {len(outputs)}")
        r = api.consistency(outputs, threshold=args.threshold, rules=rules,
                            embedding_model=args.embedding_model)
        results[prompt] = r
        worst = min(worst, r.share_same if r.share_same is not None else 0.0)
    if args.json:
        payload = {p: r.to_dict() for p, r in results.items()}
        print(json.dumps(payload if len(payload) > 1 else next(iter(payload.values())), indent=2, default=str))
    else:
        for i, (prompt, r) in enumerate(results.items()):
            if len(results) > 1:
                print(("\n" if i else "") + f"== {prompt}")
            print(r)
    if args.min_same is not None and worst < args.min_same:
        if not args.json:
            sys.stdout.flush()
            print(f"\nFAIL: share of same pairs {worst:.0%} < --min-same {args.min_same:.0%}", file=sys.stderr)
        return EXIT_DIFFERENT
    return EXIT_OK


def _files(root: Path, pattern: str) -> Dict[str, Path]:
    return {str(p.relative_to(root)): p for p in sorted(root.rglob(pattern)) if p.is_file()}


def _cmd_diff(args) -> int:
    from . import api

    a_root, b_root = Path(args.before), Path(args.after)
    for root in (a_root, b_root):
        if not root.is_dir():
            raise InputError(f"{root}: not a directory")
    fa, fb = _files(a_root, args.glob), _files(b_root, args.glob)
    if not fa and not fb:
        raise InputError(f"no files matching {args.glob!r} in {a_root} or {b_root}")
    rules = _rules(args)
    report: List[Dict[str, Any]] = []
    for name in sorted(set(fa) | set(fb)):
        if name not in fb:
            report.append({"file": name, "status": "missing_after"})
            continue
        if name not in fa:
            report.append({"file": name, "status": "new_after"})
            continue
        try:
            r = api.compare(_read_json(str(fa[name])), _read_json(str(fb[name])), threshold=args.threshold,
                            rules=rules, embedding_model=args.embedding_model)
        except (InputError, ValueError) as e:
            report.append({"file": name, "status": "error", "error": str(e)})
            continue
        report.append({"file": name, "status": "same" if r.same else "different", "result": r})

    counts: Dict[str, int] = {}
    for row in report:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    failed = any(row["status"] != "same" for row in report)
    if args.json:
        out = [{**{k: v for k, v in row.items() if k != "result"},
                **({"comparison": row["result"].to_dict()} if "result" in row else {})} for row in report]
        print(json.dumps({"files": out, "summary": counts}, indent=2, default=str))
    else:
        for row in report:
            st = row["status"]
            if st == "same" and not args.verbose:
                continue
            if "result" in row:
                r = row["result"]
                lines = str(r).splitlines()
                print(f"{st.upper():<10} {row['file']}   min {r.min_readout:.3f}")
                for line in lines[1:]:
                    print(line)
            else:
                print(f"{st.upper():<10} {row['file']}" + (f"   {row['error']}" if "error" in row else ""))
        summary = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(counts.items()))
        print(f"\n{len(report)} files: {summary}")
    return EXIT_DIFFERENT if failed else EXIT_OK


# ---------------------------------------------------------------- parser

def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("field rules")
    g.add_argument("--ignore", action="append", metavar="PATTERN", help="drop matching fields (repeatable)")
    g.add_argument("--unordered", action="append", metavar="PATTERN", help="compare matching arrays as sets")
    g.add_argument("--tolerance", action="append", metavar="PATTERN=VALUE",
                   help="absolute numeric tolerance for matching fields")
    g.add_argument("--rules", metavar="FILE", help="JSON/YAML file with ignore / unordered / tolerance")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="min readout at or above which two outputs count as the same (default 0.95)")
    p.add_argument("--embedding-model", metavar="NAME",
                   help='Sentence-Transformers model for text fields, or "none" for character similarity '
                        "(default: all-MiniLM-L6-v2 if installed)")
    p.add_argument("--json", action="store_true", help="print JSON instead of the text summary")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sted", description="Compare structured LLM outputs field by field and rate their consistency.",
        epilog="Path patterns: request_id (any depth), arguments.tags, items[*].id, **.timestamp")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compare", help="which fields differ between two JSON outputs (exit 1 if different)")
    p.add_argument("a", help="first JSON file, or - for stdin")
    p.add_argument("b", help="second JSON file")
    _add_common(p)
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("consistency", help="which fields are unstable across repeated runs")
    p.add_argument("file", help="JSON array or JSONL of outputs, or rows of {prompt, outputs}; - for stdin")
    p.add_argument("--min-same", type=float, metavar="SHARE",
                   help="exit 1 if the share of same pairs is below this (e.g. 0.9), for any prompt")
    _add_common(p)
    p.set_defaults(func=_cmd_consistency)

    p = sub.add_parser("diff", help="compare same-named JSON files in two directories (exit 1 if any differ)")
    p.add_argument("before", help="directory with the reference outputs")
    p.add_argument("after", help="directory with the new outputs")
    p.add_argument("--glob", default="*.json", help="which files to compare (default *.json, recursive)")
    p.add_argument("-v", "--verbose", action="store_true", help="also list files that are the same")
    _add_common(p)
    p.set_defaults(func=_cmd_diff)
    return parser


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:  # pragma: no cover
        return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        print("sted: --threshold must be in [0, 1]", file=sys.stderr)
        return EXIT_ERROR
    try:
        return args.func(args)
    except (InputError, OSError) as e:
        print(f"sted: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
