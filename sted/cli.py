"""Command-line interface for sted.

Usage:
    sted-eval --input logs.jsonl --runs-key runs --output report.json
    sted-eval --help

The input file is JSONL where each line is:
    {"prompt": "...", "outputs": [{...}, {...}, ...]}

Output is a JSON report compatible with ConsistencyReport.to_dict().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from . import AgentConsistencyEvaluator
from ._logging import get_logger, set_log_level

logger = get_logger("cli")


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("line %d not valid JSON: %s", i, e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sted-eval",
        description="Evaluate LLM agent consistency from production logs.",
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="JSONL file with one row per prompt. Each row must have "
             "'prompt' (str) and 'outputs' (list of JSON-serializable values).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write JSON report to this path (default: stdout).",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Number of evaluator threads (default 4).",
    )
    parser.add_argument(
        "--bottom-k", type=int, default=10,
        help="Surface this many least-consistent prompts in the report.",
    )
    parser.add_argument(
        "--no-precompute", action="store_true",
        help="Disable cross-prompt batch embedding precompute (slower).",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Per-pair STED timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print human-readable summary instead of full JSON report.",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default WARNING).",
    )
    args = parser.parse_args(argv)

    set_log_level(args.log_level)

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    logger.info("loading prompts from %s", args.input)
    outputs_per_prompt: dict[str, list] = {}
    for row in _read_jsonl(args.input):
        prompt = row.get("prompt")
        outputs = row.get("outputs")
        if prompt is None or outputs is None:
            logger.warning("skipping row missing 'prompt' or 'outputs': %s",
                           list(row.keys()))
            continue
        outputs_per_prompt[prompt] = outputs
    logger.info("loaded %d prompts", len(outputs_per_prompt))

    if not outputs_per_prompt:
        print("error: no valid prompts found in input", file=sys.stderr)
        return 2

    eval_kwargs = {"n_workers": args.workers}
    if args.timeout is not None:
        eval_kwargs["timeout_seconds"] = args.timeout
    evaluator = AgentConsistencyEvaluator(**eval_kwargs)

    logger.info("evaluating %d prompts with %d workers",
                len(outputs_per_prompt), args.workers)
    report = evaluator.evaluate_outputs(
        outputs_per_prompt,
        bottom_k=args.bottom_k,
        precompute_embeddings=not args.no_precompute,
    )

    if args.summary_only:
        out_text = report.summary()
    else:
        out_text = json.dumps(report.to_dict(), indent=2, default=str)

    if args.output:
        args.output.write_text(out_text + "\n")
        logger.info("wrote report to %s", args.output)
    else:
        print(out_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
