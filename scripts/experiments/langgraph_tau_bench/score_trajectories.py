"""Score the JSONL produced by run_trajectories.py with STED trajectory mode.

Usage:
    python score_trajectories.py \\
        --input results/full_n115_k5/trajectories.jsonl \\
        --output results/full_n115_k5/report.json \\
        --bottom-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root so `import sted` works without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sted import AgentConsistencyEvaluator  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Trajectories JSONL from run_trajectories.py.")
    p.add_argument("--output", type=Path, required=True,
                   help="JSON output for the report.")
    p.add_argument("--bottom-k", type=int, default=10)
    p.add_argument("--n-workers", type=int, default=4,
                   help="Threads for prompt-level scoring.")
    args = p.parse_args()

    # Group trajectories by (task_idx, instruction).
    groups: dict = defaultdict(list)
    n_records = n_errors = n_empty = 0
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_records += 1
            if rec.get("error"):
                n_errors += 1
                continue
            traj = rec.get("trajectory") or []
            if not traj:
                n_empty += 1
                continue
            # Use the instruction as the prompt key — that's what shows up in
            # the report's worst-prompts list. Prepend task_idx for clarity.
            key = f"task_{rec['task_idx']:03d}: {rec['instruction'][:80]}"
            groups[key].append(traj)

    # Drop tasks with fewer than 2 valid trajectories — STED needs pairs.
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    n_dropped = len(groups) - len(valid_groups)

    print(f"Loaded {n_records} records from {args.input}")
    print(f"  errors:           {n_errors}")
    print(f"  empty trajectories: {n_empty}")
    print(f"  tasks with <2 valid runs (dropped): {n_dropped}")
    print(f"  tasks scored:     {len(valid_groups)}")
    print()

    if not valid_groups:
        print("No tasks have >= 2 valid trajectories — nothing to score.",
              file=sys.stderr)
        return 1

    evaluator = AgentConsistencyEvaluator.for_trajectory(n_workers=args.n_workers)
    report = evaluator.evaluate_outputs(
        valid_groups, bottom_k=args.bottom_k, progress="tqdm",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "summary": {
                    "n_tasks_scored": len(valid_groups),
                    "n_records_total": n_records,
                    "n_errors": n_errors,
                    "n_empty": n_empty,
                    "n_dropped_lt2": n_dropped,
                },
                "report": report.to_dict(),
            },
            f, indent=2,
        )

    print()
    print(report.summary())
    print()
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
