#!/usr/bin/env python3
"""
Run full analysis pipeline for COLM consistency predictor.

Steps:
1. Calculate STED metrics for json-instruct
2. Run combined analysis (Toucan + ShareGPT + json-instruct)
3. Train predictor on combined data

Usage:
    python run_full_analysis.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


def run_command(cmd: list, desc: str):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"{desc}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\nWARNING: Command failed with return code {result.returncode}")
        return False
    return True


def main():
    print("=" * 70)
    print("COLM 2026 Full Analysis Pipeline")
    print("=" * 70)

    # Step 1: Calculate STED metrics
    run_command(
        [sys.executable, str(SCRIPT_DIR / "calculate_json_instruct_sted.py")],
        "Step 1/3: Calculate STED metrics for json-instruct"
    )

    # Step 2: Run combined analysis
    run_command(
        [sys.executable, str(SCRIPT_DIR / "experiments" / "run_combined_analysis.py")],
        "Step 2/3: Run combined analysis (Toucan + ShareGPT + json-instruct)"
    )

    # Step 3: Train predictor
    run_command(
        [sys.executable, str(SCRIPT_DIR / "experiments" / "exp3_train_predictor.py")],
        "Step 3/3: Train consistency predictor on combined data"
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: experiments/colm_2026_consistency_predicto_20260210_154446/results/")


if __name__ == "__main__":
    main()
