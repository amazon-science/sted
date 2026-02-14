#!/usr/bin/env python3
"""
ACL 2026 Paper: Master Experiment Runner

Orchestrates all phases of the ACL linguistic variation experiments:
1. Data Preparation (stratified by schema complexity)
2. Phase 1: Multi-model baseline collection
3. Phase 2: Stratified linguistic interventions
4. Phase 3: Schema × linguistic interaction analysis
5. Analysis: Ceiling effect, cross-model transfer

Building on KDD findings:
- Schema complexity is #1 factor (19% SHAP importance)
- Per-model R² = 0.67 vs Pooled R² = 0.10 (model heterogeneity)
- Ceiling effect: interventions help difficult, harm easy prompts

Usage:
    # Full pipeline
    python run_acl_experiments.py --all

    # Individual phases
    python run_acl_experiments.py --prepare-data
    python run_acl_experiments.py --phase1 --model MODEL_ID
    python run_acl_experiments.py --phase2 --model MODEL_ID
    python run_acl_experiments.py --phase3 --model MODEL_ID
    python run_acl_experiments.py --analyze
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPT_DIR = Path(__file__).parent

# Primary models for ACL analysis
ACL_MODELS = [
    "us.anthropic.claude-sonnet-4-20250514-v1:0",  # High consistency baseline
    "openai/gpt-4.1-mini",                          # KDD SHAP outlier (rho=0.07)
    "us.meta.llama3-3-70b-instruct-v1:0",          # Good SHAP correlation (0.53)
    "qwen.qwen3-235b-a22b-2507-v1:0",              # Good generalization
    "google/gemini-2.5-flash-lite",                 # Different architecture
]

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "acl_stratified"
RESULTS_DIR = PROJECT_ROOT / "results" / "acl_linguistic"


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)

    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def prepare_data(args):
    """Phase 0: Prepare stratified data."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_stratified_data.py"),
        "--output", str(DATA_DIR / "stratified_samples.json"),
        "--seed", str(args.seed),
        "--relaxed-filter",  # Use relaxed filter for adequate sample sizes
        "--use-all"          # Use all available samples (balanced)
    ]
    if args.input_data:
        cmd.extend(["--input", args.input_data])

    return run_command(cmd, "Preparing stratified sample data (relaxed filter for power)")


def run_phase1(args):
    """Phase 1: Baseline collection (uses existing results if available)."""
    samples_file = DATA_DIR / "stratified_samples.json"
    if not samples_file.exists():
        print(f"Error: Samples file not found: {samples_file}")
        print("Run --prepare-data first.")
        return False

    # Check if we can use existing results
    existing_results = PROJECT_ROOT / "llm_gen_results" / "toucan"
    if existing_results.exists() and not args.force_new:
        print("\n*** Using existing results to extract baseline (saves API calls) ***")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "extract_baseline_from_existing.py"),
            "--samples", str(samples_file),
            "--results-base", str(PROJECT_ROOT / "llm_gen_results"),
            "--output-dir", str(RESULTS_DIR / "phase1_baseline")
        ]

        if args.model:
            cmd.extend(["--model", args.model])
        elif args.all_models:
            cmd.append("--all-models")

        return run_command(cmd, "Phase 1: Extract baseline from existing results (NO API CALLS)")
    else:
        # Fall back to running new experiments
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "phase1_baseline_collection.py"),
            "--samples", str(samples_file),
            "--num-runs", str(args.num_runs),
            "--output-dir", str(RESULTS_DIR / "phase1_baseline")
        ]

        if args.model:
            cmd.extend(["--model", args.model])
        elif args.all_models:
            cmd.append("--primary-only")

        return run_command(cmd, "Phase 1: Multi-model baseline collection (new API calls)")


def run_phase2(args):
    """Phase 2: Stratified linguistic interventions."""
    samples_file = DATA_DIR / "stratified_samples.json"

    # Find baseline file for the model
    if args.model:
        # Try to find corresponding baseline file
        baseline_dir = RESULTS_DIR / "phase1_baseline"
        baseline_files = list(baseline_dir.glob("*_baseline.json"))

        if not baseline_files:
            print(f"Error: No baseline files found in {baseline_dir}")
            print("Run --phase1 first.")
            return False

        # Match model to baseline file
        baseline_file = None
        for bf in baseline_files:
            # Simple matching by checking if model name is in filename
            model_short = args.model.split("/")[-1].split(":")[0].lower()
            if model_short.replace("-", "_") in bf.name.lower():
                baseline_file = bf
                break

        if not baseline_file:
            print(f"Error: No baseline file found for model {args.model}")
            print(f"Available: {[f.name for f in baseline_files]}")
            return False

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "phase2_linguistic_interventions.py"),
            "--model", args.model,
            "--baseline-file", str(baseline_file),
            "--samples", str(samples_file),
            "--num-runs", str(args.num_runs),
            "--output-dir", str(RESULTS_DIR / "phase2_interventions")
        ]

        return run_command(cmd, f"Phase 2: Stratified interventions for {args.model}")

    else:
        print("Please specify --model for Phase 2")
        return False


def run_phase3(args):
    """Phase 3: Schema × linguistic interaction."""
    samples_file = DATA_DIR / "stratified_samples.json"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "phase3_schema_interaction.py"),
        "--samples", str(samples_file),
        "--num-runs", str(args.num_runs),
        "--output-dir", str(RESULTS_DIR / "phase3_interaction")
    ]

    if args.model:
        cmd.extend(["--model", args.model])
    elif args.all_models:
        cmd.append("--all-models")

    return run_command(cmd, "Phase 3: Schema × linguistic interaction analysis")


def run_analysis(args):
    """Run all analysis scripts."""
    success = True

    # Ceiling effect analysis
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "analyze_ceiling_effect.py"),
        "--results-dir", str(RESULTS_DIR / "phase2_interventions"),
        "--output", str(RESULTS_DIR / "analysis" / "ceiling_effect_analysis.json")
    ]
    success &= run_command(cmd, "Analysis: Ceiling effect")

    # Cross-model transfer analysis
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "analyze_cross_model_transfer.py"),
        "--results-dir", str(RESULTS_DIR / "phase2_interventions"),
        "--output", str(RESULTS_DIR / "analysis" / "cross_model_transfer.json")
    ]
    success &= run_command(cmd, "Analysis: Cross-model transfer (LOMO)")

    return success


def run_all(args):
    """Run full pipeline."""
    print("\n" + "=" * 70)
    print("ACL 2026 EXPERIMENT PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1: Prepare data
    if not prepare_data(args):
        print("FAILED: Data preparation")
        return False

    # Step 2: Phase 1 for all models
    for model in ACL_MODELS:
        args.model = model
        if not run_phase1(args):
            print(f"FAILED: Phase 1 for {model}")
            # Continue with other models

    # Step 3: Phase 2 for all models
    for model in ACL_MODELS:
        args.model = model
        if not run_phase2(args):
            print(f"FAILED: Phase 2 for {model}")

    # Step 4: Phase 3 for all models
    args.all_models = True
    if not run_phase3(args):
        print("FAILED: Phase 3")

    # Step 5: Analysis
    if not run_analysis(args):
        print("FAILED: Analysis")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='ACL 2026 Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare stratified data
    python run_acl_experiments.py --prepare-data

    # Run Phase 1 baseline for specific model
    python run_acl_experiments.py --phase1 --model us.anthropic.claude-sonnet-4-20250514-v1:0

    # Run Phase 2 interventions
    python run_acl_experiments.py --phase2 --model us.anthropic.claude-sonnet-4-20250514-v1:0

    # Run Phase 3 interactions for all models
    python run_acl_experiments.py --phase3 --all-models

    # Run analysis scripts
    python run_acl_experiments.py --analyze

    # Run full pipeline
    python run_acl_experiments.py --all
        """
    )

    # Phase selection
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline')
    parser.add_argument('--prepare-data', action='store_true',
                        help='Prepare stratified sample data')
    parser.add_argument('--phase1', action='store_true',
                        help='Run Phase 1: baseline collection')
    parser.add_argument('--phase2', action='store_true',
                        help='Run Phase 2: stratified interventions')
    parser.add_argument('--phase3', action='store_true',
                        help='Run Phase 3: schema interaction')
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis scripts')

    # Model selection
    parser.add_argument('--model', type=str,
                        help='Specific model to run')
    parser.add_argument('--all-models', action='store_true',
                        help='Run on all ACL models')

    # Data preparation options
    parser.add_argument('--input-data', type=str,
                        default='data/toucan/toucan_tool_calls_1006.json',
                        help='Input dataset for data preparation')
    parser.add_argument('--samples-per-stratum', type=int, default=40,
                        help='Samples per complexity stratum')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Experiment options
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Runs per condition')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for experiments')
    parser.add_argument('--force-new', action='store_true',
                        help='Force new API calls even if existing results available')

    args = parser.parse_args()

    # Execute requested phase(s)
    if args.all:
        run_all(args)
    elif args.prepare_data:
        prepare_data(args)
    elif args.phase1:
        run_phase1(args)
    elif args.phase2:
        run_phase2(args)
    elif args.phase3:
        run_phase3(args)
    elif args.analyze:
        run_analysis(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
