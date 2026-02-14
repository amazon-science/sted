#!/usr/bin/env python3
"""
COLM 2026: Master Experiment Runner

Orchestrates all COLM architecture analysis experiments:
1. Architecture Analysis: Size scaling, MoE vs Dense, family patterns
2. Inference Verification: Calibration and evaluation
3. Theoretical Bounds: Estimation and validation

Deadline: March 31, 2026 (Abstract: March 26, 2026)

Usage:
    # Run all analyses on existing data (NO API CALLS)
    python run_colm_experiments.py --analyze-existing

    # Run architecture analysis only
    python run_colm_experiments.py --architecture

    # Calibrate verification methods
    python run_colm_experiments.py --calibrate-verification

    # Run everything
    python run_colm_experiments.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPT_DIR = Path(__file__).parent

# Existing data location
EXISTING_RESULTS = PROJECT_ROOT / "llm_gen_results" / "toucan"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results" / "colm_architecture"

# Models of interest for COLM (architecture diversity)
COLM_MODELS = {
    "dense_large": [
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "openai/gpt-4.1",
        "us.meta.llama3-3-70b-instruct-v1:0",
    ],
    "dense_small": [
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "openai/gpt-4.1-nano",
        "us.meta.llama3-2-3b-instruct-v1:0",
    ],
    "moe": [
        "qwen.qwen3-235b-a22b-2507-v1:0",
        "google/gemini-2.5-flash-lite",
    ],
}


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


def run_architecture_analysis(args) -> bool:
    """Run architecture analysis on existing data."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "architecture_analysis.py"),
        "--results-base", str(EXISTING_RESULTS),
        "--output", str(RESULTS_DIR / "architecture_analysis.json"),
        "--analyze-all"
    ]

    if args.temperatures:
        cmd.extend(["--temperatures"] + [str(t) for t in args.temperatures])

    return run_command(cmd, "Architecture Analysis (Size, MoE vs Dense, Families)")


def run_verification_calibration(args) -> bool:
    """Calibrate verification methods using existing data."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "inference_verification.py"),
        "--calibrate",
        "--results-base", str(EXISTING_RESULTS),
        "--output", str(RESULTS_DIR / "verification_calibration.json")
    ]

    return run_command(cmd, "Verification Method Calibration")


def run_verification_demo(args) -> bool:
    """Run verification demo."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "inference_verification.py"),
        "--demo"
    ]

    return run_command(cmd, "Verification Method Demo")


def generate_summary_report(args) -> bool:
    """Generate summary report from all analyses."""
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)

    report = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "venue": "COLM 2026",
            "deadline": "March 31, 2026"
        },
        "analyses": {}
    }

    # Load architecture analysis
    arch_file = RESULTS_DIR / "architecture_analysis.json"
    if arch_file.exists():
        with open(arch_file) as f:
            arch_data = json.load(f)
        report["analyses"]["architecture"] = {
            "n_models": arch_data.get("metadata", {}).get("n_models", 0),
            "key_findings": []
        }

        # Extract key findings
        if "size_scaling" in arch_data:
            corr = arch_data["size_scaling"].get("correlation", {})
            if corr:
                report["analyses"]["architecture"]["key_findings"].append(
                    f"Size-Consistency correlation: r={corr.get('r', 0):.3f}"
                )

        if "moe_vs_dense" in arch_data:
            comparison = arch_data["moe_vs_dense"].get("comparison", {})
            if comparison:
                report["analyses"]["architecture"]["key_findings"].append(
                    f"MoE vs Dense difference: p={comparison.get('p_value', 1):.4f}"
                )

    # Load verification calibration
    verify_file = RESULTS_DIR / "verification_calibration.json"
    if verify_file.exists():
        with open(verify_file) as f:
            verify_data = json.load(f)
        report["analyses"]["verification"] = {
            "n_calibration_samples": verify_data.get("n_samples", 0),
            "methods": {}
        }

        for method_name, method_data in verify_data.get("methods", {}).items():
            report["analyses"]["verification"]["methods"][method_name] = {
                "correlation": method_data.get("correlation", 0),
                "auc": method_data.get("auc", 0.5)
            }

    # Save report
    report_path = RESULTS_DIR / "colm_summary_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved summary report to {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("COLM 2026 EXPERIMENT SUMMARY")
    print("=" * 70)

    if "architecture" in report["analyses"]:
        print(f"\nArchitecture Analysis: {report['analyses']['architecture']['n_models']} models")
        for finding in report['analyses']['architecture']['key_findings']:
            print(f"  - {finding}")

    if "verification" in report["analyses"]:
        print(f"\nVerification Calibration: {report['analyses']['verification']['n_calibration_samples']} samples")
        for method, data in report['analyses']['verification']['methods'].items():
            print(f"  - {method}: corr={data['correlation']:.3f}, AUC={data['auc']:.3f}")

    return True


def run_all(args) -> bool:
    """Run all COLM experiments."""
    print("\n" + "=" * 70)
    print("COLM 2026 FULL EXPERIMENT PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    success = True

    # Check for existing data
    if not EXISTING_RESULTS.exists():
        print(f"\nWARNING: Existing results not found at {EXISTING_RESULTS}")
        print("Architecture analysis requires llm_gen_results data.")
        print("Please ensure data is available before running.")
        return False

    # Step 1: Architecture Analysis
    if not run_architecture_analysis(args):
        print("WARNING: Architecture analysis had issues")
        success = False

    # Step 2: Verification Calibration
    if not run_verification_calibration(args):
        print("WARNING: Verification calibration had issues")
        success = False

    # Step 3: Generate Summary Report
    if not generate_summary_report(args):
        print("WARNING: Summary report generation had issues")
        success = False

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 70)

    return success


def main():
    parser = argparse.ArgumentParser(
        description='COLM 2026 Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all analyses on existing data (NO API CALLS)
    python run_colm_experiments.py --all

    # Architecture analysis only
    python run_colm_experiments.py --architecture

    # Calibrate verification methods
    python run_colm_experiments.py --calibrate-verification

    # Generate summary report
    python run_colm_experiments.py --report
        """
    )

    # Phase selection
    parser.add_argument('--all', action='store_true',
                        help='Run full experiment pipeline')
    parser.add_argument('--architecture', action='store_true',
                        help='Run architecture analysis')
    parser.add_argument('--calibrate-verification', action='store_true',
                        help='Calibrate verification methods')
    parser.add_argument('--verification-demo', action='store_true',
                        help='Run verification demo')
    parser.add_argument('--report', action='store_true',
                        help='Generate summary report')

    # Options
    parser.add_argument('--temperatures', type=float, nargs='+',
                        default=[0.3, 0.7, 1.0],
                        help='Temperatures to analyze')

    args = parser.parse_args()

    # Execute requested phase(s)
    if args.all:
        run_all(args)
    elif args.architecture:
        run_architecture_analysis(args)
    elif args.calibrate_verification:
        run_verification_calibration(args)
    elif args.verification_demo:
        run_verification_demo(args)
    elif args.report:
        generate_summary_report(args)
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("COLM 2026 KEY INFORMATION")
        print("=" * 70)
        print("Deadline: March 31, 2026 (Abstract: March 26, 2026)")
        print("\nDifferentiation from KDD:")
        print("  1. Architecture-focused analysis (MoE vs Dense, size scaling)")
        print("  2. Inference-time verification methods")
        print("  3. Theoretical consistency bounds")
        print("\nData: Uses existing llm_gen_results (~2.1M outputs, 20 models)")
        print("      NO NEW API CALLS NEEDED for analysis phase")


if __name__ == '__main__':
    main()
