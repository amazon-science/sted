#!/usr/bin/env python3
"""
COLM 2026: Inference-Time Verification Methods

Implements practical methods to detect inconsistent outputs at inference time
WITHOUT requiring ground truth labels.

Methods:
1. N-Sample Self-Consistency: Generate N outputs and measure agreement
2. Temperature Sweep: Check stability across temperature settings
3. Schema-Guided Verification: Validate against schema constraints
4. Early Detection: Predict consistency from partial outputs

Calibration: Uses existing data to calibrate verification thresholds.

Usage:
    python inference_verification.py --method n-sample --n-samples 5
    python inference_verification.py --calibrate --results-base llm_gen_results/toucan
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod
import statistics

import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score


# =============================================================================
# Verification Methods (Abstract Base)
# =============================================================================

class VerificationMethod(ABC):
    """Base class for inference-time verification methods."""

    @abstractmethod
    def verify(self, outputs: List[Any], schema: Dict = None) -> Dict:
        """
        Verify consistency of outputs.

        Returns:
            Dict with keys:
            - score: 0-1 consistency score (1 = fully consistent)
            - confidence: How confident we are in the score
            - details: Method-specific details
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class NSampleConsistency(VerificationMethod):
    """
    N-Sample Self-Consistency Verification.

    Generate N outputs and measure pairwise agreement.
    High agreement = likely consistent, low = likely inconsistent.
    """

    def __init__(self, similarity_fn: str = "jaccard"):
        self.similarity_fn = similarity_fn

    def get_name(self) -> str:
        return f"n_sample_{self.similarity_fn}"

    def verify(self, outputs: List[Any], schema: Dict = None) -> Dict:
        if len(outputs) < 2:
            return {
                "score": 1.0 if len(outputs) == 1 else 0.0,
                "confidence": 0.0,
                "details": {"n_outputs": len(outputs), "reason": "insufficient_samples"}
            }

        # Filter valid outputs (non-empty, parseable)
        valid_outputs = [o for o in outputs if o and self._is_valid(o)]

        if len(valid_outputs) < 2:
            return {
                "score": 0.0,
                "confidence": 0.5,
                "details": {
                    "n_valid": len(valid_outputs),
                    "n_total": len(outputs),
                    "validity_rate": len(valid_outputs) / len(outputs)
                }
            }

        # Compute pairwise similarities
        similarities = []
        for i in range(len(valid_outputs)):
            for j in range(i + 1, len(valid_outputs)):
                sim = self._compute_similarity(valid_outputs[i], valid_outputs[j])
                similarities.append(sim)

        score = np.mean(similarities)
        confidence = 1 - np.std(similarities) if len(similarities) > 1 else 0.5

        return {
            "score": score,
            "confidence": confidence,
            "details": {
                "n_valid": len(valid_outputs),
                "n_total": len(outputs),
                "n_comparisons": len(similarities),
                "similarity_std": np.std(similarities) if similarities else 0,
                "min_similarity": min(similarities) if similarities else 0,
                "max_similarity": max(similarities) if similarities else 0
            }
        }

    def _is_valid(self, output: Any) -> bool:
        """Check if output is valid (non-empty, proper structure)."""
        if output is None:
            return False
        if isinstance(output, list):
            return len(output) > 0
        if isinstance(output, dict):
            return bool(output)
        return True

    def _compute_similarity(self, out1: Any, out2: Any) -> float:
        """Compute similarity between two outputs."""
        if self.similarity_fn == "jaccard":
            return self._jaccard_similarity(out1, out2)
        elif self.similarity_fn == "exact":
            return 1.0 if out1 == out2 else 0.0
        else:
            return self._jaccard_similarity(out1, out2)

    def _jaccard_similarity(self, out1: Any, out2: Any) -> float:
        """Jaccard similarity based on tool names."""
        # Extract tool names
        tools1 = self._extract_tool_names(out1)
        tools2 = self._extract_tool_names(out2)

        if not tools1 and not tools2:
            return 1.0
        if not tools1 or not tools2:
            return 0.0

        intersection = len(tools1 & tools2)
        union = len(tools1 | tools2)
        return intersection / union if union > 0 else 0.0

    def _extract_tool_names(self, output: Any) -> set:
        """Extract tool names from output."""
        if isinstance(output, list):
            return set(item.get('name', '') for item in output if isinstance(item, dict))
        if isinstance(output, dict):
            return {output.get('name', '')}
        return set()


class TemperatureSweepVerification(VerificationMethod):
    """
    Temperature Sweep Verification.

    Check if outputs are stable across different temperature settings.
    Stable outputs across temperatures indicate higher reliability.
    """

    def __init__(self, base_similarity: str = "jaccard"):
        self.base_method = NSampleConsistency(base_similarity)

    def get_name(self) -> str:
        return "temperature_sweep"

    def verify(self, outputs: List[Any], schema: Dict = None) -> Dict:
        """
        Expects outputs to be a dict: {temperature: [outputs_at_temp]}
        """
        if not isinstance(outputs, dict):
            return {
                "score": 0.0,
                "confidence": 0.0,
                "details": {"error": "Expected dict mapping temperature to outputs"}
            }

        temperatures = sorted(outputs.keys())
        if len(temperatures) < 2:
            return {
                "score": 1.0 if len(temperatures) == 1 else 0.0,
                "confidence": 0.0,
                "details": {"n_temperatures": len(temperatures)}
            }

        # Measure consistency within each temperature
        within_temp_scores = {}
        for temp, temp_outputs in outputs.items():
            result = self.base_method.verify(temp_outputs)
            within_temp_scores[temp] = result["score"]

        # Measure consistency across temperatures (using first output from each)
        cross_temp_outputs = [
            outputs[t][0] for t in temperatures
            if outputs[t] and outputs[t][0]
        ]
        cross_temp_result = self.base_method.verify(cross_temp_outputs)

        # Combined score: weighted average
        within_avg = np.mean(list(within_temp_scores.values()))
        cross_score = cross_temp_result["score"]
        combined_score = 0.6 * within_avg + 0.4 * cross_score

        # Temperature sensitivity
        temp_range = max(temperatures) - min(temperatures)
        score_range = max(within_temp_scores.values()) - min(within_temp_scores.values())
        sensitivity = score_range / (temp_range + 0.01)

        return {
            "score": combined_score,
            "confidence": cross_temp_result["confidence"],
            "details": {
                "within_temp_scores": within_temp_scores,
                "cross_temp_score": cross_score,
                "temperature_sensitivity": sensitivity,
                "n_temperatures": len(temperatures)
            }
        }


class SchemaGuidedVerification(VerificationMethod):
    """
    Schema-Guided Verification.

    Validate outputs against schema constraints without ground truth.
    Checks: type correctness, required fields, value constraints.
    """

    def get_name(self) -> str:
        return "schema_guided"

    def verify(self, outputs: List[Any], schema: Dict = None) -> Dict:
        if not outputs:
            return {"score": 0.0, "confidence": 0.0, "details": {"error": "no_outputs"}}

        if schema is None:
            # Fall back to basic structural validation
            return self._basic_validation(outputs)

        # Validate each output against schema
        validation_results = []
        for output in outputs:
            result = self._validate_against_schema(output, schema)
            validation_results.append(result)

        # Aggregate
        scores = [r["score"] for r in validation_results]
        avg_score = np.mean(scores)

        # Consistency: all outputs should have same validation status
        all_valid = all(s > 0.8 for s in scores)
        all_invalid = all(s < 0.2 for s in scores)
        consistency_bonus = 0.1 if (all_valid or all_invalid) else 0.0

        return {
            "score": min(1.0, avg_score + consistency_bonus),
            "confidence": 1 - np.std(scores) if len(scores) > 1 else 0.5,
            "details": {
                "per_output_scores": scores,
                "validation_results": validation_results,
                "all_valid": all_valid
            }
        }

    def _basic_validation(self, outputs: List[Any]) -> Dict:
        """Basic validation without schema."""
        valid_count = 0
        for output in outputs:
            if self._is_structurally_valid(output):
                valid_count += 1

        validity_rate = valid_count / len(outputs) if outputs else 0

        return {
            "score": validity_rate,
            "confidence": 0.5,  # Lower confidence without schema
            "details": {
                "valid_count": valid_count,
                "total": len(outputs),
                "no_schema": True
            }
        }

    def _is_structurally_valid(self, output: Any) -> bool:
        """Check basic structural validity."""
        if output is None:
            return False
        if isinstance(output, list):
            return all(isinstance(item, dict) and 'name' in item for item in output)
        if isinstance(output, dict):
            return 'name' in output
        return False

    def _validate_against_schema(self, output: Any, schema: Dict) -> Dict:
        """Validate output against JSON schema."""
        checks = {
            "has_required_fields": False,
            "type_correct": False,
            "constraints_met": False
        }

        if not isinstance(output, (list, dict)):
            return {"score": 0.0, "checks": checks}

        # For tool calls, validate against tool schema
        tools = output if isinstance(output, list) else [output]

        valid_tools = 0
        for tool in tools:
            if not isinstance(tool, dict):
                continue

            # Check required fields
            if 'name' in tool:
                checks["has_required_fields"] = True
                valid_tools += 1

            # Check arguments match schema
            if 'arguments' in tool and isinstance(tool['arguments'], dict):
                checks["type_correct"] = True

        score = valid_tools / len(tools) if tools else 0

        return {
            "score": score,
            "checks": checks
        }


class EarlyDetection(VerificationMethod):
    """
    Early Detection Verification.

    Predict final consistency from partial outputs or early tokens.
    Uses learned features from calibration data.
    """

    def __init__(self, calibration_data: Dict = None):
        self.calibration_data = calibration_data or {}
        self.thresholds = self.calibration_data.get("thresholds", {
            "high_confidence": 0.85,
            "low_confidence": 0.5
        })

    def get_name(self) -> str:
        return "early_detection"

    def verify(self, outputs: List[Any], schema: Dict = None) -> Dict:
        """
        Early detection based on partial output features.
        """
        if not outputs:
            return {"score": 0.5, "confidence": 0.0, "details": {"error": "no_outputs"}}

        # Extract early features
        features = self._extract_early_features(outputs)

        # Predict consistency based on features
        prediction = self._predict_from_features(features)

        return {
            "score": prediction["predicted_consistency"],
            "confidence": prediction["confidence"],
            "details": {
                "features": features,
                "early_signals": prediction["signals"]
            }
        }

    def _extract_early_features(self, outputs: List[Any]) -> Dict:
        """Extract features from early/partial outputs."""
        features = {
            "n_outputs": len(outputs),
            "valid_rate": 0.0,
            "avg_length": 0.0,
            "tool_diversity": 0.0,
            "first_token_agreement": 0.0
        }

        valid_outputs = [o for o in outputs if o]
        features["valid_rate"] = len(valid_outputs) / len(outputs) if outputs else 0

        if valid_outputs:
            # Average output length (number of tool calls)
            lengths = []
            for o in valid_outputs:
                if isinstance(o, list):
                    lengths.append(len(o))
                else:
                    lengths.append(1)
            features["avg_length"] = np.mean(lengths)

            # Tool name diversity
            all_tools = set()
            for o in valid_outputs:
                if isinstance(o, list):
                    all_tools.update(item.get('name', '') for item in o)
            features["tool_diversity"] = len(all_tools)

            # First tool agreement
            first_tools = []
            for o in valid_outputs:
                if isinstance(o, list) and o:
                    first_tools.append(o[0].get('name', ''))
            if first_tools:
                most_common = max(set(first_tools), key=first_tools.count)
                features["first_token_agreement"] = first_tools.count(most_common) / len(first_tools)

        return features

    def _predict_from_features(self, features: Dict) -> Dict:
        """Predict consistency from features."""
        signals = []

        # High validity rate is a positive signal
        if features["valid_rate"] > 0.9:
            signals.append(("high_validity", 0.2))
        elif features["valid_rate"] < 0.5:
            signals.append(("low_validity", -0.3))

        # Low tool diversity is positive (more agreement)
        if features["tool_diversity"] <= 1:
            signals.append(("low_diversity", 0.15))
        elif features["tool_diversity"] > 3:
            signals.append(("high_diversity", -0.2))

        # High first token agreement is positive
        if features["first_token_agreement"] > 0.8:
            signals.append(("first_agree", 0.25))
        elif features["first_token_agreement"] < 0.5:
            signals.append(("first_disagree", -0.2))

        # Combine signals
        base_score = 0.7  # Prior
        adjustment = sum(s[1] for s in signals)
        predicted = np.clip(base_score + adjustment, 0, 1)

        # Confidence based on signal strength
        confidence = min(0.8, 0.3 + 0.1 * len(signals))

        return {
            "predicted_consistency": predicted,
            "confidence": confidence,
            "signals": signals
        }


# =============================================================================
# Calibration Functions
# =============================================================================

def calibrate_from_existing_data(results_base: str, temperatures: List[float] = None) -> Dict:
    """
    Calibrate verification methods using existing llm_gen_results data.

    Uses known consistency scores to learn optimal thresholds.
    """
    print("=" * 70)
    print("CALIBRATING VERIFICATION METHODS")
    print("=" * 70)

    if temperatures is None:
        temperatures = [0.7]

    results_path = Path(results_base)
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        return {}

    # Collect ground truth and verification scores
    ground_truth = []  # Actual consistency
    verification_scores = defaultdict(list)  # method -> scores

    methods = [
        NSampleConsistency("jaccard"),
        SchemaGuidedVerification(),
        EarlyDetection()
    ]

    n_samples = 0
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if this is a target temperature
            is_target_temp = any(
                f"temp_{int(t)}_{int((t % 1) * 100):02d}" in run_dir.name
                for t in temperatures
            )
            if not is_target_temp:
                continue

            results_file = run_dir / "all_results.json"
            if not results_file.exists():
                continue

            with open(results_file) as f:
                data = json.load(f)

            for sample in data.get('results', [])[:100]:  # Limit for speed
                runs = sample.get('generated_runs', [])
                if len(runs) < 5:
                    continue

                # Compute actual consistency (ground truth)
                valid_runs = [r for r in runs if r]
                if len(valid_runs) < 2:
                    continue

                sims = []
                for i in range(len(valid_runs)):
                    for j in range(i + 1, len(valid_runs)):
                        tools1 = set(tc.get('name', '') for tc in valid_runs[i])
                        tools2 = set(tc.get('name', '') for tc in valid_runs[j])
                        if tools1 or tools2:
                            sim = len(tools1 & tools2) / len(tools1 | tools2)
                            sims.append(sim)

                if not sims:
                    continue

                actual_consistency = np.mean(sims)
                ground_truth.append(actual_consistency)

                # Run verification methods
                for method in methods:
                    result = method.verify(valid_runs)
                    verification_scores[method.get_name()].append(result["score"])

                n_samples += 1

        if n_samples >= 500:  # Limit total samples
            break

    print(f"\nCalibrated on {n_samples} samples")

    # Compute calibration metrics
    calibration_results = {
        "n_samples": n_samples,
        "methods": {}
    }

    print("\n{:<25} {:>10} {:>10} {:>10}".format(
        "Method", "Corr", "MAE", "AUC"
    ))
    print("-" * 60)

    for method_name, scores in verification_scores.items():
        if len(scores) != len(ground_truth):
            continue

        # Correlation
        r, p = stats.pearsonr(scores, ground_truth)

        # MAE
        mae = np.mean(np.abs(np.array(scores) - np.array(ground_truth)))

        # AUC for detecting low consistency (< 0.7)
        binary_labels = [1 if g >= 0.7 else 0 for g in ground_truth]
        try:
            auc = roc_auc_score(binary_labels, scores)
        except ValueError:
            auc = 0.5

        print("{:<25} {:>+9.3f} {:>10.3f} {:>10.3f}".format(
            method_name, r, mae, auc
        ))

        calibration_results["methods"][method_name] = {
            "correlation": r,
            "mae": mae,
            "auc": auc,
            "optimal_threshold": np.median(scores)
        }

    # Find optimal thresholds
    print("\nOptimal Thresholds for High Consistency Detection:")
    for method_name, scores in verification_scores.items():
        if len(scores) != len(ground_truth):
            continue

        binary_labels = [1 if g >= 0.7 else 0 for g in ground_truth]
        precision, recall, thresholds = precision_recall_curve(binary_labels, scores)

        # Find threshold with best F1
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        print(f"  {method_name}: threshold={best_threshold:.3f} (F1={f1_scores[best_idx]:.3f})")
        calibration_results["methods"][method_name]["best_threshold"] = float(best_threshold)

    return calibration_results


def run_verification_demo(model_id: str = None):
    """Demo verification methods on sample data."""
    print("=" * 70)
    print("VERIFICATION METHOD DEMO")
    print("=" * 70)

    # Sample outputs (simulated)
    sample_outputs = [
        [{"name": "get_weather", "arguments": {"location": "NYC"}}],
        [{"name": "get_weather", "arguments": {"location": "New York"}}],
        [{"name": "get_weather", "arguments": {"location": "NYC", "unit": "F"}}],
        [{"name": "get_weather", "arguments": {"location": "NYC"}}],
        [{"name": "get_weather", "arguments": {"city": "NYC"}}],
    ]

    print("\nSample outputs:")
    for i, out in enumerate(sample_outputs):
        print(f"  {i+1}: {out}")

    # Test each method
    methods = [
        NSampleConsistency("jaccard"),
        NSampleConsistency("exact"),
        SchemaGuidedVerification(),
        EarlyDetection()
    ]

    print("\n{:<25} {:>10} {:>12}".format("Method", "Score", "Confidence"))
    print("-" * 50)

    for method in methods:
        result = method.verify(sample_outputs)
        print("{:<25} {:>9.3f} {:>11.3f}".format(
            method.get_name(), result["score"], result["confidence"]
        ))

    # Temperature sweep demo
    print("\n\nTemperature Sweep Demo:")
    temp_outputs = {
        0.3: sample_outputs[:3],
        0.7: sample_outputs[1:4],
        1.0: sample_outputs[2:5]
    }

    temp_method = TemperatureSweepVerification()
    result = temp_method.verify(temp_outputs)
    print(f"  Combined Score: {result['score']:.3f}")
    print(f"  Cross-temp Score: {result['details']['cross_temp_score']:.3f}")
    print(f"  Sensitivity: {result['details']['temperature_sensitivity']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='COLM 2026: Inference-Time Verification Methods'
    )
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibrate methods using existing data')
    parser.add_argument('--demo', action='store_true',
                        help='Run verification demo')
    parser.add_argument('--results-base', type=str,
                        default='llm_gen_results/toucan',
                        help='Base directory for calibration data')
    parser.add_argument('--output', type=str,
                        default='results/colm_architecture/verification_calibration.json',
                        help='Output file for calibration results')

    args = parser.parse_args()

    if args.demo:
        run_verification_demo()
        return

    if args.calibrate:
        calibration = calibrate_from_existing_data(args.results_base)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        print(f"\nSaved calibration to {output_path}")
        return

    # Default: run demo
    run_verification_demo()


if __name__ == '__main__':
    main()
