#!/usr/bin/env python3
"""
COLM 2026: Architecture-Consistency Analysis

Analyzes how model architecture affects structured output consistency.
Uses existing llm_gen_results data - NO API CALLS NEEDED.

Key analyses:
1. Size scaling: Does larger always mean more consistent?
2. Architecture type: MoE vs Dense patterns
3. Model family signatures: Characteristic consistency patterns
4. Temperature sensitivity by architecture

Usage:
    python architecture_analysis.py --results-base llm_gen_results/toucan
    python architecture_analysis.py --analyze-all
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import statistics

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Model Architecture Metadata
# =============================================================================

MODEL_METADATA = {
    # Claude family
    "claude-sonnet-4": {
        "family": "Claude",
        "architecture": "dense",
        "size_category": "large",
        "estimated_params": 100,  # billions
        "active_params": 100,
        "provider": "Anthropic"
    },
    "claude-3.5-sonnet": {
        "family": "Claude",
        "architecture": "dense",
        "size_category": "large",
        "estimated_params": 70,
        "active_params": 70,
        "provider": "Anthropic"
    },
    "claude-3.5-haiku": {
        "family": "Claude",
        "architecture": "dense",
        "size_category": "small",
        "estimated_params": 20,
        "active_params": 20,
        "provider": "Anthropic"
    },

    # GPT family
    "gpt-4.1": {
        "family": "GPT",
        "architecture": "dense",
        "size_category": "large",
        "estimated_params": 200,
        "active_params": 200,
        "provider": "OpenAI"
    },
    "gpt-4.1-mini": {
        "family": "GPT",
        "architecture": "dense",
        "size_category": "medium",
        "estimated_params": 70,
        "active_params": 70,
        "provider": "OpenAI"
    },
    "gpt-4.1-nano": {
        "family": "GPT",
        "architecture": "dense",
        "size_category": "small",
        "estimated_params": 20,
        "active_params": 20,
        "provider": "OpenAI"
    },

    # Llama family
    "llama-3.3-70b": {
        "family": "Llama",
        "architecture": "dense",
        "size_category": "large",
        "estimated_params": 70,
        "active_params": 70,
        "provider": "Meta"
    },
    "llama-3.2-3b": {
        "family": "Llama",
        "architecture": "dense",
        "size_category": "tiny",
        "estimated_params": 3,
        "active_params": 3,
        "provider": "Meta"
    },

    # Qwen family (MoE)
    "qwen3-235b-a22b": {
        "family": "Qwen",
        "architecture": "moe",
        "size_category": "xlarge",
        "estimated_params": 235,
        "active_params": 22,
        "provider": "Alibaba"
    },
    "qwen3-32b": {
        "family": "Qwen",
        "architecture": "dense",
        "size_category": "medium",
        "estimated_params": 32,
        "active_params": 32,
        "provider": "Alibaba"
    },

    # Gemini family (MoE)
    "gemini-2.5-flash": {
        "family": "Gemini",
        "architecture": "moe",
        "size_category": "medium",
        "estimated_params": 100,
        "active_params": 30,
        "provider": "Google"
    },
    "gemini-2.5-flash-lite": {
        "family": "Gemini",
        "architecture": "moe",
        "size_category": "small",
        "estimated_params": 50,
        "active_params": 15,
        "provider": "Google"
    },

    # Nova family
    "nova2-lite": {
        "family": "Nova",
        "architecture": "dense",
        "size_category": "small",
        "estimated_params": 20,
        "active_params": 20,
        "provider": "Amazon"
    },
    "nova2-micro": {
        "family": "Nova",
        "architecture": "dense",
        "size_category": "tiny",
        "estimated_params": 8,
        "active_params": 8,
        "provider": "Amazon"
    },
    "nova2-pro": {
        "family": "Nova",
        "architecture": "dense",
        "size_category": "large",
        "estimated_params": 70,
        "active_params": 70,
        "provider": "Amazon"
    },

    # Mistral family
    "mistral-large": {
        "family": "Mistral",
        "architecture": "dense",
        "size_category": "large",
        "estimated_params": 123,
        "active_params": 123,
        "provider": "Mistral"
    },
}


def match_model_to_metadata(model_dir_name: str) -> Tuple[str, Dict]:
    """Match a model directory name to metadata."""
    name_lower = model_dir_name.lower()

    for key, meta in MODEL_METADATA.items():
        if key.replace("-", "") in name_lower.replace("-", "").replace("_", ""):
            return key, meta

    # Fallback: infer from name
    architecture = "moe" if "moe" in name_lower else "dense"
    family = "Unknown"
    for fam in ["claude", "gpt", "llama", "qwen", "gemini", "nova", "mistral"]:
        if fam in name_lower:
            family = fam.capitalize()
            break

    return model_dir_name, {
        "family": family,
        "architecture": architecture,
        "size_category": "unknown",
        "estimated_params": 0,
        "active_params": 0,
        "provider": "Unknown"
    }


def load_model_results(results_dir: Path, temperatures: List[float] = None) -> Dict:
    """Load results for a single model across temperatures."""
    if temperatures is None:
        temperatures = [0.3, 0.7, 1.0]

    results = {"temperatures": {}, "samples": {}}

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Parse temperature from directory name
        dir_name = run_dir.name
        temp = None
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            temp_str = f"temp_{int(t)}_{int((t % 1) * 100):02d}"
            if temp_str in dir_name:
                temp = t
                break

        if temp is None or temp not in temperatures:
            continue

        results_file = run_dir / "all_results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        # Compute consistency metrics
        consistency_scores = []
        for sample in data.get('results', []):
            runs = sample.get('generated_runs', [])
            valid_runs = [r for r in runs if r]

            if len(valid_runs) >= 2:
                # Pairwise tool set similarity
                sims = []
                for i in range(len(valid_runs)):
                    for j in range(i + 1, len(valid_runs)):
                        tools1 = set(tc.get('name', '') for tc in valid_runs[i])
                        tools2 = set(tc.get('name', '') for tc in valid_runs[j])
                        if tools1 or tools2:
                            sim = len(tools1 & tools2) / len(tools1 | tools2)
                            sims.append(sim)

                if sims:
                    consistency_scores.append(np.mean(sims))

                    sample_id = sample.get('sample_id')
                    if sample_id:
                        if sample_id not in results["samples"]:
                            results["samples"][sample_id] = {}
                        results["samples"][sample_id][temp] = np.mean(sims)

        if consistency_scores:
            results["temperatures"][temp] = {
                "mean": np.mean(consistency_scores),
                "std": np.std(consistency_scores),
                "median": np.median(consistency_scores),
                "n": len(consistency_scores)
            }

    return results


def analyze_size_scaling(model_results: Dict[str, Dict]) -> Dict:
    """Analyze: Does larger always mean more consistent?"""
    print("\n" + "=" * 70)
    print("SIZE SCALING ANALYSIS")
    print("=" * 70)

    # Group by size category
    by_size = defaultdict(list)
    size_order = ["tiny", "small", "medium", "large", "xlarge"]

    for model_name, data in model_results.items():
        meta = data.get("metadata", {})
        size = meta.get("size_category", "unknown")
        active_params = meta.get("active_params", 0)

        # Get T=0.7 consistency
        temp_data = data.get("results", {}).get("temperatures", {})
        if 0.7 in temp_data:
            c_mean = temp_data[0.7]["mean"]
            by_size[size].append({
                "model": model_name,
                "consistency": c_mean,
                "active_params": active_params
            })

    print("\n{:<10} {:>12} {:>12} {:>15}".format(
        "Size", "Mean C", "Std C", "Models"
    ))
    print("-" * 55)

    size_summary = {}
    for size in size_order:
        if size not in by_size:
            continue
        models = by_size[size]
        consistencies = [m["consistency"] for m in models]

        print("{:<10} {:>11.3f} {:>11.3f} {:>15}".format(
            size, np.mean(consistencies), np.std(consistencies), len(models)
        ))

        size_summary[size] = {
            "mean": np.mean(consistencies),
            "std": np.std(consistencies),
            "n": len(models),
            "models": [m["model"] for m in models]
        }

    # Correlation: active params vs consistency
    all_params = []
    all_consistency = []
    for models in by_size.values():
        for m in models:
            if m["active_params"] > 0:
                all_params.append(m["active_params"])
                all_consistency.append(m["consistency"])

    if len(all_params) >= 3:
        r, p = stats.pearsonr(all_params, all_consistency)
        print(f"\nActive Params vs Consistency: r={r:+.3f}, p={p:.4f}")
        size_summary["correlation"] = {"r": r, "p": p}

    return size_summary


def analyze_moe_vs_dense(model_results: Dict[str, Dict]) -> Dict:
    """Analyze: MoE vs Dense architecture patterns."""
    print("\n" + "=" * 70)
    print("MOE vs DENSE ARCHITECTURE ANALYSIS")
    print("=" * 70)

    moe_models = []
    dense_models = []

    for model_name, data in model_results.items():
        meta = data.get("metadata", {})
        arch = meta.get("architecture", "dense")

        temp_data = data.get("results", {}).get("temperatures", {})
        if not temp_data:
            continue

        model_info = {
            "model": model_name,
            "family": meta.get("family", "Unknown"),
            "active_params": meta.get("active_params", 0),
            "temperatures": {}
        }

        for temp, temp_stats in temp_data.items():
            model_info["temperatures"][temp] = temp_stats["mean"]

        if arch == "moe":
            moe_models.append(model_info)
        else:
            dense_models.append(model_info)

    print(f"\nMoE models: {len(moe_models)}")
    print(f"Dense models: {len(dense_models)}")

    # Compare at T=0.7
    print("\n{:<15} {:>12} {:>12} {:>12}".format(
        "Architecture", "Mean C", "Std C", "n"
    ))
    print("-" * 55)

    results = {}

    if moe_models:
        moe_c = [m["temperatures"].get(0.7, 0) for m in moe_models if 0.7 in m["temperatures"]]
        if moe_c:
            print("{:<15} {:>11.3f} {:>11.3f} {:>12}".format(
                "MoE", np.mean(moe_c), np.std(moe_c), len(moe_c)
            ))
            results["moe"] = {
                "mean": np.mean(moe_c),
                "std": np.std(moe_c),
                "n": len(moe_c),
                "models": [m["model"] for m in moe_models]
            }

    if dense_models:
        dense_c = [m["temperatures"].get(0.7, 0) for m in dense_models if 0.7 in m["temperatures"]]
        if dense_c:
            print("{:<15} {:>11.3f} {:>11.3f} {:>12}".format(
                "Dense", np.mean(dense_c), np.std(dense_c), len(dense_c)
            ))
            results["dense"] = {
                "mean": np.mean(dense_c),
                "std": np.std(dense_c),
                "n": len(dense_c),
                "models": [m["model"] for m in dense_models]
            }

    # Statistical test
    if len(moe_c) >= 2 and len(dense_c) >= 2:
        t_stat, p_value = stats.ttest_ind(moe_c, dense_c)
        print(f"\nt-test: t={t_stat:+.3f}, p={p_value:.4f}")
        results["comparison"] = {"t_stat": t_stat, "p_value": p_value}

    # Temperature sensitivity comparison
    print("\nTemperature Sensitivity:")
    print("{:<15} {:>12} {:>12}".format("Architecture", "T=0.3 - T=1.0", "Sensitivity"))
    print("-" * 45)

    for arch_name, models in [("MoE", moe_models), ("Dense", dense_models)]:
        sensitivities = []
        for m in models:
            if 0.3 in m["temperatures"] and 1.0 in m["temperatures"]:
                sens = m["temperatures"][0.3] - m["temperatures"][1.0]
                sensitivities.append(sens)

        if sensitivities:
            print("{:<15} {:>+11.3f} {:>12}".format(
                arch_name, np.mean(sensitivities),
                "High" if np.mean(sensitivities) > 0.1 else "Low"
            ))
            results[f"{arch_name.lower()}_sensitivity"] = np.mean(sensitivities)

    return results


def analyze_model_families(model_results: Dict[str, Dict]) -> Dict:
    """Analyze: Model family characteristic patterns."""
    print("\n" + "=" * 70)
    print("MODEL FAMILY ANALYSIS")
    print("=" * 70)

    by_family = defaultdict(list)

    for model_name, data in model_results.items():
        meta = data.get("metadata", {})
        family = meta.get("family", "Unknown")

        temp_data = data.get("results", {}).get("temperatures", {})
        if 0.7 in temp_data:
            by_family[family].append({
                "model": model_name,
                "consistency": temp_data[0.7]["mean"],
                "std": temp_data[0.7]["std"]
            })

    print("\n{:<15} {:>12} {:>12} {:>12} {:>8}".format(
        "Family", "Mean C", "Within-Std", "Between-Std", "n"
    ))
    print("-" * 65)

    family_results = {}
    for family in sorted(by_family.keys()):
        models = by_family[family]
        consistencies = [m["consistency"] for m in models]
        within_stds = [m["std"] for m in models]

        print("{:<15} {:>11.3f} {:>11.3f} {:>11.3f} {:>8}".format(
            family,
            np.mean(consistencies),
            np.mean(within_stds),  # Average within-sample variance
            np.std(consistencies),  # Between-model variance
            len(models)
        ))

        family_results[family] = {
            "mean_consistency": np.mean(consistencies),
            "within_model_std": np.mean(within_stds),
            "between_model_std": np.std(consistencies),
            "n": len(models),
            "models": [m["model"] for m in models]
        }

    # ANOVA across families
    family_groups = [
        [m["consistency"] for m in models]
        for models in by_family.values()
        if len(models) >= 2
    ]

    if len(family_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*family_groups)
        print(f"\nANOVA: F={f_stat:.3f}, p={p_value:.4f}")
        family_results["anova"] = {"f_stat": f_stat, "p_value": p_value}

    return family_results


def cluster_by_consistency_pattern(model_results: Dict[str, Dict]) -> Dict:
    """Cluster models by their consistency patterns across temperatures."""
    print("\n" + "=" * 70)
    print("CONSISTENCY PATTERN CLUSTERING")
    print("=" * 70)

    # Build feature matrix: [c_0.3, c_0.7, c_1.0, sensitivity]
    model_names = []
    features = []

    for model_name, data in model_results.items():
        temp_data = data.get("results", {}).get("temperatures", {})

        if 0.3 in temp_data and 0.7 in temp_data and 1.0 in temp_data:
            feature_vec = [
                temp_data[0.3]["mean"],
                temp_data[0.7]["mean"],
                temp_data[1.0]["mean"],
                temp_data[0.3]["mean"] - temp_data[1.0]["mean"],  # sensitivity
            ]
            model_names.append(model_name)
            features.append(feature_vec)

    if len(features) < 3:
        print("Not enough models with complete temperature data for clustering.")
        return {}

    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Hierarchical clustering
    linkage_matrix = linkage(features_scaled, method='ward')

    # Cut into 3 clusters
    clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')

    # Analyze clusters
    cluster_results = defaultdict(list)
    for model_name, cluster_id, feat in zip(model_names, clusters, features):
        meta = model_results[model_name].get("metadata", {})
        cluster_results[cluster_id].append({
            "model": model_name,
            "family": meta.get("family", "Unknown"),
            "architecture": meta.get("architecture", "unknown"),
            "c_0.3": feat[0],
            "c_0.7": feat[1],
            "c_1.0": feat[2],
            "sensitivity": feat[3]
        })

    print("\nClustering Results (3 clusters):")
    cluster_summary = {}

    for cluster_id in sorted(cluster_results.keys()):
        members = cluster_results[cluster_id]
        print(f"\nCluster {cluster_id} ({len(members)} models):")

        sensitivities = [m["sensitivity"] for m in members]
        consistencies = [m["c_0.7"] for m in members]

        # Characterize cluster
        avg_sens = np.mean(sensitivities)
        avg_c = np.mean(consistencies)

        if avg_c > 0.85:
            char = "High Consistency"
        elif avg_c < 0.7:
            char = "Low Consistency"
        else:
            char = "Medium Consistency"

        if avg_sens > 0.15:
            char += " / Temp-Sensitive"
        else:
            char += " / Temp-Stable"

        print(f"  Characterization: {char}")
        print(f"  Mean C@0.7: {avg_c:.3f}, Sensitivity: {avg_sens:+.3f}")
        print(f"  Models: {', '.join(m['model'] for m in members)}")

        cluster_summary[int(cluster_id)] = {
            "characterization": char,
            "mean_consistency": avg_c,
            "mean_sensitivity": avg_sens,
            "models": [m["model"] for m in members],
            "families": list(set(m["family"] for m in members))
        }

    return cluster_summary


def main():
    parser = argparse.ArgumentParser(
        description='COLM 2026: Architecture-Consistency Analysis'
    )
    parser.add_argument('--results-base', type=str,
                        default='llm_gen_results/toucan',
                        help='Base directory for model results')
    parser.add_argument('--temperatures', type=float, nargs='+',
                        default=[0.3, 0.7, 1.0],
                        help='Temperatures to analyze')
    parser.add_argument('--output', type=str,
                        default='results/colm_architecture/architecture_analysis.json',
                        help='Output file for analysis')
    parser.add_argument('--analyze-all', action='store_true',
                        help='Run all analyses')

    args = parser.parse_args()

    results_base = Path(args.results_base)
    if not results_base.exists():
        print(f"Error: Results directory not found: {results_base}")
        sys.exit(1)

    print("=" * 70)
    print("COLM 2026: ARCHITECTURE-CONSISTENCY ANALYSIS")
    print("=" * 70)
    print(f"Results base: {results_base}")
    print(f"Temperatures: {args.temperatures}")

    # Load all model results
    model_results = {}

    for model_dir in results_base.iterdir():
        if not model_dir.is_dir():
            continue

        model_key, metadata = match_model_to_metadata(model_dir.name)
        print(f"\nLoading: {model_dir.name} -> {model_key}")

        results = load_model_results(model_dir, args.temperatures)
        if results["temperatures"]:
            model_results[model_key] = {
                "dir_name": model_dir.name,
                "metadata": metadata,
                "results": results
            }

            # Print summary
            for temp, temp_stats in sorted(results["temperatures"].items()):
                print(f"  T={temp}: mean={temp_stats['mean']:.3f}, n={temp_stats['n']}")

    print(f"\n\nLoaded {len(model_results)} models")

    # Run analyses
    analysis_results = {
        "metadata": {
            "n_models": len(model_results),
            "temperatures": args.temperatures,
            "models": list(model_results.keys())
        }
    }

    if args.analyze_all or True:  # Always run all
        analysis_results["size_scaling"] = analyze_size_scaling(model_results)
        analysis_results["moe_vs_dense"] = analyze_moe_vs_dense(model_results)
        analysis_results["model_families"] = analyze_model_families(model_results)
        analysis_results["clustering"] = cluster_by_consistency_pattern(model_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=json_serializer)

    print(f"\n\nSaved analysis to {output_path}")


if __name__ == '__main__':
    main()
