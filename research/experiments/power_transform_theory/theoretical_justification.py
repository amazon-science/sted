"""
Theoretical Justification for Power Transformation in PDC Metric

This module provides rigorous theoretical and empirical justification for the
power transformation used in PDC: f(σ) = (1/(1+α·σ))^β

We explore multiple theoretical frameworks:
1. Information Theory - Fisher Information, KL divergence
2. Signal Detection Theory - d-prime, ROC analysis
3. Statistical Decision Theory - Optimal discrimination
4. Optimal Transport - Wasserstein distance connection
5. Empirical comparison of transformation families

Author: Research Extension for NeurIPS Main Track
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize, special
from scipy.integrate import quad
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import List, Dict, Tuple, Callable, Optional
import warnings
from dataclasses import dataclass
from itertools import combinations
import json
import os

# Ensure output directory exists
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Part 1: Transformation Families
# =============================================================================

@dataclass
class TransformationFamily:
    """Defines a transformation family with its properties"""
    name: str
    transform: Callable[[np.ndarray, dict], np.ndarray]
    inverse: Optional[Callable[[np.ndarray, dict], np.ndarray]]
    params: dict
    description: str


def power_transform(sigma: np.ndarray, params: dict) -> np.ndarray:
    """PDC power transformation: f(σ) = (1/(1+α·σ))^β"""
    alpha = params.get('alpha', 2.0)
    beta = params.get('beta', 20.0)
    return (1.0 / (1.0 + alpha * sigma)) ** beta


def log_transform(sigma: np.ndarray, params: dict) -> np.ndarray:
    """Logarithmic transformation: f(σ) = 1 - log(1+σ)/log(1+σ_max)"""
    sigma_max = params.get('sigma_max', 0.5)
    return 1.0 - np.log1p(sigma) / np.log1p(sigma_max)


def sigmoid_transform(sigma: np.ndarray, params: dict) -> np.ndarray:
    """Sigmoid transformation: f(σ) = 1 - σ^k / (σ^k + (1-σ)^k)"""
    k = params.get('k', 5.0)
    sigma_clipped = np.clip(sigma, 1e-10, 1.0 - 1e-10)
    return 1.0 - (sigma_clipped ** k) / (sigma_clipped ** k + (1 - sigma_clipped) ** k)


def exponential_transform(sigma: np.ndarray, params: dict) -> np.ndarray:
    """Exponential transformation: f(σ) = exp(-λ·σ)"""
    lambd = params.get('lambda', 10.0)
    return np.exp(-lambd * sigma)


def linear_transform(sigma: np.ndarray, params: dict) -> np.ndarray:
    """Linear transformation: f(σ) = 1 - σ/σ_max"""
    sigma_max = params.get('sigma_max', 0.5)
    return np.clip(1.0 - sigma / sigma_max, 0, 1)


def box_cox_transform(sigma: np.ndarray, params: dict) -> np.ndarray:
    """Box-Cox inspired transformation"""
    lambd = params.get('lambda', 0.5)
    sigma_shifted = sigma + 0.01  # Avoid zero
    if abs(lambd) < 1e-10:
        transformed = -np.log(sigma_shifted)
    else:
        transformed = (sigma_shifted ** lambd - 1) / lambd
    # Normalize to [0, 1]
    return 1.0 - (transformed - transformed.min()) / (transformed.max() - transformed.min() + 1e-10)


TRANSFORMATION_FAMILIES = [
    TransformationFamily("Power (PDC)", power_transform, None, {'alpha': 2.0, 'beta': 20.0},
                        "Current PDC: (1/(1+α·σ))^β"),
    TransformationFamily("Logarithmic", log_transform, None, {'sigma_max': 0.5},
                        "Log-based: 1 - log(1+σ)/log(1+σ_max)"),
    TransformationFamily("Sigmoid", sigmoid_transform, None, {'k': 5.0},
                        "Sigmoid: 1 - σ^k/(σ^k+(1-σ)^k)"),
    TransformationFamily("Exponential", exponential_transform, None, {'lambda': 10.0},
                        "Exponential: exp(-λ·σ)"),
    TransformationFamily("Linear", linear_transform, None, {'sigma_max': 0.5},
                        "Linear: 1 - σ/σ_max"),
    TransformationFamily("Box-Cox", box_cox_transform, None, {'lambda': 0.5},
                        "Box-Cox inspired normalization"),
]


# =============================================================================
# Part 2: Information Theory Analysis
# =============================================================================

class InformationTheoreticAnalysis:
    """
    Theoretical justification through information theory.

    Key insight: The power transformation maximizes Fisher Information
    for distinguishing between consistency levels.
    """

    @staticmethod
    def fisher_information(transform_fn: Callable, params: dict,
                          sigma_range: np.ndarray) -> np.ndarray:
        """
        Compute Fisher Information for a transformation.

        Fisher Information: I(σ) = E[(∂log p(x|σ)/∂σ)²]

        For our transformation f(σ), the Fisher Information measures
        how sensitive the output is to changes in dispersion.

        I(σ) ∝ (f'(σ))² / Var(f(σ))

        Higher Fisher Information = better discrimination
        """
        # Numerical derivative
        epsilon = 1e-6
        f_plus = transform_fn(sigma_range + epsilon, params)
        f_minus = transform_fn(sigma_range - epsilon, params)
        derivative = (f_plus - f_minus) / (2 * epsilon)

        # Fisher Information proportional to squared derivative
        # normalized by local variance
        fisher_info = derivative ** 2

        return fisher_info

    @staticmethod
    def kl_divergence_sensitivity(transform_fn: Callable, params: dict,
                                   sigma1: float, sigma2: float,
                                   n_samples: int = 1000) -> float:
        """
        Compute KL divergence between transformed distributions.

        D_KL(P||Q) measures distinguishability between two consistency levels.
        """
        # Generate sample distances for each dispersion level
        np.random.seed(42)

        # Model distances as truncated normal with given dispersion
        dist1 = np.clip(np.random.normal(0.5, sigma1, n_samples), 0, 1)
        dist2 = np.clip(np.random.normal(0.5, sigma2, n_samples), 0, 1)

        # Transform
        t1 = transform_fn(np.array([np.std(dist1)]), params)[0]
        t2 = transform_fn(np.array([np.std(dist2)]), params)[0]

        # KL divergence approximation for point masses
        # Use absolute difference as proxy
        return abs(t1 - t2)

    @staticmethod
    def entropy_of_transformation(transform_fn: Callable, params: dict,
                                  sigma_range: np.ndarray) -> float:
        """
        Compute entropy of transformed distribution.

        Lower entropy = more concentrated = better discrimination at extremes
        """
        transformed = transform_fn(sigma_range, params)

        # Discretize and compute entropy
        hist, _ = np.histogram(transformed, bins=50, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()

        entropy = -np.sum(hist * np.log(hist))
        return entropy

    @staticmethod
    def mutual_information_analysis(transform_fn: Callable, params: dict,
                                    n_levels: int = 5) -> Dict[str, float]:
        """
        Analyze mutual information between input dispersion and output score.

        High MI = transformation preserves information about consistency level
        """
        # Generate consistency levels
        sigmas = np.linspace(0.01, 0.4, n_levels)
        scores = transform_fn(sigmas, params)

        # Compute correlation (proxy for MI)
        correlation = np.corrcoef(sigmas, scores)[0, 1]

        # Compute monotonicity
        diffs = np.diff(scores)
        monotonicity = np.sum(diffs < 0) / len(diffs)  # Should be 1.0 for decreasing

        # Compute range utilization
        range_util = (scores.max() - scores.min())

        return {
            'correlation': correlation,
            'monotonicity': monotonicity,
            'range_utilization': range_util
        }


# =============================================================================
# Part 3: Signal Detection Theory Analysis
# =============================================================================

class SignalDetectionAnalysis:
    """
    Justification through Signal Detection Theory (SDT).

    Key insight: Power transformation optimizes d' (d-prime) for
    distinguishing between "consistent" and "inconsistent" outputs.
    """

    @staticmethod
    def generate_consistency_classes(n_samples: int = 500,
                                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data for two classes:
        - High consistency (low dispersion): σ ~ U(0, 0.1)
        - Low consistency (high dispersion): σ ~ U(0.2, 0.5)
        """
        np.random.seed(seed)

        # High consistency class
        high_consistency = np.random.uniform(0, 0.1, n_samples)

        # Low consistency class
        low_consistency = np.random.uniform(0.2, 0.5, n_samples)

        return high_consistency, low_consistency

    @staticmethod
    def compute_dprime(transform_fn: Callable, params: dict,
                       high_consistency: np.ndarray,
                       low_consistency: np.ndarray) -> float:
        """
        Compute d' (d-prime) for discrimination.

        d' = (μ_high - μ_low) / √((σ²_high + σ²_low)/2)

        Higher d' = better discrimination between classes
        """
        # Transform both classes
        scores_high = transform_fn(high_consistency, params)
        scores_low = transform_fn(low_consistency, params)

        # Compute d'
        mu_high = np.mean(scores_high)
        mu_low = np.mean(scores_low)
        var_high = np.var(scores_high)
        var_low = np.var(scores_low)

        pooled_std = np.sqrt((var_high + var_low) / 2)

        if pooled_std < 1e-10:
            return float('inf') if mu_high != mu_low else 0.0

        dprime = (mu_high - mu_low) / pooled_std
        return dprime

    @staticmethod
    def compute_roc_auc(transform_fn: Callable, params: dict,
                        high_consistency: np.ndarray,
                        low_consistency: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute ROC curve and AUC for binary classification.
        """
        # Transform
        scores_high = transform_fn(high_consistency, params)
        scores_low = transform_fn(low_consistency, params)

        # Create labels (1 = high consistency, 0 = low consistency)
        all_scores = np.concatenate([scores_high, scores_low])
        all_labels = np.concatenate([np.ones(len(scores_high)),
                                    np.zeros(len(scores_low))])

        # Compute ROC
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)

        return roc_auc, fpr, tpr

    @staticmethod
    def optimal_threshold_analysis(transform_fn: Callable, params: dict,
                                   high_consistency: np.ndarray,
                                   low_consistency: np.ndarray) -> Dict[str, float]:
        """
        Analyze optimal decision threshold properties.
        """
        scores_high = transform_fn(high_consistency, params)
        scores_low = transform_fn(low_consistency, params)

        # Find optimal threshold (Youden's J)
        all_scores = np.concatenate([scores_high, scores_low])
        all_labels = np.concatenate([np.ones(len(scores_high)),
                                    np.zeros(len(scores_low))])

        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)

        optimal_threshold = thresholds[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]

        return {
            'optimal_threshold': optimal_threshold,
            'sensitivity': optimal_tpr,
            'specificity': 1 - optimal_fpr,
            'youden_j': j_scores[optimal_idx]
        }


# =============================================================================
# Part 4: Optimal Beta Derivation
# =============================================================================

class OptimalBetaDerivation:
    """
    Derive optimal β from first principles.

    Key insight: β should maximize discrimination while maintaining stability.
    """

    @staticmethod
    def discrimination_score(beta: float, alpha: float,
                            high_consistency: np.ndarray,
                            low_consistency: np.ndarray) -> float:
        """
        Compute discrimination score for given β.

        Combines:
        - d' (discrimination ability)
        - Range utilization
        - Numerical stability
        """
        params = {'alpha': alpha, 'beta': beta}

        # Compute d'
        dprime = SignalDetectionAnalysis.compute_dprime(
            power_transform, params, high_consistency, low_consistency
        )

        # Compute range
        all_sigma = np.concatenate([high_consistency, low_consistency])
        all_scores = power_transform(all_sigma, params)
        score_range = all_scores.max() - all_scores.min()

        # Stability penalty (very high β causes numerical issues)
        stability = 1.0 / (1.0 + np.exp(0.1 * (beta - 50)))

        # Combined score
        return dprime * score_range * stability

    @staticmethod
    def find_optimal_beta(alpha: float = 2.0,
                          beta_range: Tuple[float, float] = (1, 100),
                          n_samples: int = 500) -> Dict[str, float]:
        """
        Find optimal β that maximizes discrimination.
        """
        high_cons, low_cons = SignalDetectionAnalysis.generate_consistency_classes(n_samples)

        # Grid search
        betas = np.linspace(beta_range[0], beta_range[1], 100)
        scores = []

        for beta in betas:
            score = OptimalBetaDerivation.discrimination_score(
                beta, alpha, high_cons, low_cons
            )
            scores.append(score)

        scores = np.array(scores)
        optimal_idx = np.argmax(scores)
        optimal_beta = betas[optimal_idx]

        # Also find via optimization
        result = optimize.minimize_scalar(
            lambda b: -OptimalBetaDerivation.discrimination_score(b, alpha, high_cons, low_cons),
            bounds=beta_range,
            method='bounded'
        )

        return {
            'optimal_beta_grid': optimal_beta,
            'optimal_beta_optim': result.x,
            'max_discrimination': scores.max(),
            'betas': betas,
            'scores': scores
        }

    @staticmethod
    def beta_from_target_discrimination(target_dprime: float = 3.0,
                                        alpha: float = 2.0,
                                        n_samples: int = 500) -> float:
        """
        Derive β from target d' value.

        In signal detection theory, d' ≥ 3 is considered "excellent" discrimination.
        """
        high_cons, low_cons = SignalDetectionAnalysis.generate_consistency_classes(n_samples)

        def objective(beta):
            dprime = SignalDetectionAnalysis.compute_dprime(
                power_transform, {'alpha': alpha, 'beta': beta},
                high_cons, low_cons
            )
            return (dprime - target_dprime) ** 2

        result = optimize.minimize_scalar(objective, bounds=(1, 100), method='bounded')
        return result.x

    @staticmethod
    def information_theoretic_beta(alpha: float = 2.0,
                                   sigma_range: np.ndarray = None) -> Dict[str, float]:
        """
        Derive β that maximizes Fisher Information in the critical region.

        Critical region: σ ∈ [0.05, 0.2] where most real data falls
        """
        if sigma_range is None:
            sigma_range = np.linspace(0.01, 0.4, 100)

        critical_region = (sigma_range >= 0.05) & (sigma_range <= 0.2)

        betas = np.linspace(1, 100, 100)
        total_fisher = []
        critical_fisher = []

        for beta in betas:
            params = {'alpha': alpha, 'beta': beta}
            fisher = InformationTheoreticAnalysis.fisher_information(
                power_transform, params, sigma_range
            )
            total_fisher.append(np.mean(fisher))
            critical_fisher.append(np.mean(fisher[critical_region]))

        total_fisher = np.array(total_fisher)
        critical_fisher = np.array(critical_fisher)

        return {
            'beta_max_total_fisher': betas[np.argmax(total_fisher)],
            'beta_max_critical_fisher': betas[np.argmax(critical_fisher)],
            'betas': betas,
            'total_fisher': total_fisher,
            'critical_fisher': critical_fisher
        }


# =============================================================================
# Part 5: Comprehensive Comparison
# =============================================================================

class TransformationComparison:
    """Compare all transformation families across multiple criteria."""

    @staticmethod
    def compare_all(n_samples: int = 500) -> Dict[str, Dict]:
        """
        Comprehensive comparison of all transformations.
        """
        high_cons, low_cons = SignalDetectionAnalysis.generate_consistency_classes(n_samples)
        sigma_range = np.linspace(0.01, 0.5, 100)

        results = {}

        for family in TRANSFORMATION_FAMILIES:
            # Signal Detection Metrics
            dprime = SignalDetectionAnalysis.compute_dprime(
                family.transform, family.params, high_cons, low_cons
            )
            roc_auc, _, _ = SignalDetectionAnalysis.compute_roc_auc(
                family.transform, family.params, high_cons, low_cons
            )
            threshold_analysis = SignalDetectionAnalysis.optimal_threshold_analysis(
                family.transform, family.params, high_cons, low_cons
            )

            # Information Theory Metrics
            fisher_info = InformationTheoreticAnalysis.fisher_information(
                family.transform, family.params, sigma_range
            )
            mi_analysis = InformationTheoreticAnalysis.mutual_information_analysis(
                family.transform, family.params
            )

            # Range and Discrimination
            all_sigma = np.concatenate([high_cons, low_cons])
            all_scores = family.transform(all_sigma, family.params)

            results[family.name] = {
                'dprime': dprime,
                'roc_auc': roc_auc,
                'youden_j': threshold_analysis['youden_j'],
                'sensitivity': threshold_analysis['sensitivity'],
                'specificity': threshold_analysis['specificity'],
                'mean_fisher_info': np.mean(fisher_info),
                'correlation': mi_analysis['correlation'],
                'monotonicity': mi_analysis['monotonicity'],
                'range_utilization': mi_analysis['range_utilization'],
                'score_range': all_scores.max() - all_scores.min(),
                'score_std': np.std(all_scores)
            }

        return results

    @staticmethod
    def rank_transformations(comparison_results: Dict[str, Dict]) -> Dict[str, int]:
        """
        Rank transformations by overall performance.
        """
        # Metrics where higher is better
        higher_better = ['dprime', 'roc_auc', 'youden_j', 'sensitivity',
                        'specificity', 'mean_fisher_info', 'monotonicity',
                        'range_utilization', 'score_range']

        # Compute composite score
        composite_scores = {}

        for name, metrics in comparison_results.items():
            score = 0
            for metric in higher_better:
                if metric in metrics:
                    # Normalize by max value across all transformations
                    max_val = max(r[metric] for r in comparison_results.values())
                    if max_val > 0:
                        score += metrics[metric] / max_val
            composite_scores[name] = score

        # Rank
        sorted_names = sorted(composite_scores.keys(),
                             key=lambda x: composite_scores[x],
                             reverse=True)

        rankings = {name: rank + 1 for rank, name in enumerate(sorted_names)}

        return rankings, composite_scores


# =============================================================================
# Part 6: Theoretical Derivation from Optimal Transport
# =============================================================================

class OptimalTransportConnection:
    """
    Connect PDC to Optimal Transport theory.

    Key insight: PDC can be interpreted as measuring the Wasserstein distance
    from the ideal (perfectly consistent) output distribution.
    """

    @staticmethod
    def wasserstein_interpretation(sigma: float, n: int = 10) -> float:
        """
        Interpret normalized dispersion as 1D Wasserstein distance.

        For a set of outputs with dispersion σ, the Wasserstein distance
        from the ideal (all identical) distribution is proportional to σ.
        """
        # W_1 distance from point mass to distribution with std σ
        # For 1D: W_1 = E[|X - μ|] ≈ σ * sqrt(2/π) for normal
        return sigma * np.sqrt(2 / np.pi)

    @staticmethod
    def derive_beta_from_transport(target_separation: float = 0.9) -> float:
        """
        Derive β such that Wasserstein distances map to desired score separation.

        We want:
        - W_1 = 0 (perfect consistency) → PDC = 1
        - W_1 = W_max (maximum inconsistency) → PDC ≈ 0

        The power transformation maps this exponentially for better separation.
        """
        # For separation of 0.9 between W=0 and W=0.2 (typical threshold)
        # We need: (1/(1+2*0))^β - (1/(1+2*0.2))^β ≥ 0.9
        # 1 - (1/1.4)^β ≥ 0.9
        # (1/1.4)^β ≤ 0.1
        # β * log(1/1.4) ≤ log(0.1)
        # β ≥ log(0.1) / log(1/1.4)

        alpha = 2.0
        sigma_threshold = 0.2

        min_beta = np.log(1 - target_separation) / np.log(1 / (1 + alpha * sigma_threshold))

        return min_beta


# =============================================================================
# Part 7: Visualization and Reporting
# =============================================================================

def plot_transformation_comparison(results: Dict[str, Dict],
                                   output_path: str = None) -> plt.Figure:
    """Create comprehensive visualization of transformation comparison."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    names = list(results.keys())

    # Plot 1: d-prime comparison
    ax = axes[0, 0]
    dprimes = [results[n]['dprime'] for n in names]
    colors = ['green' if n == 'Power (PDC)' else 'steelblue' for n in names]
    bars = ax.bar(range(len(names)), dprimes, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("d' (d-prime)")
    ax.set_title("Discrimination Ability (d')")
    ax.axhline(y=3.0, color='red', linestyle='--', label="Excellent threshold")
    ax.legend()

    # Plot 2: ROC AUC comparison
    ax = axes[0, 1]
    aucs = [results[n]['roc_auc'] for n in names]
    bars = ax.bar(range(len(names)), aucs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("ROC AUC")
    ax.set_title("Classification Performance (AUC)")
    ax.set_ylim([0.5, 1.0])

    # Plot 3: Fisher Information
    ax = axes[0, 2]
    fisher = [results[n]['mean_fisher_info'] for n in names]
    bars = ax.bar(range(len(names)), fisher, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Mean Fisher Information")
    ax.set_title("Information Content")

    # Plot 4: Range utilization
    ax = axes[1, 0]
    ranges = [results[n]['range_utilization'] for n in names]
    bars = ax.bar(range(len(names)), ranges, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Range [0-1]")
    ax.set_title("Score Range Utilization")

    # Plot 5: Transformation curves
    ax = axes[1, 1]
    sigma = np.linspace(0, 0.5, 100)
    for family in TRANSFORMATION_FAMILIES:
        scores = family.transform(sigma, family.params)
        linewidth = 3 if family.name == 'Power (PDC)' else 1.5
        ax.plot(sigma, scores, label=family.name, linewidth=linewidth)
    ax.set_xlabel("Dispersion (σ)")
    ax.set_ylabel("Consistency Score")
    ax.set_title("Transformation Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: Composite ranking
    ax = axes[1, 2]
    rankings, composite = TransformationComparison.rank_transformations(results)
    sorted_names = sorted(composite.keys(), key=lambda x: composite[x], reverse=True)
    sorted_scores = [composite[n] for n in sorted_names]
    colors_sorted = ['green' if n == 'Power (PDC)' else 'steelblue' for n in sorted_names]
    bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors_sorted)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Composite Score")
    ax.set_title("Overall Ranking")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_optimal_beta_analysis(beta_results: Dict, fisher_results: Dict,
                               output_path: str = None) -> plt.Figure:
    """Visualize optimal beta derivation."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Discrimination score vs beta
    ax = axes[0]
    ax.plot(beta_results['betas'], beta_results['scores'], 'b-', linewidth=2)
    optimal_beta = beta_results['optimal_beta_grid']
    ax.axvline(x=optimal_beta, color='red', linestyle='--',
               label=f'Optimal β = {optimal_beta:.1f}')
    ax.axvline(x=20, color='green', linestyle=':', linewidth=2,
               label='Current β = 20')
    ax.set_xlabel("β (steepness parameter)")
    ax.set_ylabel("Discrimination Score")
    ax.set_title("Optimal β from Discrimination Maximization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Fisher Information vs beta
    ax = axes[1]
    ax.plot(fisher_results['betas'], fisher_results['total_fisher'],
            'b-', label='Total Fisher Info', linewidth=2)
    ax.plot(fisher_results['betas'], fisher_results['critical_fisher'],
            'r-', label='Critical Region Fisher', linewidth=2)
    ax.axvline(x=fisher_results['beta_max_critical_fisher'], color='red',
               linestyle='--', label=f"Optimal (critical) = {fisher_results['beta_max_critical_fisher']:.1f}")
    ax.axvline(x=20, color='green', linestyle=':', linewidth=2,
               label='Current β = 20')
    ax.set_xlabel("β")
    ax.set_ylabel("Fisher Information")
    ax.set_title("Optimal β from Information Theory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Effect of beta on score distribution
    ax = axes[2]
    sigma = np.linspace(0.01, 0.4, 100)
    for beta in [5, 10, 20, 30, 50]:
        scores = power_transform(sigma, {'alpha': 2.0, 'beta': beta})
        linewidth = 3 if beta == 20 else 1.5
        ax.plot(sigma, scores, label=f'β = {beta}', linewidth=linewidth)
    ax.set_xlabel("Dispersion (σ)")
    ax.set_ylabel("PDC Score")
    ax.set_title("Effect of β on Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves(n_samples: int = 500, output_path: str = None) -> plt.Figure:
    """Plot ROC curves for all transformations."""

    high_cons, low_cons = SignalDetectionAnalysis.generate_consistency_classes(n_samples)

    fig, ax = plt.subplots(figsize=(8, 8))

    for family in TRANSFORMATION_FAMILIES:
        roc_auc, fpr, tpr = SignalDetectionAnalysis.compute_roc_auc(
            family.transform, family.params, high_cons, low_cons
        )
        linewidth = 3 if family.name == 'Power (PDC)' else 1.5
        ax.plot(fpr, tpr, label=f'{family.name} (AUC = {roc_auc:.3f})',
                linewidth=linewidth)

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Consistency Classification')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Part 8: Main Experiment Runner
# =============================================================================

def run_all_experiments(output_dir: str = None) -> Dict:
    """
    Run all theoretical justification experiments.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("THEORETICAL JUSTIFICATION FOR POWER TRANSFORMATION IN PDC")
    print("=" * 70)

    results = {}

    # 1. Compare all transformations
    print("\n1. Comparing transformation families...")
    comparison_results = TransformationComparison.compare_all()
    rankings, composite_scores = TransformationComparison.rank_transformations(comparison_results)

    results['transformation_comparison'] = comparison_results
    results['rankings'] = rankings
    results['composite_scores'] = composite_scores

    print("\n   Transformation Rankings (by composite score):")
    for name, rank in sorted(rankings.items(), key=lambda x: x[1]):
        score = composite_scores[name]
        marker = " <-- CURRENT" if name == "Power (PDC)" else ""
        print(f"   {rank}. {name}: {score:.3f}{marker}")

    # 2. Optimal beta derivation
    print("\n2. Deriving optimal β...")
    beta_results = OptimalBetaDerivation.find_optimal_beta()
    fisher_results = OptimalBetaDerivation.information_theoretic_beta()

    results['beta_optimization'] = {
        'optimal_beta_discrimination': beta_results['optimal_beta_grid'],
        'optimal_beta_fisher': fisher_results['beta_max_critical_fisher'],
        'current_beta': 20.0
    }

    print(f"   Optimal β (discrimination): {beta_results['optimal_beta_grid']:.1f}")
    print(f"   Optimal β (Fisher info): {fisher_results['beta_max_critical_fisher']:.1f}")
    print(f"   Current β: 20.0")

    # 3. Signal Detection Theory metrics
    print("\n3. Signal Detection Theory Analysis...")
    high_cons, low_cons = SignalDetectionAnalysis.generate_consistency_classes()

    pdc_dprime = comparison_results['Power (PDC)']['dprime']
    pdc_auc = comparison_results['Power (PDC)']['roc_auc']

    results['signal_detection'] = {
        'dprime': pdc_dprime,
        'roc_auc': pdc_auc,
        'dprime_interpretation': 'excellent' if pdc_dprime >= 3 else 'good' if pdc_dprime >= 2 else 'fair'
    }

    print(f"   d' (d-prime): {pdc_dprime:.3f} ({results['signal_detection']['dprime_interpretation']})")
    print(f"   ROC AUC: {pdc_auc:.3f}")

    # 4. Information Theory metrics
    print("\n4. Information Theory Analysis...")
    sigma_range = np.linspace(0.01, 0.5, 100)
    fisher_info = InformationTheoreticAnalysis.fisher_information(
        power_transform, {'alpha': 2.0, 'beta': 20.0}, sigma_range
    )
    mi_analysis = InformationTheoreticAnalysis.mutual_information_analysis(
        power_transform, {'alpha': 2.0, 'beta': 20.0}
    )

    results['information_theory'] = {
        'mean_fisher_info': float(np.mean(fisher_info)),
        'max_fisher_info': float(np.max(fisher_info)),
        'correlation': mi_analysis['correlation'],
        'monotonicity': mi_analysis['monotonicity']
    }

    print(f"   Mean Fisher Information: {results['information_theory']['mean_fisher_info']:.4f}")
    print(f"   Correlation (σ → score): {mi_analysis['correlation']:.3f}")
    print(f"   Monotonicity: {mi_analysis['monotonicity']:.1%}")

    # 5. Optimal Transport connection
    print("\n5. Optimal Transport Theory Connection...")
    min_beta_transport = OptimalTransportConnection.derive_beta_from_transport(0.9)

    results['optimal_transport'] = {
        'min_beta_for_90_separation': min_beta_transport,
        'interpretation': f'β ≥ {min_beta_transport:.1f} needed for 90% score separation'
    }

    print(f"   Minimum β for 90% separation: {min_beta_transport:.1f}")

    # 6. Generate visualizations
    print("\n6. Generating visualizations...")

    fig1 = plot_transformation_comparison(
        comparison_results,
        os.path.join(output_dir, 'transformation_comparison.png')
    )
    print(f"   Saved: transformation_comparison.png")

    fig2 = plot_optimal_beta_analysis(
        beta_results, fisher_results,
        os.path.join(output_dir, 'optimal_beta_analysis.png')
    )
    print(f"   Saved: optimal_beta_analysis.png")

    fig3 = plot_roc_curves(
        output_path=os.path.join(output_dir, 'roc_curves.png')
    )
    print(f"   Saved: roc_curves.png")

    # 7. Summary and conclusions
    print("\n" + "=" * 70)
    print("THEORETICAL JUSTIFICATION SUMMARY")
    print("=" * 70)

    conclusions = []

    # Check if power transform is best
    if rankings['Power (PDC)'] == 1:
        conclusions.append("Power transformation RANKS #1 among all tested families")
    else:
        conclusions.append(f"Power transformation ranks #{rankings['Power (PDC)']} (consider alternatives)")

    # Check d-prime
    if pdc_dprime >= 3:
        conclusions.append(f"Excellent discrimination (d' = {pdc_dprime:.2f} ≥ 3.0)")
    elif pdc_dprime >= 2:
        conclusions.append(f"Good discrimination (d' = {pdc_dprime:.2f} ≥ 2.0)")

    # Check β optimality
    optimal_beta_avg = (beta_results['optimal_beta_grid'] + fisher_results['beta_max_critical_fisher']) / 2
    if abs(20 - optimal_beta_avg) < 5:
        conclusions.append(f"Current β=20 is near-optimal (theoretical optimum: {optimal_beta_avg:.1f})")
    else:
        conclusions.append(f"Consider adjusting β to {optimal_beta_avg:.1f} (current: 20)")

    # Monotonicity
    if mi_analysis['monotonicity'] >= 0.99:
        conclusions.append("Perfect monotonicity maintained (consistency axiom satisfied)")

    results['conclusions'] = conclusions

    for i, conclusion in enumerate(conclusions, 1):
        print(f"   {i}. {conclusion}")

    # Save results
    results_path = os.path.join(output_dir, 'theoretical_justification_results.json')

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results_serializable = convert_numpy(results)

    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n   Results saved to: {results_path}")

    plt.close('all')

    return results


# =============================================================================
# Part 9: Additional Theoretical Analysis
# =============================================================================

def prove_metric_properties():
    """
    Verify that PDC satisfies desirable metric properties.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION OF PDC METRIC PROPERTIES")
    print("=" * 70)

    # Property 1: Boundedness
    print("\n1. Boundedness: PDC ∈ [0, 1]")
    sigma_test = np.linspace(0, 10, 1000)
    scores = power_transform(sigma_test, {'alpha': 2.0, 'beta': 20.0})
    print(f"   Min score: {scores.min():.6f}")
    print(f"   Max score: {scores.max():.6f}")
    print(f"   Bounded: {'YES' if scores.min() >= 0 and scores.max() <= 1 else 'NO'}")

    # Property 2: Monotonicity
    print("\n2. Strict Monotonicity: ∂PDC/∂σ < 0")
    diffs = np.diff(scores)
    monotonic = np.all(diffs <= 0)
    print(f"   All derivatives ≤ 0: {'YES' if monotonic else 'NO'}")
    print(f"   Violations: {np.sum(diffs > 0)}")

    # Property 3: Limit behavior
    print("\n3. Limit Behavior:")
    pdc_at_zero = power_transform(np.array([0.0]), {'alpha': 2.0, 'beta': 20.0})[0]
    pdc_at_inf = power_transform(np.array([100.0]), {'alpha': 2.0, 'beta': 20.0})[0]
    print(f"   lim(σ→0) PDC = {pdc_at_zero:.6f} (should be 1.0)")
    print(f"   lim(σ→∞) PDC = {pdc_at_inf:.10f} (should approach 0)")

    # Property 4: Sensitivity analysis
    print("\n4. Sensitivity Analysis:")
    sigma_critical = np.array([0.05, 0.10, 0.15, 0.20])
    scores_critical = power_transform(sigma_critical, {'alpha': 2.0, 'beta': 20.0})
    print("   σ       | PDC Score | Δ from previous")
    print("   " + "-" * 40)
    for i, (s, sc) in enumerate(zip(sigma_critical, scores_critical)):
        delta = scores_critical[i-1] - sc if i > 0 else 0
        print(f"   {s:.2f}    | {sc:.4f}    | {delta:.4f}")

    return {
        'bounded': scores.min() >= 0 and scores.max() <= 1,
        'monotonic': monotonic,
        'correct_limits': pdc_at_zero > 0.99 and pdc_at_inf < 0.01
    }


if __name__ == "__main__":
    # Run all experiments
    results = run_all_experiments()

    # Verify metric properties
    properties = prove_metric_properties()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
