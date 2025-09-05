import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

# Load data
with open('./experiments/experiment-1/semantic_variation_progression_results.json', 'r') as f:
    semantic_data = json.load(f)
with open('./experiments/experiment-1/expression_variation_progression_results.json', 'r') as f:
    expression_data = json.load(f)

variation_ratios = semantic_data['variation_ratios']
semantic_similarities = semantic_data['average_similarities']
expression_similarities = expression_data['average_similarities']

# Extract raw results for statistical analysis
semantic_raw = semantic_data['raw_results']
expression_raw = expression_data['raw_results']

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

metrics = ['ted', 'sted', 'bertscore', 'deepdiff']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Function to calculate confidence intervals
def calculate_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

# Semantic variations with error bars
for i, metric in enumerate(metrics):
    label = f"{metric.upper()} (N/A)" if metric == 'ted' else metric.upper()
    means = semantic_similarities[metric]
    
    # Calculate standard errors for error bars
    stds = []
    if metric in semantic_raw:
        for ratio in variation_ratios:
            ratio_key = str(ratio)
            if ratio_key in semantic_raw[metric]:
                values = semantic_raw[metric][ratio_key]
                if isinstance(values, list) and len(values) > 1:
                    stds.append(np.std(values) / np.sqrt(len(values)))  # Standard error
                else:
                    stds.append(0)
            else:
                stds.append(0)
    else:
        stds = [0] * len(variation_ratios)
    
    ax1.errorbar(variation_ratios, means, yerr=stds, 
                marker='o', linewidth=2, label=label, color=colors[i], 
                capsize=3, capthick=1)

ax1.set_title('Semantic Variations', fontweight='bold', fontsize=14)
ax1.set_xlabel('Variation Ratio')
ax1.set_ylabel('Similarity Score')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0.6, 1.01)

# Expression variations with error bars
for i, metric in enumerate(metrics):
    label = f"{metric.upper()} (N/A)" if metric == 'ted' else metric.upper()
    means = expression_similarities[metric]
    
    # Calculate standard errors for error bars
    stds = []
    if metric in expression_raw:
        for ratio in variation_ratios:
            ratio_key = str(ratio)
            if ratio_key in expression_raw[metric]:
                values = expression_raw[metric][ratio_key]
                if isinstance(values, list) and len(values) > 1:
                    stds.append(np.std(values) / np.sqrt(len(values)))  # Standard error
                else:
                    stds.append(0)
            else:
                stds.append(0)
    else:
        stds = [0] * len(variation_ratios)
    
    ax2.errorbar(variation_ratios, means, yerr=stds,
                marker='s', linewidth=2, label=label, color=colors[i],
                capsize=3, capthick=1)

ax2.set_title('Expression Variations', fontweight='bold', fontsize=14)
ax2.set_xlabel('Variation Ratio')
ax2.set_ylabel('Similarity Score')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0.6, 1.01)

plt.tight_layout()
plt.savefig('similarity_progression_combined.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical Analysis
print("="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# Function to perform statistical tests
def perform_statistical_tests(raw_data, metric1, metric2, variation_type):
    print(f"\n{variation_type} - {metric1} vs {metric2}:")
    
    if metric1 not in raw_data or metric2 not in raw_data:
        print(f"  Data not available for comparison")
        return
    
    all_p_values = []
    for ratio in variation_ratios:
        ratio_key = str(ratio)
        if ratio_key in raw_data[metric1] and ratio_key in raw_data[metric2]:
            values1 = raw_data[metric1][ratio_key]
            values2 = raw_data[metric2][ratio_key]
            
            if isinstance(values1, list) and isinstance(values2, list) and len(values1) > 1 and len(values2) > 1:
                # Perform Mann-Whitney U test (non-parametric)
                try:
                    statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    all_p_values.append(p_value)
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  Ratio {ratio}: p={p_value:.4f} {significance}")
                except:
                    print(f"  Ratio {ratio}: Test failed")
    
    # Overall significance across all ratios
    if all_p_values:
        mean_p = np.mean(all_p_values)
        print(f"  Mean p-value: {mean_p:.4f}")

# Compare STED vs other methods
for data_type, raw_data in [("Semantic", semantic_raw), ("Expression", expression_raw)]:
    print(f"\n{data_type.upper()} VARIATIONS - STED vs Other Methods:")
    for metric in ['ted', 'bertscore', 'deepdiff']:
        if metric != 'sted':
            perform_statistical_tests(raw_data, 'sted', metric, data_type)

# Create summary statistics table
def create_summary_table(raw_data, variation_type):
    summary_stats = []
    
    for metric in metrics:
        if metric in raw_data:
            all_values = []
            for ratio_key in raw_data[metric].keys():
                values = raw_data[metric][ratio_key]
                if isinstance(values, list):
                    all_values.extend(values)
                else:
                    all_values.append(values)
            
            if all_values:
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                ci_low, ci_high = calculate_ci(all_values)
                
                summary_stats.append({
                    'Metric': metric.upper(),
                    'Mean': mean_val,
                    'Std': std_val,
                    'CI_Low': ci_low,
                    'CI_High': ci_high,
                    'N_Samples': len(all_values)
                })
    
    return pd.DataFrame(summary_stats)

# Generate summary tables
semantic_summary = create_summary_table(semantic_raw, "Semantic")
expression_summary = create_summary_table(expression_raw, "Expression")

print(f"\n{'-'*60}")
print("SEMANTIC VARIATIONS - SUMMARY STATISTICS")
print(f"{'-'*60}")
print(semantic_summary.to_string(index=False, float_format='%.4f'))

print(f"\n{'-'*60}")
print("EXPRESSION VARIATIONS - SUMMARY STATISTICS")
print(f"{'-'*60}")
print(expression_summary.to_string(index=False, float_format='%.4f'))

# Save detailed results
if not semantic_summary.empty:
    semantic_summary.to_csv('semantic_summary_statistics.csv', index=False, float_format='%.4f')
    # Save as LaTeX tables with confidence intervals
    semantic_summary['Mean_CI'] = semantic_summary.apply(lambda x: f"{x['Mean']:.3f} ± {x['Std']:.3f}", axis=1)
    with open('semantic_summary_table.tex', 'w') as f:
        f.write(semantic_summary[['Metric', 'Mean_CI', 'N_Samples']].to_latex(index=False))

if not expression_summary.empty:
    expression_summary.to_csv('expression_summary_statistics.csv', index=False, float_format='%.4f')
    # Save as LaTeX tables with confidence intervals
    expression_summary['Mean_CI'] = expression_summary.apply(lambda x: f"{x['Mean']:.3f} ± {x['Std']:.3f}", axis=1)
    with open('expression_summary_table.tex', 'w') as f:
        f.write(expression_summary[['Metric', 'Mean_CI', 'N_Samples']].to_latex(index=False))

print(f"\nStatistical analysis complete. Files saved:")
print("- similarity_progression_combined.png (with error bars)")
if not semantic_summary.empty:
    print("- semantic_summary_statistics.csv")
    print("- semantic_summary_table.tex")
if not expression_summary.empty:
    print("- expression_summary_statistics.csv") 
    print("- expression_summary_table.tex")
