import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import argparse
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze consistency scores from LLM benchmarking results')
parser.add_argument('--combined', required=True, help='Path to combined consistency metrics results JSON file')
parser.add_argument('--content', required=True, help='Path to content consistency metrics results JSON file')
parser.add_argument('--structural', required=True, help='Path to structural consistency metrics results JSON file')
parser.add_argument('--output-dir', default='results', help='Directory to save output files')
parser.add_argument('--exclude', nargs='+', default=[], help='Models to exclude from visualization (e.g., --exclude "GPT-4.1-Mini" "Grok-4.1-Fast")')
parser.add_argument('--include', nargs='+', default=[], help='Only include these models (e.g., --include "Claude-Opus-4" "Claude-Sonnet-4")')
parser.add_argument('--final-models-only', action='store_true', help='Only include the 19 final models (working on both ShareGPT and Toucan)')
parser.add_argument('--top-n', type=int, default=None, help='Only show top N models in heatmaps (by ranking score)')
parser.add_argument('--top-n-boxplot', type=int, default=None, help='Only show top N models in stability distribution boxplot (by median score)')
parser.add_argument('--pdf', action='store_true', help='Also save figures as PDF for LaTeX (vector graphics)')
parser.add_argument('--exclude-samples', nargs='+', type=int, default=[], help='Sample indices to exclude (e.g., --exclude-samples 10 17 18 68 75)')

args = parser.parse_args()

# Define the 18 final models that work on both ShareGPT and Toucan datasets
FINAL_MODELS = [
    'Qwen3-235B-A22B',
    'Claude-3.5-Sonnet',
    'Claude-Haiku-4.5',
    'Claude-3.7-Sonnet',
    'Claude-3.5-Haiku',
    'Claude-Opus-4.5',
    'Claude-Opus-4',
    'Claude-Sonnet-4',
    'Claude-Sonnet-4.5',
    'Qwen3-32B',
    'Llama-3.3-70B',
    'Nova-2-Lite',
    'Mimo-V2-Flash',  # Also matches 'Mimo-V2-Flash:free'
    'Grok-4.1-Fast',
    'Minimax-M2',
    'GPT-4.1-Mini',
    'Gemini-2.5-Flash-Lite',
    'GPT-OSS-120B',
]

import os
os.makedirs(args.output_dir, exist_ok=True)

# Try to import adjustText for better label placement (optional)
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("Note: Install adjustText for better label placement: pip install adjustText")


def save_figure(fig, base_path, dpi=300):
    """Save figure as PNG and optionally as PDF for LaTeX."""
    # Save PNG
    fig.savefig(base_path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {base_path}")

    # Save PDF if requested
    if args.pdf:
        pdf_path = base_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"  Saved: {pdf_path}")

# Load data from all three files
files = {
    'Overall': args.combined,
    'Semantic': args.content, 
    'Structural': args.structural
}

all_results = []
for metric_type, filename in files.items():
    with open(filename, 'r') as f:
        data = json.load(f)

    for model, entries in data.items():
        for entry in entries:
            # penalized_stability_score already includes empty_ratio penalty
            # (computed in structural_consistency_analyzer._calculate_advanced_metrics)
            all_results.append({
                'metric_type': metric_type,
                'model': model,
                'temperature': entry['temperature'],
                'sample_idx': entry.get('sample_idx'),  # May be None for aggregated data
                'normalized_cv': entry['normalized_cv'],
                'consistency_score': entry.get('penalized_stability_score', 0.0),
                'raw_consistency_score': entry.get('stability_score', 0.0),
                'empty_ratio': entry.get('empty_ratio', 0.0),
                'validity_rate': entry.get('validity_rate', 1.0)
            })

df = pd.DataFrame(all_results)

# Model name mapping for friendly display names
MODEL_NAME_MAP = {
    'us.anthropic.claude-opus-4-20250514-v1': 'Claude-Opus-4',
}
df['model'] = df['model'].replace(MODEL_NAME_MAP)

# Filter to include all temperatures from 0.0 to 1.0 in 0.1 increments
valid_temps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
df = df[df['temperature'].isin(valid_temps)]

# Filter out excluded samples (for removing noisy/invalid ground truth samples)
if args.exclude_samples:
    original_count = len(df)
    df = df[~df['sample_idx'].isin(args.exclude_samples)]
    excluded_count = original_count - len(df)
    print(f"Excluded {excluded_count} entries from samples: {args.exclude_samples}")

# Filter models based on options
if args.final_models_only:
    # Use partial matching for model names (handles suffixes like ':free')
    def matches_final_model(model_name):
        for final_model in FINAL_MODELS:
            if final_model in model_name or model_name in final_model:
                return True
        return False

    original_models = df['model'].unique()
    df = df[df['model'].apply(matches_final_model)]
    filtered_models = df['model'].unique()
    excluded = set(original_models) - set(filtered_models)
    print(f"Final models filter: kept {len(filtered_models)} models, excluded {len(excluded)}: {excluded}")

if args.include:
    print(f"Including only models: {args.include}")
    df = df[df['model'].isin(args.include)]

# Filter out excluded models
if args.exclude:
    print(f"Excluding models: {args.exclude}")
    df = df[~df['model'].isin(args.exclude)]

print(f"Models in dataset: {sorted(df['model'].unique())}")

# Calculate statistics (mean, std, confidence intervals)
def calculate_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

# Calculate detailed statistics
detailed_stats = df.groupby(['metric_type', 'model', 'temperature']).agg({
    'normalized_cv': ['mean', 'std', 'count'],
    'consistency_score': ['mean', 'std', 'count']
}).reset_index()

# Flatten column names
detailed_stats.columns = ['metric_type', 'model', 'temperature', 
                         'cv_mean', 'cv_std', 'cv_count',
                         'cs_mean', 'cs_std', 'cs_count']

# Calculate confidence intervals
def add_confidence_intervals(group):
    for metric in ['cv', 'cs']:
        data_col = f'{metric}_data'
        mean_col = f'{metric}_mean'
        ci_low_col = f'{metric}_ci_low'
        ci_high_col = f'{metric}_ci_high'
        
        # Get raw data for CI calculation
        raw_data = df[(df['metric_type'] == group['metric_type'].iloc[0]) & 
                     (df['model'] == group['model'].iloc[0]) & 
                     (df['temperature'] == group['temperature'].iloc[0])]
        
        if metric == 'cv':
            values = raw_data['normalized_cv'].values
        else:
            values = raw_data['consistency_score'].values
            
        if len(values) > 1:
            ci_low, ci_high = calculate_ci(values)
            group[ci_low_col] = ci_low
            group[ci_high_col] = ci_high
        else:
            group[ci_low_col] = group[mean_col]
            group[ci_high_col] = group[mean_col]
    
    return group

# Add confidence intervals
for idx, row in detailed_stats.iterrows():
    raw_data = df[(df['metric_type'] == row['metric_type']) & 
                  (df['model'] == row['model']) & 
                  (df['temperature'] == row['temperature'])]
    
    # CV confidence intervals
    cv_values = raw_data['normalized_cv'].values
    if len(cv_values) > 1:
        ci_low, ci_high = calculate_ci(cv_values)
        detailed_stats.loc[idx, 'cv_ci_low'] = ci_low
        detailed_stats.loc[idx, 'cv_ci_high'] = ci_high
        detailed_stats.loc[idx, 'cv_se'] = stats.sem(cv_values)
    else:
        detailed_stats.loc[idx, 'cv_ci_low'] = row['cv_mean']
        detailed_stats.loc[idx, 'cv_ci_high'] = row['cv_mean']
        detailed_stats.loc[idx, 'cv_se'] = 0
    
    # Consistency Score confidence intervals
    cs_values = raw_data['consistency_score'].values
    if len(cs_values) > 1:
        ci_low, ci_high = calculate_ci(cs_values)
        detailed_stats.loc[idx, 'cs_ci_low'] = ci_low
        detailed_stats.loc[idx, 'cs_ci_high'] = ci_high
        detailed_stats.loc[idx, 'cs_se'] = stats.sem(cs_values)
    else:
        detailed_stats.loc[idx, 'cs_ci_low'] = row['cs_mean']
        detailed_stats.loc[idx, 'cs_ci_high'] = row['cs_mean']
        detailed_stats.loc[idx, 'cs_se'] = 0

# Save detailed statistics
detailed_stats.to_csv(os.path.join(args.output_dir, 'detailed_consistency_statistics.csv'), index=False)
print("Detailed statistics saved to 'detailed_consistency_statistics.csv'")

# Statistical significance tests
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*80)

def perform_significance_tests(metric_type, temperature, metric_name):
    print(f"\n{metric_type} - Temperature {temperature} - {metric_name}:")
    
    # Get data for this specific condition
    condition_data = df[(df['metric_type'] == metric_type) & 
                       (df['temperature'] == temperature)]
    
    models = condition_data['model'].unique()
    if len(models) < 2:
        print("  Not enough models for comparison")
        return
    
    # Compare all pairs of models
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            data1 = condition_data[condition_data['model'] == model1][metric_name].values
            data2 = condition_data[condition_data['model'] == model2][metric_name].values
            
            if len(data1) > 1 and len(data2) > 1:
                # Perform Mann-Whitney U test
                try:
                    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  {model1} vs {model2}: p={p_value:.4f} {significance}")
                except:
                    print(f"  {model1} vs {model2}: Test failed")

# Run significance tests for key conditions
for metric_type in ['Overall', 'Semantic', 'Structural']:
    for temp in [0.1, 0.5, 0.9]:  # Test at low, medium, high temperatures
        perform_significance_tests(metric_type, temp, 'consistency_score')
        perform_significance_tests(metric_type, temp, 'normalized_cv')

# Create summary statistics table
print("\n" + "="*80)
print("SUMMARY STATISTICS (Mean ± Std)")
print("="*80)

summary_stats = df.groupby(['metric_type', 'model']).agg({
    'normalized_cv': ['mean', 'std', 'count'],
    'consistency_score': ['mean', 'std', 'count']
}).round(4)

for metric_type in ['Overall', 'Semantic', 'Structural']:
    print(f"\n{metric_type} Consistency:")
    subset = summary_stats.loc[metric_type]
    print("Model\t\tConsistency Score\t\tNormalized CV\t\tN")
    print("-" * 70)
    for model in subset.index:
        cs_mean = subset.loc[model, ('consistency_score', 'mean')]
        cs_std = subset.loc[model, ('consistency_score', 'std')]
        cv_mean = subset.loc[model, ('normalized_cv', 'mean')]
        cv_std = subset.loc[model, ('normalized_cv', 'std')]
        n = subset.loc[model, ('consistency_score', 'count')]
        print(f"{model}\t\t{cs_mean:.3f} ± {cs_std:.3f}\t\t{cv_mean:.3f} ± {cv_std:.3f}\t\t{n}")

# Create enhanced visualizations with error bars
import matplotlib.colors as mcolors

# Define distinct colors for each model
models = sorted(detailed_stats['model'].unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
model_colors = dict(zip(models, colors))

for metric in ['consistency_score']:
    # Create figure with 3 rows (vertical layout for paper column format)
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    for i, metric_type in enumerate(['Overall', 'Semantic', 'Structural']):
        ax = axes[i]

        for model in models:
            data_subset = detailed_stats[(detailed_stats['metric_type'] == metric_type) &
                                       (detailed_stats['model'] == model)]

            means = data_subset['cs_mean']
            errors = data_subset['cs_se']

            ax.errorbar(data_subset['temperature'], means, yerr=errors,
                       marker='o', label=model, linewidth=1.5, capsize=2, capthick=1,
                       color=model_colors[model], markersize=3)

        ax.set_xlabel('Temperature', fontsize=10)
        ax.set_ylabel('Penalized Stability Score ($S_\\alpha \\cdot r_v$)', fontsize=10)
        ax.set_title(f'{metric_type} Consistency ($S_\\alpha = (1/(1+2\\hat{{D}}_{{std}}))^{{\\alpha}}$, $\\alpha$=20)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Place legend below the plots, centered and in multiple columns
    handles, labels = axes[0].get_legend_handles_labels()
    n_cols = min(4, len(models))  # 4 columns for vertical layout

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Leave space at bottom for legend
    # Add legend below with proper spacing - closer to the diagram
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.11),
               fontsize=8, frameon=True, ncol=n_cols, columnspacing=0.8)

    output_path = os.path.join(args.output_dir, f'{metric}_by_consistency_type_with_errors.png')
    save_figure(fig, output_path)
    plt.show()

# Print LaTeX tables with confidence intervals for 0.1 and 0.9 temperatures
temp_filter = [0.1, 0.9]
filtered_stats = detailed_stats[detailed_stats['temperature'].isin(temp_filter)]

print("\n" + "="*80)
print("LATEX TABLES WITH CONFIDENCE INTERVALS")
print("="*80)

print("\n=== CONSISTENCY SCORE WITH CI ===")
for metric_type in ['Overall', 'Semantic', 'Structural']:
    print(f"\n% {metric_type} Consistency Score (Mean ± 95% CI)")
    subset = filtered_stats[filtered_stats['metric_type'] == metric_type]
    
    models = subset['model'].unique()
    print("\\begin{tabular}{l" + "c" * len(models) + "}")
    print("\\hline")
    print("Temperature & " + " & ".join(models) + " \\\\")
    print("\\hline")
    
    for temp in temp_filter:
        temp_data = subset[subset['temperature'] == temp]
        row_values = []
        for model in models:
            model_data = temp_data[temp_data['model'] == model]
            if not model_data.empty:
                mean = model_data['cs_mean'].iloc[0]
                ci_low = model_data['cs_ci_low'].iloc[0]
                ci_high = model_data['cs_ci_high'].iloc[0]
                row_values.append(f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
            else:
                row_values.append("N/A")
        print(f"{temp} & " + " & ".join(row_values) + " \\\\")
    
    print("\\hline")
    print("\\end{tabular}")

# Save enhanced results
enhanced_summary = df.groupby(['metric_type', 'model']).agg({
    'normalized_cv': ['mean', 'std'],
    'consistency_score': ['mean', 'std']
}).round(4)

enhanced_summary.columns = ['CV_Mean', 'CV_Std', 'CS_Mean', 'CS_Std']
enhanced_summary['CV_MeanStd'] = enhanced_summary.apply(lambda x: f"{x['CV_Mean']:.3f} ± {x['CV_Std']:.3f}", axis=1)
enhanced_summary['CS_MeanStd'] = enhanced_summary.apply(lambda x: f"{x['CS_Mean']:.3f} ± {x['CS_Std']:.3f}", axis=1)

enhanced_summary.to_csv(os.path.join(args.output_dir, 'enhanced_consistency_summary.csv'))

# ============================================================================
# Validity Rate Visualization
# ============================================================================
print("\n" + "="*80)
print("VALIDITY RATE ANALYSIS")
print("="*80)

# Calculate validity rate statistics (only for Overall metric type to avoid duplication)
validity_df = df[df['metric_type'] == 'Overall'].copy()
validity_by_model = validity_df.groupby('model').agg({
    'validity_rate': ['mean', 'std', 'min', 'max']
}).round(4)
validity_by_model.columns = ['validity_mean', 'validity_std', 'validity_min', 'validity_max']
validity_by_model = validity_by_model.sort_values('validity_mean', ascending=True)

print("\nValidity Rate by Model (sorted by mean):")
print(validity_by_model.to_string())

# Save validity rate statistics
validity_by_model.to_csv(os.path.join(args.output_dir, 'validity_rate_by_model.csv'))

# Create validity rate bar chart
fig, ax = plt.subplots(figsize=(14, 8))
models_sorted = validity_by_model.index.tolist()
means = validity_by_model['validity_mean'].values
stds = validity_by_model['validity_std'].values

# Color bars by validity rate (red for low, green for high)
colors = plt.cm.RdYlGn(means)

bars = ax.barh(models_sorted, means, xerr=stds, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Validity Rate', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('Validity Rate by Model (Mean ± Std across temperatures)', fontsize=14)
ax.set_xlim(0, 1.05)
ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
validity_bar_path = os.path.join(args.output_dir, 'validity_rate_by_model.png')
save_figure(fig, validity_bar_path)
plt.show()

# Create validity rate heatmap by model and temperature
validity_pivot = validity_df.pivot_table(
    values='validity_rate',
    index='model',
    columns='temperature',
    aggfunc='mean'
).round(3)

# Sort by mean validity rate
validity_pivot = validity_pivot.loc[models_sorted]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(validity_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks
ax.set_xticks(np.arange(len(validity_pivot.columns)))
ax.set_yticks(np.arange(len(validity_pivot.index)))
ax.set_xticklabels([f'{t:.1f}' for t in validity_pivot.columns])
ax.set_yticklabels(validity_pivot.index)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Validity Rate', rotation=-90, va="bottom")

# Add text annotations
for i in range(len(validity_pivot.index)):
    for j in range(len(validity_pivot.columns)):
        val = validity_pivot.iloc[i, j]
        text_color = 'white' if val < 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)

ax.set_xlabel('Temperature', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('Validity Rate by Model and Temperature', fontsize=14)

plt.tight_layout()
validity_heatmap_path = os.path.join(args.output_dir, 'validity_rate_heatmap.png')
save_figure(fig, validity_heatmap_path)
plt.show()

# Save validity pivot table
validity_pivot.to_csv(os.path.join(args.output_dir, 'validity_rate_by_temperature.csv'))

# ============================================================================
# Generate LaTeX Tables for Paper
# ============================================================================
print("\n" + "="*80)
print("LATEX TABLES FOR PAPER")
print("="*80)

# Table 1: Overall Consistency Score by Model (averaged across temperatures)
overall_summary = df[df['metric_type'] == 'Overall'].groupby('model').agg({
    'consistency_score': ['mean', 'std'],
    'validity_rate': 'mean'
}).round(3)
overall_summary.columns = ['CS_Mean', 'CS_Std', 'Validity']
overall_summary = overall_summary.sort_values('CS_Mean', ascending=False)

latex_table1 = """
% Table: Overall Consistency Scores by Model
\\begin{table}[h]
\\centering
\\caption{Overall STED Consistency Scores by Model (averaged across temperatures 0.1-0.9)}
\\label{tab:overall_consistency}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Model} & \\textbf{Consistency Score} & \\textbf{Validity Rate} \\\\
\\midrule
"""
for model, row in overall_summary.iterrows():
    latex_table1 += f"{model} & {row['CS_Mean']:.3f} $\\pm$ {row['CS_Std']:.3f} & {row['Validity']:.1%} \\\\\n"
latex_table1 += """\\bottomrule
\\end{tabular}
\\end{table}
"""
print(latex_table1)

# Table 2: Consistency at T=0.1 vs T=0.9 (showing temperature sensitivity)
temp_comparison = df[df['metric_type'] == 'Overall'].pivot_table(
    values='consistency_score',
    index='model',
    columns='temperature',
    aggfunc='mean'
).round(3)

latex_table2 = """
% Table: Temperature Sensitivity (T=0.1 vs T=0.9)
\\begin{table}[h]
\\centering
\\caption{STED Consistency Score at Low (T=0.1) vs High (T=0.9) Temperature}
\\label{tab:temp_sensitivity}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Model} & \\textbf{T=0.1} & \\textbf{T=0.9} & \\textbf{$\\Delta$} \\\\
\\midrule
"""
for model in temp_comparison.index:
    t01 = temp_comparison.loc[model, 0.1] if 0.1 in temp_comparison.columns else 0
    t09 = temp_comparison.loc[model, 0.9] if 0.9 in temp_comparison.columns else 0
    delta = t01 - t09
    latex_table2 += f"{model} & {t01:.3f} & {t09:.3f} & {delta:+.3f} \\\\\n"
latex_table2 += """\\bottomrule
\\end{tabular}
\\end{table}
"""
print(latex_table2)

# Table 3: Comparison of Semantic vs Structural Consistency
comparison_df = df.pivot_table(
    values='consistency_score',
    index='model',
    columns='metric_type',
    aggfunc='mean'
).round(3)

latex_table3 = """
% Table: Semantic vs Structural Consistency
\\begin{table}[h]
\\centering
\\caption{Comparison of Semantic vs Structural Consistency (averaged across temperatures)}
\\label{tab:semantic_vs_structural}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Model} & \\textbf{Semantic} & \\textbf{Structural} & \\textbf{Overall} \\\\
\\midrule
"""
for model in comparison_df.index:
    sem = comparison_df.loc[model, 'Semantic'] if 'Semantic' in comparison_df.columns else 0
    struct = comparison_df.loc[model, 'Structural'] if 'Structural' in comparison_df.columns else 0
    overall = comparison_df.loc[model, 'Overall'] if 'Overall' in comparison_df.columns else 0
    latex_table3 += f"{model} & {sem:.3f} & {struct:.3f} & {overall:.3f} \\\\\n"
latex_table3 += """\\bottomrule
\\end{tabular}
\\end{table}
"""
print(latex_table3)

# ============================================================================
# Appendix: Dataset Details
# ============================================================================
print("\n" + "="*80)
print("APPENDIX: DATASET DETAILS")
print("="*80)

# Calculate dataset statistics
models_list = df['model'].unique()
temperatures_list = sorted(df['temperature'].unique())
unique_prompts = df.groupby('model')['temperature'].apply(lambda x: len(x) // len(temperatures_list)).iloc[0] if len(temperatures_list) > 0 else 0

# For calculating runs per sample, we need to check the original data
# The validity_rate * total_runs gives us info about valid runs
runs_per_sample = 10  # Standard configuration

print(f"\n## Dataset Summary:")
print(f"  - Models evaluated: {len(models_list)}")
print(f"  - Temperature settings: {len(temperatures_list)} ({min(temperatures_list):.1f} to {max(temperatures_list):.1f})")
print(f"  - Unique prompts per model: {unique_prompts}")
print(f"  - Runs per sample: {runs_per_sample}")
print(f"  - Total evaluations: {len(df) // 3}")  # Divided by 3 metric types

# Create Appendix Table A1: Dataset Summary
latex_appendix_summary = f"""
% Appendix Table A1: Dataset Summary
\\begin{{table}}[h]
\\centering
\\caption{{Dataset Summary for ShareGPT Structured Output Evaluation}}
\\label{{tab:dataset_summary}}
\\begin{{tabular}}{{ll}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Models Evaluated & {len(models_list)} \\\\
Temperature Settings & {len(temperatures_list)} ({min(temperatures_list):.1f}, {', '.join([f'{t:.1f}' for t in temperatures_list[1:-1]])}, {max(temperatures_list):.1f}) \\\\
Unique Prompts per Model & {unique_prompts} \\\\
Runs per Sample & {runs_per_sample} \\\\
Total Model-Temperature Combinations & {len(models_list) * len(temperatures_list)} \\\\
Total Consistency Evaluations & {len(df) // 3:,} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
print(latex_appendix_summary)

# Create Appendix Table A2: Model Coverage Details
validity_summary = df[df['metric_type'] == 'Overall'].groupby('model').agg({
    'validity_rate': ['mean', 'std', 'min', 'max'],
    'temperature': 'nunique'
}).round(3)
validity_summary.columns = ['validity_mean', 'validity_std', 'validity_min', 'validity_max', 'num_temps']
validity_summary = validity_summary.sort_values('validity_mean', ascending=False)

# Count samples per model
samples_per_model = df[df['metric_type'] == 'Overall'].groupby('model').size()

latex_appendix_coverage = """
% Appendix Table A2: Model Coverage and Validity Rates
\\begin{table}[h]
\\centering
\\caption{Model Coverage Details and Validity Rates}
\\label{tab:model_coverage}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{Samples} & \\textbf{Temps} & \\textbf{Validity Rate} & \\textbf{Range} \\\\
\\midrule
"""
for model in validity_summary.index:
    row = validity_summary.loc[model]
    n_samples = samples_per_model.get(model, 0)
    latex_appendix_coverage += f"{model} & {n_samples} & {int(row['num_temps'])} & {row['validity_mean']:.1%} $\\pm$ {row['validity_std']:.1%} & [{row['validity_min']:.0%}, {row['validity_max']:.0%}] \\\\\n"

latex_appendix_coverage += """\\bottomrule
\\end{tabular}
\\end{table}
"""
print(latex_appendix_coverage)

# Create Appendix Table A3: Temperature Distribution
temp_sample_counts = df[df['metric_type'] == 'Overall'].groupby('temperature').size()
temp_model_counts = df[df['metric_type'] == 'Overall'].groupby('temperature')['model'].nunique()

latex_appendix_temps = """
% Appendix Table A3: Temperature Distribution
\\begin{table}[h]
\\centering
\\caption{Sample Distribution Across Temperatures}
\\label{tab:temp_distribution}
\\begin{tabular}{ccc}
\\toprule
\\textbf{Temperature} & \\textbf{Models} & \\textbf{Total Samples} \\\\
\\midrule
"""
for temp in sorted(temp_sample_counts.index):
    latex_appendix_temps += f"{temp:.1f} & {temp_model_counts[temp]} & {temp_sample_counts[temp]} \\\\\n"
latex_appendix_temps += """\\bottomrule
\\end{tabular}
\\end{table}
"""
print(latex_appendix_temps)

# Save validity summary to CSV
validity_summary_path = os.path.join(args.output_dir, 'appendix_model_coverage.csv')
validity_summary.to_csv(validity_summary_path)
print(f"Model coverage saved to: {validity_summary_path}")

# Save LaTeX tables to file
latex_output_path = os.path.join(args.output_dir, 'latex_tables.tex')
with open(latex_output_path, 'w') as f:
    f.write("% Auto-generated LaTeX tables for STED consistency analysis\n")
    f.write("% Requires: \\usepackage{booktabs}\n\n")
    f.write("% ============================================================\n")
    f.write("% Main Paper Tables\n")
    f.write("% ============================================================\n\n")
    f.write(latex_table1)
    f.write("\n")
    f.write(latex_table2)
    f.write("\n")
    f.write(latex_table3)
    f.write("\n")
    f.write("% ============================================================\n")
    f.write("% Appendix Tables: Dataset Details\n")
    f.write("% ============================================================\n\n")
    f.write(latex_appendix_summary)
    f.write("\n")
    f.write(latex_appendix_coverage)
    f.write("\n")
    f.write(latex_appendix_temps)

print(f"\nLaTeX tables saved to: {latex_output_path}")

print(f"\nStatistical analysis complete. Files saved:")
print("- detailed_consistency_statistics.csv")
print("- enhanced_consistency_summary.csv")
print("- consistency_score_by_consistency_type_with_errors.png")
print("- validity_rate_by_model.csv")
print("- validity_rate_by_model.png")
print("- validity_rate_heatmap.png")
print("- validity_rate_by_temperature.csv")
print("- appendix_model_coverage.csv")
print("- latex_tables.tex (includes Appendix tables)")

# ============================================================================
# Enhanced Visualizations for ICML 2026 Paper
# ============================================================================
print("\n" + "="*80)
print("GENERATING ENHANCED VISUALIZATIONS")
print("="*80)

# First, load additional metrics from the JSON files for richer visualization
all_metrics = []
for metric_type, filename in files.items():
    with open(filename, 'r') as f:
        data = json.load(f)

    for model, entries in data.items():
        for entry in entries:
            all_metrics.append({
                'metric_type': metric_type,
                'model': model,
                'temperature': entry['temperature'],
                'c_mean': entry.get('c_mean', entry.get('mean_similarity', 0.0)),
                'd_std': entry.get('d_std', entry.get('std_deviation', 0.0)),
                'd_std_normalized': entry.get('d_std_normalized', 0.0),
                'r_v': entry.get('r_v', entry.get('validity_rate', 1.0)),
                'c_adj': entry.get('c_adj', 0.0),
                'stability_score': entry.get('stability_score', 0.0),
                'ranking_score': entry.get('ranking_score', 0.0),
                'penalized_stability_score': entry.get('penalized_stability_score', 0.0),
                'validity_rate': entry.get('validity_rate', 1.0),
            })

metrics_df = pd.DataFrame(all_metrics)
metrics_df['model'] = metrics_df['model'].replace(MODEL_NAME_MAP)

# Filter temperatures and excluded models
metrics_df = metrics_df[metrics_df['temperature'].isin(valid_temps)]

# Apply the same filtering as df
if args.final_models_only:
    metrics_df = metrics_df[metrics_df['model'].apply(matches_final_model)]

if args.include:
    metrics_df = metrics_df[metrics_df['model'].isin(args.include)]

if args.exclude:
    metrics_df = metrics_df[~metrics_df['model'].isin(args.exclude)]

# ============================================================================
# 1. Scatter Plot: C_mean vs D_std (Mean-Variance Tradeoff)
# ============================================================================
print("\n1. Creating C_mean vs D_std scatter plot...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, metric_type in enumerate(['Overall', 'Semantic', 'Structural']):
    ax = axes[idx]
    subset = metrics_df[metrics_df['metric_type'] == metric_type]

    # Average across temperatures for each model
    model_avg = subset.groupby('model').agg({
        'c_mean': 'mean',
        'd_std': 'mean',
        'r_v': 'mean'
    }).reset_index()

    # Scatter with size proportional to validity rate
    scatter = ax.scatter(model_avg['c_mean'], model_avg['d_std'],
                        s=model_avg['r_v'] * 200 + 50,
                        c=[model_colors.get(m, 'gray') for m in model_avg['model']],
                        alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add model labels
    for _, row in model_avg.iterrows():
        ax.annotate(row['model'], (row['c_mean'], row['d_std']),
                   fontsize=6, alpha=0.8, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')

    ax.set_xlabel('$C_{mean}$ (Mean Pairwise Consistency)', fontsize=10)
    ax.set_ylabel('$D_{std}$ (Dispersion)', fontsize=10)
    ax.set_title(f'{metric_type}: Mean vs Dispersion Tradeoff', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, max(0.3, model_avg['d_std'].max() * 1.2))
    ax.grid(True, alpha=0.3)

    # Add ideal region annotation (high mean, low std)
    ax.annotate('Ideal Region\n(High $C_{mean}$, Low $D_{std}$)',
               xy=(0.9, 0.02), fontsize=8, color='green', alpha=0.7,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
scatter_path = os.path.join(args.output_dir, 'cmean_vs_dstd_scatter.png')
save_figure(fig, scatter_path)
plt.show()

# ============================================================================
# 2. Radar/Spider Chart: Multi-dimensional Model Comparison
# ============================================================================
print("\n2. Creating radar chart for multi-dimensional comparison...")

from math import pi

# Select top models for radar chart (avoid clutter)
overall_avg = metrics_df[metrics_df['metric_type'] == 'Overall'].groupby('model').agg({
    'c_mean': 'mean',
    'd_std_normalized': 'mean',
    'r_v': 'mean',
    'stability_score': 'mean',
    'ranking_score': 'mean'
}).reset_index()

# Sort by ranking score and take top 8 models
top_models = overall_avg.nlargest(8, 'ranking_score')['model'].tolist()

# Prepare radar data
categories = ['$C_{mean}$', '$1-\\hat{D}_{std}$', '$r_v$', '$S_\\alpha$', '$R$']
n_cats = len(categories)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

angles = [n / float(n_cats) * 2 * pi for n in range(n_cats)]
angles += angles[:1]  # Complete the loop

for i, model in enumerate(top_models):
    model_data = overall_avg[overall_avg['model'] == model].iloc[0]

    values = [
        model_data['c_mean'],
        1 - model_data['d_std_normalized'],  # Invert so higher is better
        model_data['r_v'],
        model_data['stability_score'],
        model_data['ranking_score']
    ]
    values += values[:1]  # Complete the loop

    color = model_colors.get(model, plt.cm.tab10(i / len(top_models)))
    ax.plot(angles, values, 'o-', linewidth=1.5, label=model, color=color, markersize=4)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Multi-dimensional Model Comparison (Top 8 by Ranking Score)',
            fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
radar_path = os.path.join(args.output_dir, 'radar_model_comparison.png')
save_figure(fig, radar_path)
plt.show()

# ============================================================================
# 3. Box Plots: Distribution of Raw Similarities
# ============================================================================
print("\n3. Creating box plots for similarity distributions...")

# We need raw similarities - aggregate c_mean distributions across temperatures
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for idx, metric_type in enumerate(['Overall', 'Semantic', 'Structural']):
    ax = axes[idx]
    subset = metrics_df[metrics_df['metric_type'] == metric_type]

    # Get models sorted by median c_mean
    model_medians = subset.groupby('model')['c_mean'].median().sort_values(ascending=False)
    sorted_models = model_medians.index.tolist()

    # Create box plot data
    box_data = [subset[subset['model'] == m]['c_mean'].values for m in sorted_models]

    bp = ax.boxplot(box_data, labels=sorted_models, patch_artist=True,
                   showfliers=True, flierprops={'marker': 'o', 'markersize': 3})

    # Color boxes by model
    for patch, model in zip(bp['boxes'], sorted_models):
        patch.set_facecolor(model_colors.get(model, 'lightblue'))
        patch.set_alpha(0.7)

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('$C_{mean}$ Distribution', fontsize=10)
    ax.set_title(f'{metric_type}: Consistency Distribution by Model', fontsize=11, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
boxplot_path = os.path.join(args.output_dir, 'cmean_boxplots.png')
save_figure(fig, boxplot_path)
plt.show()

# ============================================================================
# 4. Heatmap: C_adj (Validity-Adjusted Consistency) by Model x Temperature
# ============================================================================
print("\n4. Creating C_adj heatmap...")

# Create pivot table for Overall metric type
overall_metrics = metrics_df[metrics_df['metric_type'] == 'Overall']
cadj_pivot = overall_metrics.pivot_table(
    values='c_adj',
    index='model',
    columns='temperature',
    aggfunc='mean'
).round(3)

# Sort by mean C_adj
cadj_pivot = cadj_pivot.loc[cadj_pivot.mean(axis=1).sort_values(ascending=False).index]

# Apply --top-n filter if specified
if args.top_n and args.top_n < len(cadj_pivot):
    cadj_pivot = cadj_pivot.head(args.top_n)
    print(f"  Showing top {args.top_n} models")

# Adjust figure height based on number of models
n_models = len(cadj_pivot)
fig_height = max(6, min(12, n_models * 0.5 + 2))
fig, ax = plt.subplots(figsize=(14, fig_height))
im = ax.imshow(cadj_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks
ax.set_xticks(np.arange(len(cadj_pivot.columns)))
ax.set_yticks(np.arange(len(cadj_pivot.index)))
ax.set_xticklabels([f'{t:.1f}' for t in cadj_pivot.columns])
ax.set_yticklabels(cadj_pivot.index)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('$C_{adj}$ (Validity-Adjusted Consistency)', rotation=-90, va="bottom")

# Add text annotations
for i in range(len(cadj_pivot.index)):
    for j in range(len(cadj_pivot.columns)):
        val = cadj_pivot.iloc[i, j]
        if not np.isnan(val):
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)

ax.set_xlabel('Temperature', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('$C_{adj} = r_v \\times C_{mean}$ (Validity-Adjusted Consistency)', fontsize=14, fontweight='bold')

plt.tight_layout()
cadj_heatmap_path = os.path.join(args.output_dir, 'cadj_heatmap.png')
save_figure(fig, cadj_heatmap_path)
plt.show()

# ============================================================================
# 5. Grouped Bar Chart: Structural vs Semantic Consistency
# ============================================================================
print("\n5. Creating grouped bar chart for Structural vs Semantic comparison...")

# Average across temperatures
comparison_data = metrics_df.groupby(['model', 'metric_type']).agg({
    'c_mean': 'mean',
    'stability_score': 'mean'
}).reset_index()

# Pivot for grouped bars
semantic_data = comparison_data[comparison_data['metric_type'] == 'Semantic'].set_index('model')['c_mean']
structural_data = comparison_data[comparison_data['metric_type'] == 'Structural'].set_index('model')['c_mean']
overall_data = comparison_data[comparison_data['metric_type'] == 'Overall'].set_index('model')['c_mean']

# Get common models and sort by overall
common_models = list(set(semantic_data.index) & set(structural_data.index) & set(overall_data.index))
common_models = sorted(common_models, key=lambda m: overall_data.get(m, 0), reverse=True)

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(common_models))
width = 0.25

bars1 = ax.bar(x - width, [structural_data.get(m, 0) for m in common_models],
               width, label='Structural', color='steelblue', alpha=0.8)
bars2 = ax.bar(x, [semantic_data.get(m, 0) for m in common_models],
               width, label='Semantic', color='coral', alpha=0.8)
bars3 = ax.bar(x + width, [overall_data.get(m, 0) for m in common_models],
               width, label='Overall (Combined)', color='forestgreen', alpha=0.8)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('$C_{mean}$ (Mean Pairwise Consistency)', fontsize=12)
ax.set_title('Structural vs Semantic Consistency by Model', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(common_models, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0.05:
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2), textcoords="offset points",
                       ha='center', va='bottom', fontsize=6, rotation=90)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()
grouped_bar_path = os.path.join(args.output_dir, 'structural_vs_semantic_bars.png')
save_figure(fig, grouped_bar_path)
plt.show()

# ============================================================================
# 6. Pareto Frontier: C_adj vs S_α (Identifying Optimal Models)
# ============================================================================
print("\n6. Creating Pareto frontier plot...")

# Get overall metrics averaged across temperatures
pareto_data = metrics_df[metrics_df['metric_type'] == 'Overall'].groupby('model').agg({
    'c_adj': 'mean',
    'stability_score': 'mean',
    'r_v': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter all models and collect text annotations for adjustText
texts = []
for _, row in pareto_data.iterrows():
    color = model_colors.get(row['model'], 'gray')
    ax.scatter(row['c_adj'], row['stability_score'],
              s=row['r_v'] * 200 + 50, c=[color], alpha=0.7,
              edgecolors='black', linewidth=0.5)
    # Collect text objects for adjustment
    txt = ax.text(row['c_adj'], row['stability_score'], row['model'],
                  fontsize=7, alpha=0.9, ha='left', va='bottom')
    texts.append(txt)

# Use adjustText to prevent label overlaps if available
if HAS_ADJUST_TEXT and texts:
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5),
                expand_points=(1.5, 1.5),
                force_text=(0.5, 0.5),
                force_points=(0.3, 0.3))

# Find and plot Pareto frontier
def is_pareto_efficient(costs):
    """Find Pareto-efficient points (maximize both dimensions)"""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

costs = pareto_data[['c_adj', 'stability_score']].values
pareto_mask = is_pareto_efficient(costs)
pareto_points = pareto_data[pareto_mask].sort_values('c_adj')

# Draw Pareto frontier line
if len(pareto_points) > 1:
    ax.plot(pareto_points['c_adj'], pareto_points['stability_score'],
           'g--', linewidth=2, alpha=0.7, label='Pareto Frontier')
    ax.scatter(pareto_points['c_adj'], pareto_points['stability_score'],
              s=100, facecolors='none', edgecolors='green', linewidth=2)

ax.set_xlabel('$C_{adj}$ (Validity-Adjusted Consistency)', fontsize=12)
ax.set_ylabel('$S_\\alpha$ (Stability Score)', fontsize=12)
ax.set_title('Pareto Frontier: Consistency $\\times$ Stability', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')

# Add quadrant labels
ax.annotate('High Consistency\nHigh Stability\n(OPTIMAL)',
           xy=(0.88, 0.92), fontsize=9, color='green', alpha=0.7,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
           ha='center')
ax.annotate('Low Consistency\nLow Stability',
           xy=(0.15, 0.15), fontsize=9, color='red', alpha=0.7,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
           ha='center')

plt.tight_layout()
pareto_path = os.path.join(args.output_dir, 'pareto_frontier.png')
save_figure(fig, pareto_path)
plt.show()

# ============================================================================
# 7. Ranking Score Heatmap (R = r_v * C_mean * S_α)
# ============================================================================
print("\n7. Creating Ranking Score heatmap...")

ranking_pivot = overall_metrics.pivot_table(
    values='ranking_score',
    index='model',
    columns='temperature',
    aggfunc='mean'
).round(3)

# Sort by mean ranking score
ranking_pivot = ranking_pivot.loc[ranking_pivot.mean(axis=1).sort_values(ascending=False).index]

# Apply --top-n filter if specified
if args.top_n and args.top_n < len(ranking_pivot):
    ranking_pivot = ranking_pivot.head(args.top_n)
    print(f"  Showing top {args.top_n} models")

# Adjust figure height based on number of models
n_models = len(ranking_pivot)
fig_height = max(6, min(12, n_models * 0.5 + 2))
fig, ax = plt.subplots(figsize=(14, fig_height))
im = ax.imshow(ranking_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(ranking_pivot.columns)))
ax.set_yticks(np.arange(len(ranking_pivot.index)))
ax.set_xticklabels([f'{t:.1f}' for t in ranking_pivot.columns])
ax.set_yticklabels(ranking_pivot.index)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('$R = r_v \\times C_{mean} \\times S_\\alpha$', rotation=-90, va="bottom")

for i in range(len(ranking_pivot.index)):
    for j in range(len(ranking_pivot.columns)):
        val = ranking_pivot.iloc[i, j]
        if not np.isnan(val):
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)

ax.set_xlabel('Temperature', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('$R = r_v \\times C_{mean} \\times S_\\alpha$ (Combined Ranking Score)', fontsize=14, fontweight='bold')

plt.tight_layout()
ranking_heatmap_path = os.path.join(args.output_dir, 'ranking_score_heatmap.png')
save_figure(fig, ranking_heatmap_path)
plt.show()

# Save pivot tables as CSV
cadj_pivot.to_csv(os.path.join(args.output_dir, 'cadj_by_temperature.csv'))
ranking_pivot.to_csv(os.path.join(args.output_dir, 'ranking_score_by_temperature.csv'))

# ============================================================================
# 8. Appendix Figure: Stability Distribution + Key Models Temperature Curves
# ============================================================================
print("\n8. Creating appendix stability distribution figure...")

# Define key representative models for panel (b)
KEY_MODELS = [
    'Claude-Opus-4.5',      # Most powerful proprietary
    'Qwen3-235B-A22B',      # Most powerful open source
    'GPT-4.1-Mini',         # Cost-effective option
    'Grok-4.1-Fast',        # Fast inference
    'Nova-2-Lite',          # AWS option
]

# Filter to models that exist in the dataset
available_models = df['model'].unique()
key_models_available = [m for m in KEY_MODELS if m in available_models]

# Create two-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.2, 1]})

# Panel (a): Box plots of stability score by model
overall_df = df[df['metric_type'] == 'Overall']

# Get models sorted by median consistency score
model_medians = overall_df.groupby('model')['consistency_score'].median().sort_values(ascending=False)
sorted_models = model_medians.index.tolist()

# Apply top-N filter if specified
if args.top_n_boxplot and args.top_n_boxplot < len(sorted_models):
    sorted_models = sorted_models[:args.top_n_boxplot]
    print(f"  Showing top {args.top_n_boxplot} models in boxplot")

# Create box plot data
box_data = [overall_df[overall_df['model'] == m]['consistency_score'].values for m in sorted_models]

# Create gradient colors (red to green based on rank)
n_models_box = len(sorted_models)
gradient_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_models_box))

bp = ax1.boxplot(box_data, labels=sorted_models, patch_artist=True,
                 showfliers=True, flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.5},
                 medianprops={'color': 'red', 'linewidth': 1.5})

# Color boxes with gradient
for patch, color in zip(bp['boxes'], gradient_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_xlabel('Model', fontsize=10)
ax1.set_ylabel('Stability Score ($S_\\alpha$)', fontsize=10)
ax1.set_title('(a) Stability Score Distribution by Model', fontsize=11, fontweight='bold')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3, axis='y')

# Panel (b): Line plot showing key models' consistency vs temperature
# Define colors for key models
key_model_colors = {
    'Claude-Opus-4.5': '#E63946',      # Red
    'Qwen3-235B-A22B': '#457B9D',      # Blue
    'GPT-4.1-Mini': '#2A9D8F',         # Teal
    'Grok-4.1-Fast': '#E9C46A',        # Yellow
    'Nova-2-Lite': '#9B5DE5',          # Purple
}

key_model_markers = {
    'Claude-Opus-4.5': 'o',
    'Qwen3-235B-A22B': 's',
    'GPT-4.1-Mini': '^',
    'Grok-4.1-Fast': 'D',
    'Nova-2-Lite': 'v',
}

# Calculate mean consistency score by model and temperature
temp_curves = overall_df.groupby(['model', 'temperature']).agg({
    'consistency_score': 'mean'
}).reset_index()
temp_curves.columns = ['model', 'temperature', 'cs_mean']

for model in key_models_available:
    model_data = temp_curves[temp_curves['model'] == model].sort_values('temperature')
    color = key_model_colors.get(model, 'gray')
    marker = key_model_markers.get(model, 'o')

    ax2.plot(model_data['temperature'], model_data['cs_mean'],
             marker=marker, label=model, linewidth=2.5,
             color=color, markersize=7, alpha=0.9)

ax2.set_xlabel('Temperature', fontsize=10)
ax2.set_ylabel('Stability Score ($S_\\alpha$)', fontsize=10)
ax2.set_title('(b) Key Models: Stability vs Temperature', fontsize=11, fontweight='bold')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower left', fontsize=8, framealpha=0.9)

# Add annotation for temperature effect
ax2.annotate('Higher temperature\n→ Lower stability',
            xy=(0.85, 0.25), fontsize=8, color='gray', alpha=0.8,
            ha='center', style='italic')

plt.tight_layout()
appendix_stability_path = os.path.join(args.output_dir, 'appendix_stability_distribution.png')
save_figure(fig, appendix_stability_path)
plt.show()

print("\n" + "="*80)
print("ENHANCED VISUALIZATIONS COMPLETE")
print("="*80)
print("\nAdditional files saved:")
print("- cmean_vs_dstd_scatter.png (Mean-Variance Tradeoff)")
print("- radar_model_comparison.png (Multi-dimensional Comparison)")
print("- cmean_boxplots.png (Consistency Distributions)")
print("- cadj_heatmap.png (Validity-Adjusted Consistency)")
print("- structural_vs_semantic_bars.png (Component Comparison)")
print("- pareto_frontier.png (Optimal Model Selection)")
print("- ranking_score_heatmap.png (Combined Ranking Score)")
print("- appendix_stability_distribution.png (Stability Distribution + Key Models)")
print("- cadj_by_temperature.csv")
print("- ranking_score_by_temperature.csv")