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
        ax.set_title(f'{metric_type} Stability ($S_\\alpha = (1/(1+2\\hat{{\\sigma}}))^{{\\alpha}}$, $\\alpha$=20)', fontsize=11, fontweight='bold')
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
    plt.close(fig)

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
validity_by_model.to_csv(os.path.join(args.output_dir, 'rv_by_model.csv'))

# Create validity rate bar chart
fig, ax = plt.subplots(figsize=(14, 8))
models_sorted = validity_by_model.index.tolist()
means = validity_by_model['validity_mean'].values
stds = validity_by_model['validity_std'].values

# Color bars by validity rate (red for low, green for high)
colors = plt.cm.RdYlGn(means)

bars = ax.barh(models_sorted, means, xerr=stds, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('$r_v$ (Validity)', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('$r_v$ (Validity Rate) by Model', fontsize=14)
ax.set_xlim(0, 1.05)
ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
validity_bar_path = os.path.join(args.output_dir, 'rv_by_model.png')
save_figure(fig, validity_bar_path)
plt.close(fig)

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
cbar.ax.set_ylabel('$r_v$ (Validity)', rotation=-90, va="bottom")

# Add text annotations
for i in range(len(validity_pivot.index)):
    for j in range(len(validity_pivot.columns)):
        val = validity_pivot.iloc[i, j]
        text_color = 'white' if val < 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)

ax.set_xlabel('Temperature', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('$r_v$ (Validity) by Model and Temperature', fontsize=14)

plt.tight_layout()
validity_heatmap_path = os.path.join(args.output_dir, 'rv_heatmap.png')
save_figure(fig, validity_heatmap_path)
plt.close(fig)

# Save validity pivot table
validity_pivot.to_csv(os.path.join(args.output_dir, 'rv_by_temperature.csv'))

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
print("- rv_by_model.csv")
print("- rv_by_model.png")
print("- rv_heatmap.png")
print("- rv_by_temperature.csv")
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

    ax.set_xlabel('$C_{mean}$ (Consistency)', fontsize=10)
    ax.set_ylabel('$\\sigma$ (Std. Deviation)', fontsize=10)
    ax.set_title(f'{metric_type}: Consistency vs Dispersion Tradeoff', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, max(0.3, model_avg['d_std'].max() * 1.2))
    ax.grid(True, alpha=0.3)

    # Add ideal region annotation (high mean, low std)
    ax.annotate('Ideal Region\n(High $C_{mean}$, Low $D_{std}$)',
               xy=(0.9, 0.02), fontsize=8, color='green', alpha=0.7,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
scatter_path = os.path.join(args.output_dir, 'Cmean_vs_sigma_scatter.png')
save_figure(fig, scatter_path)
plt.close(fig)

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
plt.close(fig)

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
    ax.set_ylabel('$C_{mean}$ (Consistency)', fontsize=10)
    ax.set_title(f'{metric_type}: $C_{{mean}}$ Distribution by Model', fontsize=11, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
boxplot_path = os.path.join(args.output_dir, 'Cmean_boxplots.png')
save_figure(fig, boxplot_path)
plt.close(fig)

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
cbar.ax.set_ylabel('$C_{adj} = r_v \\times C_{mean}$', rotation=-90, va="bottom")

# Add text annotations
for i in range(len(cadj_pivot.index)):
    for j in range(len(cadj_pivot.columns)):
        val = cadj_pivot.iloc[i, j]
        if not np.isnan(val):
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)

ax.set_xlabel('Temperature', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('$C_{adj} = r_v \\times C_{mean}$ (Adjusted Consistency)', fontsize=14, fontweight='bold')

plt.tight_layout()
cadj_heatmap_path = os.path.join(args.output_dir, 'cadj_heatmap.png')
save_figure(fig, cadj_heatmap_path)
plt.close(fig)

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
ax.set_ylabel('$C_{mean}$ (Consistency)', fontsize=12)
ax.set_title('Structural vs Semantic $C_{mean}$ by Model', fontsize=14, fontweight='bold')
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
plt.close(fig)

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

ax.set_xlabel('$C_{adj} = r_v \\times C_{mean}$ (Adjusted Consistency)', fontsize=12)
ax.set_ylabel('$S_\\alpha$ (Stability)', fontsize=12)
ax.set_title('Pareto Frontier: $C_{adj}$ vs $S_\\alpha$', fontsize=14, fontweight='bold')
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
plt.close(fig)

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
ax.set_title('$R = r_v \\times C_{mean} \\times S_\\alpha$ (Ranking Score)', fontsize=14, fontweight='bold')

plt.tight_layout()
ranking_heatmap_path = os.path.join(args.output_dir, 'R_heatmap.png')
save_figure(fig, ranking_heatmap_path)
plt.close(fig)

# Save pivot tables as CSV
cadj_pivot.to_csv(os.path.join(args.output_dir, 'cadj_by_temperature.csv'))
ranking_pivot.to_csv(os.path.join(args.output_dir, 'R_by_temperature.csv'))

# ============================================================================
# SECTION 3.7: CONSISTENCY AGGREGATION (per ICML paper)
# ============================================================================
# This section implements the aggregation metrics from Section 3.7:
#   - C_mean: Average pairwise STED similarity (2/m(m-1) * sum(s_ij))
#   - r_v: Validity rate (fraction of parseable outputs = m/n)
#   - S_alpha: Stability Score = (1/(1 + 2*sigma_hat))^alpha
#     where sigma_hat = sigma / sigma_max, sigma_max = 0.5
#   - R: Ranking Score = r_v * C_mean * S_alpha
# ============================================================================
print("\n" + "="*80)
print("SECTION 3.7: CONSISTENCY AGGREGATION METRICS BY TYPE")
print("="*80)

def calculate_metrics_by_type(metrics_subset, metric_type_name):
    """
    Calculate consistency aggregation metrics per ICML paper Section 3.7.

    Metrics computed:
    - C_mean: Mean pairwise consistency ("how similar are two random runs?")
    - r_v: Validity rate (fraction of successful parses)
    - S_alpha: Stability score with power transformation (alpha=20)
    - R: Ranking score = r_v * C_mean * S_alpha
    """

    # Create C_mean pivot table for each model at each temperature
    cmean_pivot = metrics_subset.pivot_table(
        values='c_mean',
        index='model',
        columns='temperature',
        aggfunc='mean'
    ).round(4)

    # Sort by mean C_mean
    cmean_pivot = cmean_pivot.loc[cmean_pivot.mean(axis=1).sort_values(ascending=False).index]

    # Create stability score pivot table
    stability_pivot = metrics_subset.pivot_table(
        values='stability_score',
        index='model',
        columns='temperature',
        aggfunc='mean'
    ).round(4)
    stability_pivot = stability_pivot.reindex(cmean_pivot.index)

    # Create ranking score pivot table
    ranking_pivot = metrics_subset.pivot_table(
        values='ranking_score',
        index='model',
        columns='temperature',
        aggfunc='mean'
    ).round(4)
    ranking_pivot = ranking_pivot.reindex(cmean_pivot.index)

    # Create r_v (validity rate) pivot table - Section 3.7 metric
    rv_pivot = metrics_subset.pivot_table(
        values='r_v',
        index='model',
        columns='temperature',
        aggfunc='mean'
    ).round(4)
    rv_pivot = rv_pivot.reindex(cmean_pivot.index)

    # Calculate scalability metrics for each model
    scalability_results = []
    for model in cmean_pivot.index:
        model_data = metrics_subset[metrics_subset['model'] == model]

        # Get r_v (validity rate) at different temperatures - Section 3.7
        rv_t01 = model_data[model_data['temperature'] == 0.1]['r_v'].mean() if 0.1 in model_data['temperature'].values else np.nan
        rv_t05 = model_data[model_data['temperature'] == 0.5]['r_v'].mean() if 0.5 in model_data['temperature'].values else np.nan
        rv_t09 = model_data[model_data['temperature'] == 0.9]['r_v'].mean() if 0.9 in model_data['temperature'].values else np.nan
        rv_t10 = model_data[model_data['temperature'] == 1.0]['r_v'].mean() if 1.0 in model_data['temperature'].values else np.nan
        rv_mean = model_data['r_v'].mean()

        # Get C_mean at different temperatures
        cmean_t01 = model_data[model_data['temperature'] == 0.1]['c_mean'].mean() if 0.1 in model_data['temperature'].values else np.nan
        cmean_t05 = model_data[model_data['temperature'] == 0.5]['c_mean'].mean() if 0.5 in model_data['temperature'].values else np.nan
        cmean_t09 = model_data[model_data['temperature'] == 0.9]['c_mean'].mean() if 0.9 in model_data['temperature'].values else np.nan
        cmean_t10 = model_data[model_data['temperature'] == 1.0]['c_mean'].mean() if 1.0 in model_data['temperature'].values else np.nan
        cmean_mean = model_data['c_mean'].mean()

        # Get stability scores at different temperatures
        stability_t01 = model_data[model_data['temperature'] == 0.1]['stability_score'].mean() if 0.1 in model_data['temperature'].values else np.nan
        stability_t05 = model_data[model_data['temperature'] == 0.5]['stability_score'].mean() if 0.5 in model_data['temperature'].values else np.nan
        stability_t09 = model_data[model_data['temperature'] == 0.9]['stability_score'].mean() if 0.9 in model_data['temperature'].values else np.nan
        stability_t10 = model_data[model_data['temperature'] == 1.0]['stability_score'].mean() if 1.0 in model_data['temperature'].values else np.nan
        stability_mean = model_data['stability_score'].mean()

        # Get ranking scores at different temperatures
        ranking_t01 = model_data[model_data['temperature'] == 0.1]['ranking_score'].mean() if 0.1 in model_data['temperature'].values else np.nan
        ranking_t05 = model_data[model_data['temperature'] == 0.5]['ranking_score'].mean() if 0.5 in model_data['temperature'].values else np.nan
        ranking_t09 = model_data[model_data['temperature'] == 0.9]['ranking_score'].mean() if 0.9 in model_data['temperature'].values else np.nan
        ranking_t10 = model_data[model_data['temperature'] == 1.0]['ranking_score'].mean() if 1.0 in model_data['temperature'].values else np.nan
        ranking_mean = model_data['ranking_score'].mean()

        # Calculate scalability metrics
        retention_09_01 = cmean_t09 / cmean_t01 if cmean_t01 > 0 else np.nan
        retention_10_01 = cmean_t10 / cmean_t01 if cmean_t01 > 0 else np.nan
        decay_rate = (cmean_t01 - cmean_t09) / cmean_t01 if cmean_t01 > 0 else np.nan
        scalability_score = cmean_mean * retention_09_01 if not np.isnan(retention_09_01) else np.nan
        temp_cv = model_data.groupby('temperature')['c_mean'].mean().std() / cmean_mean if cmean_mean > 0 else np.nan
        temp_robustness = 1 - temp_cv if not np.isnan(temp_cv) else np.nan

        scalability_results.append({
            'model': model,
            'metric_type': metric_type_name,
            # Section 3.7 Core Metrics: r_v (validity rate)
            'r_v_T0.1': rv_t01,
            'r_v_T0.5': rv_t05,
            'r_v_T0.9': rv_t09,
            'r_v_T1.0': rv_t10,
            'r_v_avg': rv_mean,
            # Section 3.7 Core Metrics: C_mean (mean pairwise consistency)
            'c_mean_T0.1': cmean_t01,
            'c_mean_T0.5': cmean_t05,
            'c_mean_T0.9': cmean_t09,
            'c_mean_T1.0': cmean_t10,
            'c_mean_avg': cmean_mean,
            # Section 3.7 Core Metrics: S_alpha (stability score)
            'stability_T0.1': stability_t01,
            'stability_T0.5': stability_t05,
            'stability_T0.9': stability_t09,
            'stability_T1.0': stability_t10,
            'stability_avg': stability_mean,
            # Section 3.7 Core Metrics: R = r_v * C_mean * S_alpha (ranking score)
            'ranking_T0.1': ranking_t01,
            'ranking_T0.5': ranking_t05,
            'ranking_T0.9': ranking_t09,
            'ranking_T1.0': ranking_t10,
            'ranking_avg': ranking_mean,
            # Scalability metrics (temperature robustness analysis)
            'retention_ratio_0.9/0.1': retention_09_01,
            'retention_ratio_1.0/0.1': retention_10_01,
            'decay_rate': decay_rate,
            'scalability_score': scalability_score,
            'temp_robustness': temp_robustness
        })

    scalability_df = pd.DataFrame(scalability_results)
    scalability_df = scalability_df.sort_values('scalability_score', ascending=False)

    return cmean_pivot, stability_pivot, ranking_pivot, rv_pivot, scalability_df

# Process each metric type
all_scalability_results = []
for metric_type in ['Overall', 'Semantic', 'Structural']:
    print(f"\n{'='*80}")
    print(f"{metric_type.upper()} METRICS (Section 3.7 Consistency Aggregation)")
    print("="*80)

    metric_subset = metrics_df[metrics_df['metric_type'] == metric_type]

    cmean_pivot, stability_pivot, ranking_pivot, rv_pivot, scalability_df = calculate_metrics_by_type(
        metric_subset, metric_type
    )

    # Print summary tables (Section 3.7 metrics)
    print(f"\n=== {metric_type} r_v (Validity Rate) by Temperature ===")
    print(rv_pivot.to_string())

    print(f"\n=== {metric_type} C_mean (Mean Pairwise Consistency) by Temperature ===")
    print(cmean_pivot.to_string())

    print(f"\n=== {metric_type} S_alpha (Stability Score) by Temperature ===")
    print(stability_pivot.to_string())

    print(f"\n=== {metric_type} R (Ranking Score = r_v * C_mean * S_alpha) by Temperature ===")
    print(ranking_pivot.to_string())

    print(f"\n=== {metric_type} Section 3.7 Aggregation Summary ===")
    print(f"{'Model':<25} {'r_v(avg)':<10} {'C_mean(avg)':<12} {'S_alpha(avg)':<12} {'R(avg)':<12}")
    print("-" * 80)
    for _, row in scalability_df.iterrows():
        rv_avg = row.get('r_v_avg', np.nan)
        cmean_avg = row['c_mean_avg']
        stability_avg = row['stability_avg']
        ranking_avg = row['ranking_avg']
        print(f"{row['model']:<25} {rv_avg:.4f}    {cmean_avg:.4f}       {stability_avg:.4f}       {ranking_avg:.4f}")

    # Save to CSV
    metric_type_lower = metric_type.lower()
    rv_pivot.to_csv(os.path.join(args.output_dir, f'rv_by_temperature_{metric_type_lower}.csv'))
    cmean_pivot.to_csv(os.path.join(args.output_dir, f'Cmean_by_temperature_{metric_type_lower}.csv'))
    stability_pivot.to_csv(os.path.join(args.output_dir, f'S_alpha_by_temperature_{metric_type_lower}.csv'))
    ranking_pivot.to_csv(os.path.join(args.output_dir, f'R_by_temperature_{metric_type_lower}.csv'))
    scalability_df.to_csv(os.path.join(args.output_dir, f'scalability_metrics_{metric_type_lower}.csv'), index=False)

    print(f"\nSaved: rv_by_temperature_{metric_type_lower}.csv")
    print(f"Saved: Cmean_by_temperature_{metric_type_lower}.csv")
    print(f"Saved: S_alpha_by_temperature_{metric_type_lower}.csv")
    print(f"Saved: R_by_temperature_{metric_type_lower}.csv")
    print(f"Saved: scalability_metrics_{metric_type_lower}.csv")

    # Collect for combined analysis
    all_scalability_results.append(scalability_df)

    # Create visualization: C_mean bar chart for this metric type
    print(f"\nCreating {metric_type} C_mean bar chart...")
    cmean_avg = cmean_pivot.mean(axis=1).sort_values(ascending=True)
    n_models_cmean = len(cmean_avg)
    fig_height_cmean = max(6, n_models_cmean * 0.4 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height_cmean))

    colors = plt.cm.RdYlGn(cmean_avg.values / max(cmean_avg.max(), 0.01))
    bars = ax.barh(range(n_models_cmean), cmean_avg.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(n_models_cmean))
    ax.set_yticklabels([f'{i+1}. {m}' for i, m in enumerate(reversed(cmean_avg.index))], fontsize=11)
    ax.invert_yaxis()  # Top rank at top

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, cmean_avg.values)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('$C_{mean}$ (Average across temperatures)', fontsize=12)
    ax.set_title(f'{metric_type} $C_{{mean}}$ Model Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    cmean_bar_path = os.path.join(args.output_dir, f'Cmean_ranking_{metric_type_lower}.png')
    save_figure(fig, cmean_bar_path)
    plt.close(fig)

    # Create visualization: R (Ranking Score) bar chart for this metric type
    print(f"Creating {metric_type} R (Ranking Score) bar chart...")
    ranking_avg = ranking_pivot.mean(axis=1).sort_values(ascending=True)
    n_models_r = len(ranking_avg)
    fig_height_r = max(6, n_models_r * 0.4 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height_r))

    colors = plt.cm.RdYlGn(ranking_avg.values / max(ranking_avg.max(), 0.01))
    bars = ax.barh(range(n_models_r), ranking_avg.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(n_models_r))
    ax.set_yticklabels([f'{i+1}. {m}' for i, m in enumerate(reversed(ranking_avg.index))], fontsize=11)
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, ranking_avg.values)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('$R = r_v \\times C_{mean} \\times S_\\alpha$ (Average)', fontsize=12)
    ax.set_title(f'{metric_type} Ranking Score ($R$) Model Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    r_bar_path = os.path.join(args.output_dir, f'R_ranking_{metric_type_lower}.png')
    save_figure(fig, r_bar_path)
    plt.close(fig)

    # Create visualization: S_alpha (Stability Score) bar chart for this metric type
    print(f"Creating {metric_type} S_alpha (Stability Score) bar chart...")
    stability_avg = stability_pivot.mean(axis=1).sort_values(ascending=True)
    n_models_s = len(stability_avg)
    fig_height_s = max(6, n_models_s * 0.4 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height_s))

    colors = plt.cm.RdYlGn(stability_avg.values / max(stability_avg.max(), 0.01))
    bars = ax.barh(range(n_models_s), stability_avg.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(n_models_s))
    ax.set_yticklabels([f'{i+1}. {m}' for i, m in enumerate(reversed(stability_avg.index))], fontsize=11)
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, stability_avg.values)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('$S_\\alpha$ (Average across temperatures)', fontsize=12)
    ax.set_title(f'{metric_type} $S_\\alpha$ (Stability) Model Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    s_bar_path = os.path.join(args.output_dir, f'S_alpha_ranking_{metric_type_lower}.png')
    save_figure(fig, s_bar_path)
    plt.close(fig)

    # Create visualization: r_v (Validity Rate) bar chart for this metric type
    print(f"Creating {metric_type} r_v (Validity Rate) bar chart...")
    rv_avg = rv_pivot.mean(axis=1).sort_values(ascending=True)
    n_models_rv = len(rv_avg)
    fig_height_rv = max(6, n_models_rv * 0.4 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height_rv))

    colors = plt.cm.RdYlGn(rv_avg.values)
    bars = ax.barh(range(n_models_rv), rv_avg.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(n_models_rv))
    ax.set_yticklabels([f'{i+1}. {m}' for i, m in enumerate(reversed(rv_avg.index))], fontsize=11)
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, rv_avg.values)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.1%}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('$r_v$ (Validity Rate, Average)', fontsize=12)
    ax.set_title(f'{metric_type} $r_v$ (Validity) Model Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    rv_bar_path = os.path.join(args.output_dir, f'rv_ranking_{metric_type_lower}.png')
    save_figure(fig, rv_bar_path)
    plt.close(fig)

    print(f"Saved: Cmean_ranking_{metric_type_lower}.png")
    print(f"Saved: R_ranking_{metric_type_lower}.png")
    print(f"Saved: S_alpha_ranking_{metric_type_lower}.png")
    print(f"Saved: rv_ranking_{metric_type_lower}.png")

# Combine all scalability results
combined_scalability = pd.concat(all_scalability_results, ignore_index=True)
combined_scalability.to_csv(os.path.join(args.output_dir, 'scalability_metrics_all.csv'), index=False)
print(f"\nSaved: scalability_metrics_all.csv (all metric types combined)")

# Create comparison visualization: Scalability by metric type
print("\nCreating scalability comparison chart by metric type...")
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for idx, metric_type in enumerate(['Overall', 'Semantic', 'Structural']):
    ax = axes[idx]
    subset = combined_scalability[combined_scalability['metric_type'] == metric_type]
    subset_sorted = subset.sort_values('scalability_score', ascending=True)

    colors = plt.cm.RdYlGn(subset_sorted['scalability_score'].values / max(subset_sorted['scalability_score'].max(), 0.01))
    bars = ax.barh(subset_sorted['model'], subset_sorted['scalability_score'],
                   color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Scalability Score', fontsize=10)
    if idx == 0:
        ax.set_ylabel('Model', fontsize=10)
    ax.set_title(f'{metric_type}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1)

    # Add value labels
    for bar, val in zip(bars, subset_sorted['scalability_score']):
        if not np.isnan(val):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                   va='center', fontsize=7)

plt.suptitle('Scalability Score by Metric Type ($C_{mean}^{avg} \\times$ Retention Ratio)', fontsize=14, fontweight='bold')
plt.tight_layout()
scalability_comparison_path = os.path.join(args.output_dir, 'scalability_comparison_by_type.png')
save_figure(fig, scalability_comparison_path)
plt.close(fig)

# Create comparison visualization: Ranking score by metric type
print("Creating ranking score comparison chart by metric type...")
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for idx, metric_type in enumerate(['Overall', 'Semantic', 'Structural']):
    ax = axes[idx]
    subset = combined_scalability[combined_scalability['metric_type'] == metric_type]
    subset_sorted = subset.sort_values('ranking_avg', ascending=True)

    colors = plt.cm.RdYlGn(subset_sorted['ranking_avg'].values / max(subset_sorted['ranking_avg'].max(), 0.01))
    bars = ax.barh(subset_sorted['model'], subset_sorted['ranking_avg'],
                   color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Ranking Score (avg)', fontsize=10)
    if idx == 0:
        ax.set_ylabel('Model', fontsize=10)
    ax.set_title(f'{metric_type}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1)

    # Add value labels
    for bar, val in zip(bars, subset_sorted['ranking_avg']):
        if not np.isnan(val):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                   va='center', fontsize=7)

plt.suptitle('Ranking Score by Metric Type ($R = r_v \\times C_{mean} \\times S_\\alpha$)', fontsize=14, fontweight='bold')
plt.tight_layout()
ranking_comparison_path = os.path.join(args.output_dir, 'ranking_comparison_by_type.png')
save_figure(fig, ranking_comparison_path)
plt.close(fig)

print(f"\nSaved: scalability_comparison_by_type.png")
print(f"Saved: ranking_comparison_by_type.png")

# ============================================================================
# 8. Appendix Figures: Stability Distribution + Temperature Curves (SEPARATE)
# ============================================================================
print("\n8. Creating appendix stability figures (separate)...")

# Panel (a): Box plots of stability score by model
overall_df = df[df['metric_type'] == 'Overall']

# Get models sorted by median consistency score
model_medians = overall_df.groupby('model')['consistency_score'].median().sort_values(ascending=False)
sorted_models = model_medians.index.tolist()

# Apply top-N filter if specified
if args.top_n_boxplot and args.top_n_boxplot < len(sorted_models):
    sorted_models = sorted_models[:args.top_n_boxplot]
    print(f"  Showing top {args.top_n_boxplot} models in boxplot")

n_models_box = len(sorted_models)

# ============================================================================
# 8a. FIGURE 1: Box plots showing stability distribution for all models
# ============================================================================
print("  Creating stability distribution boxplot (all models)...")

# Calculate figure height based on number of models
fig_height_box = max(8, n_models_box * 0.45 + 2)

fig1, ax1 = plt.subplots(figsize=(10, fig_height_box))

# Create box plot data
box_data = [overall_df[overall_df['model'] == m]['consistency_score'].values for m in sorted_models]

# Create gradient colors (red to green based on rank)
gradient_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_models_box))

bp = ax1.boxplot(box_data, labels=sorted_models, patch_artist=True, vert=False,
                 showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5},
                 medianprops={'color': 'red', 'linewidth': 1.5})

# Color boxes with gradient
for patch, color in zip(bp['boxes'], gradient_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_ylabel('Model', fontsize=12)
ax1.set_xlabel('$S_\\alpha$ (Stability Score)', fontsize=12)
ax1.set_title('$S_\\alpha$ Distribution by Model (All Models)', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 1.05)
ax1.grid(True, alpha=0.3, axis='x')
ax1.tick_params(axis='y', labelsize=11)

plt.tight_layout()
boxplot_path = os.path.join(args.output_dir, 'appendix_stability_boxplot.png')
save_figure(fig1, boxplot_path)
plt.close(fig1)

# ============================================================================
# 8b. FIGURE 2: Temperature curves for ALL models
# ============================================================================
print("  Creating temperature curves (all models)...")

# Calculate mean consistency score by model and temperature
temp_curves = overall_df.groupby(['model', 'temperature']).agg({
    'consistency_score': 'mean'
}).reset_index()
temp_curves.columns = ['model', 'temperature', 'cs_mean']

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Use distinct colors for all models
all_model_colors = plt.cm.tab20(np.linspace(0, 1, n_models_box))
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P', '<', '>', '8', 'H', 'd', '1', '2', '3']

for i, model in enumerate(sorted_models):
    model_data = temp_curves[temp_curves['model'] == model].sort_values('temperature')
    color = all_model_colors[i]
    marker = markers[i % len(markers)]

    ax2.plot(model_data['temperature'], model_data['cs_mean'],
             marker=marker, label=model, linewidth=2,
             color=color, markersize=6, alpha=0.85)

ax2.set_xlabel('Temperature', fontsize=12)
ax2.set_ylabel('$S_\\alpha$ (Stability Score)', fontsize=12)
ax2.set_title('$S_\\alpha$ vs Temperature (All Models)', fontsize=13, fontweight='bold')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

# Place legend outside the plot
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)

plt.tight_layout()
temp_curves_path = os.path.join(args.output_dir, 'appendix_stability_temperature.png')
save_figure(fig2, temp_curves_path)
plt.close(fig2)

# Also create combined figure - HORIZONTAL LAYOUT (models in one row)
print("  Creating combined figure (horizontal layout)...")
fig_width_combined = max(16, n_models_box * 0.9 + 4)
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(fig_width_combined, 10), gridspec_kw={'height_ratios': [1, 1]})

# Panel (a): Box plots - VERTICAL (models on X-axis in one row)
box_data = [overall_df[overall_df['model'] == m]['consistency_score'].values for m in sorted_models]
bp3 = ax3a.boxplot(box_data, labels=sorted_models, patch_artist=True, vert=True,
                   showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5},
                   medianprops={'color': 'red', 'linewidth': 1.5})
for patch, color in zip(bp3['boxes'], gradient_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax3a.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3a.set_xlabel('Model', fontsize=12)
ax3a.set_ylabel('$S_\\alpha$ (Stability)', fontsize=12)
ax3a.set_title('(a) $S_\\alpha$ Distribution by Model (All Models)', fontsize=13, fontweight='bold')
ax3a.set_ylim(0, 1.05)
ax3a.grid(True, alpha=0.3, axis='y')
plt.setp(ax3a.get_xticklabels(), rotation=45, ha='right', fontsize=10)

# Panel (b): Temperature curves for all models
for i, model in enumerate(sorted_models):
    model_data = temp_curves[temp_curves['model'] == model].sort_values('temperature')
    color = all_model_colors[i]
    marker = markers[i % len(markers)]
    ax3b.plot(model_data['temperature'], model_data['cs_mean'],
             marker=marker, label=model, linewidth=2,
             color=color, markersize=6, alpha=0.85)

ax3b.set_xlabel('Temperature', fontsize=12)
ax3b.set_ylabel('$S_\\alpha$ (Stability)', fontsize=12)
ax3b.set_title('(b) $S_\\alpha$ vs Temperature (All Models)', fontsize=13, fontweight='bold')
ax3b.set_xlim(-0.05, 1.05)
ax3b.set_ylim(0, 1.05)
ax3b.grid(True, alpha=0.3)
ax3b.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)

plt.tight_layout()
appendix_stability_path = os.path.join(args.output_dir, 'appendix_stability_distribution.png')
save_figure(fig3, appendix_stability_path)
plt.close(fig3)

print("\n" + "="*80)
print("ENHANCED VISUALIZATIONS COMPLETE")
print("="*80)
print("\nAdditional files saved:")
print("- Cmean_vs_sigma_scatter.png (Mean-Variance Tradeoff)")
print("- radar_model_comparison.png (Multi-dimensional Comparison)")
print("- Cmean_boxplots.png (Consistency Distributions)")
print("- cadj_heatmap.png (Validity-Adjusted Consistency)")
print("- structural_vs_semantic_bars.png (Component Comparison)")
print("- pareto_frontier.png (Optimal Model Selection)")
print("- R_heatmap.png (Combined Ranking Score)")
print("- scalability_comparison_by_type.png (Scalability Score Comparison)")
print("- ranking_comparison_by_type.png (Ranking Score Comparison)")
print("- appendix_stability_distribution.png (Stability Distribution + Key Models)")
print("- cadj_by_temperature.csv")
print("- R_by_temperature.csv")
print("- scalability_metrics_all.csv (All metric types combined)")
print("\nPer-type files (for each of Overall, Semantic, Structural):")
print("- Cmean_ranking_{type}.png (C_mean Model Ranking Bar Chart)")
print("- R_ranking_{type}.png (Ranking Score Model Ranking Bar Chart)")
print("- S_alpha_ranking_{type}.png (Stability Score Model Ranking Bar Chart)")
print("- rv_ranking_{type}.png (Validity Rate Model Ranking Bar Chart)")
print("- rv_by_temperature_{type}.csv")
print("- Cmean_by_temperature_{type}.csv")
print("- S_alpha_by_temperature_{type}.csv")
print("- R_by_temperature_{type}.csv")
print("- scalability_metrics_{type}.csv")

# ============================================================================
# 9. Structural vs Semantic Temperature Behavior Analysis
# ============================================================================
# Expected behavior:
#   - Schema/Structural: Should be STABLE across all temperatures
#   - Content/Semantic: Should VARY more as temperature increases (expected)
# ============================================================================
print("\n" + "="*80)
print("STRUCTURAL VS SEMANTIC TEMPERATURE BEHAVIOR ANALYSIS")
print("="*80)
print("Expectation: Schema STABLE across temperatures, Content VARIES with temperature")

# Load structural and semantic scalability metrics
struct_path = os.path.join(args.output_dir, 'scalability_metrics_structural.csv')
sem_path = os.path.join(args.output_dir, 'scalability_metrics_semantic.csv')

if os.path.exists(struct_path) and os.path.exists(sem_path):
    struct_df = pd.read_csv(struct_path)
    sem_df = pd.read_csv(sem_path)

    # Calculate temperature drop for each model
    temp_behavior_data = []
    for _, row in struct_df.iterrows():
        model = row['model']
        sem_row = sem_df[sem_df['model'] == model]
        if len(sem_row) == 0:
            continue

        struct_t01 = row['stability_T0.1']
        struct_t10 = row['stability_T1.0']
        sem_t01 = sem_row['stability_T0.1'].values[0]
        sem_t10 = sem_row['stability_T1.0'].values[0]

        struct_drop = (struct_t01 - struct_t10) / struct_t01 * 100 if struct_t01 > 0 else 0
        sem_drop = (sem_t01 - sem_t10) / sem_t01 * 100 if sem_t01 > 0 else 0

        schema_stable = struct_drop < 15  # <15% drop considered stable

        temp_behavior_data.append({
            'model': model,
            'struct_t01': struct_t01,
            'struct_t10': struct_t10,
            'struct_drop_pct': struct_drop,
            'sem_t01': sem_t01,
            'sem_t10': sem_t10,
            'sem_drop_pct': sem_drop,
            'schema_stable': schema_stable
        })

    temp_behavior_df = pd.DataFrame(temp_behavior_data)
    temp_behavior_df = temp_behavior_df.sort_values('struct_drop_pct', ascending=True)

    # Print summary table
    print(f"\n{'Model':<22} | Struct Drop | Sem Drop | Schema Stable?")
    print("-"*65)
    for _, row in temp_behavior_df.iterrows():
        stable_str = 'YES' if row['schema_stable'] else 'NO'
        print(f"{row['model']:<22} | {row['struct_drop_pct']:5.0f}%     | {row['sem_drop_pct']:5.0f}%   | {stable_str}")

    # Save to CSV
    temp_behavior_path = os.path.join(args.output_dir, 'temperature_behavior_analysis.csv')
    temp_behavior_df.to_csv(temp_behavior_path, index=False)
    print(f"\nSaved: temperature_behavior_analysis.csv")

    # Create visualization: Structural vs Semantic Drop Scatter Plot
    print("\n9. Creating structural vs semantic temperature drop scatter plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by schema stability
    colors = ['green' if s else 'red' for s in temp_behavior_df['schema_stable']]

    scatter = ax.scatter(temp_behavior_df['struct_drop_pct'],
                        temp_behavior_df['sem_drop_pct'],
                        c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add model labels
    for _, row in temp_behavior_df.iterrows():
        ax.annotate(row['model'],
                   (row['struct_drop_pct'], row['sem_drop_pct']),
                   fontsize=8, alpha=0.9, ha='left', va='bottom',
                   xytext=(3, 3), textcoords='offset points')

    # Add reference lines
    ax.axvline(x=15, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Schema Stability Threshold (15%)')
    ax.axhline(y=30, color='blue', linestyle=':', alpha=0.5, linewidth=2, label='Expected Semantic Variation (30%)')

    # Add quadrant labels
    ax.annotate('IDEAL: Schema stable,\nContent varies',
               xy=(5, 50), fontsize=10, color='green', alpha=0.8,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
               ha='center')
    ax.annotate('PROBLEM: Schema\nunstable',
               xy=(35, 50), fontsize=10, color='red', alpha=0.8,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
               ha='center')

    ax.set_xlabel('Structural Stability Drop (T0.1 → T1.0) %', fontsize=12)
    ax.set_ylabel('Semantic Stability Drop (T0.1 → T1.0) %', fontsize=12)
    ax.set_title('Temperature Behavior: Schema vs Content Stability Drop', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, max(60, temp_behavior_df['struct_drop_pct'].max() + 5))
    ax.set_ylim(-5, max(80, temp_behavior_df['sem_drop_pct'].max() + 5))

    plt.tight_layout()
    temp_scatter_path = os.path.join(args.output_dir, 'temperature_behavior_scatter.png')
    save_figure(fig, temp_scatter_path)
    plt.close(fig)

    # Create bar chart: Side-by-side structural vs semantic drop
    print("Creating structural vs semantic drop comparison bar chart...")

    n_models = len(temp_behavior_df)
    fig_height = max(8, n_models * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_pos = np.arange(n_models)
    bar_height = 0.35

    # Sort by structural drop for better visualization
    temp_behavior_sorted = temp_behavior_df.sort_values('struct_drop_pct', ascending=False)

    bars1 = ax.barh(y_pos - bar_height/2, temp_behavior_sorted['struct_drop_pct'],
                   bar_height, label='Structural (Schema) Drop', color='steelblue', alpha=0.8)
    bars2 = ax.barh(y_pos + bar_height/2, temp_behavior_sorted['sem_drop_pct'],
                   bar_height, label='Semantic (Content) Drop', color='coral', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(temp_behavior_sorted['model'], fontsize=10)
    ax.axvline(x=15, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Schema Stability Threshold')
    ax.set_xlabel('Stability Drop (T0.1 → T1.0) %', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Structural vs Semantic Stability Drop by Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    temp_bar_path = os.path.join(args.output_dir, 'temperature_behavior_bars.png')
    save_figure(fig, temp_bar_path)
    plt.close(fig)

    # Summary statistics
    stable_models = temp_behavior_df[temp_behavior_df['schema_stable']]['model'].tolist()
    unstable_models = temp_behavior_df[~temp_behavior_df['schema_stable']]['model'].tolist()

    print(f"\n=== Summary ===")
    print(f"Schema Stable Models ({len(stable_models)}): {', '.join(stable_models)}")
    print(f"Schema Unstable Models ({len(unstable_models)}): {', '.join(unstable_models)}")
    print(f"\nAvg Structural Drop: {temp_behavior_df['struct_drop_pct'].mean():.1f}%")
    print(f"Avg Semantic Drop: {temp_behavior_df['sem_drop_pct'].mean():.1f}%")

    print(f"\nSaved: temperature_behavior_scatter.png")
    print(f"Saved: temperature_behavior_bars.png")
else:
    print("Skipping temperature behavior analysis - structural/semantic CSV files not found")

print("\n" + "="*80)
print("ALL ANALYSIS COMPLETE")
print("="*80)