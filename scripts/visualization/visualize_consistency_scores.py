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

args = parser.parse_args()

import os
os.makedirs(args.output_dir, exist_ok=True)

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
            all_results.append({
                'metric_type': metric_type,
                'model': model,
                'temperature': entry['temperature'],
                'normalized_cv': entry['normalized_cv'],
                'consistency_score': entry['stability_score']
            })

df = pd.DataFrame(all_results)

# Filter to only include temperatures 0.1 to 0.9 in 0.1 increments
valid_temps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
df = df[df['temperature'].isin(valid_temps)]

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric_type in enumerate(['Overall', 'Semantic', 'Structural']):
        ax = axes[i]
        
        for model in models:
            data_subset = detailed_stats[(detailed_stats['metric_type'] == metric_type) & 
                                       (detailed_stats['model'] == model)]
            
            means = data_subset['cs_mean']
            errors = data_subset['cs_se']
            
            ax.errorbar(data_subset['temperature'], means, yerr=errors,
                       marker='o', label=model, linewidth=2, capsize=3, capthick=1,
                       color=model_colors[model])
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Consistency Score')
        ax.set_title(f'{metric_type} Consistency')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f'{metric}_by_consistency_type_with_errors.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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

print(f"\nStatistical analysis complete. Files saved:")
print("- detailed_consistency_statistics.csv")
print("- enhanced_consistency_summary.csv")
print("- consistency_score_by_consistency_type_with_errors.png")