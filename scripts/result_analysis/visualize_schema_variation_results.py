#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def visualize_schema_results():
    """Visualize schema variation analysis results with statistical analysis"""
    
    with open('schema_variation_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    methods = ['ted', 'sted', 'bertscore', 'deepdiff']
    method_labels = ['TED', 'STED', 'BERTSCORE', 'DEEPDIFF']
    method_legends = ['TED', 'STED', 'BERTSCORE', 'DEEPDIFF']
    colors = ['blue', 'red', 'green', 'orange']
    
    # Function to calculate confidence intervals
    def calculate_ci(data, confidence=0.95):
        if len(data) <= 1:
            return 0, 0
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return mean - h, mean + h
    
    # Function to get raw data for statistical analysis
    def get_raw_data(data_section, method, ratio_idx=None):
        if 'raw_data' in data_section and method in data_section['raw_data']:
            if ratio_idx is not None:
                # For field name change progression
                if isinstance(data_section['raw_data'][method], dict):
                    ratio_key = str(results['field_name_change']['ratios'][ratio_idx])
                    return data_section['raw_data'][method].get(ratio_key, [])
                elif isinstance(data_section['raw_data'][method], list) and ratio_idx < len(data_section['raw_data'][method]):
                    return data_section['raw_data'][method][ratio_idx]
            else:
                # For flat/nested scenarios
                return data_section['raw_data'][method]
        return []
    
    # Create visualization with error bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot field_name_change progression with error bars
    if 'field_name_change' in results:
        ax = axes[0]
        ratios = results['field_name_change']['ratios']
        
        for i, method in enumerate(methods):
            means = results['field_name_change']['averages'][method]
            
            # Calculate error bars
            errors = []
            for ratio_idx in range(len(ratios)):
                raw_values = get_raw_data(results['field_name_change'], method, ratio_idx)
                if len(raw_values) > 1:
                    errors.append(stats.sem(raw_values))
                else:
                    errors.append(0)
            
            ax.errorbar(ratios, means, yerr=errors, marker='o', color=colors[i], 
                       label=method_legends[i], linewidth=2, capsize=3, capthick=1)
        
        ax.set_xlabel('Variation Ratio')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Field Name Change Progression (with Error Bars)', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Plot flat_structure comparison with error bars
    if 'flat_structure' in results:
        ax = axes[1]
        available_methods = [m for m in methods if m in results['flat_structure']['average']]
        method_names = [method_labels[methods.index(method)] for method in available_methods]
        scores = [results['flat_structure']['average'][method] for method in available_methods]
        
        # Calculate error bars
        errors = []
        for method in available_methods:
            raw_values = get_raw_data(results['flat_structure'], method)
            if len(raw_values) > 1:
                errors.append(stats.sem(raw_values))
            else:
                errors.append(0)
        
        bars = ax.bar(method_names, scores, yerr=errors, color=colors[:len(method_names)], 
                     capsize=3)
        ax.set_ylabel('Similarity Score')
        ax.set_title('Flat Structure Similarity (with Error Bars)', pad=20)
        ax.set_ylim(0, 1)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom')
    
    # Plot nested_change comparison with error bars
    if 'nested_change' in results:
        ax = axes[2]
        available_methods = [m for m in methods if m in results['nested_change']['average']]
        method_names = [method_labels[methods.index(method)] for method in available_methods]
        scores = [results['nested_change']['average'][method] for method in available_methods]
        
        # Calculate error bars
        errors = []
        for method in available_methods:
            raw_values = get_raw_data(results['nested_change'], method)
            if len(raw_values) > 1:
                errors.append(stats.sem(raw_values))
            else:
                errors.append(0)
        
        bars = ax.bar(method_names, scores, yerr=errors, color=colors[:len(method_names)], 
                     capsize=3)
        ax.set_ylabel('Similarity Score')
        ax.set_title('Nested Change Similarity (with Error Bars)', pad=20)
        ax.set_ylim(0, 1)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('schema_variation_analysis_with_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical Analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    # Perform significance tests
    def perform_significance_tests():
        print("\nStatistical Significance Tests (STED vs Other Methods):")
        print("-" * 60)
        
        # Field name change tests
        if 'field_name_change' in results:
            print("\nField Name Change Progression:")
            ratios = results['field_name_change']['ratios']
            
            for ratio_idx, ratio in enumerate(ratios):
                print(f"\nRatio {ratio}:")
                sted_data = get_raw_data(results['field_name_change'], 'sted', ratio_idx)
                
                for method in ['ted', 'bertscore', 'deepdiff']:
                    other_data = get_raw_data(results['field_name_change'], method, ratio_idx)
                    
                    if len(sted_data) > 1 and len(other_data) > 1:
                        try:
                            statistic, p_value = stats.mannwhitneyu(sted_data, other_data, alternative='two-sided')
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                            print(f"  STED vs {method.upper()}: p={p_value:.4f} {significance}")
                        except:
                            print(f"  STED vs {method.upper()}: Test failed")
                    else:
                        print(f"  STED vs {method.upper()}: Insufficient data")
        
        # Scenario-based tests
        for scenario in ['flat_structure', 'nested_change']:
            if scenario in results:
                print(f"\n{scenario.replace('_', ' ').title()}:")
                sted_data = get_raw_data(results[scenario], 'sted')
                
                for method in ['ted', 'bertscore', 'deepdiff']:
                    if method in results[scenario]['average']:
                        other_data = get_raw_data(results[scenario], method)
                        
                        if len(sted_data) > 1 and len(other_data) > 1:
                            try:
                                statistic, p_value = stats.mannwhitneyu(sted_data, other_data, alternative='two-sided')
                                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                                print(f"  STED vs {method.upper()}: p={p_value:.4f} {significance}")
                            except:
                                print(f"  STED vs {method.upper()}: Test failed")
                        else:
                            print(f"  STED vs {method.upper()}: Insufficient data")
    
    perform_significance_tests()
    
    # Summary Statistics with Confidence Intervals
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Mean ± Std, 95% CI)")
    print("="*80)
    
    summary_data = []
    
    # Field name change overall statistics
    if 'field_name_change' in results:
        print("\nField Name Change (All Ratios Combined):")
        print("Method\t\tMean ± Std\t\t95% CI\t\t\tN")
        print("-" * 70)
        
        for method in methods:
            all_values = []
            ratios = results['field_name_change']['ratios']
            
            for ratio_idx in range(len(ratios)):
                raw_values = get_raw_data(results['field_name_change'], method, ratio_idx)
                all_values.extend(raw_values)
            
            if all_values:
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                ci_low, ci_high = calculate_ci(all_values)
                
                print(f"{method.upper()}\t\t{mean_val:.3f} ± {std_val:.3f}\t\t[{ci_low:.3f}, {ci_high:.3f}]\t{len(all_values)}")
                
                summary_data.append({
                    'Scenario': 'Field Name Change',
                    'Method': method.upper(),
                    'Mean': mean_val,
                    'Std': std_val,
                    'CI_Low': ci_low,
                    'CI_High': ci_high,
                    'N_Samples': len(all_values)
                })
    
    # Scenario-specific statistics
    for scenario in ['flat_structure', 'nested_change']:
        if scenario in results:
            print(f"\n{scenario.replace('_', ' ').title()}:")
            print("Method\t\tMean ± Std\t\t95% CI\t\t\tN")
            print("-" * 70)
            
            for method in methods:
                if method in results[scenario]['average']:
                    raw_values = get_raw_data(results[scenario], method)
                    
                    if raw_values:
                        mean_val = np.mean(raw_values)
                        std_val = np.std(raw_values)
                        ci_low, ci_high = calculate_ci(raw_values)
                        
                        print(f"{method.upper()}\t\t{mean_val:.3f} ± {std_val:.3f}\t\t[{ci_low:.3f}, {ci_high:.3f}]\t{len(raw_values)}")
                        
                        summary_data.append({
                            'Scenario': scenario.replace('_', ' ').title(),
                            'Method': method.upper(),
                            'Mean': mean_val,
                            'Std': std_val,
                            'CI_Low': ci_low,
                            'CI_High': ci_high,
                            'N_Samples': len(raw_values)
                        })
                    else:
                        avg_val = results[scenario]['average'][method]
                        print(f"{method.upper()}\t\t{avg_val:.3f} ± N/A\t\t[N/A, N/A]\t\t1")
    
    # Save summary statistics
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('schema_variation_detailed_statistics.csv', index=False)
        print(f"\nDetailed statistics saved to 'schema_variation_detailed_statistics.csv'")
    
    # LaTeX table with confidence intervals
    print("\n" + "="*80)
    print("LATEX TABLE FOR PUBLICATION")
    print("="*80)
    
    print("\n% Schema Variation Results with Confidence Intervals")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Scenario & TED & STED & BERTScore & DeepDiff \\\\")
    print("\\hline")
    
    scenarios = [
        ('Field Change (50\\%)', 'field_name_change', 4),  # 50% ratio index
        ('Flat Structure', 'flat_structure', None),
        ('Nested Change', 'nested_change', None)
    ]
    
    for scenario_name, scenario_key, ratio_idx in scenarios:
        if scenario_key in results:
            row_values = []
            for method in methods:
                if scenario_key == 'field_name_change':
                    if ratio_idx is not None:
                        value = results[scenario_key]['averages'][method][ratio_idx]
                        raw_values = get_raw_data(results[scenario_key], method, ratio_idx)
                    else:
                        value = 0
                        raw_values = []
                else:
                    if method in results[scenario_key]['average']:
                        value = results[scenario_key]['average'][method]
                        raw_values = get_raw_data(results[scenario_key], method)
                    else:
                        value = 0
                        raw_values = []
                
                if len(raw_values) > 1:
                    ci_low, ci_high = calculate_ci(raw_values)
                    row_values.append(f"{value:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
                else:
                    row_values.append(f"{value:.3f}")
            
            print(f"{scenario_name} & " + " & ".join(row_values) + " \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    
    # Print original results for comparison
    print("\n" + "="*80)
    print("ORIGINAL RESULTS (for reference)")
    print("="*80)
    
    if 'field_name_change' in results:
        print("\nField Name Change Progression:")
        print(f"{'Ratio':<6} {'TED':<8} {'STED':<8} {'BERT':<8} {'DEEP':<8}")
        print("-" * 42)
        for i, ratio in enumerate(results['field_name_change']['ratios']):
            row = f"{ratio:<6}"
            for method in methods:
                row += f" {results['field_name_change']['averages'][method][i]:<7.3f}"
            print(row)
    
    for var_type in ['flat_structure', 'nested_change']:
        if var_type in results:
            print(f"\n{var_type.replace('_', ' ').title()} Average Similarities:")
            for method in methods:
                if method in results[var_type]['average']:
                    score = results[var_type]['average'][method]
                    print(f"  {method.upper()}: {score:.3f}")
    
    print(f"\nAnalysis complete. Files saved:")
    print("- schema_variation_analysis_with_errors.png")
    if summary_data:
        print("- schema_variation_detailed_statistics.csv")

if __name__ == "__main__":
    visualize_schema_results()
