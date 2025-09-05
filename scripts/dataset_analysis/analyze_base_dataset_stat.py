#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_field_types(obj, type_counts=None):
    """Recursively analyze field types in JSON object"""
    if type_counts is None:
        type_counts = Counter()
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                type_counts['string'] += 1
            elif isinstance(value, int):
                type_counts['integer'] += 1
            elif isinstance(value, float):
                type_counts['float'] += 1
            elif isinstance(value, bool):
                type_counts['boolean'] += 1
            elif isinstance(value, list):
                type_counts['array'] += 1
                for item in value:
                    analyze_field_types(item, type_counts)
            elif isinstance(value, dict):
                type_counts['object'] += 1
                analyze_field_types(value, type_counts)
            elif value is None:
                type_counts['null'] += 1
    elif isinstance(obj, list):
        for item in obj:
            analyze_field_types(item, type_counts)
    
    return type_counts

def calculate_comprehensive_metrics(obj, depth=0):
    """Calculate comprehensive complexity metrics for a JSON object"""
    metrics = {
        'max_depth': depth,
        'total_fields': 0,
        'nested_objects': 0,
        'arrays': 0,
        'total_nodes': 1,
        'leaf_nodes': 0,
        'array_elements': 0,
        'max_array_length': 0
    }
    
    # Get field type distribution
    type_counts = analyze_field_types(obj)
    for field_type in ['string', 'integer', 'float', 'boolean', 'array', 'object', 'null']:
        metrics[f'{field_type}_fields'] = type_counts.get(field_type, 0)
    
    if isinstance(obj, dict):
        metrics['total_fields'] = len(obj)
        if depth > 0:
            metrics['nested_objects'] = 1
        
        if not obj:
            metrics['leaf_nodes'] = 1
        
        for value in obj.values():
            child_metrics = calculate_comprehensive_metrics(value, depth + 1)
            metrics['max_depth'] = max(metrics['max_depth'], child_metrics['max_depth'])
            metrics['total_fields'] += child_metrics['total_fields']
            metrics['nested_objects'] += child_metrics['nested_objects']
            metrics['arrays'] += child_metrics['arrays']
            metrics['total_nodes'] += child_metrics['total_nodes']
            metrics['leaf_nodes'] += child_metrics['leaf_nodes']
            metrics['array_elements'] += child_metrics['array_elements']
            metrics['max_array_length'] = max(metrics['max_array_length'], child_metrics['max_array_length'])
            
            for field_type in ['string', 'integer', 'float', 'boolean', 'array', 'object', 'null']:
                metrics[f'{field_type}_fields'] += child_metrics[f'{field_type}_fields']
    
    elif isinstance(obj, list):
        metrics['arrays'] = 1
        metrics['array_elements'] = len(obj)
        metrics['max_array_length'] = len(obj)
        
        if not obj:
            metrics['leaf_nodes'] = 1
        
        for item in obj:
            child_metrics = calculate_comprehensive_metrics(item, depth + 1)
            metrics['max_depth'] = max(metrics['max_depth'], child_metrics['max_depth'])
            metrics['total_fields'] += child_metrics['total_fields']
            metrics['nested_objects'] += child_metrics['nested_objects']
            metrics['arrays'] += child_metrics['arrays']
            metrics['total_nodes'] += child_metrics['total_nodes']
            metrics['leaf_nodes'] += child_metrics['leaf_nodes']
            metrics['array_elements'] += child_metrics['array_elements']
            metrics['max_array_length'] = max(metrics['max_array_length'], child_metrics['max_array_length'])
            
            for field_type in ['string', 'integer', 'float', 'boolean', 'array', 'object', 'null']:
                metrics[f'{field_type}_fields'] += child_metrics[f'{field_type}_fields']
    
    else:
        metrics['leaf_nodes'] = 1
    
    return metrics

def main():
    # Load schema dataset and extract the 75 "flat" samples as base samples
    with open('schema_variation_dataset_2025-08-28_14-02-39-full-dataset.json', 'r') as f:
        schema_data = json.load(f)
    
    # Extract only the flat structure samples (these are the 75 base samples)
    base_samples = []
    for item in schema_data:
        if item.get('variation_type') == 'flat_structure':
            sample_id = item.get('sample_id', 'unknown')
            base_sample = item['ground_truth']
            
            metrics = calculate_comprehensive_metrics(base_sample)
            metrics['sample_id'] = sample_id
            metrics['base_id'] = sample_id  # These are the base samples
            base_samples.append(metrics)
    
    df = pd.DataFrame(base_samples)
    
    print("=== ANALYSIS OF 75 BASE SAMPLES ===\n")
    print(f"Total base samples: {len(df)}")
    print()
    
    # Structural complexity metrics
    structural_metrics = ['max_depth', 'total_fields', 'nested_objects', 'arrays', 'total_nodes', 'leaf_nodes', 'array_elements', 'max_array_length']
    
    print("=== STRUCTURAL COMPLEXITY STATISTICS ===")
    stats_df = df[structural_metrics].describe()
    print(stats_df.round(2))
    print()
    
    # Field type distribution
    field_type_metrics = ['string_fields', 'integer_fields', 'float_fields', 'boolean_fields', 'array_fields', 'object_fields', 'null_fields']
    
    print("=== FIELD TYPE DISTRIBUTION STATISTICS ===")
    type_stats_df = df[field_type_metrics].describe()
    print(type_stats_df.round(2))
    print()
    
    # Field type percentages
    print("=== FIELD TYPE PERCENTAGES (across 75 base samples) ===")
    total_fields_by_type = df[field_type_metrics].sum()
    total_all_fields = total_fields_by_type.sum()
    
    for field_type in field_type_metrics:
        type_name = field_type.replace('_fields', '')
        percentage = (total_fields_by_type[field_type] / total_all_fields) * 100
        print(f"{type_name.capitalize()}: {total_fields_by_type[field_type]} fields ({percentage:.1f}%)")
    print()
    
    # Depth distribution
    print("=== DEPTH DISTRIBUTION ===")
    depth_dist = df['max_depth'].value_counts().sort_index()
    for depth, count in depth_dist.items():
        print(f"Depth {depth}: {count} samples ({count/len(df)*100:.1f}%)")
    print()
    
    # Field count distribution
    print("=== FIELD COUNT DISTRIBUTION ===")
    field_bins = pd.cut(df['total_fields'], bins=[0, 10, 25, 50, 100, float('inf')], labels=['1-10', '11-25', '26-50', '51-100', '100+'])
    field_dist = field_bins.value_counts()
    for bin_name, count in field_dist.items():
        print(f"{bin_name} fields: {count} samples ({count/len(df)*100:.1f}%)")
    print()
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('75 Base Samples - Comprehensive Analysis', fontsize=16)
    
    # Subplot 1: Depth distribution
    plt.subplot(3, 3, 1)
    plt.hist(df['max_depth'], bins=range(df['max_depth'].min(), df['max_depth'].max()+2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('JSON Depth Distribution')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Field count distribution
    plt.subplot(3, 3, 2)
    plt.hist(df['total_fields'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Total Fields Distribution')
    plt.xlabel('Total Fields')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Field type distribution
    plt.subplot(3, 3, 3)
    field_type_sums = df[field_type_metrics].sum()
    field_type_names = [name.replace('_fields', '').capitalize() for name in field_type_metrics]
    colors = plt.cm.Set3(np.linspace(0, 1, len(field_type_names)))
    bars = plt.bar(field_type_names, field_type_sums.values, alpha=0.8, color=colors, edgecolor='black')
    plt.title('Field Type Distribution')
    plt.xlabel('Field Type')
    plt.ylabel('Total Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Subplot 4: Depth vs Fields scatter
    plt.subplot(3, 3, 4)
    scatter = plt.scatter(df['max_depth'], df['total_fields'], 
                         c=df['total_nodes'], alpha=0.7, cmap='viridis', s=60, edgecolors='black')
    plt.title('Depth vs Fields (colored by total nodes)')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Total Fields')
    plt.colorbar(scatter, label='Total Nodes')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Arrays vs Objects
    plt.subplot(3, 3, 5)
    plt.scatter(df['arrays'], df['nested_objects'], alpha=0.7, color='coral', s=60, edgecolors='black')
    plt.title('Arrays vs Nested Objects')
    plt.xlabel('Number of Arrays')
    plt.ylabel('Number of Nested Objects')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: String vs Numeric fields
    plt.subplot(3, 3, 6)
    numeric_fields = df['integer_fields'] + df['float_fields']
    plt.scatter(df['string_fields'], numeric_fields, alpha=0.7, color='orange', s=60, edgecolors='black')
    plt.title('String vs Numeric Fields')
    plt.xlabel('String Fields')
    plt.ylabel('Numeric Fields (int + float)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 7: Node distribution
    plt.subplot(3, 3, 7)
    plt.hist(df['total_nodes'], bins=15, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Total Nodes Distribution')
    plt.xlabel('Total Nodes')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Subplot 8: Leaf nodes vs total nodes
    plt.subplot(3, 3, 8)
    plt.scatter(df['total_nodes'], df['leaf_nodes'], alpha=0.7, color='brown', s=60, edgecolors='black')
    plt.title('Leaf Nodes vs Total Nodes')
    plt.xlabel('Total Nodes')
    plt.ylabel('Leaf Nodes')
    plt.grid(True, alpha=0.3)
    
    # Subplot 9: Array elements distribution
    plt.subplot(3, 3, 9)
    plt.hist(df['array_elements'], bins=15, alpha=0.7, color='pink', edgecolor='black')
    plt.title('Array Elements Distribution')
    plt.xlabel('Total Array Elements')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('75_base_samples_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    df.to_csv('75_base_samples_metrics.csv', index=False)
    
    # Create summary table for paper
    summary_stats = df[structural_metrics + field_type_metrics].describe().loc[['min', 'max', 'mean', 'std']]
    summary_stats.to_csv('75_samples_summary_statistics.csv')
    
    print("=== REPRESENTATIVENESS EVIDENCE FOR 75 BASE SAMPLES ===")
    print(f"✓ Depth coverage: {df['max_depth'].nunique()} levels ({df['max_depth'].min()}-{df['max_depth'].max()})")
    print(f"✓ Field range: {df['total_fields'].min()}-{df['total_fields'].max()} fields")
    print(f"✓ Node range: {df['total_nodes'].min()}-{df['total_nodes'].max()} nodes")
    print(f"✓ Field type diversity: Strings ({total_fields_by_type['string_fields']}), Integers ({total_fields_by_type['integer_fields']}), Arrays ({total_fields_by_type['array_fields']}), Objects ({total_fields_by_type['object_fields']})")
    print(f"✓ Structural variety: {df['arrays'].sum()} arrays, {df['nested_objects'].sum()} nested objects, {df['leaf_nodes'].sum()} leaf nodes")
    print(f"✓ Array complexity: Max array length {df['max_array_length'].max()}, Total array elements {df['array_elements'].sum()}")
    
    print(f"\nFiles saved:")
    print(f"- 75_base_samples_metrics.csv (detailed metrics for each sample)")
    print(f"- 75_samples_summary_statistics.csv (summary statistics table)")
    print(f"- 75_base_samples_analysis.png (comprehensive visualizations)")
    
    # Create a summary table for the paper
    print(f"\n=== SUMMARY TABLE FOR PAPER ===")
    print("Metric | Min | Max | Mean | Std")
    print("-------|-----|-----|------|----")
    for metric in ['max_depth', 'total_fields', 'total_nodes', 'arrays', 'nested_objects']:
        min_val = df[metric].min()
        max_val = df[metric].max()
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"{metric.replace('_', ' ').title()} | {min_val} | {max_val} | {mean_val:.1f} | {std_val:.1f}")

if __name__ == "__main__":
    main()
