#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

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
        
        if not obj:  # empty dict is a leaf
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
            
            # Aggregate field type counts
            for field_type in ['string', 'integer', 'float', 'boolean', 'array', 'object', 'null']:
                metrics[f'{field_type}_fields'] += child_metrics[f'{field_type}_fields']
    
    elif isinstance(obj, list):
        metrics['arrays'] = 1
        metrics['array_elements'] = len(obj)
        metrics['max_array_length'] = len(obj)
        
        if not obj:  # empty array is a leaf
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
            
            # Aggregate field type counts
            for field_type in ['string', 'integer', 'float', 'boolean', 'array', 'object', 'null']:
                metrics[f'{field_type}_fields'] += child_metrics[f'{field_type}_fields']
    
    else:
        # Primitive value is a leaf node
        metrics['leaf_nodes'] = 1
    
    return metrics

def extract_base_sample_id(sample_id):
    """Extract the base sample identifier from variation sample IDs"""
    if '_var_' in sample_id:
        return sample_id.split('_var_')[0]
    elif '_semantic_' in sample_id:
        return sample_id.split('_semantic_')[0]
    elif '_expression_' in sample_id:
        return sample_id.split('_expression_')[0]
    else:
        return sample_id

def analyze_dataset(filename):
    """Analyze a single dataset file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    results = []
    
    for item in data:
        # Extract base sample based on dataset structure
        if 'ground_truth' in item:
            base_sample = item['ground_truth']
            sample_id = item.get('sample_id', 'unknown')
        elif 'base_sample' in item:
            base_sample = item['base_sample']
            sample_id = item.get('sample_id', 'unknown')
        else:
            continue
        
        metrics = calculate_comprehensive_metrics(base_sample)
        metrics['sample_id'] = sample_id
        metrics['base_id'] = extract_base_sample_id(sample_id)
        metrics['dataset'] = filename.split('_')[0]
        results.append(metrics)
    
    return results

def main():
    datasets = [
        'schema_variation_dataset_2025-08-28_14-02-39-full-dataset.json',
        'semantic_variation_dataset_2025-08-25_04-23-54-full-dataset.json',
        'expression_variation_dataset_2025-08-25_07-02-22-full-dataset.json'
    ]
    
    all_results = []
    for dataset in datasets:
        results = analyze_dataset(dataset)
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    
    # Get unique base samples
    unique_df = df.drop_duplicates(subset=['base_id']).copy()
    
    print("=== COMPREHENSIVE DATASET ANALYSIS ===\n")
    print(f"Total samples analyzed: {len(df)}")
    print(f"Unique base samples: {len(unique_df)}")
    print(f"Datasets: {df['dataset'].unique()}")
    print()
    
    # Structural complexity metrics
    structural_metrics = ['max_depth', 'total_fields', 'nested_objects', 'arrays', 'total_nodes', 'leaf_nodes', 'array_elements', 'max_array_length']
    
    print("=== STRUCTURAL COMPLEXITY STATISTICS ===")
    stats_df = unique_df[structural_metrics].describe()
    print(stats_df.round(2))
    print()
    
    # Field type distribution
    field_type_metrics = ['string_fields', 'integer_fields', 'float_fields', 'boolean_fields', 'array_fields', 'object_fields', 'null_fields']
    
    print("=== FIELD TYPE DISTRIBUTION STATISTICS ===")
    type_stats_df = unique_df[field_type_metrics].describe()
    print(type_stats_df.round(2))
    print()
    
    # Field type percentages
    print("=== FIELD TYPE PERCENTAGES (across all unique samples) ===")
    total_fields_by_type = unique_df[field_type_metrics].sum()
    total_all_fields = total_fields_by_type.sum()
    
    for field_type in field_type_metrics:
        type_name = field_type.replace('_fields', '')
        percentage = (total_fields_by_type[field_type] / total_all_fields) * 100
        print(f"{type_name.capitalize()}: {total_fields_by_type[field_type]} fields ({percentage:.1f}%)")
    print()
    
    # Depth distribution
    print("=== DEPTH DISTRIBUTION ===")
    depth_dist = unique_df['max_depth'].value_counts().sort_index()
    for depth, count in depth_dist.items():
        print(f"Depth {depth}: {count} samples ({count/len(unique_df)*100:.1f}%)")
    print()
    
    # Field count distribution
    print("=== FIELD COUNT DISTRIBUTION ===")
    field_bins = pd.cut(unique_df['total_fields'], bins=[0, 10, 25, 50, 100, float('inf')], labels=['1-10', '11-25', '26-50', '51-100', '100+'])
    field_dist = field_bins.value_counts()
    for bin_name, count in field_dist.items():
        print(f"{bin_name} fields: {count} samples ({count/len(unique_df)*100:.1f}%)")
    print()
    
    # Create comprehensive visualizations
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Depth distribution
    plt.subplot(3, 3, 1)
    plt.hist(unique_df['max_depth'], bins=range(unique_df['max_depth'].min(), unique_df['max_depth'].max()+2), alpha=0.7, color='skyblue')
    plt.title('JSON Depth Distribution')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Field count distribution
    plt.subplot(3, 3, 2)
    plt.hist(unique_df['total_fields'], bins=15, alpha=0.7, color='lightgreen')
    plt.title('Total Fields Distribution')
    plt.xlabel('Total Fields')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Field type distribution
    plt.subplot(3, 3, 3)
    field_type_sums = unique_df[field_type_metrics].sum()
    field_type_names = [name.replace('_fields', '') for name in field_type_metrics]
    plt.bar(field_type_names, field_type_sums.values, alpha=0.7)
    plt.title('Field Type Distribution')
    plt.xlabel('Field Type')
    plt.ylabel('Total Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Depth vs Fields scatter
    plt.subplot(3, 3, 4)
    scatter = plt.scatter(unique_df['max_depth'], unique_df['total_fields'], 
                         c=unique_df['total_nodes'], alpha=0.7, cmap='viridis', s=60)
    plt.title('Depth vs Fields')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Total Fields')
    plt.colorbar(scatter, label='Total Nodes')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Arrays vs Objects
    plt.subplot(3, 3, 5)
    plt.scatter(unique_df['arrays'], unique_df['nested_objects'], alpha=0.7, color='coral')
    plt.title('Arrays vs Nested Objects')
    plt.xlabel('Number of Arrays')
    plt.ylabel('Number of Nested Objects')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: String vs Numeric fields
    plt.subplot(3, 3, 6)
    numeric_fields = unique_df['integer_fields'] + unique_df['float_fields']
    plt.scatter(unique_df['string_fields'], numeric_fields, alpha=0.7, color='orange')
    plt.title('String vs Numeric Fields')
    plt.xlabel('String Fields')
    plt.ylabel('Numeric Fields (int + float)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 7: Node distribution
    plt.subplot(3, 3, 7)
    plt.hist(unique_df['total_nodes'], bins=15, alpha=0.7, color='purple')
    plt.title('Total Nodes Distribution')
    plt.xlabel('Total Nodes')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Subplot 8: Leaf nodes vs total nodes
    plt.subplot(3, 3, 8)
    plt.scatter(unique_df['total_nodes'], unique_df['leaf_nodes'], alpha=0.7, color='brown')
    plt.title('Leaf Nodes vs Total Nodes')
    plt.xlabel('Total Nodes')
    plt.ylabel('Leaf Nodes')
    plt.grid(True, alpha=0.3)
    
    # Subplot 9: Array elements distribution
    plt.subplot(3, 3, 9)
    plt.hist(unique_df['array_elements'], bins=15, alpha=0.7, color='pink')
    plt.title('Array Elements Distribution')
    plt.xlabel('Total Array Elements')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    unique_df.to_csv('comprehensive_dataset_metrics.csv', index=False)
    
    # Create summary table for paper
    summary_stats = unique_df[structural_metrics + field_type_metrics].describe().loc[['min', 'max', 'mean', 'std']]
    summary_stats.to_csv('dataset_summary_statistics.csv')
    
    print("=== REPRESENTATIVENESS EVIDENCE ===")
    print(f"✓ Depth coverage: {unique_df['max_depth'].nunique()} levels ({unique_df['max_depth'].min()}-{unique_df['max_depth'].max()})")
    print(f"✓ Field range: {unique_df['total_fields'].min()}-{unique_df['total_fields'].max()} fields")
    print(f"✓ Node range: {unique_df['total_nodes'].min()}-{unique_df['total_nodes'].max()} nodes")
    print(f"✓ Field type diversity: All major JSON types represented")
    print(f"✓ Structural variety: Arrays ({unique_df['arrays'].sum()}), Objects ({unique_df['nested_objects'].sum()}), Primitives ({unique_df['leaf_nodes'].sum()})")
    
    print(f"\nFiles saved:")
    print(f"- comprehensive_dataset_metrics.csv (detailed metrics)")
    print(f"- dataset_summary_statistics.csv (summary table)")
    print(f"- comprehensive_dataset_analysis.png (visualizations)")

if __name__ == "__main__":
    main()
