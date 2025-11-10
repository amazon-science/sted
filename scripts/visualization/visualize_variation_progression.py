import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Visualize variation progression results')
    parser.add_argument('--semantic', default='experiments/experiment-1/semantic_variation_progression_results.json', 
                        help='Path to semantic variation results JSON')
    parser.add_argument('--expression', default='experiments/experiment-1/expression_variation_progression_results.json',
                        help='Path to expression variation results JSON')
    parser.add_argument('--output-dir', default='results', help='Directory to save output files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    with open(args.semantic, 'r') as f:
        semantic_data = json.load(f)
    
    with open(args.expression, 'r') as f:
        expression_data = json.load(f)
    
    # Extract data
    variation_ratios = semantic_data['variation_ratios']
    semantic_similarities = semantic_data['average_similarities']
    expression_similarities = expression_data['average_similarities']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Similarity Progression: Semantic vs Expression Variations', fontsize=16, fontweight='bold')
    
    # Metrics to plot
    metrics = ['ted', 'sted', 'bertscore', 'deepdiff', 'gnn']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Plot semantic variation
        ax.plot(variation_ratios, semantic_similarities[metric], 
                marker='o', linewidth=2, markersize=6, 
                color=colors[i], label='Semantic', alpha=0.8)
        
        # Plot expression variation
        ax.plot(variation_ratios, expression_similarities[metric], 
                marker='s', linewidth=2, markersize=6, 
                color=colors[i], linestyle='--', label='Expression', alpha=0.8)
        
        ax.set_title(f'{metric.upper()} Similarity', fontweight='bold')
        ax.set_xlabel('Variation Ratio')
        ax.set_ylabel('Similarity Score')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0.1, 1.0)
        
        # Set y-axis limits based on data range for better visibility
        if metric == 'deepdiff':
            ax.set_ylim(0.4, 1.0)
        elif metric == 'gnn':
            ax.set_ylim(0.986, 1.0)
        else:
            ax.set_ylim(0.9, 1.0)
    
    # Hide the empty subplot
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, 'similarity_progression_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

# Print summary statistics
print("Summary Statistics:")
print("==================")
print("\nSemantic Variations:")
for metric in ['sted', 'bertscore', 'deepdiff', 'gnn']:
    values = semantic_similarities[metric]
    print(f"{metric.upper()}: {values[0]:.4f} → {values[-1]:.4f} (drop: {values[0] - values[-1]:.4f})")

print("\nExpression Variations:")
for metric in ['sted', 'bertscore', 'deepdiff', 'gnn']:
    values = expression_similarities[metric]
    print(f"{metric.upper()}: {values[0]:.4f} → {values[-1]:.4f} (drop: {values[0] - values[-1]:.4f})")
