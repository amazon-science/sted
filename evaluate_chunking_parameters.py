#!/usr/bin/env python3
"""
Script to evaluate the impact of chunk_size and overlap parameters when using LangChain splitter
for long string comparison in semantic JSON tree consistency evaluation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import pandas as pd
from itertools import product
import time
from semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

def create_test_data() -> List[Dict[str, Any]]:
    """Create test JSON objects with varying lengths of text content."""
    
    # Short text samples
    short_text1 = "This is a brief description of the product."
    short_text2 = "A short product description."
    
    # Medium text samples
    medium_text1 = """
    This product is designed for professionals who need reliable performance in demanding environments. 
    It features advanced technology that ensures consistent results across various applications. 
    The ergonomic design makes it comfortable to use for extended periods, while the durable construction 
    guarantees long-lasting performance. Users appreciate its intuitive interface and comprehensive feature set.
    """
    
    medium_text2 = """
    Engineered for professional use, this item delivers dependable performance in challenging conditions.
    It incorporates cutting-edge technology to provide consistent outcomes across different use cases.
    The user-friendly design ensures comfort during prolonged usage, and the robust build quality
    promises durability. Customers value its straightforward interface and complete range of features.
    """
    
    # Long text samples
    long_text1 = """
    In the rapidly evolving landscape of modern technology, artificial intelligence has emerged as one of the most 
    transformative forces of our time. From machine learning algorithms that can predict consumer behavior to 
    natural language processing systems that can understand and generate human-like text, AI is revolutionizing 
    industries across the board. Healthcare professionals are using AI to diagnose diseases more accurately and 
    develop personalized treatment plans. Financial institutions leverage AI for fraud detection and algorithmic 
    trading. Transportation companies are developing autonomous vehicles that promise to make roads safer and 
    more efficient. The entertainment industry uses AI for content recommendation and even content creation.
    
    However, with great power comes great responsibility. As AI systems become more sophisticated and ubiquitous, 
    we must carefully consider the ethical implications of their deployment. Issues such as algorithmic bias, 
    privacy concerns, and the potential for job displacement require thoughtful consideration and proactive 
    solutions. Researchers and policymakers are working together to establish guidelines and regulations that 
    ensure AI development proceeds in a manner that benefits society as a whole.
    
    The future of AI holds immense promise. Advances in quantum computing may unlock new possibilities for 
    AI algorithms, while improvements in hardware efficiency could make AI more accessible to smaller 
    organizations and developing countries. As we continue to push the boundaries of what's possible with 
    artificial intelligence, we must remain committed to developing these technologies responsibly and 
    ensuring they serve the greater good of humanity.
    """
    
    long_text2 = """
    Artificial intelligence represents a paradigm shift in how we approach complex problems and automate 
    decision-making processes. The field has witnessed remarkable progress in recent years, with deep learning 
    models achieving superhuman performance in tasks ranging from image recognition to strategic game playing. 
    These advances have practical applications across numerous sectors. In medicine, AI assists in medical 
    imaging analysis and drug discovery. The finance sector employs AI for risk assessment and automated 
    trading strategies. Automotive companies are investing heavily in self-driving technology that could 
    transform transportation. Media and entertainment platforms use AI to personalize user experiences and 
    optimize content delivery.
    
    Despite these exciting developments, the widespread adoption of AI technologies raises important questions 
    about fairness, transparency, and accountability. Biased training data can lead to discriminatory outcomes, 
    while the complexity of modern AI systems can make their decision-making processes opaque. The potential 
    impact on employment markets and economic inequality also demands careful attention from researchers, 
    industry leaders, and government officials.
    
    Looking ahead, the trajectory of AI development suggests even more profound changes on the horizon. 
    Emerging technologies like neuromorphic computing and advanced neural architectures may enable new 
    breakthroughs in artificial general intelligence. As these technologies mature, it will be crucial to 
    maintain a balance between innovation and responsible development, ensuring that the benefits of AI 
    are distributed equitably across society.
    """
    
    # Code samples
    code_text1 = """
    def calculate_similarity(text1, text2):
        # Tokenize the texts
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        # Calculate Jaccard similarity
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if len(union) == 0:
            return 0.0
        
        jaccard_sim = len(intersection) / len(union)
        
        # Calculate cosine similarity
        from collections import Counter
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        
        # Get all unique words
        all_words = set(counter1.keys()).union(set(counter2.keys()))
        
        # Create vectors
        vector1 = [counter1[word] for word in all_words]
        vector2 = [counter2[word] for word in all_words]
        
        # Calculate dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = sum(a * a for a in vector1) ** 0.5
        magnitude2 = sum(b * b for b in vector2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            cosine_sim = 0.0
        else:
            cosine_sim = dot_product / (magnitude1 * magnitude2)
        
        # Return average of both similarities
        return (jaccard_sim + cosine_sim) / 2
    """
    
    code_text2 = """
    def compute_text_similarity(str1, str2):
        # Preprocess and tokenize
        words1 = str1.lower().split()
        words2 = str2.lower().split()
        
        # Jaccard index calculation
        word_set1 = set(words1)
        word_set2 = set(words2)
        common_words = word_set1 & word_set2
        all_words = word_set1 | word_set2
        
        jaccard_score = len(common_words) / len(all_words) if all_words else 0.0
        
        # Cosine similarity computation
        from collections import defaultdict
        freq1 = defaultdict(int)
        freq2 = defaultdict(int)
        
        for word in words1:
            freq1[word] += 1
        for word in words2:
            freq2[word] += 1
        
        # Build feature vectors
        vocabulary = set(freq1.keys()) | set(freq2.keys())
        vec1 = [freq1[term] for term in vocabulary]
        vec2 = [freq2[term] for term in vocabulary]
        
        # Compute cosine similarity
        numerator = sum(x * y for x, y in zip(vec1, vec2))
        norm1 = sum(x * x for x in vec1) ** 0.5
        norm2 = sum(y * y for y in vec2) ** 0.5
        
        cosine_score = numerator / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        # Combined similarity score
        return (jaccard_score + cosine_score) / 2.0
    """
    
    # Create test JSON objects with different text lengths
    test_cases = [
        # Short text comparison
        {
            "case_name": "short_text",
            "json1": {"description": short_text1, "type": "product", "id": 1},
            "json2": {"description": short_text2, "type": "product", "id": 1}
        },
        # Medium text comparison
        {
            "case_name": "medium_text", 
            "json1": {"content": medium_text1, "category": "review", "rating": 5},
            "json2": {"content": medium_text2, "category": "review", "rating": 5}
        },
        # Long text comparison
        {
            "case_name": "long_text",
            "json1": {"article": long_text1, "topic": "AI", "author": "John Doe"},
            "json2": {"article": long_text2, "topic": "AI", "author": "Jane Smith"}
        },
        # Code comparison
        {
            "case_name": "code_text",
            "json1": {"implementation": code_text1, "language": "python", "version": "1.0"},
            "json2": {"implementation": code_text2, "language": "python", "version": "2.0"}
        },
        # Mixed content
        {
            "case_name": "mixed_content",
            "json1": {
                "title": "AI Research Paper",
                "abstract": medium_text1,
                "full_text": long_text1,
                "code_sample": code_text1,
                "metadata": {"year": 2024, "authors": ["John Doe"]}
            },
            "json2": {
                "title": "AI Research Article", 
                "abstract": medium_text2,
                "full_text": long_text2,
                "code_sample": code_text2,
                "metadata": {"year": 2024, "authors": ["Jane Smith"]}
            }
        }
    ]
    
    return test_cases

def evaluate_chunking_parameters(
    test_cases: List[Dict[str, Any]], 
    chunk_sizes: List[int] = [100, 200, 300, 500, 800],
    overlaps: List[int] = [0, 25, 50, 100, 150]
) -> pd.DataFrame:
    """
    Evaluate different chunk_size and overlap combinations.
    
    Args:
        test_cases: List of test case dictionaries
        chunk_sizes: List of chunk sizes to test
        overlaps: List of overlap values to test
        
    Returns:
        DataFrame with results
    """
    results = []
    
    # Test all combinations of chunk_size and overlap
    for chunk_size, overlap in product(chunk_sizes, overlaps):
        # Skip invalid combinations where overlap >= chunk_size
        if overlap >= chunk_size:
            continue
            
        print(f"Testing chunk_size={chunk_size}, overlap={overlap}")
        
        for test_case in test_cases:
            case_name = test_case["case_name"]
            json1 = test_case["json1"]
            json2 = test_case["json2"]
            
            # Test with LangChain splitter
            start_time = time.time()
            evaluator_langchain = SemanticJsonTreeConsistencyEvaluator(
                use_semantic_similarity=True,
                use_langchain_splitter=True,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                long_string_method='hungarian'
            )
            
            similarity_langchain, operations_langchain = evaluator_langchain.calculate_tree_edit_distance(json1, json2)
            time_langchain = time.time() - start_time
            
            # Test without LangChain splitter (custom splitter)
            start_time = time.time()
            evaluator_custom = SemanticJsonTreeConsistencyEvaluator(
                use_semantic_similarity=True,
                use_langchain_splitter=False,
                long_string_method='hungarian'
            )
            
            similarity_custom, operations_custom = evaluator_custom.calculate_tree_edit_distance(json1, json2)
            time_custom = time.time() - start_time
            
            # Test direct comparison (no chunking)
            start_time = time.time()
            evaluator_direct = SemanticJsonTreeConsistencyEvaluator(
                use_semantic_similarity=True,
                use_langchain_splitter=False,
                long_string_method='direct'
            )
            
            similarity_direct, operations_direct = evaluator_direct.calculate_tree_edit_distance(json1, json2)
            time_direct = time.time() - start_time
            
            # Store results
            results.append({
                'case_name': case_name,
                'chunk_size': chunk_size,
                'overlap': overlap,
                'similarity_langchain': similarity_langchain,
                'similarity_custom': similarity_custom,
                'similarity_direct': similarity_direct,
                'time_langchain': time_langchain,
                'time_custom': time_custom,
                'time_direct': time_direct,
                'improvement_vs_custom': similarity_langchain - similarity_custom,
                'improvement_vs_direct': similarity_langchain - similarity_direct,
                'num_operations_langchain': len(operations_langchain),
                'num_operations_custom': len(operations_custom),
                'num_operations_direct': len(operations_direct)
            })
    
    return pd.DataFrame(results)

def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the results and provide insights."""
    
    analysis = {}
    
    # Overall statistics
    analysis['overall_stats'] = {
        'mean_similarity_langchain': df['similarity_langchain'].mean(),
        'mean_similarity_custom': df['similarity_custom'].mean(),
        'mean_similarity_direct': df['similarity_direct'].mean(),
        'mean_improvement_vs_custom': df['improvement_vs_custom'].mean(),
        'mean_improvement_vs_direct': df['improvement_vs_direct'].mean(),
        'mean_time_langchain': df['time_langchain'].mean(),
        'mean_time_custom': df['time_custom'].mean(),
        'mean_time_direct': df['time_direct'].mean()
    }
    
    # Best parameters overall
    best_overall = df.loc[df['similarity_langchain'].idxmax()]
    analysis['best_overall'] = {
        'chunk_size': int(best_overall['chunk_size']),
        'overlap': int(best_overall['overlap']),
        'similarity': float(best_overall['similarity_langchain']),
        'case_name': best_overall['case_name']
    }
    
    # Best parameters by case type
    analysis['best_by_case'] = {}
    for case_name in df['case_name'].unique():
        case_df = df[df['case_name'] == case_name]
        best_case = case_df.loc[case_df['similarity_langchain'].idxmax()]
        analysis['best_by_case'][case_name] = {
            'chunk_size': int(best_case['chunk_size']),
            'overlap': int(best_case['overlap']),
            'similarity': float(best_case['similarity_langchain']),
            'improvement_vs_custom': float(best_case['improvement_vs_custom']),
            'improvement_vs_direct': float(best_case['improvement_vs_direct'])
        }
    
    # Parameter sensitivity analysis
    chunk_size_impact = df.groupby('chunk_size')['similarity_langchain'].agg(['mean', 'std']).reset_index()
    overlap_impact = df.groupby('overlap')['similarity_langchain'].agg(['mean', 'std']).reset_index()
    
    analysis['parameter_sensitivity'] = {
        'chunk_size_impact': chunk_size_impact.to_dict('records'),
        'overlap_impact': overlap_impact.to_dict('records')
    }
    
    # Performance analysis
    analysis['performance'] = {
        'langchain_vs_custom_speedup': df['time_custom'].mean() / df['time_langchain'].mean(),
        'langchain_vs_direct_speedup': df['time_direct'].mean() / df['time_langchain'].mean(),
        'correlation_time_vs_similarity': df[['time_langchain', 'similarity_langchain']].corr().iloc[0, 1]
    }
    
    return analysis

def create_visualizations(df: pd.DataFrame, output_dir: str = "./chunking_analysis"):
    """Create visualizations for the chunking parameter analysis."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Heatmap of similarity scores by chunk_size and overlap
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Similarity Scores by Chunk Size and Overlap', fontsize=16)
    
    case_names = df['case_name'].unique()
    for i, case_name in enumerate(case_names):
        if i >= 6:  # Limit to 6 subplots
            break
        
        row, col = i // 3, i % 3
        case_df = df[df['case_name'] == case_name]
        
        # Create pivot table for heatmap
        pivot_table = case_df.pivot_table(
            values='similarity_langchain', 
            index='chunk_size', 
            columns='overlap', 
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt='.3f', 
            cmap='YlOrRd',
            ax=axes[row, col],
            cbar_kws={'label': 'Similarity Score'}
        )
        axes[row, col].set_title(f'{case_name.replace("_", " ").title()}')
        axes[row, col].set_xlabel('Overlap')
        axes[row, col].set_ylabel('Chunk Size')
    
    # Hide unused subplots
    for i in range(len(case_names), 6):
        row, col = i // 3, i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/similarity_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Comparison of methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Different String Comparison Methods', fontsize=16)
    
    # Similarity comparison
    methods = ['similarity_langchain', 'similarity_custom', 'similarity_direct']
    method_labels = ['LangChain Splitter', 'Custom Splitter', 'Direct Comparison']
    
    similarity_data = [df[method].values for method in methods]
    
    axes[0, 0].boxplot(similarity_data, labels=method_labels)
    axes[0, 0].set_title('Similarity Score Distribution')
    axes[0, 0].set_ylabel('Similarity Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Time comparison
    time_methods = ['time_langchain', 'time_custom', 'time_direct']
    time_data = [df[method].values for method in time_methods]
    
    axes[0, 1].boxplot(time_data, labels=method_labels)
    axes[0, 1].set_title('Processing Time Distribution')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Improvement over custom method
    axes[1, 0].hist(df['improvement_vs_custom'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(df['improvement_vs_custom'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["improvement_vs_custom"].mean():.3f}')
    axes[1, 0].set_title('Improvement over Custom Splitter')
    axes[1, 0].set_xlabel('Similarity Improvement')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Improvement over direct method
    axes[1, 1].hist(df['improvement_vs_direct'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].axvline(df['improvement_vs_direct'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["improvement_vs_direct"].mean():.3f}')
    axes[1, 1].set_title('Improvement over Direct Comparison')
    axes[1, 1].set_xlabel('Similarity Improvement')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Parameter sensitivity analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
    
    # Chunk size impact
    chunk_size_stats = df.groupby('chunk_size')['similarity_langchain'].agg(['mean', 'std']).reset_index()
    axes[0].errorbar(chunk_size_stats['chunk_size'], chunk_size_stats['mean'], 
                     yerr=chunk_size_stats['std'], marker='o', capsize=5, capthick=2)
    axes[0].set_title('Impact of Chunk Size')
    axes[0].set_xlabel('Chunk Size')
    axes[0].set_ylabel('Mean Similarity Score')
    axes[0].grid(True, alpha=0.3)
    
    # Overlap impact
    overlap_stats = df.groupby('overlap')['similarity_langchain'].agg(['mean', 'std']).reset_index()
    axes[1].errorbar(overlap_stats['overlap'], overlap_stats['mean'], 
                     yerr=overlap_stats['std'], marker='s', capsize=5, capthick=2, color='orange')
    axes[1].set_title('Impact of Overlap')
    axes[1].set_xlabel('Overlap')
    axes[1].set_ylabel('Mean Similarity Score')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Case-specific analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    case_means = df.groupby(['case_name', 'chunk_size', 'overlap'])['similarity_langchain'].mean().reset_index()
    
    for case_name in df['case_name'].unique():
        case_data = case_means[case_means['case_name'] == case_name]
        # Create a combined parameter for x-axis
        case_data['param_combo'] = case_data['chunk_size'].astype(str) + '_' + case_data['overlap'].astype(str)
        ax.plot(range(len(case_data)), case_data['similarity_langchain'], 
                marker='o', label=case_name.replace('_', ' ').title(), linewidth=2)
    
    ax.set_title('Similarity Scores by Case Type and Parameters')
    ax.set_xlabel('Parameter Combinations (chunk_size_overlap)')
    ax.set_ylabel('Similarity Score')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/case_specific_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def main():
    """Main function to run the chunking parameter evaluation."""
    print("=== Evaluating Chunk Size and Overlap Parameters ===\n")
    
    # Create test data
    print("Creating test data...")
    test_cases = create_test_data()
    print(f"Created {len(test_cases)} test cases: {[case['case_name'] for case in test_cases]}\n")
    
    # Define parameter ranges to test
    chunk_sizes = [100, 200, 300, 500, 800]
    overlaps = [0, 25, 50, 100, 150]
    
    print(f"Testing chunk sizes: {chunk_sizes}")
    print(f"Testing overlaps: {overlaps}")
    print(f"Total combinations: {len(chunk_sizes) * len(overlaps)} (excluding invalid combinations)\n")
    
    # Run evaluation
    print("Running evaluation...")
    results_df = evaluate_chunking_parameters(test_cases, chunk_sizes, overlaps)
    
    # Save raw results
    results_df.to_csv('chunking_parameter_results.csv', index=False)
    print(f"Raw results saved to chunking_parameter_results.csv")
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results_df)
    
    # Save analysis
    with open('chunking_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("Analysis saved to chunking_analysis.json")
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print(f"Overall mean similarity (LangChain): {analysis['overall_stats']['mean_similarity_langchain']:.4f}")
    print(f"Overall mean similarity (Custom): {analysis['overall_stats']['mean_similarity_custom']:.4f}")
    print(f"Overall mean similarity (Direct): {analysis['overall_stats']['mean_similarity_direct']:.4f}")
    print(f"Mean improvement vs Custom: {analysis['overall_stats']['mean_improvement_vs_custom']:.4f}")
    print(f"Mean improvement vs Direct: {analysis['overall_stats']['mean_improvement_vs_direct']:.4f}")
    
    print(f"\nBest overall parameters:")
    print(f"  Chunk size: {analysis['best_overall']['chunk_size']}")
    print(f"  Overlap: {analysis['best_overall']['overlap']}")
    print(f"  Similarity: {analysis['best_overall']['similarity']:.4f}")
    print(f"  Test case: {analysis['best_overall']['case_name']}")
    
    print(f"\nBest parameters by case type:")
    for case_name, params in analysis['best_by_case'].items():
        print(f"  {case_name}: chunk_size={params['chunk_size']}, overlap={params['overlap']}, "
              f"similarity={params['similarity']:.4f}")
    
    print(f"\nPerformance analysis:")
    print(f"  LangChain vs Custom speedup: {analysis['performance']['langchain_vs_custom_speedup']:.2f}x")
    print(f"  LangChain vs Direct speedup: {analysis['performance']['langchain_vs_direct_speedup']:.2f}x")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df)
    
    print("\n=== RECOMMENDATIONS ===")
    
    # Analyze parameter sensitivity
    chunk_impact = analysis['parameter_sensitivity']['chunk_size_impact']
    overlap_impact = analysis['parameter_sensitivity']['overlap_impact']
    
    best_chunk_size = max(chunk_impact, key=lambda x: x['mean'])['chunk_size']
    best_overlap = max(overlap_impact, key=lambda x: x['mean'])['overlap']
    
    print(f"Recommended chunk_size: {best_chunk_size}")
    print(f"Recommended overlap: {best_overlap}")
    
    # Check if LangChain splitter is beneficial
    if analysis['overall_stats']['mean_improvement_vs_custom'] > 0.01:
        print("✓ LangChain splitter shows significant improvement over custom splitter")
    else:
        print("⚠ LangChain splitter shows minimal improvement over custom splitter")
    
    if analysis['overall_stats']['mean_improvement_vs_direct'] > 0.01:
        print("✓ Chunking approach shows significant improvement over direct comparison")
    else:
        print("⚠ Chunking approach shows minimal improvement over direct comparison")
    
    print("\nEvaluation complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
