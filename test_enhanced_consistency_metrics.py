#!/usr/bin/env python
"""
Test Enhanced Consistency Metrics with LangChain Text Splitting

This script tests the enhanced semantic_json_tree_consistency.py with LangChain text splitting
to evaluate its effectiveness on long text comparison.

Usage:
    python test_enhanced_consistency_metrics.py --input-file sample_long_article.txt
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import random

# Import the semantic tree consistency evaluator
from semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

# Create evaluator with different configurations
evaluators = {
    "hungarian_langchain": SemanticJsonTreeConsistencyEvaluator(
        use_semantic_similarity=True,
        string_method='semantic',
        use_hungarian=True,
        long_string_method='hungarian',
        use_langchain_splitter=True  # Use LangChain's text splitter
    ),
    "hungarian_custom": SemanticJsonTreeConsistencyEvaluator(
        use_semantic_similarity=True,
        string_method='semantic',
        use_hungarian=True,
        long_string_method='hungarian',
        use_langchain_splitter=False  # Use custom text splitting
    ),
    "direct": SemanticJsonTreeConsistencyEvaluator(
        use_semantic_similarity=True,
        string_method='semantic',
        use_hungarian=False,
        long_string_method='direct',
        use_langchain_splitter=False  # Use custom text splitting
    ),
    "cosine": SemanticJsonTreeConsistencyEvaluator(
        use_semantic_similarity=True,
        string_method='semantic',
        use_hungarian=False,
        long_string_method='cosine',
        use_langchain_splitter=False  # Use custom text splitting
    )
}

def load_text(input_file: str) -> str:
    """
    Load text from a file.
    
    Args:
        input_file: Path to the input file
        
    Returns:
        The text content
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()

def modify_text(text: str, modification_level: float = 0.2) -> str:
    """
    Create a modified version of the text with controlled changes.
    
    Args:
        text: Original text
        modification_level: Level of modification (0.0 to 1.0)
        
    Returns:
        Modified text
    """
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # Determine how many paragraphs to modify
    num_to_modify = max(1, int(len(paragraphs) * modification_level))
    
    # Choose paragraphs to modify
    indices_to_modify = random.sample(range(len(paragraphs)), num_to_modify)
    
    # Apply modifications
    for i in indices_to_modify:
        if i < len(paragraphs):
            # Choose a modification type
            mod_type = random.choice(['rephrase', 'delete', 'add', 'reorder'])
            
            if mod_type == 'rephrase':
                # Rephrase by changing some words
                words = paragraphs[i].split()
                for j in range(min(5, len(words))):
                    idx = random.randint(0, len(words) - 1)
                    words[idx] = random.choice(['very', 'quite', 'extremely', 'somewhat', 'rather']) + ' ' + words[idx]
                paragraphs[i] = ' '.join(words)
            
            elif mod_type == 'delete':
                # Delete a sentence if paragraph is long enough
                import nltk
                try:
                    nltk.download('punkt', quiet=True)
                    sentences = nltk.sent_tokenize(paragraphs[i])
                    if len(sentences) > 1:
                        del sentences[random.randint(0, len(sentences) - 1)]
                        paragraphs[i] = ' '.join(sentences)
                    else:
                        paragraphs[i] = ''  # Delete the whole paragraph if it's just one sentence
                except:
                    # If NLTK fails, just delete the paragraph
                    paragraphs[i] = ''
            
            elif mod_type == 'add':
                # Add a sentence
                added_text = random.choice([
                    " This is an important consideration.",
                    " Many experts agree on this point.",
                    " Further research is needed in this area.",
                    " This has significant implications.",
                    " This finding is consistent with previous studies."
                ])
                paragraphs[i] += added_text
            
            elif mod_type == 'reorder':
                # Reorder sentences if there are multiple
                import nltk
                try:
                    nltk.download('punkt', quiet=True)
                    sentences = nltk.sent_tokenize(paragraphs[i])
                    if len(sentences) > 1:
                        random.shuffle(sentences)
                        paragraphs[i] = ' '.join(sentences)
                except:
                    # If NLTK fails, leave the paragraph as is
                    pass
    
    # Randomly reorder some paragraphs
    if len(paragraphs) > 2:
        reorder_count = max(1, int(len(paragraphs) * modification_level * 0.5))
        for _ in range(reorder_count):
            i, j = random.sample(range(len(paragraphs)), 2)
            paragraphs[i], paragraphs[j] = paragraphs[j], paragraphs[i]
    
    return '\n\n'.join(paragraphs)

def compare_texts(text1: str, text2: str) -> Dict[str, Any]:
    """
    Compare two texts using the enhanced semantic tree consistency evaluator.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    # Compare using each evaluator
    for name, evaluator in evaluators.items():
        start_time = time.time()
        similarity = evaluator._compare_long_strings(text1, text2)
        end_time = time.time()
        
        # Get chunk information
        chunks1 = evaluator._split_into_chunks(text1)
        chunks2 = evaluator._split_into_chunks(text2)
        
        results[name] = {
            "similarity": float(similarity),
            "num_chunks1": len(chunks1),
            "num_chunks2": len(chunks2),
            "time_taken": end_time - start_time
        }
    
    return results

def run_experiment(args):
    """
    Run the experiment.
    
    Args:
        args: Command-line arguments
    """
    # Load the text
    print(f"Loading text from {args.input_file}...")
    text = load_text(args.input_file)
    
    print(f"Loaded text: {len(text)} characters")
    
    # Create modified versions at different levels
    modification_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for level in modification_levels:
        print(f"\nTesting modification level {level:.1f}...")
        modified_text = modify_text(text, level)
        
        # Compare original and modified text
        comparison_results = compare_texts(text, modified_text)
        
        # Add to results
        results.append({
            "modification_level": level,
            "results": comparison_results
        })
        
        # Print results
        print(f"\nModification level: {level:.1f}")
        print(f"Original text length: {len(text)} chars, Modified text length: {len(modified_text)} chars")
        for method, metrics in comparison_results.items():
            print(f"  {method}: {metrics['similarity']:.4f} (chunks: {metrics['num_chunks1']}/{metrics['num_chunks2']}, time: {metrics['time_taken']:.2f}s)")
        
        # Print detailed chunk information for the first level only
        if level == modification_levels[0]:
            print("\nDetailed chunk information for first modification level:")
            for method, metrics in comparison_results.items():
                print(f"\n{method}:")
                print(f"  Number of chunks in original text: {metrics['num_chunks1']}")
                print(f"  Number of chunks in modified text: {metrics['num_chunks2']}")
                
                # Get the evaluator for this method
                evaluator = evaluators[method]
                
                # Get chunks for original text
                chunks1 = evaluator._split_into_chunks(text)
                
                # Print first few chunks
                print(f"\n  First 2 chunks from original text:")
                for i, chunk in enumerate(chunks1[:2]):
                    print(f"    Chunk {i+1} ({len(chunk)} chars): {chunk[:50]}...")
                    
                # Print information about the evaluator
                print(f"\n  Evaluator configuration:")
                print(f"    use_hungarian: {evaluator.use_hungarian}")
                print(f"    long_string_method: {evaluator.long_string_method}")
                print(f"    use_langchain_splitter: {evaluator.use_langchain_splitter}")
                print(f"    string_method: {evaluator.string_method}")
    
    # Create output directory if needed
    os.makedirs("./enhanced_metrics_results", exist_ok=True)
    
    # Save results
    output_file = os.path.join("./enhanced_metrics_results", "results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Create visualization
    create_visualization(results)

def create_visualization(results: List[Dict[str, Any]]):
    """
    Create visualization of the results.
    
    Args:
        results: List of results
    """
    # Extract data
    modification_levels = [r["modification_level"] for r in results]
    
    # Extract similarity scores for each method
    methods = list(results[0]["results"].keys())
    scores = {method: [] for method in methods}
    
    for result in results:
        for method in methods:
            scores[method].append(result["results"][method]["similarity"])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    for method in methods:
        plt.plot(modification_levels, scores[method], 'o-', label=method, linewidth=2)
    
    plt.title('Text Similarity Methods vs. Modification Level')
    plt.xlabel('Modification Level')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join("./enhanced_metrics_results", "comparison.png")
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    
    # Calculate correlations
    print("\nCorrelation with modification level:")
    for method in methods:
        correlation = np.corrcoef(modification_levels, scores[method])[0, 1]
        print(f"  {method}: {correlation:.4f}")
    
    # Calculate ranges
    print("\nScore ranges (Max - Min):")
    for method in methods:
        score_range = max(scores[method]) - min(scores[method])
        print(f"  {method}: {score_range:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Test enhanced consistency metrics with LangChain text splitting.")
    parser.add_argument("--input-file", type=str, default="sample_long_article.txt", help="Path to the input text file.")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()