#!/usr/bin/env python
"""
Test Long Text Comparison Methods

This script compares different methods for evaluating similarity between long texts,
with a focus on demonstrating the effectiveness of the Hungarian algorithm approach.

Usage:
    python test_long_text_comparison.py --input-file path/to/long_article.txt --output-dir ./text_comparison_results
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import re
import random
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from difflib import SequenceMatcher
import torch

# Import from semantic_json_tree_consistency.py
try:
    from semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
    SEMANTIC_TREE_AVAILABLE = True
except ImportError:
    print("Warning: semantic_json_tree_consistency module not available. Install it for Hungarian algorithm.")
    SEMANTIC_TREE_AVAILABLE = False

# Try to import BERTScore
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: BERTScore not available. Install with 'pip install bert-score'")
    BERT_SCORE_AVAILABLE = False

# Download NLTK data if needed
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class TextComparer:
    """Class for comparing long texts using various methods."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the text comparer.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize Rouge
        self.rouge = Rouge()
        
        # Initialize smoothing function for BLEU
        self.smoothie = SmoothingFunction().method1
        
        # Initialize SemanticJsonTreeConsistencyEvaluator for Hungarian algorithm
        if SEMANTIC_TREE_AVAILABLE:
            self.tree_evaluator = SemanticJsonTreeConsistencyEvaluator(
                array_order_matters=False,
                use_semantic_similarity=True,
                semantic_threshold=0.7,
                string_method='semantic',
                use_hungarian=True,
                long_string_method='hungarian'
            )
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into meaningful chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # First try to split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If we have enough paragraphs, use those
        if len(paragraphs) >= 3:
            return paragraphs
        
        # Otherwise, split by sentences
        sentences = nltk.sent_tokenize(text)
        
        # If we have enough sentences, use those
        if len(sentences) >= 5:
            return sentences
        
        # Otherwise, use a sliding window approach
        words = text.split()
        chunk_size = max(20, len(words) // 10)  # Aim for about 10 chunks
        chunks = []
        
        for i in range(0, len(words), chunk_size // 2):  # 50% overlap
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def modify_text(self, text: str, modification_level: float = 0.2) -> str:
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
                    sentences = nltk.sent_tokenize(paragraphs[i])
                    if len(sentences) > 1:
                        del sentences[random.randint(0, len(sentences) - 1)]
                        paragraphs[i] = ' '.join(sentences)
                    else:
                        paragraphs[i] = ''  # Delete the whole paragraph if it's just one sentence
                
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
                    sentences = nltk.sent_tokenize(paragraphs[i])
                    if len(sentences) > 1:
                        random.shuffle(sentences)
                        paragraphs[i] = ' '.join(sentences)
        
        # Randomly reorder some paragraphs
        if len(paragraphs) > 2:
            reorder_count = max(1, int(len(paragraphs) * modification_level * 0.5))
            for _ in range(reorder_count):
                i, j = random.sample(range(len(paragraphs)), 2)
                paragraphs[i], paragraphs[j] = paragraphs[j], paragraphs[i]
        
        return '\n\n'.join(paragraphs)
    
    def create_text_pair(self, text: str, modification_levels: List[float] = [0.1, 0.3, 0.5]) -> List[Tuple[str, str, float]]:
        """
        Create pairs of original and modified texts at different modification levels.
        
        Args:
            text: Original text
            modification_levels: List of modification levels to apply
            
        Returns:
            List of (original, modified, level) tuples
        """
        pairs = []
        for level in modification_levels:
            modified = self.modify_text(text, level)
            pairs.append((text, modified, level))
        return pairs
    
    def compare_with_hungarian(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare texts using Hungarian algorithm with chunking.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with similarity metrics
        """
        if SEMANTIC_TREE_AVAILABLE:
            # Use the implementation from semantic_json_tree_consistency.py
            similarity = self.tree_evaluator._compare_long_strings_hungarian(text1, text2)
            
            # Get additional information about chunks
            chunks1 = self.tree_evaluator._split_into_chunks(text1)
            chunks2 = self.tree_evaluator._split_into_chunks(text2)
            
            return {
                "method": "hungarian",
                "score": float(similarity),
                "num_chunks1": len(chunks1),
                "num_chunks2": len(chunks2),
                "implementation": "semantic_json_tree_consistency"
            }
        else:
            # Fallback implementation if semantic_json_tree_consistency is not available
            # Split texts into chunks
            chunks1 = self.split_into_chunks(text1)
            chunks2 = self.split_into_chunks(text2)
            
            # Get embeddings for all chunks
            embeddings1 = self.embedding_model.encode(chunks1)
            embeddings2 = self.embedding_model.encode(chunks2)
            
            # Calculate similarity matrix
            similarity_matrix = np.zeros((len(chunks1), len(chunks2)))
            
            for i in range(len(chunks1)):
                for j in range(len(chunks2)):
                    # Calculate cosine similarity between embeddings
                    sim = cosine_similarity([embeddings1[i]], [embeddings2[j]])[0][0]
                    similarity_matrix[i, j] = sim
            
            # Use Hungarian algorithm to find optimal matching
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Negate for max similarity
            
            # Calculate metrics
            matched_similarities = [similarity_matrix[i, j] for i, j in zip(row_ind, col_ind)]
            
            # Calculate average similarity of matched chunks
            avg_similarity = sum(matched_similarities) / len(matched_similarities) if matched_similarities else 0.0
            
            # Calculate coverage (what percentage of chunks are matched well)
            good_matches = sum(1 for sim in matched_similarities if sim > 0.7)
            coverage = good_matches / max(len(chunks1), len(chunks2))
            
            # Combine similarity and coverage
            hungarian_score = 0.7 * avg_similarity + 0.3 * coverage
            
            return {
                "method": "hungarian",
                "score": float(hungarian_score),
                "avg_similarity": float(avg_similarity),
                "coverage": float(coverage),
                "num_chunks1": len(chunks1),
                "num_chunks2": len(chunks2),
                "matched_similarities": [float(sim) for sim in matched_similarities],
                "implementation": "fallback"
            }
    
    def compare_with_bertscore(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare texts using BERTScore.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with similarity metrics
        """
        if not BERT_SCORE_AVAILABLE:
            return {
                "method": "bertscore",
                "score": 0.0,
                "error": "BERTScore not available"
            }
        
        try:
            P, R, F1 = bert_score([text1], [text2], lang="en")
            
            return {
                "method": "bertscore",
                "score": float(F1.item()),
                "precision": float(P.item()),
                "recall": float(R.item())
            }
        except Exception as e:
            return {
                "method": "bertscore",
                "score": 0.0,
                "error": str(e)
            }
    
    def compare_with_cosine(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare texts using cosine similarity of embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with similarity metrics
        """
        # Get embeddings
        embedding1 = self.embedding_model.encode(text1)
        embedding2 = self.embedding_model.encode(text2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        
        return {
            "method": "cosine",
            "score": float(similarity)
        }
    
    def compare_with_traditional(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare texts using traditional metrics (BLEU, ROUGE).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with similarity metrics
        """
        results = {
            "method": "traditional"
        }
        
        # Tokenize texts
        tokens1 = nltk.word_tokenize(text1.lower())
        tokens2 = nltk.word_tokenize(text2.lower())
        
        # Calculate BLEU score
        try:
            bleu = sentence_bleu([tokens1], tokens2, smoothing_function=self.smoothie)
            results["bleu"] = float(bleu)
        except Exception as e:
            results["bleu_error"] = str(e)
        
        # Calculate ROUGE scores
        try:
            rouge_scores = self.rouge.get_scores(text1, text2)[0]
            results["rouge_1"] = float(rouge_scores['rouge-1']['f'])
            results["rouge_2"] = float(rouge_scores['rouge-2']['f'])
            results["rouge_l"] = float(rouge_scores['rouge-l']['f'])
        except Exception as e:
            results["rouge_error"] = str(e)
        
        # Calculate Levenshtein ratio
        try:
            levenshtein_ratio = SequenceMatcher(None, text1, text2).ratio()
            results["levenshtein"] = float(levenshtein_ratio)
        except Exception as e:
            results["levenshtein_error"] = str(e)
        
        return results
    
    def compare_all_methods(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare texts using all available methods.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with results from all methods
        """
        results = {
            "text_length_1": len(text1),
            "text_length_2": len(text2),
            "word_count_1": len(text1.split()),
            "word_count_2": len(text2.split()),
            "sentence_count_1": len(nltk.sent_tokenize(text1)),
            "sentence_count_2": len(nltk.sent_tokenize(text2))
        }
        
        # Compare with Hungarian algorithm
        hungarian_results = self.compare_with_hungarian(text1, text2)
        results["hungarian"] = hungarian_results
        
        # Compare with BERTScore
        bertscore_results = self.compare_with_bertscore(text1, text2)
        results["bertscore"] = bertscore_results
        
        # Compare with cosine similarity
        cosine_results = self.compare_with_cosine(text1, text2)
        results["cosine"] = cosine_results
        
        # Compare with traditional metrics
        traditional_results = self.compare_with_traditional(text1, text2)
        results["traditional"] = traditional_results
        
        return results

def run_experiment(args):
    """
    Run the long text comparison experiment.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the input text
    print(f"Loading text from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded text: {len(text)} characters, {len(text.split())} words")
    
    # Initialize text comparer
    comparer = TextComparer()
    
    # Create text pairs with different modification levels
    modification_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"Creating text pairs with modification levels: {modification_levels}")
    text_pairs = comparer.create_text_pair(text, modification_levels)
    
    # Compare each pair with all methods
    results = []
    for original, modified, level in text_pairs:
        print(f"Comparing texts with modification level {level:.1f}...")
        comparison_results = comparer.compare_all_methods(original, modified)
        comparison_results["modification_level"] = level
        results.append(comparison_results)
    
    # Save results
    results_file = os.path.join(args.output_dir, "text_comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Create visualizations
    create_visualizations(results, args.output_dir)
    
    # Print summary
    print_summary(results)
    
    # Print implementation info
    if SEMANTIC_TREE_AVAILABLE:
        print("\nUsing Hungarian algorithm implementation from semantic_json_tree_consistency.py")
    else:
        print("\nUsing fallback Hungarian algorithm implementation")
        print("To use the implementation from semantic_json_tree_consistency.py, make sure it's in your path.")

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """
    Create visualizations of the text comparison results.
    
    Args:
        results: List of comparison results
        output_dir: Directory to save visualizations
    """
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Extract data for plotting
    modification_levels = [r["modification_level"] for r in results]
    hungarian_scores = [r["hungarian"]["score"] for r in results]
    bertscore_scores = [r["bertscore"].get("score", 0) for r in results]
    cosine_scores = [r["cosine"]["score"] for r in results]
    rouge_l_scores = [r["traditional"].get("rouge_l", 0) for r in results]
    
    # 1. Line plot comparing all methods
    plt.figure(figsize=(12, 8))
    plt.plot(modification_levels, hungarian_scores, 'o-', label='Hungarian', linewidth=2)
    plt.plot(modification_levels, bertscore_scores, 's-', label='BERTScore', linewidth=2)
    plt.plot(modification_levels, cosine_scores, '^-', label='Cosine', linewidth=2)
    plt.plot(modification_levels, rouge_l_scores, 'd-', label='ROUGE-L', linewidth=2)
    
    plt.title('Text Similarity Methods vs. Modification Level')
    plt.xlabel('Modification Level')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300)
    
    # 2. Bar chart comparing methods at each modification level
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(modification_levels))
    width = 0.2
    
    plt.bar(x - width*1.5, hungarian_scores, width, label='Hungarian')
    plt.bar(x - width*0.5, bertscore_scores, width, label='BERTScore')
    plt.bar(x + width*0.5, cosine_scores, width, label='Cosine')
    plt.bar(x + width*1.5, rouge_l_scores, width, label='ROUGE-L')
    
    plt.title('Comparison of Text Similarity Methods')
    plt.xlabel('Modification Level')
    plt.ylabel('Similarity Score')
    plt.xticks(x, [f"{level:.1f}" for level in modification_levels])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison_bar.png'), dpi=300)
    
    # 3. Scatter plot of Hungarian vs. BERTScore
    plt.figure(figsize=(10, 8))
    plt.scatter(hungarian_scores, bertscore_scores, s=100, c=modification_levels, cmap='viridis')
    
    plt.title('Hungarian Algorithm vs. BERTScore')
    plt.xlabel('Hungarian Score')
    plt.ylabel('BERTScore')
    plt.colorbar(label='Modification Level')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hungarian_vs_bertscore.png'), dpi=300)
    
    # 4. Heatmap of all methods
    plt.figure(figsize=(12, 8))
    
    # Create a matrix of scores
    methods = ['Hungarian', 'BERTScore', 'Cosine', 'ROUGE-L']
    scores_matrix = np.array([
        hungarian_scores,
        bertscore_scores,
        cosine_scores,
        rouge_l_scores
    ])
    
    sns.heatmap(
        scores_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="YlGnBu",
        xticklabels=[f"{level:.1f}" for level in modification_levels],
        yticklabels=methods
    )
    
    plt.title('Similarity Scores Across Methods and Modification Levels')
    plt.xlabel('Modification Level')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scores_heatmap.png'), dpi=300)
    
    print(f"Visualizations saved to {output_dir}")

def print_summary(results: List[Dict[str, Any]]):
    """
    Print a summary of the text comparison results.
    
    Args:
        results: List of comparison results
    """
    print("\n" + "=" * 80)
    print("TEXT COMPARISON SUMMARY")
    print("=" * 80)
    
    # Print text statistics
    first_result = results[0]
    print(f"\nText Statistics:")
    print(f"  Length: {first_result['text_length_1']} characters")
    print(f"  Words: {first_result['word_count_1']} words")
    print(f"  Sentences: {first_result['sentence_count_1']} sentences")
    
    # Print results for each modification level
    print("\nResults by Modification Level:")
    for result in results:
        level = result["modification_level"]
        print(f"\n  Modification Level: {level:.1f}")
        print(f"    Hungarian: {result['hungarian']['score']:.4f}")
        print(f"    BERTScore: {result['bertscore'].get('score', 0):.4f}")
        print(f"    Cosine: {result['cosine']['score']:.4f}")
        print(f"    ROUGE-L: {result['traditional'].get('rouge_l', 0):.4f}")
    
    # Calculate correlations with modification level
    modification_levels = [r["modification_level"] for r in results]
    hungarian_scores = [r["hungarian"]["score"] for r in results]
    bertscore_scores = [r["bertscore"].get("score", 0) for r in results]
    cosine_scores = [r["cosine"]["score"] for r in results]
    rouge_l_scores = [r["traditional"].get("rouge_l", 0) for r in results]
    
    # Calculate correlation coefficients
    hungarian_corr = np.corrcoef(modification_levels, hungarian_scores)[0, 1]
    bertscore_corr = np.corrcoef(modification_levels, bertscore_scores)[0, 1]
    cosine_corr = np.corrcoef(modification_levels, cosine_scores)[0, 1]
    rouge_corr = np.corrcoef(modification_levels, rouge_l_scores)[0, 1]
    
    print("\nCorrelation with Modification Level:")
    print(f"  Hungarian: {hungarian_corr:.4f}")
    print(f"  BERTScore: {bertscore_corr:.4f}")
    print(f"  Cosine: {cosine_corr:.4f}")
    print(f"  ROUGE-L: {rouge_corr:.4f}")
    
    # Calculate average scores
    avg_hungarian = sum(hungarian_scores) / len(hungarian_scores)
    avg_bertscore = sum(bertscore_scores) / len(bertscore_scores)
    avg_cosine = sum(cosine_scores) / len(cosine_scores)
    avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores)
    
    print("\nAverage Scores:")
    print(f"  Hungarian: {avg_hungarian:.4f}")
    print(f"  BERTScore: {avg_bertscore:.4f}")
    print(f"  Cosine: {avg_cosine:.4f}")
    print(f"  ROUGE-L: {avg_rouge:.4f}")
    
    # Calculate score ranges
    range_hungarian = max(hungarian_scores) - min(hungarian_scores)
    range_bertscore = max(bertscore_scores) - min(bertscore_scores)
    range_cosine = max(cosine_scores) - min(cosine_scores)
    range_rouge = max(rouge_l_scores) - min(rouge_l_scores)
    
    print("\nScore Ranges (Max - Min):")
    print(f"  Hungarian: {range_hungarian:.4f}")
    print(f"  BERTScore: {range_bertscore:.4f}")
    print(f"  Cosine: {range_cosine:.4f}")
    print(f"  ROUGE-L: {range_rouge:.4f}")
    
    # Determine which method has the strongest correlation with modification level
    correlations = [
        ("Hungarian", abs(hungarian_corr)),
        ("BERTScore", abs(bertscore_corr)),
        ("Cosine", abs(cosine_corr)),
        ("ROUGE-L", abs(rouge_corr))
    ]
    
    best_method = max(correlations, key=lambda x: x[1])
    
    print(f"\nMethod with strongest correlation to modification level: {best_method[0]} ({best_method[1]:.4f})")
    
    # Determine which method has the widest range (most sensitive)
    ranges = [
        ("Hungarian", range_hungarian),
        ("BERTScore", range_bertscore),
        ("Cosine", range_cosine),
        ("ROUGE-L", range_rouge)
    ]
    
    most_sensitive = max(ranges, key=lambda x: x[1])
    
    print(f"Most sensitive method (widest score range): {most_sensitive[0]} ({most_sensitive[1]:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Test long text comparison methods.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--output-dir", type=str, default="./text_comparison_results", help="Directory to save results.")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()