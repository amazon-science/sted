#!/usr/bin/env python3
"""
Quick test script to evaluate the impact of chunk_size and overlap parameters
when using LangChain splitter for long string comparison.
"""

import json
import time
from semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

def create_sample_texts():
    """Create sample texts of different lengths for testing."""
    
    # Long article text
    long_text1 = """
    Artificial intelligence has revolutionized numerous industries and continues to shape the future of technology. 
    Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with 
    remarkable accuracy. Deep learning models have achieved breakthrough performance in computer vision, natural 
    language processing, and speech recognition tasks. These advances have enabled practical applications such as 
    autonomous vehicles, medical diagnosis systems, and intelligent virtual assistants.
    
    The development of transformer architectures has particularly accelerated progress in natural language understanding. 
    Models like GPT and BERT have demonstrated the ability to generate human-like text and comprehend complex linguistic 
    nuances. This has opened up new possibilities for automated content creation, language translation, and conversational AI.
    
    However, the rapid advancement of AI also raises important ethical considerations. Issues such as algorithmic bias, 
    privacy protection, and the potential displacement of human workers require careful attention from researchers, 
    policymakers, and industry leaders. Ensuring that AI development proceeds responsibly and benefits society as a 
    whole remains a critical challenge for the coming decades.
    """
    
    long_text2 = """
    The field of artificial intelligence has experienced unprecedented growth and innovation in recent years. Advanced 
    machine learning techniques now enable computers to analyze complex datasets and extract meaningful insights that 
    were previously impossible to obtain. Neural networks with deep architectures have achieved superhuman performance 
    in various domains, including image classification, game playing, and protein folding prediction. These technological 
    breakthroughs have led to practical implementations in healthcare, finance, transportation, and entertainment.
    
    The introduction of attention mechanisms and transformer models has been particularly transformative for natural 
    language processing applications. Large language models trained on diverse text corpora can now generate coherent 
    text, answer questions, and even write code. This has created new opportunities for automated writing assistance, 
    intelligent search systems, and human-computer interaction.
    
    Nevertheless, the widespread adoption of AI technologies brings significant challenges that must be addressed. 
    Concerns about fairness, transparency, and accountability in AI systems have prompted discussions about regulation 
    and governance. The potential societal impact of automation and the need for workforce adaptation also require 
    proactive planning and policy interventions to ensure equitable outcomes for all members of society.
    """
    
    # Code sample
    code_text1 = """
    def process_data(input_file, output_file):
        '''
        Process data from input file and save results to output file.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str): Path to output JSON file
        '''
        import pandas as pd
        import json
        
        # Read the input data
        try:
            df = pd.read_csv(input_file)
            print(f"Successfully loaded {len(df)} records from {input_file}")
        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found")
            return False
        except Exception as e:
            print(f"Error reading input file: {e}")
            return False
        
        # Process the data
        processed_data = []
        for index, row in df.iterrows():
            record = {
                'id': row.get('id', index),
                'name': row.get('name', '').strip(),
                'value': float(row.get('value', 0)),
                'category': row.get('category', 'unknown').lower(),
                'processed_at': pd.Timestamp.now().isoformat()
            }
            processed_data.append(record)
        
        # Save the results
        try:
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            print(f"Successfully saved {len(processed_data)} processed records to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving output file: {e}")
            return False
    """
    
    code_text2 = """
    def data_processor(source_path, destination_path):
        '''
        Load data from source file, process it, and save to destination.
        
        Parameters:
            source_path (str): Input CSV file path
            destination_path (str): Output JSON file path
        
        Returns:
            bool: True if successful, False otherwise
        '''
        import pandas as pd
        import json
        from datetime import datetime
        
        # Load source data
        try:
            data_frame = pd.read_csv(source_path)
            print(f"Loaded {data_frame.shape[0]} rows from {source_path}")
        except FileNotFoundError:
            print(f"Source file not found: {source_path}")
            return False
        except Exception as error:
            print(f"Failed to read source file: {error}")
            return False
        
        # Transform the data
        results = []
        for idx, record in data_frame.iterrows():
            transformed_record = {
                'identifier': record.get('id', idx),
                'title': str(record.get('name', '')).strip(),
                'amount': float(record.get('value', 0.0)),
                'type': str(record.get('category', 'default')).lower(),
                'timestamp': datetime.now().isoformat()
            }
            results.append(transformed_record)
        
        # Write output
        try:
            with open(destination_path, 'w') as output_file:
                json.dump(results, output_file, indent=4)
            print(f"Saved {len(results)} transformed records to {destination_path}")
            return True
        except Exception as error:
            print(f"Failed to write output file: {error}")
            return False
    """
    
    return {
        "long_article": (long_text1, long_text2),
        "code_sample": (code_text1, code_text2)
    }

def test_chunking_parameters():
    """Test different chunk_size and overlap combinations."""
    
    print("=== Testing Chunk Size and Overlap Impact ===\n")
    
    # Get sample texts
    sample_texts = create_sample_texts()
    
    # Parameter combinations to test
    test_params = [
        {"chunk_size": 100, "overlap": 0},
        {"chunk_size": 100, "overlap": 25},
        {"chunk_size": 200, "overlap": 0},
        {"chunk_size": 200, "overlap": 50},
        {"chunk_size": 300, "overlap": 0},
        {"chunk_size": 300, "overlap": 50},
        {"chunk_size": 500, "overlap": 0},
        {"chunk_size": 500, "overlap": 100},
        {"chunk_size": 800, "overlap": 0},
        {"chunk_size": 800, "overlap": 150},
    ]
    
    results = {}
    
    for text_type, (text1, text2) in sample_texts.items():
        print(f"\n--- Testing {text_type.replace('_', ' ').title()} ---")
        print(f"Text 1 length: {len(text1)} characters")
        print(f"Text 2 length: {len(text2)} characters")
        
        # Create test JSON objects
        json1 = {"content": text1, "type": text_type, "id": 1}
        json2 = {"content": text2, "type": text_type, "id": 2}
        
        results[text_type] = []
        
        # Test baseline (direct comparison, no chunking)
        print("\nBaseline (Direct comparison):")
        start_time = time.time()
        evaluator_direct = SemanticJsonTreeConsistencyEvaluator(
            use_semantic_similarity=True,
            use_langchain_splitter=False,
            long_string_method='direct'
        )
        similarity_direct, _ = evaluator_direct.calculate_tree_edit_distance(json1, json2)
        time_direct = time.time() - start_time
        print(f"  Similarity: {similarity_direct:.4f}, Time: {time_direct:.3f}s")
        
        # Test custom splitter (no LangChain)
        print("\nCustom splitter:")
        start_time = time.time()
        evaluator_custom = SemanticJsonTreeConsistencyEvaluator(
            use_semantic_similarity=True,
            use_langchain_splitter=False,
            long_string_method='hungarian'
        )
        similarity_custom, _ = evaluator_custom.calculate_tree_edit_distance(json1, json2)
        time_custom = time.time() - start_time
        print(f"  Similarity: {similarity_custom:.4f}, Time: {time_custom:.3f}s")
        
        # Test different LangChain parameters
        print("\nLangChain splitter with different parameters:")
        for params in test_params:
            chunk_size = params["chunk_size"]
            overlap = params["overlap"]
            
            start_time = time.time()
            evaluator_langchain = SemanticJsonTreeConsistencyEvaluator(
                use_semantic_similarity=True,
                use_langchain_splitter=True,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                long_string_method='hungarian'
            )
            
            similarity_langchain, _ = evaluator_langchain.calculate_tree_edit_distance(json1, json2)
            time_langchain = time.time() - start_time
            
            improvement_vs_direct = similarity_langchain - similarity_direct
            improvement_vs_custom = similarity_langchain - similarity_custom
            
            print(f"  chunk_size={chunk_size:3d}, overlap={overlap:3d}: "
                  f"similarity={similarity_langchain:.4f}, time={time_langchain:.3f}s, "
                  f"vs_direct={improvement_vs_direct:+.4f}, vs_custom={improvement_vs_custom:+.4f}")
            
            results[text_type].append({
                "chunk_size": chunk_size,
                "overlap": overlap,
                "similarity": similarity_langchain,
                "time": time_langchain,
                "improvement_vs_direct": improvement_vs_direct,
                "improvement_vs_custom": improvement_vs_custom,
                "similarity_direct": similarity_direct,
                "similarity_custom": similarity_custom
            })
    
    return results

def analyze_and_recommend(results):
    """Analyze results and provide recommendations."""
    
    print("\n\n=== ANALYSIS AND RECOMMENDATIONS ===\n")
    
    all_results = []
    for text_type, type_results in results.items():
        for result in type_results:
            result["text_type"] = text_type
            all_results.append(result)
    
    if not all_results:
        print("No results to analyze.")
        return
    
    # Find best overall parameters
    best_overall = max(all_results, key=lambda x: x["similarity"])
    print(f"Best overall parameters:")
    print(f"  Chunk size: {best_overall['chunk_size']}")
    print(f"  Overlap: {best_overall['overlap']}")
    print(f"  Similarity: {best_overall['similarity']:.4f}")
    print(f"  Text type: {best_overall['text_type']}")
    print(f"  Improvement vs direct: {best_overall['improvement_vs_direct']:+.4f}")
    print(f"  Improvement vs custom: {best_overall['improvement_vs_custom']:+.4f}")
    
    # Analyze by text type
    print(f"\nBest parameters by text type:")
    for text_type in results.keys():
        type_results = [r for r in all_results if r["text_type"] == text_type]
        best_for_type = max(type_results, key=lambda x: x["similarity"])
        
        print(f"  {text_type.replace('_', ' ').title()}:")
        print(f"    Chunk size: {best_for_type['chunk_size']}, Overlap: {best_for_type['overlap']}")
        print(f"    Similarity: {best_for_type['similarity']:.4f}")
        print(f"    Improvements: vs_direct={best_for_type['improvement_vs_direct']:+.4f}, "
              f"vs_custom={best_for_type['improvement_vs_custom']:+.4f}")
    
    # Parameter sensitivity analysis
    print(f"\nParameter sensitivity analysis:")
    
    # Group by chunk_size
    chunk_sizes = {}
    for result in all_results:
        cs = result["chunk_size"]
        if cs not in chunk_sizes:
            chunk_sizes[cs] = []
        chunk_sizes[cs].append(result["similarity"])
    
    print(f"  Average similarity by chunk size:")
    for cs in sorted(chunk_sizes.keys()):
        avg_sim = sum(chunk_sizes[cs]) / len(chunk_sizes[cs])
        print(f"    {cs:3d}: {avg_sim:.4f} (n={len(chunk_sizes[cs])})")
    
    # Group by overlap
    overlaps = {}
    for result in all_results:
        ov = result["overlap"]
        if ov not in overlaps:
            overlaps[ov] = []
        overlaps[ov].append(result["similarity"])
    
    print(f"  Average similarity by overlap:")
    for ov in sorted(overlaps.keys()):
        avg_sim = sum(overlaps[ov]) / len(overlaps[ov])
        print(f"    {ov:3d}: {avg_sim:.4f} (n={len(overlaps[ov])})")
    
    # Performance analysis
    avg_improvement_direct = sum(r["improvement_vs_direct"] for r in all_results) / len(all_results)
    avg_improvement_custom = sum(r["improvement_vs_custom"] for r in all_results) / len(all_results)
    avg_time = sum(r["time"] for r in all_results) / len(all_results)
    
    print(f"\nOverall performance:")
    print(f"  Average improvement vs direct: {avg_improvement_direct:+.4f}")
    print(f"  Average improvement vs custom: {avg_improvement_custom:+.4f}")
    print(f"  Average processing time: {avg_time:.3f}s")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if avg_improvement_direct > 0.01:
        print(f"✓ LangChain chunking shows significant improvement over direct comparison")
    else:
        print(f"⚠ LangChain chunking shows minimal improvement over direct comparison")
    
    if avg_improvement_custom > 0.01:
        print(f"✓ LangChain chunking shows significant improvement over custom splitter")
    else:
        print(f"⚠ LangChain chunking shows minimal improvement over custom splitter")
    
    # Find most consistent parameters
    param_consistency = {}
    for result in all_results:
        key = (result["chunk_size"], result["overlap"])
        if key not in param_consistency:
            param_consistency[key] = []
        param_consistency[key].append(result["similarity"])
    
    # Calculate coefficient of variation for each parameter combination
    param_cv = {}
    for key, similarities in param_consistency.items():
        if len(similarities) > 1:
            mean_sim = sum(similarities) / len(similarities)
            std_sim = (sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)) ** 0.5
            cv = std_sim / mean_sim if mean_sim > 0 else float('inf')
            param_cv[key] = (mean_sim, cv)
    
    if param_cv:
        # Find parameters with good performance and low variability
        best_consistent = min(param_cv.items(), key=lambda x: x[1][1])  # Lowest CV
        best_performance = max(param_cv.items(), key=lambda x: x[1][0])  # Highest mean
        
        print(f"\nMost consistent parameters (lowest variability):")
        print(f"  Chunk size: {best_consistent[0][0]}, Overlap: {best_consistent[0][1]}")
        print(f"  Mean similarity: {best_consistent[1][0]:.4f}, CV: {best_consistent[1][1]:.4f}")
        
        print(f"\nBest performing parameters (highest mean):")
        print(f"  Chunk size: {best_performance[0][0]}, Overlap: {best_performance[0][1]}")
        print(f"  Mean similarity: {best_performance[1][0]:.4f}, CV: {best_performance[1][1]:.4f}")

def main():
    """Main function to run the chunking parameter test."""
    
    # Run the tests
    results = test_chunking_parameters()
    
    # Save results to file
    with open('chunking_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to chunking_test_results.json")
    
    # Analyze and provide recommendations
    analyze_and_recommend(results)

if __name__ == "__main__":
    main()
