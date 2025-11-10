# STED Computational Complexity Analysis

## Overview
This document analyzes the computational complexity of the Semantic Tree Edit Distance (STED) algorithm implemented in `src/semantic_json_tree_consistency.py`.

## Key Components and Their Complexities

### 1. Tree Construction
**Function**: `JsonNode.from_dict()`
- **Time Complexity**: O(n) where n is the number of elements in the JSON
- **Space Complexity**: O(n) for storing the tree structure
- **Analysis**: Linear traversal of JSON structure to build tree nodes

### 2. Embedding Computation
**Function**: `_get_embedding()`
- **Time Complexity**: O(k) where k is the text length (with caching: O(1) for cached items)
- **Space Complexity**: O(d) where d is embedding dimension (typically 384-1024)
- **Caching**: LRU cache with maxsize=2000 reduces repeated computations

### 3. Semantic Similarity Calculation
**Function**: `_calculate_semantic_similarity()`
- **Time Complexity**: O(d) for cosine similarity computation
- **Space Complexity**: O(d) for embedding vectors
- **Analysis**: Constant time for vector operations once embeddings are computed

### 4. Tree Edit Distance Algorithms

#### 4.1 Original ZSS Algorithm (`calculate_tree_edit_distance` with `original_zss=True`)
**Function**: `zss.simple_distance()`
- **Time Complexity**: O(n₁ × n₂) where n₁, n₂ are tree sizes
- **Space Complexity**: O(n₁ × n₂) for dynamic programming table
- **Analysis**: Standard tree edit distance using Zhang-Shasha algorithm

#### 4.2 Optimized STED Algorithm (`calculate_tree_edit_distance_opt`)
**Function**: `_calculate_optimal_matching_cost()`
- **Time Complexity**: O(n₁ × n₂ × (n₁ + n₂)) due to Hungarian algorithm
- **Space Complexity**: O(max(n₁, n₂)²) for cost matrix
- **Analysis**: 
  - Creates cost matrix: O(n₁ × n₂)
  - Hungarian algorithm: O(max(n₁, n₂)³)
  - Recursive calls: O(depth × branching_factor)

### 5. Array Comparison
**Function**: `_compare_arrays_unordered()`
- **Time Complexity**: O(m × n × C) where m, n are array lengths, C is comparison cost
- **Space Complexity**: O(m × n) for cost matrix
- **Hungarian Algorithm**: O(max(m,n)³) for optimal matching

### 6. String Comparison with Chunking
**Function**: `_compare_strings()`
- **Time Complexity**: 
  - Short strings: O(d) for embedding similarity
  - Long strings: O(L/chunk_size × chunk_comparisons)
- **Space Complexity**: O(number_of_chunks × d)

## Overall Complexity Analysis

### Best Case Scenario
- **Identical JSONs**: O(n) - only tree construction needed
- **Cached embeddings**: Significant speedup for repeated comparisons
- **Simple structures**: Linear complexity dominates

### Average Case Scenario
- **Time Complexity**: O(n₁ × n₂ × max(n₁, n₂)) for STED algorithm
- **Space Complexity**: O(max(n₁, n₂)²) for cost matrices
- **Embedding overhead**: O(unique_strings × embedding_time)

### Worst Case Scenario
- **Deep nested structures**: Exponential in depth due to recursive calls
- **Large arrays**: O(m³) for each array comparison
- **No caching**: All embeddings computed fresh
- **Complex strings**: Chunking overhead for long text values

## Scalability Factors

### 1. JSON Structure Characteristics
- **Depth**: Deeper nesting increases recursive overhead
- **Branching Factor**: More children per node increases comparison matrix size
- **Array Sizes**: Large arrays trigger expensive Hungarian algorithm
- **String Lengths**: Long strings require chunking and multiple comparisons

### 2. Optimization Strategies Implemented
- **LRU Caching**: Reduces embedding computation (maxsize=2000)
- **Early Termination**: Identical values return immediately
- **Structural Filtering**: Low structural similarity skips content comparison
- **Batch Processing**: Efficient for multiple comparisons

### 3. Memory Usage Patterns
- **Embedding Cache**: O(cache_size × embedding_dimension)
- **Cost Matrices**: O(max_nodes²) per comparison
- **Tree Storage**: O(total_json_elements)

## Performance Characteristics

### Computational Bottlenecks
1. **Hungarian Algorithm**: O(n³) for large arrays/objects
2. **Embedding Computation**: Network calls for Bedrock models
3. **Recursive Tree Traversal**: Exponential in pathological cases
4. **String Chunking**: Overhead for very long text values

### Optimization Opportunities
1. **Parallel Processing**: Independent subtree comparisons
2. **Approximate Algorithms**: Trade accuracy for speed
3. **Structural Pruning**: Skip obviously different subtrees
4. **Embedding Precomputation**: Batch embedding generation

## Comparison with Baseline Methods

| Method | Time Complexity | Space Complexity | Notes |
|--------|----------------|------------------|-------|
| TED (ZSS) | O(n₁ × n₂) | O(n₁ × n₂) | Standard tree edit distance |
| STED | O(n₁ × n₂ × max(n₁,n₂)) | O(max(n₁,n₂)²) | Semantic-aware with Hungarian |
| BERTScore | O(text_length) | O(embedding_dim) | Linear in text length |
| DeepDiff | O(n₁ + n₂) | O(n₁ + n₂) | Efficient structural diff |
| GNN | O(nodes × edges) | O(graph_size) | Graph neural network |

## Practical Performance Implications

### Small JSONs (< 100 elements)
- **STED**: ~10-100ms per comparison
- **Bottleneck**: Embedding computation
- **Recommendation**: Use caching aggressively

### Medium JSONs (100-1000 elements)
- **STED**: ~100ms-1s per comparison
- **Bottleneck**: Hungarian algorithm for large arrays
- **Recommendation**: Consider structural filtering

### Large JSONs (> 1000 elements)
- **STED**: 1s+ per comparison
- **Bottleneck**: Recursive tree traversal and cost matrices
- **Recommendation**: Use approximate methods or sampling

## Conclusion

STED provides semantic-aware JSON comparison at the cost of increased computational complexity compared to purely structural methods. The algorithm scales reasonably for typical JSON sizes but may require optimization for very large or deeply nested structures. The semantic benefits often justify the computational overhead for applications requiring nuanced similarity assessment.
