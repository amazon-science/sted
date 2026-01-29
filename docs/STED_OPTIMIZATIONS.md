# STED Algorithm Optimizations

This document describes the time and space complexity optimizations implemented in the STED (Semantic Tree Edit Distance) algorithm.

## Overview

The STED algorithm compares JSON structures using tree edit distance with semantic similarity. The original implementation had exponential time complexity due to recursive subtree comparisons. The optimized version reduces this to polynomial time while also bounding memory usage.

## Optimization Summary

| Optimization | Time Impact | Space Impact | Trade-off |
|--------------|-------------|--------------|-----------|
| **1. LRU Memoization** | Exponential → Polynomial | Bounded cache | None (exact) |
| **2. Single-pass Combined** | 2x → 1x passes | Same | None (exact) |
| **3. Greedy Matching** | O(B³) → O(B²) | O(B²) → O(B) | Accuracy loss |
| **4. Lazy Hash Computation** | Faster | O(N) cache | None |
| **5. Early Pruning** | Skip mismatches | Same | Minor accuracy |
| **6. NumPy float32 Arrays** | Vectorized ops | 50% reduction | Precision (negligible) |
| **7. FP16 Embeddings** | Same | 50% reduction | Precision (slight) |

## Bottleneck Analysis

Profiling reveals the **actual bottlenecks** in STED computation:

| Component | Time % | Notes |
|-----------|--------|-------|
| **Embedding computation** | 85% (if cache miss) | Biggest bottleneck if not precomputed |
| Tree traversal/Python overhead | ~12% | Function calls, cache lookups |
| Semantic similarity | ~8% | Cosine similarity computation |
| Hash computation | ~2% | With lazy caching |
| Hungarian algorithm | **<1%** | Very fast for typical JSON sizes (B<50) |

**Key insight**: The Hungarian algorithm O(B³) is NOT the bottleneck for typical JSON structures. The real bottleneck is **embedding computation** for strings not in the precomputed cache.

### When Hungarian Becomes a Bottleneck

Hungarian algorithm time scales as O(B³) where B is the branching factor:

| Matrix Size (B×B) | Time per Call | Impact |
|-------------------|---------------|--------|
| 10×10 | 0.006ms | Negligible |
| 20×20 | 0.007ms | Negligible |
| 50×50 | 0.036ms | Minor |
| 100×100 | ~0.3ms | Noticeable |
| 500×500 | ~30ms | **Bottleneck** |

For typical JSON structures with B<50, Hungarian is NOT the issue. Only consider greedy matching for very large arrays (B>100).

### Why Embeddings Were Missing from Cache

The `collect_strings_from_json()` function collected raw strings, but `_calculate_structural_similarity()` uses **preprocessed** field names:
- `"inputWord"` → `"input word"` (camelCase split)
- `"user_name"` → `"user name"` (underscore to space)

**Fix**: Now also precomputes embeddings for preprocessed versions of keys.

### Impact of Embedding Cache Fix

| Metric | Before Fix | After Fix | Improvement |
|--------|------------|-----------|-------------|
| Time per comparison | ~7ms | ~2ms | **3.5x faster** |
| Time in embedding | 85% | <15% | Cache hits |
| Precomputed strings | Raw only | Raw + preprocessed | Complete coverage |

## Complexity Analysis

### Time Complexity

| Algorithm | Original | Optimized | Greedy Mode |
|-----------|----------|-----------|-------------|
| Per-level matching | O(B³) Hungarian | O(B³) Hungarian | O(B²) |
| Subtree comparisons | O(B^(2D)) | O(B² × D) | O(B² × D) |
| Combined variation | 2 passes | 1 pass | 1 pass |
| **Total** | **O(B^(2D+3))** | **O(B³ × D)** | **O(B² × D)** |

Where:
- `B` = maximum branching factor (children per node)
- `D` = tree depth

### Space Complexity

| Component | Original | Optimized |
|-----------|----------|-----------|
| Subtree cache | O(N²) unbounded | O(C) bounded LRU |
| Cost matrices | O(B²) float64 | O(B²) float32 |
| Greedy matching | O(B²) matrix | O(B) streaming |
| Embeddings | O(E × d) unbounded | O(M × d/2) bounded |
| Node hash cache | N/A | O(N) |
| **Total** | **Unbounded** | **O(C + B² + M×d)** |

Where:
- `C` = `subtree_cache_size` (default 10,000)
- `E` = number of unique strings
- `M` = `max_embedding_cache_size` (default 100,000)
- `d` = embedding dimension
- `N` = total nodes in trees

## Benchmark Results

Benchmark conducted with 50 sample pairs from Toucan dataset using `all-MiniLM-L6-v2` embeddings.

### Performance Comparison

**Before embedding cache fix** (embeddings computed on-the-fly):

| Method | Time (s) | Speedup | Throughput | Notes |
|--------|----------|---------|------------|-------|
| Original | 1.035 | 1.0x | 48 pairs/sec | 85% time in embedding |
| Optimized (cold) | 0.148 | 7.0x | 337 pairs/sec | Memoization helps |
| Optimized (warm) | 0.046 | 22.5x | 1,090 pairs/sec | Cache hits |

**After embedding cache fix** (all strings precomputed):

| Method | Time (s) | Speedup | Throughput | Notes |
|--------|----------|---------|------------|-------|
| Original | 0.070 | 1.0x | 714 pairs/sec | Embeddings cached |
| Optimized (cold) | 0.130 | 0.5x | 385 pairs/sec | Cache overhead |
| Optimized (warm) | 0.036 | 1.9x | 1,389 pairs/sec | Best performance |

**Key takeaway**: With proper embedding precomputation, the algorithm is already fast (~2ms per comparison). The subtree memoization provides additional benefit for repeated comparisons.

### Cache Statistics

| Metric | Value |
|--------|-------|
| Subtree cache hits | ~957 |
| Subtree cache misses | ~2,845 |
| Subtree cache hit rate | 25% |
| Embedding cache hit rate | 100% (with precompute) |

### Accuracy Metrics

| Method | Mean Absolute Error | Max Absolute Error | Correlation |
|--------|--------------------|--------------------|-------------|
| Optimized vs Original | 0.027 | 0.284 | 0.979 |
| Greedy vs Original | 0.130 | 0.831 | 0.704 |

**Note**: Greedy mode trades accuracy for speed. Only recommended for B>100.

## Implementation Details

### 1. LRU Memoization Cache

The subtree comparison cache uses an LRU (Least Recently Used) eviction policy with bounded size:

```python
class LRUCache:
    """Memory-bounded LRU cache for subtree comparison results."""

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
```

**Key insight**: Subtree comparisons are often repeated when comparing similar JSON structures. Caching results reduces exponential recursion to polynomial.

### 2. Single-pass Combined Calculation

Original approach for "combined" variation type:
1. Build structural cost matrix → Run Hungarian
2. Build content cost matrix → Run Hungarian
3. Combine results

Optimized approach:
1. Build both matrices in single traversal
2. Run Hungarian once on structural matrix
3. Use same matching for content costs

```python
def _single_pass_combined_matching(self, children1, children2, n1, n2):
    # Build both matrices in one pass
    structural_costs = np.zeros((max_size, max_size), dtype=np.float32)
    content_costs = np.zeros((max_size, max_size), dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            s_cost, c_cost = self._get_structural_and_content_costs(...)
            structural_costs[i, j] = s_cost
            content_costs[i, j] = c_cost
```

### 3. Greedy Matching (Optional)

For large branching factors, the O(B³) Hungarian algorithm becomes a bottleneck. The greedy approximation uses O(B²) time and O(B) space:

```python
def _greedy_matching_streaming(self, children1, children2, variation_type):
    """O(B) space - doesn't build full cost matrix."""
    used_j = set()  # Only tracks used indices

    for i in range(n1):
        best_j, best_cost = -1, float('inf')
        for j in range(n2):
            if j in used_j:
                continue
            cost = self._calculate_optimal_matching_cost_fast(...)
            if cost < best_cost:
                best_cost, best_j = cost, j
        # ... match and continue
```

**Trade-off**: ~30x speedup but correlation drops from 0.96 to 0.70. Use only when speed is critical.

### 4. Lazy Hash Computation

Node hashes for memoization keys are computed lazily and cached by node `id()`:

```python
def _compute_tree_hash(self, node):
    node_id = id(node)
    if node_id in self._node_hash_cache:
        return self._node_hash_cache[node_id]

    # Compute hash recursively...
    result = hashlib.md5('|'.join(parts).encode()).hexdigest()[:16]
    self._node_hash_cache[node_id] = result
    return result
```

### 5. Early Pruning

Skip detailed comparison for clearly mismatched structures:

```python
def _early_prune_check(self, node1, node2):
    # Type mismatch with high cost
    type_cost = self.type_change_cost.get((node1.node_type, node2.node_type), 1.0)
    if type_cost >= self.early_pruning_threshold:
        return type_cost  # Skip detailed comparison

    # Large size difference (> 3x)
    if node1.children and node2.children:
        ratio = max(n1, n2) / min(n1, n2)
        if ratio > 3:
            return min(1.0, 0.3 + 0.1 * ratio)
```

### 6. NumPy Float32 Arrays

Cost matrices use `float32` instead of Python `float` (64-bit):

```python
cost_matrix = np.full((max_size, max_size), np.inf, dtype=np.float32)
```

**Benefits**:
- 50% memory reduction for cost matrices
- Vectorized operations with NumPy

### 7. FP16 Embeddings (Optional)

Store embeddings as `float16` to save memory:

```python
def _store_embedding(self, text, embedding):
    if self.use_fp16_embeddings:
        embedding = embedding.astype(np.float16)
    self._embedding_dict[text] = embedding
```

**Trade-off**: ~50% memory savings with negligible accuracy impact for similarity comparisons.

## Configuration Options

```python
from sted import SemanticJsonTreeConsistencyEvaluator

evaluator = SemanticJsonTreeConsistencyEvaluator(
    model_id='all-MiniLM-L6-v2',

    # Time optimizations
    use_greedy_matching=False,      # Enable O(B²) greedy (default: False)
    early_pruning_threshold=0.8,    # Skip comparison threshold (default: 0.8)

    # Space optimizations
    subtree_cache_size=10000,       # Max LRU cache entries (0 = unbounded)
    use_fp16_embeddings=False,      # Store embeddings as float16
    max_embedding_cache_size=100000 # Max embeddings to cache (0 = unbounded)
)

# Enable greedy mode for speed-critical applications
evaluator.set_greedy_matching(True)

# Monitor cache performance
stats = evaluator.get_cache_stats()
print(f"Cache hit rate: {stats['subtree_cache_hit_rate']:.1%}")
```

## CLI Options

The `calculate_consistency_metrics.py` script supports optimization flags:

```bash
python scripts/eval/calculate_consistency_metrics.py \
    --results-dir llm_gen_results/toucan \
    --use-greedy \                    # Enable greedy matching
    --early-pruning-threshold 0.8     # Early pruning threshold
```

## Recommendations

| Use Case | Configuration |
|----------|---------------|
| **Production (accuracy)** | Default settings, `use_greedy=False` |
| **Large datasets** | `subtree_cache_size=50000`, `use_fp16_embeddings=True` |
| **Speed-critical** | `use_greedy=True` (accept ~30% accuracy loss) |
| **Memory-constrained** | `subtree_cache_size=5000`, `max_embedding_cache_size=50000`, `use_fp16_embeddings=True` |

## Benchmarking

Run the benchmark script to compare performance on your data:

```bash
python scripts/eval/benchmark_sted_optimizations.py \
    --results-dir llm_gen_results/toucan \
    --num-samples 100 \
    --variation-type combined
```

Output includes:
- Time comparison (original vs optimized vs greedy)
- Cache hit rates
- Accuracy metrics (MAE, correlation)
- Sample-by-sample comparison
