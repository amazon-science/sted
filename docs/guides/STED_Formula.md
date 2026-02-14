# Complete STED Similarity Formula

This document describes the mathematical formula for Semantic Tree Edit Distance (STED) similarity calculation.

## 1. Base Cost Functions

### Path Weight

$$w(n) = \lambda^{depth(n)} \times \begin{cases} 1.5 & \text{if } n \in \text{required\_fields} \\ 1.0 & \text{otherwise} \end{cases}$$

where:
- $\lambda$ = `path_weight_decay` (default 1.0)
- $depth$ = count of `.` and `[` in path

### Insert/Delete Cost

$$C_{ins}(n) = C_{del}(n) = 1.0 \times w(n)$$

### Structural Update Cost

$$C_{struct}(n_1, n_2) = (1 - S_{struct}(n_1, n_2)) \times \frac{w(n_1) + w(n_2)}{2}$$

where structural similarity is calculated as:

$$S_{struct}(n_1, n_2) = \begin{cases}
1.0 & \text{if } label(n_1) = label(n_2) \text{ and } type(n_1) = type(n_2) \text{ and } |children(n_1)| = |children(n_2)| \\
\frac{S_{field}(n_1, n_2) + P_{match}(n_1, n_2)}{2} & \text{otherwise}
\end{cases}$$

where:
- $S_{field}(n_1, n_2)$ = semantic similarity between field names (using embeddings)
  - Field names are extracted from the last component of the label
  - Array indices and special characters are normalized
- $P_{match}(n_1, n_2)$ = path match indicator:
  - $P_{match} = 1$ if paths are identical (after normalizing array indices `[0], [1], ...` to `[*]`)
  - $P_{match} = 0$ otherwise

### Content Update Cost

$$C_{content}(n_1, n_2) = \begin{cases}
(1 - S_{value}(n_1, n_2)) \times \frac{w(n_1) + w(n_2)}{2} & \text{if } S_{struct}(n_1, n_2) > \theta_{struct} \\
1.0 \times \frac{w(n_1) + w(n_2)}{2} & \text{otherwise}
\end{cases}$$

where:
- $\theta_{struct}$ = `structural_sim_threshold` (default 0.3)
- $S_{struct}(n_1, n_2)$ = structural similarity between nodes (defined in Structural Update Cost section above)
- $S_{value}(n_1, n_2)$ = pure value-based similarity (no structural penalty):
  - For same type: value equality or semantic similarity
  - For strings: embedding similarity or BERTScore
  - For numbers: equality check
  - For different types: semantic similarity of string representations with small type penalty

**Key insight**: Structural similarity acts as a **threshold gate**, not a penalty. Once the threshold is passed, content similarity is calculated purely based on values without structural penalties. This allows cross-matching of values between different keys (e.g., `{"a": "X", "b": "Y"}` vs `{"a": "Y", "b": "X"}` can achieve perfect content similarity).

### Combined Update Cost

$$C_{combined}(n_1, n_2) = 0.5 \times C_{struct}(n_1, n_2) + 0.5 \times C_{content}(n_1, n_2)$$

---

## 2. Tree Edit Distance (Recursive with Hungarian Algorithm)

For two trees $T_1$ and $T_2$ with children $\{c_1^{(1)}, ..., c_{n_1}^{(1)}\}$ and $\{c_1^{(2)}, ..., c_{n_2}^{(2)}\}$:

### Base Case (Leaf Nodes)

$$TED(T_1, T_2) = C_{type}(T_1, T_2)$$

where $type \in \{struct, content, combined\}$

### Recursive Case (Internal Nodes)

**Step 1: Build cost matrix** $M$ of size $\max(n_1, n_2) \times \max(n_1, n_2)$:

$$M[i][j] = \begin{cases}
TED(c_i^{(1)}, c_j^{(2)}) & \text{if } i < n_1 \text{ and } j < n_2 \\
C_{del}(c_i^{(1)}) & \text{if } i < n_1 \text{ and } j \geq n_2 \\
C_{ins}(c_j^{(2)}) & \text{if } i \geq n_1 \text{ and } j < n_2 \\
\infty & \text{otherwise}
\end{cases}$$

**Step 2: Solve Hungarian algorithm** for optimal assignment:

$$(\pi^*, \sigma^*) = \arg\min_{\pi, \sigma} \sum_{k} M[\pi_k][\sigma_k]$$

**Step 3: Calculate total cost:**

$$\text{total\_cost} = \sum_{k} M[\pi^*_k][\sigma^*_k]$$

**Step 4: Add size penalty:**

$$\text{matched} = |\{(i,j) : i < n_1 \text{ and } j < n_2 \text{ in assignment}\}|$$

$$\text{unmatched} = (n_1 - \text{matched}) + (n_2 - \text{matched})$$

$$\text{penalty} = \min(\text{unmatched} \times 0.1, \max(n_1, n_2) \times 0.3)$$

**Step 5: Normalize:**

$$TED(T_1, T_2) = \min\left(\frac{\text{total\_cost} + \text{penalty}}{|\text{assignments}|}, 1.0\right)$$

---

## 3. Final Similarity

$$\text{Similarity}(J_1, J_2) = 1 - TED(T_1, T_2)$$

---

## 4. Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    _calculate_optimal_matching_cost              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  IF both leaf nodes:                                             │
│     return C_type(n1, n2)      <- update cost based on type      │
│                                                                  │
│  IF one leaf, one internal:                                      │
│     return C_del(leaf) + sum(C_ins(children))                    │
│                                                                  │
│  IF both have children:                                          │
│     ┌──────────────────────────────────────────────────────┐    │
│     │  Build Cost Matrix M[i][j]                            │    │
│     │  ┌─────────────────────────────────────────────────┐ │    │
│     │  │ TED(c1,c1') TED(c1,c2') ... C_del(c1)           │ │    │
│     │  │ TED(c2,c1') TED(c2,c2') ... C_del(c2)           │ │    │
│     │  │ C_ins(c1')  C_ins(c2')  ...    inf              │ │    │
│     │  └─────────────────────────────────────────────────┘ │    │
│     │                                                       │    │
│     │  Hungarian Algorithm -> optimal assignment            │    │
│     │                                                       │    │
│     │  total_cost = sum(M[pi*][sigma*])                    │    │
│     │  penalty = min(unmatched * 0.1, max_size * 0.3)      │    │
│     │  normalized = min((total + penalty) / |assign|, 1.0) │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│  return normalized_cost                                          │
└─────────────────────────────────────────────────────────────────┘

Final: Similarity = 1 - normalized_cost
```

---

## 5. Relationship Between Combined, Structural, and Content

### At Leaf Node Level (Formula HOLDS)

```
Combined_cost = 0.5 * Structural_cost + 0.5 * Content_cost
```

### At Tree Level (NEW APPROACH: Structure-Guided Combined)

The Hungarian algorithm is solved **differently** for each variation type:

**Structural Similarity:**
$$\pi^*_{struct} = \arg\min \sum M_{struct}[\pi][\sigma]$$

**Content Similarity:**
$$\pi^*_{content} = \arg\min \sum M_{content}[\pi][\sigma]$$

**Combined Similarity (NEW):**
Instead of solving Hungarian on combined costs, we use the structural matching:

1. **Find optimal structural matching**: $\pi^*_{struct} = \arg\min \sum M_{struct}[\pi][\sigma]$
2. **Calculate content costs on that matching**: For each pair $(i,j)$ in $\pi^*_{struct}$, compute $C_{content}(c_i, c_j)$
3. **Weighted average**:
$$TED_{combined} = 0.5 \times TED_{struct} + 0.5 \times \sum_{(i,j) \in \pi^*_{struct}} C_{content}(c_i, c_j)$$

**Key difference**: Combined similarity now **respects structural alignment** by using the structural matching as the base, then calculating content costs on that fixed alignment.

### Benefits of Structure-Guided Combined

1. **Eliminates the value swap paradox**: Combined will now be between Structural and Content
2. **More semantically meaningful**: Measures "how much content changed given structural alignment"
3. **Prioritizes key-to-key matching**: `age→age`, not `age→name`
4. **Better reflects real-world JSON edits**: Most edits preserve key structure

### Comparison: Old vs New

| Scenario | Structural | Content | Combined (Old) | Combined (New) |
|----------|-----------|---------|----------------|----------------|
| `{"a": "X", "b": "Y"}` vs `{"a": "Y", "b": "X"}` | 1.0000 | 1.0000 | **0.8878** ⚠️ | **~0.5-0.7** ✓ |
| Same keys, different values | 1.0000 | varies | could be < both | between S and C ✓ |
| Same keys, same values | 1.0000 | 1.0000 | 1.0000 | 1.0000 ✓ |

The old approach had the counter-intuitive property that combined could be **lower than both** structural and content. The new approach ensures combined is always bounded by them.

---

## 6. The Value Swap Paradox: When Combined < min(Structural, Content)

> **Note**: This paradox existed in the **old approach** where Hungarian algorithm was solved independently for combined costs. The **new structure-guided approach** (Section 5) resolves this paradox.

### Can Combined Be Smaller Than Both S and C? (Old Approach)

**Yes (in old approach)!** Combined similarity could be lower than both Structural and Content similarities. This occurred when the optimal matchings for S and C were **opposite**.

### Real Examples (Old Approach)

| JSON1 | JSON2 | Structural | Content | Combined (Old) |
|-------|-------|------------|---------|----------------|
| `{"a": "X", "b": "Y"}` | `{"a": "Y", "b": "X"}` | **1.0000** | **1.0000** | **0.8878** ⚠️ |
| `{"x": 1, "y": 2}` | `{"x": 2, "y": 1}` | **1.0000** | **1.0000** | **0.6939** ⚠️ |
| `{"name": "hello", "value": "world"}` | `{"name": "world", "value": "hello"}` | **1.0000** | **1.0000** | **0.8363** ⚠️ |

### Why This Happened (Old Approach)

```
JSON1: {"a": "X", "b": "Y"}
JSON2: {"a": "Y", "b": "X"}

STRUCTURAL perspective (match by field names):
  a="X" ↔ a="Y"  ✓ Same field name
  b="Y" ↔ b="X"  ✓ Same field name
  → Perfect structural match! Similarity = 1.0

CONTENT perspective (match by values):
  "X" ↔ "X"  ✓ Same value (found at different keys)
  "Y" ↔ "Y"  ✓ Same value (found at different keys)
  → Perfect content match! Similarity = 1.0

COMBINED perspective (OLD: must balance both):
  If match a↔a: structure ✓, but content ✗ (X≠Y)
  If match a↔b: content ✓ (X=X), but structure ✗ (different keys)
  → NO perfect matching exists! Similarity < 1.0 ⚠️
```

### How New Approach Resolves This

```
COMBINED perspective (NEW: structure-guided):
  1. Use structural matching: a↔a, b↔b (structure=perfect)
  2. Calculate content costs on that matching:
     - a="X" vs a="Y": content mismatch (cost = high)
     - b="Y" vs b="X": content mismatch (cost = high)
  3. Combined = 0.5 × 0.0 + 0.5 × high ≈ 0.5-0.7
  → Result makes sense: perfect structure, bad content ✓
```

**Key insight**: The new approach prioritizes structural alignment, then measures content differences within that alignment. This is more semantically meaningful than trying to find a compromise matching.

### Visual Explanation

```
                STRUCTURAL              CONTENT
                (by field name)         (by value)

JSON1:  a="X"  ←──────────→  a="Y"
        b="Y"  ←──────────→  b="X"
                   ↑
          Perfect match!

JSON1:  a="X" ──┐                       "X" ←─── a="Y"
        b="Y" ──┼──→ values             "Y" ←─── b="X"
                └──→ matched                  ↑
                         cross-match    Perfect match!

COMBINED: Cannot satisfy both simultaneously!
┌───────────────────────────────────────────────┐
│  Match a↔a: struct✓ content✗ (X vs Y)        │
│  Match a↔b: struct✗ content✓ (X vs X)        │
│  Either way, one constraint is violated!      │
└───────────────────────────────────────────────┘
```

### Cost Matrix Analysis

For the swapped values example:

```
                  a="Y"    b="X"
      ────────────────────────────
a="X" │  struct=0  struct=HIGH
      │  content=HIGH content=0
      │  → combined=MID
      │
b="Y" │  struct=HIGH struct=0
      │  content=0  content=HIGH
      │  → combined=MID
```

**Key insight:**
- Structural costs: Diagonal is best (0) - prefers a↔a, b↔b
- Content costs: Anti-diagonal is best (0) - prefers a↔b, b↔a
- Combined costs: All cells are mediocre - no clear winner!

### Mathematical Explanation

When S and C have **anti-correlated** optimal matchings:

$$\text{S optimal: } \pi_S = \{(a,a), (b,b)\} \text{ with } cost_S = 0$$

$$\text{C optimal: } \pi_C = \{(a,b), (b,a)\} \text{ with } cost_C = 0$$

For Combined:
- Using $\pi_S$: $cost_{combined}(\pi_S) = 0.5 \times 0 + 0.5 \times HIGH > 0$
- Using $\pi_C$: $cost_{combined}(\pi_C) = 0.5 \times HIGH + 0.5 \times 0 > 0$

Neither matching is optimal for Combined → Combined similarity < min(S, C)

### Why Content Matching Allows Cross-Matching

**Key Implementation Detail**: In `_calculate_content_similarity()` (line 362), structural similarity acts as a **threshold gate**, not a penalty:

```python
structural_sim = self._calculate_structural_similarity(node1, node2)

if not node1.children and not node2.children and structural_sim > structural_sim_threshold:
    # Content similarity is calculated here - purely based on values
    # structural_sim doesn't penalize the result!
```

**This means:**
1. **Structural similarity is just a threshold check** - if `structural_sim > 0.3` (default), then content similarity is calculated
2. **Content similarity is NOT penalized by structural differences** - once the threshold is passed, it's purely value-based
3. **Cross-matching is allowed** - for sibling keys like `a` and `b`, both have sufficient structural similarity with each other, so content comparisons happen for all pairs

**For the swap example** `{"a": "X", "b": "Y"}` vs `{"a": "Y", "b": "X"}`:

```
Content Cost Matrix (after passing threshold):
              a="Y"    b="X"
a="X"    [1.0 (X≠Y)] [0.0 (X=X)] ← "X" matches "X" perfectly
b="Y"    [0.0 (Y=Y)] [1.0 (Y≠X)] ← "Y" matches "Y" perfectly
```

- Both `a` and `b` have high structural similarity with each other (sibling keys at same level)
- All pairs pass the `structural_sim_threshold` gate
- Content similarity is calculated purely based on values
- Hungarian algorithm finds the optimal anti-diagonal matching: `a↔b` and `b↔a`
- **Result**: Content similarity = 1.0 (perfect value match via cross-matching)

**Why this design?**
- Allows flexibility in matching nodes that are structurally similar but not identical
- Prevents over-constraining the content matching process
- The threshold ensures we only compare "reasonable" node pairs (e.g., sibling keys, similar paths)

### When Does This Occur?

**Value swap scenarios:**
- Same keys with swapped values: `{"a": 1, "b": 2}` vs `{"a": 2, "b": 1}`
- Type conversions with swaps: `{"id": 100, "name": "test"}` vs `{"id": "test", "name": 100}`
- Rotated values: `{"a": 1, "b": 2, "c": 3}` vs `{"a": 2, "b": 3, "c": 1}`

**Why it's rare in practice:**
- Real JSON data usually has semantic correlation between field names and values
- `{"age": 30, "name": "John"}` rarely becomes `{"age": "John", "name": 30}`
- Most edits preserve the key-value relationship

---

## 7. Code Reference

The implementation can be found in:
- `sted/semantic_json_tree_consistency.py`
  - `_calculate_optimal_matching_cost()` - Main recursive algorithm (line 546)
  - `update_cost()` - Combined cost function (line 325)
  - `structural_update_cost()` - Structural cost (line 395)
  - `content_update_cost()` - Content cost (line 412)
  - `insert_cost()` / `delete_cost()` - Insert/delete costs (lines 234, 252)
  - `calculate_path_weight()` - Path weighting (line 209)
