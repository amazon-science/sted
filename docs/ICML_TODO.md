# STED ICML 2026 Submission TODO

**Target:** ICML 2026 Main Conference
**Current Status:** Paper draft complete, ablations partially done
**Deadline:** TBD (typically late January)

---

## Current Paper Status Summary

### Completed Sections
- [x] Abstract - comprehensive, includes key results
- [x] Introduction - motivation, contributions, key innovations
- [x] Related Work - comprehensive coverage (4 subsections)
- [x] Methodology - STED formulation, consistency score, complexity
- [x] Experiments - synthetic validation + 20-model benchmark
- [x] Conclusion - applications, limitations
- [x] Appendix - proofs, dataset analysis, statistical details

### Key Results in Paper
- 20 LLMs benchmarked across 11 providers
- 2 evaluation tasks: ShareGPT (80 samples) + Toucan (1,006 samples)
- 2.4M+ total LLM outputs evaluated
- STED achieves 0.86-0.90 for semantic equivalents, 0.0 for structural breaks
- 4× better discrimination than baselines

---

## Phase 1: Theoretical Foundations ✅ COMPLETE

- [x] Theorem 1: STED metric properties (non-negativity, identity, symmetry, triangle inequality)
- [x] Theorem 2: Hungarian algorithm optimality
- [x] Corollary: Computational complexity O(N·B³ + N·T_φ)
- [x] Proposition: Consistency score properties
- [x] Proposition: Convergence rate O(n^{-1/2})
- [x] All proofs in Appendix

---

## Phase 2: Ablation Studies 🔄 IN PROGRESS

### Completed
- [x] **α (consistency steepness)**: Table + Figure in paper
  - Values tested: [5, 10, 15, 20, 25]
  - Default: α=20 (balances range normalization + discrimination)

### In NeurIPS but NOT in ICML version
- [ ] **w (structural-content weight)**: Add to ICML paper
  - Values: [0.3, 0.5, 0.7]
  - Default: w=0.5
  - Table exists in NeurIPS version (Table 1)

- [ ] **θ (structural threshold)**: Add to ICML paper
  - Values: [0.1, 0.3, 0.5, 0.7]
  - Default: θ=0.3
  - Table exists in NeurIPS version (Table 2)

### Completed
- [x] **Embedding model ablation**: ✅ COMPLETE (2025-01-28)
  - Script: `scripts/eval/embedding_model_ablation.py`
  - Results: `results/embedding_ablation/ablation_9models.json`
  - **9 models tested** on 200 Toucan samples:
    - Local sentence-transformers (6): all-MiniLM-L6-v2, all-mpnet-base-v2, BAAI/bge-base-en-v1.5, BAAI/bge-large-en-v1.5, intfloat/e5-base-v2, intfloat/e5-large-v2
    - AWS Bedrock Titan (3): Titan-256, Titan-512, Titan-1024
  - **Key Results**:
    - All Spearman correlations > 0.995 (range: 0.9952-1.0000)
    - All Pearson correlations > 0.999 (range: 0.9993-1.0000)
    - Mean c_mean scores: 0.587-0.593 (< 1% variance)
  - **Conclusion**: STED scores highly robust to embedding model choice
  - **Decision**: Add table to paper confirming "minimal impact" claim

---

## Phase 3: Paper Improvements 📝 TODO

### Missing from ICML vs NeurIPS
- [ ] Add w and θ ablation tables (from NeurIPS version)
- [ ] Add schema robustness analysis tables (Tables 12-13 in NeurIPS)
  - Field name evolution robustness
  - Schema restructuring patterns

### Figures Status
- [x] Figure 1: schema_variation_analysis.png
- [x] Figure 2: similarity_progression_combined.png
- [x] Figure 3: consistency_score_by_consistency_type_with_errors.png
- [x] Figure 4: consistency_steepness_analysis.png
- [ ] Consider: STED overview/architecture diagram

### Tables Status
- [x] Table 1: α ablation (range normalization vs discrimination)
- [x] Table 2: Consistency summary by model
- [x] Table 3: Overall consistency across temperatures
- [ ] Add: w ablation table
- [ ] Add: θ ablation table
- [x] Embedding model comparison table (9 models, Spearman > 0.995)

---

## Phase 4: Experimental Gaps 🔬

### Scalability Analysis
- [ ] Runtime benchmarks vs JSON size (mentioned but no figure)
- [ ] Consider adding scalability plot

### Additional Baselines (Optional)
- [x] TED (Zhang-Shasha) ✅
- [x] BERTScore ✅
- [x] DeepDiff ✅
- [ ] Optional: JSONDiff, jq-based comparison

---

## Phase 5: Code & Reproducibility 📦

### Scripts Reference
| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/eval/embedding_model_ablation.py` | Embedding ablation | ✅ Complete |
| `scripts/eval/calculate_consistency_metrics.py` | Main consistency calculation | ✅ |
| `scripts/eval/generate_structured_outputs.py` | LLM output generation | ✅ |
| `scripts/data/generate_synthetic_datasets.py` | Synthetic variation generation | ✅ |

### Before Submission
- [ ] Verify all figures regenerable from scripts
- [ ] Test reproducibility on clean environment
- [ ] Update requirements.txt if needed

---

## Immediate Action Items (Priority Order)

1. ~~**HIGH**: Run embedding model ablation study~~ ✅ COMPLETE
   - Results: All Spearman correlations > 0.995, < 1% variance in mean scores
   - Action: Add brief table to ablation section confirming robustness

2. **MEDIUM**: Port w and θ ablation tables from NeurIPS to ICML
   - Copy Tables 1-2 from neurips_2025_camera_ready.tex
   - Adjust formatting for ICML style

3. **LOW**: Consider adding schema robustness analysis
   - Tables 12-13 from NeurIPS show important robustness properties

---

## Key Reviewer Concerns to Address

| Concern | Current Status | Action Needed |
|---------|---------------|---------------|
| "Is STED just TED + embeddings?" | Addressed in intro | ✅ None |
| "Why not BERTScore on JSON?" | Empirical comparison shows order sensitivity | ✅ None |
| "How does it scale?" | Complexity stated, linear scaling confirmed | Consider adding figure |
| "Are hyperparameters principled?" | α ablation complete | Add w, θ ablations |
| "Does embedding choice matter?" | ✅ Validated: r>0.995 across 9 models | Add table to paper |
| "Real LLM evaluation?" | 20 models, 2.4M outputs | ✅ Strong |

---

## Files Reference

| File | Description |
|------|-------------|
| `docs/ICML_paper/icml2026_submission.tex` | Main ICML submission |
| `docs/ICML_paper/neurips_2025_camera_ready.tex` | NeurIPS workshop version (more detailed) |
| `docs/ICML_paper/neurips_2025_camera_ready.bib` | Shared bibliography |
| `docs/ICML_paper/figures/` | All paper figures |
| `scripts/eval/embedding_model_ablation.py` | Embedding ablation script |
| `results/embedding_ablation/ablation_9models.json` | Embedding ablation results (9 models, 200 samples) |

---

## Changelog

- **2025-01-28**: Embedding model ablation study completed
  - Ran ablation with 9 embedding models on 200 Toucan samples
  - Models: 6 sentence-transformers (MiniLM, mpnet, BGE, E5) + 3 Titan variants
  - Results: All Spearman correlations > 0.995, < 1% variance in scores
  - Validates "minimal impact" claim for embedding model choice
  - Results saved: `results/embedding_ablation/ablation_9models.json`

- **2025-01-23**: Updated TODO based on current paper review
  - Marked completed sections
  - Identified missing ablations (w, θ) from NeurIPS version
  - Created embedding ablation script
  - Prioritized action items

- **2024-12-19**: Initial TODO created

---

*Last updated: 2025-01-28*
