# STED ICML 2026 Submission TODO

**Target:** ICML 2026 Main Conference
**Current Status:** NeurIPS 2025 Workshop Paper
**Deadline:** TBD (typically late January)

---

## Phase 1: Theoretical Foundations (HIGH PRIORITY)

### Completed
- [x] Formal definitions (JSON tree, embedding, semantic similarity, cost functions)
- [x] Theorem 1: STED quasi-metric properties with proof
- [x] Theorem 2: Hungarian algorithm optimality with proof
- [x] Theorem 3: Computational complexity analysis
- [x] Proposition: Bounded combined similarity (value swap paradox resolution)
- [x] Proposition: Consistency score properties and convergence
- [x] Algorithm pseudocode
- [x] LaTeX document: `docs/sted_theory_icml.tex`

### TODO
- [ ] **Tighten triangle inequality bound** - Current relaxation factor ε needs empirical validation
- [ ] **Add convergence rate proof** - Formalize O(1/√n) convergence for consistency score
- [ ] **Embedding assumption validation** - Verify L-Lipschitz property for Sentence-BERT
- [ ] **Add space complexity proof** - Complete space analysis for memoization

---

## Phase 2: Experimental Validation (HIGH PRIORITY)

### Ablation Studies
- [x] Create ablation experiment script: `scripts/experiments/run_hyperparameter_ablation.py`
- [ ] **Run full ablation study** on all parameters:
  - [ ] α (steepness): [5, 10, 15, 20, 25, 30]
  - [ ] θ (structural threshold): [0.1, 0.2, 0.3, 0.5, 0.7]
  - [ ] λ (path decay): [0.7, 0.8, 0.9, 1.0]
  - [ ] w (structural weight): [0.3, 0.5, 0.7]
- [ ] Generate publication-quality ablation figures
- [ ] Fill in Table 1 in `sted_theory_icml.tex` with empirical results

### Dataset Analysis
- [x] xLAM analysis script
- [x] Glaive analysis script
- [x] Maxscha analysis script (regular + large)
- [x] Jiraya analysis script
- [ ] **Run all dataset analyses** and save results
- [ ] **Create dataset comparison table** for paper
- [ ] **Add real LLM output evaluation** on each dataset

### Baseline Comparisons
- [ ] **Implement baselines:**
  - [ ] Edit distance (Levenshtein)
  - [ ] Tree edit distance (Zhang-Shasha)
  - [ ] BERTScore on serialized JSON
  - [ ] DeepDiff distance
  - [ ] GPTScore (if compute allows)
- [ ] **Run comparison experiments** on all datasets
- [ ] **Statistical significance tests** (paired t-test, Wilcoxon)

### Scalability Analysis
- [ ] **Runtime benchmarks** vs JSON size (10, 100, 1000, 10000 nodes)
- [ ] **Memory usage profiling**
- [ ] **Plot complexity curves** (empirical vs theoretical)

---

## Phase 3: Experiments on Real LLM Outputs (MEDIUM PRIORITY)

### Temperature Sensitivity Study
- [ ] Generate outputs at T ∈ {0.0, 0.3, 0.5, 0.7, 1.0}
- [ ] Compare consistency scores across temperatures
- [ ] Validate correlation: higher T → lower consistency

### Model Comparison Study
- [ ] Test on multiple models:
  - [ ] GPT-4 / GPT-4o
  - [ ] Claude 3.5 Sonnet
  - [ ] Llama 3 70B
  - [ ] Mistral Large
- [ ] Create model consistency leaderboard

### Cross-Domain Evaluation
- [ ] Function calling (xLAM, Glaive)
- [ ] JSON generation (Maxscha)
- [ ] Structured extraction (custom prompts)
- [ ] Agent traces (multi-step tool use)

---

## Phase 4: Paper Writing (MEDIUM PRIORITY)

### Structure
- [ ] **Abstract** - Emphasize novelty and results
- [ ] **Introduction** - Motivation, contributions
- [ ] **Related Work** - Extended comparison (started in tex file)
- [ ] **Method** - STED formulation, consistency score
- [ ] **Theoretical Analysis** - Port from `sted_theory_icml.tex`
- [ ] **Experiments** - Ablations, baselines, real LLM evaluation
- [ ] **Discussion** - Limitations, future work
- [ ] **Conclusion**

### Figures (Publication Quality)
- [ ] Figure 1: STED overview diagram
- [ ] Figure 2: Tree matching visualization
- [ ] Figure 3: Ablation results
- [ ] Figure 4: Baseline comparison
- [ ] Figure 5: Scalability curves

### Tables
- [ ] Table 1: Hyperparameter sensitivity (ablation)
- [ ] Table 2: Dataset characteristics
- [ ] Table 3: Baseline comparison results
- [ ] Table 4: Model consistency comparison

---

## Phase 5: Code & Reproducibility (LOW PRIORITY - CLOSER TO DEADLINE)

### Code Quality
- [ ] Add comprehensive docstrings
- [ ] Type hints throughout
- [ ] Unit test coverage > 80%
- [ ] Integration tests for main workflows

### Reproducibility Package
- [ ] Requirements.txt / pyproject.toml finalized
- [ ] Scripts to reproduce all experiments
- [ ] Pre-computed results for verification
- [ ] README with clear instructions

### Release Preparation
- [ ] License file (Apache 2.0 or MIT)
- [ ] CONTRIBUTING.md
- [ ] CODE_OF_CONDUCT.md
- [ ] GitHub Actions CI/CD

---

## Timeline (Suggested)

| Week | Tasks |
|------|-------|
| 1-2 | Complete ablation experiments, run baselines |
| 3-4 | Real LLM output experiments, model comparison |
| 5-6 | Paper writing (methods, experiments) |
| 7-8 | Paper writing (intro, related work, conclusion) |
| 9 | Internal review, figure polish |
| 10 | Final revisions, submission |

---

## Key Reviewer Concerns to Address

1. **"Is STED just tree edit distance with embeddings?"**
   - Emphasize: structure-guided combined matching, consistency score, order sensitivity

2. **"Why not just use BERTScore on serialized JSON?"**
   - Show: loses structural information, empirical comparison

3. **"How does it scale?"**
   - Show: O(N·B³) is practical, runtime benchmarks

4. **"Are the hyperparameter choices principled?"**
   - Show: ablation studies, sensitivity analysis

5. **"Does it work on real LLM outputs?"**
   - Show: multi-model, multi-temperature experiments

---

## Files Reference

| File | Description |
|------|-------------|
| `docs/sted_theory_icml.tex` | Theoretical foundations LaTeX |
| `docs/STED_Formula.md` | Mathematical notation reference |
| `docs/STED_and_Consistency_Scoring.pdf` | Current paper draft |
| `scripts/experiments/run_hyperparameter_ablation.py` | Ablation experiment script |
| `scripts/dataset_analysis/*.py` | Dataset analysis scripts |
| `sted/semantic_json_tree_consistency.py` | Main STED implementation |

---

*Last updated: 2024-12-19*
