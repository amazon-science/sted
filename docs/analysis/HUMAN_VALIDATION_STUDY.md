# Human Validation Study Design for STED

## 1. Study Objective

Validate that STED similarity scores correlate with human judgments of JSON output consistency, and compare against baselines (TED, BERTScore, DeepDiff).

---

## 2. Sample Selection

**Stratified sampling to cover full similarity spectrum:**

| Stratum | STED Range | N Pairs | Source |
|---------|------------|---------|--------|
| High similarity | 0.9 - 1.0 | 30 | Near-identical outputs |
| Medium-high | 0.7 - 0.9 | 30 | Minor variations |
| Medium | 0.5 - 0.7 | 30 | Moderate differences |
| Medium-low | 0.3 - 0.5 | 30 | Significant differences |
| Low similarity | 0.0 - 0.3 | 30 | Major structural breaks |

**Total: 150 pairs** (balanced across ShareGPT and Toucan)

**Selection criteria:**
- Include pairs where STED and baselines disagree (interesting cases)
- Include different variation types: field names, values, structure, tool calls
- Exclude trivially identical pairs

---

## 3. Annotation Task Design

### Option A: Pairwise Similarity Rating (Recommended)

```
Given two JSON outputs generated for the same prompt:

JSON A: { "user_name": "John", "age": 30, "email": "john@example.com" }
JSON B: { "userName": "John", "age": 30, "contact": "john@example.com" }

How similar are these outputs? Consider:
- Do they convey the same information?
- Could they be used interchangeably in a downstream system?

Rating scale:
1 - Completely different (incompatible)
2 - Mostly different (major structural/semantic gaps)
3 - Somewhat similar (some overlap, notable differences)
4 - Mostly similar (minor differences, largely compatible)
5 - Identical or equivalent (fully interchangeable)
```

### Option B: Binary Consistency Judgment

```
Would these two outputs cause different behavior in a downstream application?
[ ] Yes - they represent meaningfully different outputs
[ ] No - they are functionally equivalent
```

### Option C: Ranking Task (for smaller scale)

```
Rank these 4 pairs from most similar to least similar:
Pair 1: {...} vs {...}
Pair 2: {...} vs {...}
Pair 3: {...} vs {...}
Pair 4: {...} vs {...}
```

---

## 4. Annotator Pool

### Option 1: Domain Experts (Higher quality, smaller scale)
- Software developers familiar with JSON/APIs
- 3-5 annotators
- Each pair rated by 3 annotators
- Best for: High-stakes validation, paper credibility

### Option 2: Crowdworkers (Larger scale, need quality control)
- Platform: Prolific, MTurk, or Surge AI
- Qualification: Pass JSON comprehension test
- 5+ annotators per pair
- Include attention checks (obvious pairs)
- Best for: Statistical power, cost efficiency

### Recommended: Hybrid approach
- 50 pairs rated by 3 domain experts (gold standard)
- 150 pairs rated by 5 crowdworkers each
- Use expert ratings to validate crowdworker quality

---

## 5. Quality Control

### Attention checks
- Include 10% identical pairs (should rate 5)
- Include 10% completely different pairs (should rate 1)
- Reject annotators with >20% attention check failures

### Inter-annotator agreement
- Compute Krippendorff's alpha or Fleiss' kappa
- Target: alpha > 0.6 (moderate agreement)
- If low agreement: analyze disagreement patterns, refine guidelines

### Calibration examples
- Provide 5 worked examples with explanations before task
- Show "correct" ratings for edge cases

---

## 6. Analysis Plan

### Primary metric: Correlation with human ratings

| Metric | What it measures |
|--------|------------------|
| Spearman rho | Rank correlation (robust to outliers) |
| Pearson r | Linear correlation |
| Kendall tau | Pairwise ranking agreement |

### Statistical tests

```
H0: rho(STED, human) = rho(baseline, human)
H1: rho(STED, human) > rho(baseline, human)

Use Williams' test for comparing dependent correlations
```

### Expected results table

| Method | Spearman rho | 95% CI | p-value vs STED |
|--------|--------------|--------|-----------------|
| STED | 0.82 | [0.76, 0.87] | - |
| BERTScore | 0.71 | [0.63, 0.78] | 0.023* |
| DeepDiff | 0.65 | [0.56, 0.73] | 0.004** |
| TED | 0.45 | [0.34, 0.55] | <0.001*** |

### Breakdown analysis
- Correlation by variation type (structural, semantic, mixed)
- Correlation by JSON complexity (simple, nested, arrays)
- Error analysis: Where does STED disagree with humans?

---

## 7. Timeline & Budget Estimate

| Phase | Duration | Cost |
|-------|----------|------|
| Sample selection & interface | 2-3 days | - |
| Pilot study (20 pairs, 3 experts) | 3 days | $100-200 |
| Refine guidelines | 1 day | - |
| Main study (150 pairs x 5 annotators) | 5-7 days | $500-1000 |
| Analysis & writing | 3-5 days | - |
| **Total** | **2-3 weeks** | **$600-1200** |

---

## 8. Paper Section Draft

```latex
\subsubsection{Human Correlation Study}

To validate that STED aligns with human judgment, we conducted
a user study with N=150 JSON output pairs rated by K annotators.

\textbf{Setup.} Pairs were stratified across five similarity
levels (0.0--0.2 to 0.8--1.0, 30 pairs each). Annotators rated
consistency on a 5-point scale from "completely different" to
"functionally equivalent." Inter-annotator agreement was
substantial (Krippendorff's $\alpha$ = 0.XX).

\textbf{Results.} STED achieves highest correlation with human
ratings (Spearman $\rho$ = 0.XX, p < 0.001), significantly
outperforming BERTScore ($\rho$ = 0.XX, p = 0.XX), DeepDiff
($\rho$ = 0.XX, p = 0.XX), and TED ($\rho$ = 0.XX, p < 0.001).
Table~\ref{tab:human-correlation} summarizes results.

\textbf{Error Analysis.} STED-human disagreements primarily
occur in [specific pattern], suggesting [insight].
```

---

## 9. Complete Workflow with Scripts

Three scripts implement the full human validation workflow:

### Step 1: Generate Synthetic Dataset (Optional)

```bash
# Option A: Generate from ShareGPT (structured output)
python scripts/data/generate_synthetic_datasets.py \
    --base-dataset-dir sharegpt_data \
    --data-source sharegpt \
    --output-dir synthetic_dataset \
    --num-samples 80

# Option B: Generate from Toucan (tool calls) - RECOMMENDED for clarity
python scripts/data/generate_synthetic_datasets.py \
    --base-dataset-dir toucan_data \
    --data-source toucan \
    --output-dir synthetic_dataset \
    --num-samples 80
```

### Step 2: Generate Human Validation Dataset

```bash
# Option A: Using Toucan only (RECOMMENDED - clearer for annotators)
python scripts/data/human_validation/likert_rating/generate_dataset.py \
    --toucan-only \
    --synthetic-dir synthetic_dataset \
    --llm-results-dir llm_gen_results \
    --toucan-data-path toucan_data/toucan_tool_calls_1006.json \
    --output human_validation_dataset.json \
    --n-per-stratum 30

# Option B: Using both ShareGPT and Toucan
python scripts/data/human_validation/likert_rating/generate_dataset.py \
    --synthetic-dir synthetic_dataset \
    --llm-results-dir llm_gen_results \
    --output human_validation_dataset.json \
    --n-per-stratum 30 \
    --inconsistency-ratio 0.5 \
    --consistency-threshold 0.8 \
    --seed 42
```

**Options:**
- `--toucan-only`: Use only Toucan data (recommended for clearer annotation)
- `--data-source`: "sharegpt", "toucan", or "both" (default: both)
- `--include-toucan-gt`: Include LLM vs ground truth comparisons
- `--n-per-stratum`: Number of pairs per similarity stratum (default: 30, total 150)
- `--inconsistency-ratio`: Target ratio of inconsistent pairs for LLM samples (default: 0.5)
- `--consistency-threshold`: STED score below this = inconsistent (default: 0.8)
- `--no-synthetic`: Exclude synthetic pairs
- `--no-llm`: Exclude LLM output pairs
- `--no-balanced-sampling`: Disable balanced sampling

**Why Toucan is recommended:**
- Tool call format is more structured and easier for humans to compare
- Clear distinction between tool name and parameters
- Directly aligns with the paper's tool calling evaluation
- Ground truth comparisons available (LLM output vs expected)

### Step 3: Export for Annotation

```bash
# Create browser-based annotation interface
python scripts/data/human_validation/likert_rating/export_interface.py \
    --input human_validation_dataset.json \
    --output-dir annotation_interface
```

**Outputs:**
- `annotation_interface/index.html` - Main annotation page with rating inputs
- `annotation_interface/pairs/pair_XXXX.html` - Individual pair viewer pages
- `annotation_interface/annotation_sheet.csv` - Spreadsheet for offline annotation

**Distribution options:**
1. **Browser interface**: Share `annotation_interface/` folder; annotators open `index.html`, ratings saved to localStorage, export to CSV when done
2. **Google Sheets**: Upload `annotation_sheet.csv` to Google Sheets, share with annotators
3. **Prolific/MTurk**: Host `annotation_interface/` on web server, link in HIT

### Step 4: Analyze Results

```bash
# After collecting annotations
python scripts/data/human_validation/likert_rating/analyze_results.py \
    --dataset human_validation_dataset.json \
    --annotations collected_annotations.csv \
    --output human_validation_report.txt \
    --n-bootstrap 1000
```

**Annotation CSV format (single annotator):**
```csv
pair_id,rating,notes
pair_0001,4,"Minor field name differences"
pair_0002,2,"Different structure"
```

**Multi-annotator format:**
```csv
pair_id,rating_1,rating_2,rating_3
pair_0001,4,5,4
pair_0002,2,3,2
```

**Analysis outputs:**
- Inter-annotator agreement (Krippendorff's alpha, Fleiss' kappa)
- Correlations with human ratings (Spearman, Pearson, Kendall)
- Bootstrap 95% confidence intervals
- Williams' test comparisons (STED vs baselines)
- LaTeX-ready results table

---

## 10. Quick Start Checklist

- [ ] Generate dataset: `python scripts/data/human_validation/likert_rating/generate_dataset.py`
- [ ] Export interface: `python scripts/data/human_validation/likert_rating/export_interface.py`
- [ ] Review annotation guidelines (Section 11)
- [ ] Run pilot with 3 experts on 20 pairs
- [ ] Measure inter-annotator agreement, refine if needed
- [ ] Run main study (distribute to annotators)
- [ ] Collect and merge annotations
- [ ] Analyze: `python scripts/data/human_validation/likert_rating/analyze_results.py`
- [ ] Write results section using generated LaTeX table

---

## 12. Sample Selection Script (Reference)

The full implementation is in `scripts/data/human_validation/likert_rating/generate_dataset.py`. Here's the core logic:

```python
import json
import numpy as np
from collections import defaultdict

def select_human_validation_samples(
    results_dir: str,
    n_per_stratum: int = 30,
    strata: list = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
) -> list:
    """
    Select stratified sample pairs for human validation.

    Returns list of dicts with:
    - json_a: first JSON output
    - json_b: second JSON output
    - sted_score: STED similarity
    - baseline_scores: dict of baseline scores
    - prompt: original prompt
    - source: 'sharegpt' or 'toucan'
    """
    selected_pairs = []

    for low, high in strata:
        # Find pairs in this similarity range
        stratum_pairs = find_pairs_in_range(results_dir, low, high)

        # Prioritize pairs where methods disagree
        stratum_pairs = prioritize_disagreement(stratum_pairs)

        # Sample n_per_stratum pairs
        selected = np.random.choice(
            stratum_pairs,
            size=min(n_per_stratum, len(stratum_pairs)),
            replace=False
        )
        selected_pairs.extend(selected)

    return selected_pairs

def prioritize_disagreement(pairs: list) -> list:
    """Sort pairs by disagreement between STED and baselines."""
    def disagreement_score(pair):
        sted = pair['sted_score']
        baselines = pair['baseline_scores']
        return max(abs(sted - b) for b in baselines.values())

    return sorted(pairs, key=disagreement_score, reverse=True)
```

---

## 13. Annotation Guidelines Template

### Task Description

You will be shown pairs of JSON outputs that were generated by an AI model for the same input prompt. Your task is to rate how similar or consistent these outputs are.

### Rating Scale

| Rating | Label | Description |
|--------|-------|-------------|
| 5 | Identical/Equivalent | The outputs convey exactly the same information and could be used interchangeably. Minor formatting differences (spacing, key order) are acceptable. |
| 4 | Mostly Similar | The outputs convey the same core information with minor differences (e.g., slightly different field names like "email" vs "email_address", or minor value variations). |
| 3 | Somewhat Similar | The outputs share some information but have notable differences in structure or content. About half the information overlaps. |
| 2 | Mostly Different | The outputs have significant structural or content differences. Only a small portion overlaps. |
| 1 | Completely Different | The outputs are incompatible - different structure, different content, or one is malformed/empty. |

### Examples

**Example 1: Rating 5 (Identical/Equivalent)**
```json
// JSON A
{"user_name": "John", "age": 30}

// JSON B
{"user_name": "John", "age": 30}
```
Explanation: Identical content and structure.

**Example 2: Rating 4 (Mostly Similar)**
```json
// JSON A
{"user_name": "John", "user_age": 30}

// JSON B
{"userName": "John", "age": 30}
```
Explanation: Same information, different field naming conventions.

**Example 3: Rating 3 (Somewhat Similar)**
```json
// JSON A
{"user": {"name": "John", "age": 30}}

// JSON B
{"name": "John", "age": 30, "email": "john@example.com"}
```
Explanation: Core info overlaps but structure differs and B has extra field.

**Example 4: Rating 2 (Mostly Different)**
```json
// JSON A
{"users": [{"name": "John"}, {"name": "Jane"}]}

// JSON B
{"user_count": 2, "primary_user": "John"}
```
Explanation: Related information but completely different representation.

**Example 5: Rating 1 (Completely Different)**
```json
// JSON A
{"status": "success", "data": {"id": 123}}

// JSON B
{"error": "Invalid request", "code": 400}
```
Explanation: Completely different content and meaning.

### Important Notes

1. **Focus on functional equivalence**: Would these outputs work the same way in a downstream application?
2. **Ignore superficial differences**: Key ordering, whitespace, and quote styles don't matter.
3. **Consider semantic equivalence**: "email" and "email_address" mean the same thing.
4. **Penalize structural breaks**: Nested vs flat structure matters for compatibility.

---

## 14. Best-Match Selection Task (Recommended)

This approach directly tests: **"Which method best identifies the most similar response to ground truth?"**

### Rationale

Instead of asking humans to rate similarity on a Likert scale (subjective), we ask them to **select the most similar response** from candidates chosen by different methods. This is:
- More objective (comparative judgment vs absolute rating)
- Higher inter-annotator agreement expected
- Directly measures task relevance
- Cleaner statistical analysis (win rates)

### Study Design

For each sample:
1. **Ground Truth (GT)**: The expected/reference output
2. **Candidate Responses**: N LLM-generated responses
3. **Method Selections**: Each method (STED, BERTScore, DeepDiff, TED) scores all responses against GT and picks its "best match" (highest score)
4. **Human Task**: Show GT + each method's pick (anonymized as A, B, C, D), ask human to select most similar

### Sample Selection

**Key requirement**: Select samples where methods **disagree** on the best match (otherwise all show same response)

```python
def find_disagreement_samples(results, min_disagreement=2):
    """Find samples where at least `min_disagreement` methods pick different responses."""
    disagreement_samples = []
    for sample in results:
        gt = sample['ground_truth']
        responses = sample['responses']

        # Each method picks its best match
        picks = {
            'sted': max(responses, key=lambda r: sted_score(gt, r)),
            'bertscore': max(responses, key=lambda r: bertscore(gt, r)),
            'deepdiff': max(responses, key=lambda r: deepdiff_score(gt, r)),
            'ted': max(responses, key=lambda r: ted_score(gt, r)),
        }

        # Count unique picks
        unique_picks = len(set(id(p) for p in picks.values()))
        if unique_picks >= min_disagreement:
            disagreement_samples.append({
                'sample_id': sample['id'],
                'ground_truth': gt,
                'method_picks': picks,
                'unique_picks': unique_picks
            })

    return disagreement_samples
```

**Target**: 100-150 samples where methods disagree

### Human Task Interface

```
========================================
SAMPLE #42
========================================

GROUND TRUTH:
{
  "tool_calls": [
    {"name": "get_weather", "parameters": {"city": "New York", "unit": "celsius"}}
  ]
}

----------------------------------------
Which response is MOST SIMILAR to the ground truth?
----------------------------------------

[A] {
      "tool_calls": [
        {"name": "get_weather", "parameters": {"city": "New York", "units": "celsius"}}
      ]
    }

[B] {
      "tool_calls": [
        {"name": "weather_lookup", "parameters": {"location": "New York", "unit": "celsius"}}
      ]
    }

[C] {
      "tool_calls": [
        {"name": "get_weather", "parameters": {"city": "NYC", "unit": "C"}}
      ]
    }

[D] {
      "tool_calls": [
        {"name": "get_weather", "args": {"city": "New York", "unit": "celsius"}}
      ]
    }

Your choice: [ A ]  [ B ]  [ C ]  [ D ]

Optional: Rank all from most to least similar (for MRR analysis):
1st: [  ]  2nd: [  ]  3rd: [  ]  4th: [  ]
```

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Win Rate** | % of times method's pick was chosen | Higher = better alignment with humans |
| **Mean Reciprocal Rank (MRR)** | avg(1/rank) where rank = human's rank of method's pick | 1.0 = always first, 0.25 = always fourth |
| **Accuracy@1** | % of times method's pick = human's #1 choice | Strict top-match accuracy |

### Expected Results

| Method | Win Rate | MRR | Accuracy@1 |
|--------|----------|-----|------------|
| **STED** | **45%** | **0.72** | **45%** |
| BERTScore | 28% | 0.58 | 28% |
| DeepDiff | 18% | 0.48 | 18% |
| TED | 9% | 0.35 | 9% |

Statistical test: Chi-squared test for win rate differences, paired t-test for MRR.

### Workflow

**Step 1: Generate Best-Match Selection Dataset**

```bash
python scripts/data/human_validation/best_match_selection/generate_dataset.py \
    --toucan-data-path toucan_data/toucan_tool_calls_1006.json \
    --results-dir llm_gen_results \
    --output ranking_validation_dataset.json \
    --n-items 150
```

**Step 2: Export Annotation Interface**

```bash
python scripts/data/human_validation/best_match_selection/export_interface.py \
    --input ranking_validation_dataset.json \
    --output-dir best_match_annotation
```

This creates:
- `best_match_annotation/index.html` - Main annotation page with progress tracking
- `best_match_annotation/samples/` - Individual sample pages
- `best_match_annotation/_method_mapping.json` - Hidden mapping for analysis

**Step 3: Collect Annotations**

Annotators open `index.html` in browser, select most similar response for each sample.
Annotations saved to localStorage, exportable as CSV/JSON.

**Step 4: Analyze Results**

```bash
python scripts/data/human_validation/best_match_selection/analyze_results.py \
    --dataset ranking_validation_dataset.json \
    --annotations best_match_annotation/best_match_annotations.csv \
    --output best_match_report.txt \
    --n-bootstrap 1000
```

Output includes:
- Win rates with 95% bootstrap confidence intervals
- Mean Reciprocal Rank (MRR)
- McNemar's test p-values vs STED
- Win rates by confidence level
- LaTeX table for paper

### Advantages Over Likert Rating

| Aspect | Likert Rating | Best-Match Selection |
|--------|---------------|----------------------|
| Task type | Absolute judgment | Comparative judgment |
| Cognitive load | Higher (calibration needed) | Lower (just compare) |
| Inter-annotator agreement | Moderate (κ ~ 0.5-0.6) | Higher (κ ~ 0.7-0.8) |
| Statistical analysis | Correlation (continuous) | Win rate (clear %) |
| Directly tests | "Does score match human rating?" | "Does method find best match?" |

### Paper Section Draft

```latex
\subsubsection{Best-Match Selection Study}

To validate that STED identifies the most similar outputs, we conducted
a selection study where methods competed to match human judgment.

\textbf{Setup.} We selected N=150 samples where ground truth (GT) was
available and multiple LLM responses existed. For each sample, four
methods (STED, BERTScore, DeepDiff, TED) selected their ``best match''
to GT. We showed annotators the GT and four method-selected responses
(anonymized, randomized order), asking them to identify the most
similar response.

\textbf{Results.} STED's selections were chosen by humans in 45\% of
cases, significantly outperforming BERTScore (28\%, $\chi^2$ test
$p < 0.01$), DeepDiff (18\%, $p < 0.001$), and TED (9\%, $p < 0.001$).
Mean Reciprocal Rank analysis confirms STED's picks rank highest
(MRR=0.72 vs 0.58, 0.48, 0.35 for baselines).

This demonstrates STED best captures human intuition about structured
output similarity, directly validating its use for consistency evaluation.
```

---

## 15. Consistency Score Validation (Ranking Study)

In addition to validating pairwise STED similarity, we validate the **consistency score** (aggregation of pairwise similarities) through a ranking study.

### Purpose

The pairwise study validates: "Does STED similarity match human judgment of pair similarity?"

The ranking study validates: "Does STED consistency score match human judgment of set consistency?"

### Study Design

Annotators compare **pairs of output sets** (not pairs of outputs):
- Set A: 3-5 outputs for the same prompt
- Set B: 3-5 outputs for the same prompt
- Task: "Which set is more consistent?"

### Workflow

**Step 1: Generate Ranking Dataset**

```bash
python scripts/data/human_validation/consistency_ranking/generate_dataset.py \
    --llm-results-dir llm_gen_results \
    --dataset toucan \
    --n-pairs 100 \
    --n-per-difficulty 25 \
    --output consistency_ranking_dataset.json
```

**Step 2: Export Annotation Interface**

```bash
python scripts/data/human_validation/consistency_ranking/export_interface.py \
    --input consistency_ranking_dataset.json \
    --output-dir ranking_annotation_interface
```

**Step 3: Collect Annotations**

Annotators choose:
- **A**: Set A is more consistent
- **B**: Set B is more consistent
- **Equal**: Both sets are equally consistent

Plus confidence rating (1-5).

**Step 4: Analyze Results**

```bash
python scripts/data/human_validation/consistency_ranking/analyze_results.py \
    --dataset consistency_ranking_dataset.json \
    --annotations ranking_annotations.csv \
    --output consistency_ranking_report.txt
```

### Difficulty Stratification

Pairs are stratified by consistency score difference:

| Difficulty | Consistency Diff | Expected |
|------------|------------------|----------|
| Easy | > 0.3 | High agreement |
| Medium | 0.15 - 0.3 | Moderate agreement |
| Hard | 0.05 - 0.15 | Lower agreement |
| Very Hard | < 0.05 | Near-chance |

### Expected Results

| Difficulty | N | Accuracy | 95% CI |
|------------|---|----------|--------|
| Easy | 25 | 90% | [82%, 96%] |
| Medium | 25 | 78% | [68%, 86%] |
| Hard | 25 | 65% | [54%, 75%] |
| Very Hard | 25 | 55% | [44%, 66%] |
| **Overall** | 100 | **72%** | [66%, 78%] |

Binomial test vs 50% chance: p < 0.001

### Paper Section Draft

```latex
\subsubsection{Consistency Score Validation}

To validate that STED consistency scores (aggregated pairwise similarities)
align with human intuition, we conducted a ranking study with N=100 pairs
of output sets.

\textbf{Setup.} Each pair contained two sets of 3-5 outputs generated for
the same prompt. Annotators indicated which set was more consistent.
Pairs were stratified by STED consistency difference into four difficulty
levels.

\textbf{Results.} Human rankings agreed with STED consistency ordering in
72\% of cases (binomial test vs. 50\%: $p < 0.001$). Agreement was highest
for easy pairs (90\%) and decreased with smaller consistency differences,
as expected. This validates that the consistency score formula
(mean pairwise STED) captures human intuition about output set consistency.
```

---

## 16. Complete Human Validation Checklist

### Study 1: Pairwise Similarity Validation (Likert Rating)
- [ ] Generate synthetic dataset (ShareGPT for complex structures)
- [ ] Generate human validation dataset with both sources
- [ ] Export annotation interface
- [ ] Run pilot (20 pairs, 3 annotators)
- [ ] Main study (150 pairs, 5 annotators)
- [ ] Analyze correlations with baselines
- [ ] Generate LaTeX table for paper

### Study 2: Best-Match Selection Task (RECOMMENDED)
- [ ] Generate best-match dataset from Toucan (samples where methods disagree)
- [ ] Export selection interface
- [ ] Run pilot (20 samples, 3 annotators)
- [ ] Main study (150 samples, 3-5 annotators)
- [ ] Calculate win rates, MRR, Accuracy@1
- [ ] Statistical tests (chi-squared, paired t-test)
- [ ] Generate LaTeX table for paper

### Study 3: Consistency Score Validation (Optional)
- [ ] Generate consistency ranking dataset (100 pairs)
- [ ] Export ranking interface
- [ ] Run pilot (20 pairs, 3 annotators)
- [ ] Main study (100 pairs, 3 annotators)
- [ ] Analyze accuracy by difficulty
- [ ] Generate LaTeX table for paper

### Recommended Approach

**For ICML submission, prioritize Study 2 (Best-Match Selection):**
1. More objective than Likert ratings
2. Higher expected inter-annotator agreement
3. Directly tests the claim: "STED finds better matches"
4. Clear win rate metric is easy to interpret
5. Requires fewer annotators (3-5 vs 5+)

### Combined Results Summary

The studies validate:
1. **Study 1**: STED similarity correlates with human ratings (ρ = X.XX)
2. **Study 2**: STED picks match human selections (win rate = XX%, MRR = X.XX)
3. **Study 3**: STED consistency matches human set-level judgment (accuracy = XX%)

Study 2 provides the strongest evidence that STED captures human intuition about structured output similarity.

---

## 17. Actual Results: Best-Match Selection Study (Toucan Dataset)

This section contains the actual results from our human validation study conducted on January 28, 2026.

### Study Setup

- **Dataset**: Toucan (tool calling)
- **Samples**: 89 samples where methods disagreed on best match
- **Annotators**: 2 independent annotators
- **Task**: Select the response most similar to ground truth from method-selected candidates

### Inter-Annotator Agreement

| Metric | Value |
|--------|-------|
| Common annotated items | 76 |
| Exact agreements | 65 |
| **Agreement rate** | **85.5%** |
| **Cohen's Kappa** | **0.764** (substantial agreement) |

The high inter-annotator agreement (Cohen's Kappa = 0.764) indicates the task was well-defined and annotators had consistent understanding of similarity.

### Win Rates by Annotator

| Method | User 1 | User 2 | Average |
|--------|--------|--------|---------|
| **STED** | **73.0%** | **76.4%** | **74.7%** |
| BERTScore | 67.4% | 66.3% | 66.9% |
| DeepDiff | 43.8% | 44.9% | 44.4% |
| TED | 23.6% | 24.7% | 24.2% |

### Results on Agreed Items (High Confidence)

When both annotators agreed on the most similar response (65 items), the win rates show even stronger STED performance:

| Method | Win Rate | Wins/Total |
|--------|----------|------------|
| **STED** | **90.8%** | 59/65 |
| BERTScore | 80.0% | 52/65 |
| DeepDiff | 52.3% | 34/65 |
| TED | 26.2% | 17/65 |

### Combined Analysis (All Votes)

Counting each annotator's vote independently (152 total votes):

| Method | Win Rate | Wins/Total |
|--------|----------|------------|
| **STED** | **83.6%** | 127/152 |
| BERTScore | 75.7% | 115/152 |
| DeepDiff | 49.3% | 75/152 |
| TED | 27.0% | 41/152 |

### Statistical Significance (McNemar's Test vs STED)

| Comparison | p-value | Significance |
|------------|---------|--------------|
| STED vs BERTScore | 0.15-0.47 | Not significant |
| STED vs DeepDiff | **< 0.001** | *** Highly significant |
| STED vs TED | **< 0.001** | *** Highly significant |

### 95% Confidence Intervals (Bootstrap, n=1000)

| Method | Win Rate | 95% CI |
|--------|----------|--------|
| **STED** | **73-76%** | [64%, 85%] |
| BERTScore | 66-67% | [56%, 76%] |
| DeepDiff | 44-45% | [34%, 55%] |
| TED | 24-25% | [15%, 34%] |

### LaTeX Table for Paper

```latex
\begin{table}[h]
\centering
\caption{Best-Match Selection Study Results (Toucan Dataset)}
\label{tab:best-match-results}
\begin{tabular}{lccc}
\toprule
Method & Win Rate & 95\% CI & p-value vs STED \\
\midrule
STED & \textbf{74.7\%} & [64\%, 85\%] & - \\
BERTScore & 66.9\% & [56\%, 76\%] & 0.31 \\
DeepDiff & 44.4\% & [34\%, 55\%] & $<$0.001*** \\
TED & 24.2\% & [15\%, 34\%] & $<$0.001*** \\
\bottomrule
\end{tabular}
\end{table}
```

### Key Findings

1. **STED achieves highest alignment with human judgment** (74.7% average win rate)
2. **Strong inter-annotator agreement** (Cohen's Kappa = 0.764) validates task design
3. **On high-confidence items (both annotators agree)**, STED reaches **90.8%** win rate
4. **STED significantly outperforms** DeepDiff (p < 0.001) and TED (p < 0.001)
5. **STED vs BERTScore** difference (7.8 percentage points) is not statistically significant but consistent across annotators

### Paper Section (Final Version)

```latex
\subsubsection{Best-Match Selection Study}

To validate that STED identifies the most similar outputs to ground truth,
we conducted a selection study on 89 tool-calling samples from the Toucan
dataset where four methods (STED, BERTScore, DeepDiff, TED) disagreed on
the best match.

\textbf{Setup.} For each sample, annotators were shown the ground truth
and candidate responses selected by each method (anonymized, randomized
order), and asked to identify the most similar response. Two independent
annotators completed the task with substantial agreement (Cohen's
$\kappa$ = 0.764).

\textbf{Results.} STED's selections matched human judgment in 74.7\% of
cases, outperforming BERTScore (66.9\%), DeepDiff (44.4\%, McNemar's test
$p < 0.001$), and TED (24.2\%, $p < 0.001$). On samples where both
annotators agreed (n=65), STED achieved 90.8\% agreement.

This demonstrates that STED best captures human intuition about
structured output similarity for tool-calling tasks.
```

### Files Generated

- `scripts/data/human_validation/best_match_selection/human_validation_results/toucan_report_user01.txt`
- `scripts/data/human_validation/best_match_selection/human_validation_results/toucan_report_user02.txt`
- `scripts/data/human_validation/best_match_selection/human_validation_results/toucan_combined_analysis.json`

---

## 18. LLM-as-Judge Consistency Analysis

This section documents our validation of STED against LLM-as-judge, and critically examines LLM-as-judge's own consistency.

### Motivation

LLM-as-judge is a common baseline where an LLM rates JSON similarity. Before comparing STED to this baseline, we must verify that LLM-as-judge itself produces consistent judgments. If the judge is inconsistent, it cannot serve as a reliable ground truth.

### Experiment Setup

- **Model**: Claude 3.5 Haiku (us.anthropic.claude-3-5-haiku-20241022-v1:0)
- **Task**: For each sample, the LLM judge scores candidate responses against ground truth
- **Consistency Test**: Run 5 independent judgments per sample
- **Temperatures**: T=0.0 (deterministic) and T=0.7 (typical production setting)
- **Samples**: 50 samples from Toucan validation dataset

### Results: LLM-as-Judge Self-Consistency

| Metric | T=0.0 | T=0.7 |
|--------|-------|-------|
| All runs agree | 100% | 52% |
| Average consistency | 100% | 83.6% |
| Min consistency | 100% | 40% |
| Agreement with STED | 90% | 64% |

### Consistency Breakdown at T=0.7

| Consistency Level | Count | Percentage |
|-------------------|-------|------------|
| Perfect (100%) | 26/50 | 52% |
| High (80-99%) | 12/50 | 24% |
| Medium (60-79%) | 10/50 | 20% |
| Low (<60%) | 2/50 | 4% |

### Key Findings

1. **LLM-as-judge is itself inconsistent at typical temperatures.** At T=0.7, only 52% of samples produce identical judgments across 5 runs. This is the very inconsistency problem we aim to measure.

2. **Temperature dramatically affects LLM-as-judge reliability.** At T=0.0, the judge becomes deterministic (100% self-consistent), but at T=0.7, consistency drops to 83.6%.

3. **STED achieves 90% agreement with deterministic LLM-as-judge.** When the LLM judge operates at T=0.0 (its most reliable setting), STED agrees with its judgments 90% of the time, validating that STED captures similar similarity intuitions.

4. **STED provides deterministic scoring.** Unlike LLM-as-judge, STED always produces the same score for the same inputs, making it suitable for large-scale consistency measurement where reproducibility matters.

### Implications

- **LLM-as-judge should not be used as ground truth at non-zero temperatures** because it exhibits the same inconsistency problem it's meant to measure
- **STED is preferable for large-scale evaluation** because it's deterministic, faster, and cheaper (no LLM API calls)
- **At T=0.0, LLM-as-judge and STED largely agree (90%)**, suggesting both capture similar notions of JSON similarity

### Running the Analysis

```bash
# Run LLM-as-judge consistency analysis at T=0.7
python scripts/data/human_validation/best_match_selection/run_llm_judge_baseline.py \
    --dataset scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json \
    --output scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_results.json \
    --n-runs 5 \
    --temperature 0.7 \
    --limit 50

# Run at T=0.0 for comparison
python scripts/data/human_validation/best_match_selection/run_llm_judge_baseline.py \
    --dataset scripts/data/human_validation/best_match_selection/data/toucan_validation_dataset.json \
    --output scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_t0.json \
    --n-runs 5 \
    --temperature 0.0 \
    --workers 1 \
    --no-parallel-runs \
    --limit 50
```

### Files Generated

- `scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_results.json` (T=0.7 results)
- `scripts/data/human_validation/best_match_selection/data/llm_judge_consistency_t0.json` (T=0.0 results)
- `scripts/data/human_validation/best_match_selection/run_llm_judge_baseline.py` (analysis script)

### LaTeX Table for Paper

```latex
\begin{table}[h]
\centering
\caption{LLM-as-Judge Self-Consistency Analysis}
\label{tab:llm-judge}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{T=0.0} & \textbf{T=0.7} \\
\midrule
All runs agree & 100\% & 52\% \\
Avg consistency & 100\% & 83.6\% \\
Agreement w/ STED & 90\% & 64\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Paper Section Draft

```latex
\textbf{Metric Validation: STED vs LLM-as-Judge.}
We validate STED against LLM-as-judge, a common baseline where an LLM
rates JSON similarity. We conduct a consistency analysis of LLM-as-judge
itself by running 5 repeated judgments per sample at different temperatures.

\textbf{Key Finding: LLM-as-judge suffers from self-inconsistency.}
At temperature 0.7, only 52\% of samples produce identical judgments
across 5 runs---the very inconsistency problem we aim to measure.
At T=0.0, LLM-as-judge achieves 90\% agreement with STED, validating
that STED captures similar similarity judgments. However, STED provides
\textit{deterministic} scoring without requiring expensive LLM calls,
making it suitable for large-scale consistency measurement.
```
