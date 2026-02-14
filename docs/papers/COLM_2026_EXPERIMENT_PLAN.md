# COLM 2026 Comprehensive Experiment Plan

## Paper: Predicting LLM Structured Output Consistency from Prompt Linguistics

**Deadline**: March 31, 2026 (Abstract: March 26, 2026)
**Venue**: Conference on Language Modeling (COLM)

---

## Executive Summary

We train a lightweight model to predict LLM structured output consistency directly from prompt text, without running expensive multi-sample inference. By analyzing **67 features** spanning surface linguistics (modal verbs, hedging, politeness), semantic properties (ambiguity, underspecification), and pragmatic factors (task clarity, open-endedness), we provide both theoretical understanding and practical tooling for production LLM deployments.

**Key Insight:** Semantic/pragmatic features (ambiguity, task clarity) predict consistency more strongly than surface linguistic features (modal verbs, hedging).

---

## Differentiation from Prior Work

| Paper | Venue | Focus | This Paper's Difference |
|-------|-------|-------|-------------------------|
| KDD 2026 | Data Mining | 42-factor empirical analysis | We BUILD a predictor model |
| ICML 2026 | ML Methods | STED metric methodology | We PREDICT consistency, not measure it |
| arXiv:2601.00942 | - | MoE vs Dense architecture | We focus on PROMPT features, not model architecture |

**Novel Contributions:**
1. First model to predict structured output consistency from prompt text alone
2. Linguistic theory grounding (modal semantics, politeness theory) for LLM behavior
3. Practical deployment tool: estimate reliability before expensive inference

---

## 1. Research Questions

**RQ1 (Feature Hierarchy):** Which feature categories most strongly predict consistency?
- Hypothesis: Semantic/pragmatic features (ambiguity, task clarity) > Surface linguistic features (modals, hedging)

**RQ2 (Ambiguity Impact):** How do different types of ambiguity (lexical, referential, syntactic) affect consistency?
- Hypothesis: Referential ambiguity (unclear pronouns) has strongest negative effect

**RQ3 (Task Clarity):** Does task underspecification (missing info, open questions) predict low consistency?
- Hypothesis: Underspecified prompts show 2x higher variance than well-specified ones

**RQ4 (Causal vs Correlational):** Do prompt features causally affect consistency, or merely correlate?
- Hypothesis: Causal effect exists for ambiguity resolution and task clarification

**RQ5 (Predictor Model):** Can a small model accurately predict consistency from 67 prompt features?
- Hypothesis: >0.75 correlation with actual consistency; >0.80 AUC for low-consistency detection

**RQ6 (Generalization):** Does the predictor generalize across models and domains?
- Hypothesis: Features transfer across model families with fine-tuning

**RQ7 (Practical Utility):** Can the predictor guide prompt engineering decisions?
- Hypothesis: Predictor-guided rewrites improve consistency by >15%

---

## 2. Data Available (No New API Calls Needed)

### 2.1 Toucan Dataset (Primary)

| Field | Description | Example |
|-------|-------------|---------|
| sample_id | Unique identifier | "7900c4d1-9c88-5e4b..." |
| query | Full prompt text | "I need to find rhyming words..." |
| tools | Available tool schemas | [{name, description, parameters}] |
| ground_truth | Expected tool calls | [{name, arguments}] |
| generated_runs | 10 outputs per config | [[tool_call], [tool_call], ...] |

**Scale:**
- 1,006 unique prompts
- 19 models
- 11 temperatures (0.0 - 1.0)
- 10 runs per configuration
- **Total: ~2.1M data points**

### 2.2 Computed Consistency Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| c_mean | Mean pairwise STED similarity | [0, 1] |
| validity_rate | Fraction of valid outputs | [0, 1] |
| stability_score | Transformed variance metric | [0, 1] |
| ranking_score | Combined metric | [0, 1] |

### 2.3 Data Split Strategy

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 700 prompts (~70%) | Model training |
| Validation | 150 prompts (~15%) | Hyperparameter tuning |
| Test | 156 prompts (~15%) | Final evaluation |

**Cross-model splits:**
- Train on: Claude-Sonnet-4, GPT-4.1-Mini, Llama-3.3-70B, Qwen3-235B
- Test generalization on: Claude-3.5-Sonnet, Gemini-2.5-Flash, Nova2-Lite

---

## 3. Linguistic Feature Extraction

### 3.1 Modal Verb Features (6 features)

Based on Kratzer's modal semantics and Palmer's typology:

| Feature | Detection Pattern | Linguistic Theory |
|---------|-------------------|-------------------|
| `modal_deontic_strong` | must, need to, have to, required | Obligation/necessity |
| `modal_deontic_weak` | should, ought to | Recommendation |
| `modal_epistemic_strong` | will, would | High certainty |
| `modal_epistemic_weak` | may, might, could | Possibility |
| `modal_dynamic` | can, able to | Ability |
| `modal_count` | Total modal verbs | Aggregate |

**Hypothesis:** Strong deontic modals → Lower consistency (over-constrains model)

### 3.2 Hedging Features (5 features)

| Feature | Detection Pattern | Effect |
|---------|-------------------|--------|
| `hedge_epistemic` | maybe, perhaps, possibly | Uncertainty markers |
| `hedge_plausibility` | seems, appears, looks like | Evidential hedges |
| `hedge_conditional` | if, unless, provided that | Conditional logic |
| `hedge_approximator` | about, around, roughly | Numeric hedges |
| `hedge_count` | Total hedge words | Aggregate |

**Hypothesis:** Hedging → Higher entropy → Lower consistency

### 3.3 Politeness Features (6 features)

Based on Brown & Levinson's politeness theory:

| Feature | Detection Pattern | Face Threat Level |
|---------|-------------------|-------------------|
| `polite_bald` | Imperative without softener | 1.0 (highest) |
| `polite_positive` | please, thanks, appreciate | 0.6 |
| `polite_negative` | would you mind, could you possibly | 0.3 |
| `polite_indirect` | I was wondering if, It would be great | 0.2 |
| `polite_impersonal` | One might, It is suggested | 0.1 (lowest) |
| `politeness_score` | Weighted composite | [0, 1] |

**Hypothesis:** High face-threat → Model hesitation → Lower consistency

### 3.4 Speech Act Features (4 features)

Based on Searle's taxonomy:

| Feature | Detection Pattern | Directness |
|---------|-------------------|------------|
| `speech_directive` | Commands, imperatives | 1.0 |
| `speech_interrogative` | Questions | 0.7 |
| `speech_declarative` | Statements of intent | 0.5 |
| `speech_indirect` | Hints, suggestions | 0.3 |

### 3.5 Structural Features (8 features)

| Feature | Description |
|---------|-------------|
| `prompt_length` | Character count |
| `word_count` | Token count |
| `sentence_count` | Number of sentences |
| `question_count` | Number of questions |
| `list_markers` | Numbered/bulleted items |
| `conjunction_count` | and, or, but, then |
| `negation_count` | not, never, no, don't |
| `specificity_score` | Named entities + numbers |

### 3.6 Schema Complexity Features (6 features)

| Feature | Description |
|---------|-------------|
| `num_tools` | Number of available tools |
| `max_params` | Maximum parameters per tool |
| `total_params` | Total parameters across tools |
| `max_nesting_depth` | Deepest parameter nesting |
| `has_array_params` | Boolean: array parameters exist |
| `has_object_params` | Boolean: object parameters exist |

---

## 4. Semantic & Pragmatic Features (NEW)

### 4.1 Ambiguity Features (7 features)

| Feature | Description | Detection Method | Example |
|---------|-------------|------------------|---------|
| `lexical_ambiguity` | Words with multiple meanings | WordNet synset count | "bank" (river/financial) |
| `syntactic_ambiguity` | Multiple valid parse trees | Parser disagreement score | "I saw the man with telescope" |
| `referential_ambiguity` | Unclear pronouns/references | Unresolved coreference count | "Put it there" (what is "it"?) |
| `scope_ambiguity` | Quantifier/negation scope unclear | Scope operator analysis | "Don't call all users" |
| `attachment_ambiguity` | PP/clause attachment unclear | Dependency parse variance | "Search files in folder with X" |
| `ellipsis_count` | Missing but implied elements | Ellipsis detection | "Find users and [find] emails" |
| `ambiguity_score` | Weighted composite | Aggregate | [0, 1] |

**Detection Implementation:**
```python
def compute_ambiguity(prompt: str) -> dict:
    # Lexical: average synsets per content word
    tokens = [t for t in nlp(prompt) if t.pos_ in ('NOUN', 'VERB', 'ADJ')]
    lexical_amb = np.mean([len(wn.synsets(t.lemma_)) for t in tokens])

    # Referential: unresolved coreference chains
    coref = coref_model(prompt)
    unresolved = sum(1 for chain in coref if not chain.is_resolved)

    # Syntactic: parser disagreement (spaCy vs Stanza)
    parse1 = spacy_nlp(prompt)
    parse2 = stanza_nlp(prompt)
    syntactic_amb = compute_tree_distance(parse1, parse2)

    return {...}
```

**Hypothesis:** Referential ambiguity → Strongest negative correlation with consistency

### 4.2 Underspecification Features (6 features)

| Feature | Description | Detection Method | Example |
|---------|-------------|------------------|---------|
| `missing_arguments` | Required info not provided | Semantic role labeling gaps | "Send email" (to whom?) |
| `vague_quantifiers` | Imprecise quantities | Pattern matching | "a few", "some", "many" |
| `vague_temporals` | Imprecise time references | Temporal expression detection | "soon", "later", "recently" |
| `implicit_constraints` | Unstated requirements | Adjective vagueness | "good results" (what's good?) |
| `undefined_terms` | Domain terms without definition | OOV + context check | "Process data properly" |
| `underspec_score` | Weighted composite | Aggregate | [0, 1] |

**Vague Terms Dictionary:**
```python
VAGUE_QUANTIFIERS = {'few', 'some', 'many', 'several', 'various', 'multiple', 'numerous'}
VAGUE_TEMPORALS = {'soon', 'later', 'recently', 'eventually', 'shortly', 'quickly'}
VAGUE_QUALIFIERS = {'good', 'bad', 'nice', 'proper', 'appropriate', 'suitable', 'best'}
```

**Hypothesis:** Missing arguments → Model must guess → Higher variance

### 4.3 Task Clarity Features (8 features)

| Feature | Description | Values | Consistency Risk |
|---------|-------------|--------|------------------|
| `question_type` | Open vs closed question | open/closed/mixed | Open → High risk |
| `answer_cardinality` | Single vs multiple valid answers | single/multiple/unbounded | Unbounded → High risk |
| `success_criteria_explicit` | Clear definition of "done" | 0-1 score | Implicit → High risk |
| `task_steps` | Single vs multi-step task | count | Multi → Higher risk |
| `constraint_count` | Number of explicit constraints | count | More → Lower risk |
| `goal_explicitness` | How clearly goal is stated | 0-1 score | Vague → High risk |
| `output_format_specified` | Format requirements given | boolean | No → High risk |
| `task_clarity_score` | Weighted composite | [0, 1] | Low → High risk |

**Open Question Indicators:**
```python
OPEN_QUESTION_PATTERNS = [
    r'^(what|how|why|describe|explain|discuss)\b',
    r'(thoughts|opinion|ideas|suggestions)\??$',
    r'(could you|can you).*(help|assist|suggest)',
    r'(any|some).*(way|method|approach)',
]

CLOSED_QUESTION_PATTERNS = [
    r'^(is|are|do|does|did|will|can|should)\b.*\?$',
    r'(true|false|yes|no)\??$',
    r'(which one|select|choose)\b',
]
```

**Hypothesis:** Open questions with no constraints → Highest variance

### 4.4 Semantic Complexity Features (6 features)

| Feature | Description | Detection Method |
|---------|-------------|------------------|
| `entity_count` | Named entities mentioned | NER (spaCy) |
| `relation_count` | Relationships between entities | Dependency parsing |
| `logical_operators` | and, or, not, if-then, unless | Pattern + dependency |
| `coreference_chains` | Pronoun reference chain count | Coreference resolution |
| `negation_complexity` | Nested/complex negations | Negation scope parsing |
| `semantic_density` | (entities + relations) / words | Ratio |

**Logical Operator Detection:**
```python
LOGICAL_OPERATORS = {
    'conjunction': ['and', 'also', 'as well as', 'both', 'plus'],
    'disjunction': ['or', 'either', 'alternatively'],
    'negation': ['not', 'no', 'never', "don't", "won't", 'without'],
    'conditional': ['if', 'when', 'unless', 'provided', 'assuming'],
    'causal': ['because', 'since', 'therefore', 'so', 'thus'],
}
```

**Hypothesis:** High semantic density + many logical operators → Complex reasoning → Lower consistency

### 4.5 Pragmatic Features (5 features)

| Feature | Description | Example |
|---------|-------------|---------|
| `presupposition_count` | Assumed facts in prompt | "Stop the process" assumes it's running |
| `implicature_strength` | Implied but unstated meaning | "Can you..." implies request |
| `context_dependency` | Requires external context | "Continue from last time" |
| `speech_act_indirectness` | Indirect requests/commands | "It would be nice if..." |
| `pragmatic_load` | Composite pragmatic complexity | [0, 1] |

**Presupposition Triggers:**
```python
PRESUPPOSITION_TRIGGERS = {
    'change_of_state': ['stop', 'start', 'begin', 'continue', 'resume', 'finish'],
    'factive': ['know', 'realize', 'regret', 'notice', 'remember'],
    'definite_descriptions': ['the', 'this', 'that'],  # + NP
    'clefts': ['it is', 'it was'],  # + that clause
    'temporal_clauses': ['before', 'after', 'since', 'while'],
}
```

**Hypothesis:** High pragmatic load → More inference required → Less deterministic outputs

---

## 5. Feature Summary

| Category | Count | Type | Expected Importance |
|----------|-------|------|---------------------|
| Modal verbs | 6 | Surface linguistic | Medium |
| Hedging | 5 | Surface linguistic | Medium |
| Politeness | 6 | Surface linguistic | Low-Medium |
| Speech acts | 4 | Surface linguistic | Medium |
| Structural | 8 | Surface | Low |
| Schema complexity | 6 | Task-specific | High |
| **Ambiguity** | **7** | **Semantic** | **Very High** |
| **Underspecification** | **6** | **Semantic** | **Very High** |
| **Task clarity** | **8** | **Pragmatic** | **Very High** |
| **Semantic complexity** | **6** | **Semantic** | **High** |
| **Pragmatic** | **5** | **Pragmatic** | **High** |
| **Total** | **67** | | |

### Expected Feature Importance Ranking

| Rank | Feature Category | Expected |r| | Rationale |
|------|------------------|----------|-----------|
| 1 | Task clarity | 0.40 | Open/ambiguous tasks have many valid responses |
| 2 | Underspecification | 0.35 | Missing info forces model to guess |
| 3 | Ambiguity | 0.32 | Multiple interpretations → multiple outputs |
| 4 | Schema complexity | 0.25 | More params = more variation points |
| 5 | Semantic complexity | 0.20 | Complex reasoning less deterministic |
| 6 | Modal verbs | 0.15 | Constraint strength affects flexibility |
| 7 | Pragmatic | 0.12 | Inference overhead adds variance |
| 8 | Hedging | 0.10 | Uncertainty markers small effect |
| 9 | Politeness | 0.08 | Minimal direct effect |
| 10 | Structural | 0.05 | Length etc. weak predictors |

---

## 6. Predictor Model Architecture

### 6.1 Baseline Models

| Model | Description | Expected Performance |
|-------|-------------|---------------------|
| Linear Regression | Feature weights interpretable | Baseline |
| Ridge Regression | L2 regularization | +2-5% |
| Random Forest | Non-linear interactions | +5-10% |
| XGBoost | Gradient boosting | +10-15% |

### 6.2 Neural Models

| Model | Architecture | Parameters |
|-------|--------------|------------|
| MLP | 67 → 256 → 128 → 64 → 1 | ~30K |
| Small Transformer | 4 layers, 128 dim | ~500K |
| DistilBERT + Head | Frozen encoder + MLP | ~67M (frozen) + 30K |

### 6.3 Two-Stage Model (Proposed)

```
Stage 1: Multi-Level Feature Extractor
  Input: Raw prompt text
  Output: 67 features across 3 levels

  Level A - Surface Linguistic (25 features):
    - Modal verbs, hedging, politeness, speech acts, structural
    - Method: Pattern matching + spaCy POS

  Level B - Semantic (19 features):
    - Ambiguity, underspecification, semantic complexity
    - Method: WordNet, coreference, dependency parsing

  Level C - Pragmatic (17 features):
    - Task clarity, pragmatic load
    - Method: Pattern matching + heuristics

  Level D - Schema (6 features):
    - Tool/parameter complexity
    - Method: JSON schema analysis

Stage 2: Consistency Predictor
  Input: 67 features + [optional] prompt embedding
  Output: Predicted c_mean ∈ [0, 1]
  Method: XGBoost or MLP with feature groups
```

**Architecture Diagram:**
```
┌─────────────────────────────────────────────────────────────┐
│                      Raw Prompt Text                        │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Surface Features│ │Semantic Features│ │Pragmatic Features│
│   (25 features) │ │  (19 features)  │ │   (17 features)  │
│ • Modal verbs   │ │ • Ambiguity     │ │ • Task clarity   │
│ • Hedging       │ │ • Underspec     │ │ • Presupposition │
│ • Politeness    │ │ • Complexity    │ │ • Context dep    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Feature Vector [67]│
                  │  + Schema [6]       │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │   XGBoost / MLP     │
                  │   Predictor         │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Predicted c_mean    │
                  │     [0, 1]          │
                  └─────────────────────┘
```

**Advantages:**
- Interpretable (feature importance by category)
- Fast inference (~5ms including NLP)
- Explainable predictions ("low consistency due to high ambiguity")

### 6.4 End-to-End Model (Comparison)

```
Input: Raw prompt text
Encoder: DistilBERT (frozen)
Head: MLP (768 → 256 → 64 → 1)
Output: Predicted c_mean
```

**Advantages:**
- Captures implicit features not in hand-crafted set
- May find patterns rules miss
- Serves as upper bound for feature-based approach

### 6.5 Ablation: Feature Group Importance

| Experiment | Features Used | Expected r |
|------------|---------------|------------|
| Surface only | 25 | 0.45 |
| + Semantic | 25 + 19 = 44 | 0.65 |
| + Pragmatic | 44 + 17 = 61 | 0.75 |
| + Schema | 61 + 6 = 67 | 0.78 |
| End-to-end (DistilBERT) | - | 0.80 |

---

## 7. Experiments

### Experiment 1: Feature Correlation Analysis

**Goal:** Identify which features correlate with consistency across all 67 dimensions

**Method:**
1. Extract 67 features from all 1,006 prompts
2. Compute Pearson/Spearman correlation with c_mean
3. Test significance (p < 0.05 after Bonferroni correction for 67 tests)
4. Rank features by |correlation|

**Expected Output:**
| Rank | Feature | Category | Correlation | p-value |
|------|---------|----------|-------------|---------|
| 1 | `task_clarity_score` | Pragmatic | -0.38 | <0.001 |
| 2 | `underspec_score` | Semantic | -0.34 | <0.001 |
| 3 | `referential_ambiguity` | Semantic | -0.30 | <0.001 |
| 4 | `answer_cardinality` | Pragmatic | -0.28 | <0.001 |
| 5 | `max_nesting_depth` | Schema | -0.24 | <0.001 |
| 6 | `lexical_ambiguity` | Semantic | -0.20 | <0.001 |
| 7 | `modal_deontic_strong` | Surface | -0.15 | <0.01 |
| 8 | `hedge_conditional` | Surface | -0.12 | <0.01 |

### Experiment 2: Causal Intervention Study

**Goal:** Test if semantic/pragmatic features causally affect consistency

**Method:**
1. Take 100 prompts with varied consistency
2. Systematically modify features across categories:

**Intervention Categories:**
| Category | Intervention | Example |
|----------|--------------|---------|
| Ambiguity resolution | Clarify pronouns | "it" → "the file" |
| Underspecification | Add missing info | "Send email" → "Send email to user@example.com" |
| Task clarity | Make explicit | "Help me" → "Return a JSON with fields X, Y, Z" |
| Surface linguistic | Weaken modals | "must" → "should" |

3. Re-run LLM generation (NEW API CALLS - limited)
4. Compare consistency before/after intervention

**Cost estimate:** 100 prompts × 8 interventions × 10 runs × 4 models = 32K calls (~$150)

**Expected Output:**
| Intervention Type | Example | Δ Consistency | p-value | Effect Size |
|-------------------|---------|---------------|---------|-------------|
| Clarify reference | "it" → "the report" | +0.15 | <0.001 | Large |
| Add constraints | Open → constrained | +0.12 | <0.001 | Large |
| Specify output format | Add JSON schema | +0.10 | <0.01 | Medium |
| Resolve underspec | Add missing args | +0.08 | <0.01 | Medium |
| Weaken modal | "must" → "should" | +0.05 | <0.05 | Small |
| Add politeness | "Do X" → "Please do X" | +0.02 | 0.15 | None |

**Key Finding (Expected):** Semantic/pragmatic interventions have 2-3x larger effect than surface linguistic changes

### Experiment 3: Predictor Model Training

**Goal:** Train models to predict c_mean from 67 prompt features

**Method:**
1. Split data: 700 train, 150 val, 156 test
2. Train all baseline and neural models
3. Evaluate on held-out test set
4. Perform ablation on feature groups

**Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| Pearson r | Correlation | >0.75 |
| Spearman ρ | Rank correlation | >0.70 |
| RMSE | Root mean squared error | <0.12 |
| MAE | Mean absolute error | <0.08 |
| AUC (low-C) | Detecting c_mean < 0.7 | >0.80 |

**Expected Output:**
| Model | Features | Pearson r | RMSE | AUC |
|-------|----------|-----------|------|-----|
| Linear Regression | 67 | 0.58 | 0.16 | 0.70 |
| Ridge Regression | 67 | 0.62 | 0.15 | 0.72 |
| Random Forest | 67 | 0.70 | 0.13 | 0.76 |
| XGBoost | 67 | 0.78 | 0.10 | 0.82 |
| MLP (67→256→128→1) | 67 | 0.76 | 0.11 | 0.80 |
| DistilBERT + Head | text | 0.80 | 0.09 | 0.84 |

**Feature Group Ablation:**
| Configuration | Features | Pearson r | Δ from previous |
|---------------|----------|-----------|-----------------|
| Surface only | 25 | 0.45 | - |
| + Semantic | 44 | 0.65 | +0.20 |
| + Pragmatic | 61 | 0.74 | +0.09 |
| + Schema | 67 | 0.78 | +0.04 |

**Key Finding (Expected):** Semantic features provide largest incremental gain (+0.20)

### Experiment 4: Cross-Model Generalization

**Goal:** Test if predictor generalizes to unseen models

**Method:**
1. Train on: Claude-Sonnet-4, GPT-4.1-Mini, Llama-3.3-70B, Qwen3-235B
2. Test on: Claude-3.5-Sonnet, Gemini-2.5-Flash, Nova2-Lite (zero-shot)
3. Fine-tune with 50 samples from test models
4. Compare zero-shot vs fine-tuned performance

**Expected Output:**
| Test Model | Zero-shot r | Fine-tuned r | Δ |
|------------|-------------|--------------|---|
| Claude-3.5-Sonnet | 0.65 | 0.73 | +0.08 |
| Gemini-2.5-Flash | 0.58 | 0.70 | +0.12 |
| Nova2-Lite | 0.52 | 0.68 | +0.16 |

### Experiment 5: Practical Application - Prompt Rewriting

**Goal:** Show predictor can guide prompt engineering with focus on semantic/pragmatic fixes

**Method:**
1. Take 100 low-consistency prompts (c_mean < 0.6)
2. Use predictor to identify problematic features (ranked by importance)
3. Automatically rewrite prompts using feature-specific rules
4. Run LLM generation on rewritten prompts (NEW API CALLS)
5. Measure actual consistency improvement

**Rewriting Rules (Priority Order):**
| Priority | Problem | Feature | Rewrite Strategy | Example |
|----------|---------|---------|------------------|---------|
| 1 | Open-ended task | `task_clarity_score` | Add constraints | "Help me" → "List exactly 3 options" |
| 2 | Missing info | `underspec_score` | Add defaults/examples | "Send email" → "Send email to [recipient]" |
| 3 | Unclear reference | `referential_ambiguity` | Resolve pronouns | "Process it" → "Process the CSV file" |
| 4 | Vague terms | `vague_quantifiers` | Make specific | "a few items" → "3-5 items" |
| 5 | Multiple interpretations | `lexical_ambiguity` | Disambiguate | "bank data" → "financial bank data" |
| 6 | Strong modal | `modal_deontic_strong` | Weaken | "must" → "should" |
| 7 | Excess hedging | `hedge_count` | Remove | "maybe possibly" → "" |

**Cost estimate:** 100 prompts × 10 runs × 4 models = 4K calls (~$15)

**Expected Output:**
| Metric | Original | Rewritten | Improvement |
|--------|----------|-----------|-------------|
| Mean c_mean | 0.52 | 0.72 | +38% |
| c_mean < 0.5 | 45% | 12% | -73% |
| c_mean > 0.8 | 12% | 42% | +250% |

**Breakdown by Rewrite Type:**
| Rewrite Type | N Prompts | Avg Δ c_mean |
|--------------|-----------|--------------|
| Add constraints | 35 | +0.22 |
| Resolve ambiguity | 28 | +0.18 |
| Add missing info | 22 | +0.15 |
| Surface fixes only | 15 | +0.06 |

---

## 8. Statistical Analysis

### 8.1 Correlation Tests

| Test | Purpose | Threshold |
|------|---------|-----------|
| Pearson r | Linear correlation | p < 0.05 |
| Spearman ρ | Rank correlation | p < 0.05 |
| Bonferroni correction | Multiple comparisons | α/67 |
| FDR correction | Alternative for many tests | q < 0.05 |

### 8.2 Model Comparison

| Test | Purpose | Threshold |
|------|---------|-----------|
| Paired t-test | Compare two models | p < 0.05 |
| Wilcoxon signed-rank | Non-parametric comparison | p < 0.05 |
| Bootstrap CI | Confidence intervals | 95%, 1000 resamples |
| McNemar's test | Classification agreement | p < 0.05 |

### 8.3 Feature Importance

| Method | Description | Use Case |
|--------|-------------|----------|
| Permutation importance | Shuffle feature, measure drop | All models |
| SHAP values | Shapley additive explanations | Tree models |
| Coefficient magnitude | Standardized coefficients | Linear models |
| Attention weights | Token importance | Transformer |

### 8.4 Causal Inference

| Method | Description | Purpose |
|--------|-------------|---------|
| Paired difference test | Before/after intervention | Exp 2 |
| Effect size (Cohen's d) | Magnitude of change | Exp 2 |
| Confidence intervals | Uncertainty quantification | All |

---

## 9. Implementation Plan

### 9.1 Scripts to Create

```
scripts/experiments/colm_consistency_predictor/
├── features/
│   ├── surface_features.py          # Modal, hedging, politeness (25 features)
│   ├── semantic_features.py         # Ambiguity, underspec, complexity (19 features)
│   ├── pragmatic_features.py        # Task clarity, pragmatic load (17 features)
│   ├── schema_features.py           # Tool/param complexity (6 features)
│   └── extract_all_features.py      # Combined extraction pipeline
├── experiments/
│   ├── exp1_correlations.py         # Feature correlation analysis
│   ├── exp2_interventions.py        # Causal intervention study
│   ├── exp3_train_predictor.py      # Model training + ablation
│   ├── exp4_cross_model.py          # Generalization testing
│   └── exp5_rewriting.py            # Prompt rewriting evaluation
├── models/
│   ├── xgboost_predictor.py         # XGBoost implementation
│   ├── mlp_predictor.py             # MLP implementation
│   └── bert_predictor.py            # DistilBERT + head
├── rewriting/
│   ├── prompt_rewriter.py           # Main rewriting logic
│   ├── ambiguity_resolver.py        # Resolve referential ambiguity
│   ├── constraint_adder.py          # Add task constraints
│   └── underspec_fixer.py           # Fix missing information
├── analysis/
│   ├── generate_paper_figures.py    # All visualizations
│   └── generate_latex_tables.py     # Paper tables
└── utils/
    └── data_loader.py               # Load KDD experiment data
```

### 9.2 Dependencies

```
# Core NLP
spacy>=3.7                    # Tokenization, POS, NER, dependency parsing
nltk>=3.8                     # WordNet for lexical ambiguity
stanza>=1.7                   # Alternative parser for ambiguity detection

# Coreference Resolution
spacy-experimental            # For coreference
fastcoref                     # Alternative: fast neural coref

# Semantic Analysis
wordnet                       # Via nltk
sentence-transformers>=2.2    # For semantic similarity

# ML
scikit-learn>=1.3
xgboost>=2.0
torch>=2.0
transformers>=4.35

# Analysis & Visualization
shap>=0.42
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
scipy>=1.11

# Utilities
tqdm
jsonlines
```

### 9.3 NLP Model Downloads

```bash
# spaCy models
python -m spacy download en_core_web_lg

# Stanza models
import stanza
stanza.download('en')

# NLTK data
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```

### 9.4 Compute Requirements

| Task | Compute | Time | Notes |
|------|---------|------|-------|
| Feature extraction (67 features) | CPU | ~30 min | 1,006 prompts |
| Correlation analysis | CPU | ~5 min | |
| Model training (all) | CPU/GPU | ~1 hour | XGBoost fastest |
| Causal interventions | API calls | ~3 hours | 32K calls |
| Prompt rewriting eval | API calls | ~30 min | 4K calls |
| **Total** | | **~5 hours** | |

---

## 10. Paper Outline

### Abstract (~150 words)
- Problem: LLM structured output consistency is unpredictable
- Gap: No way to estimate consistency before expensive inference
- Key insight: Semantic/pragmatic features matter more than surface linguistics
- Contribution: 67-feature predictor model achieving 0.78 correlation
- Results: 38% consistency improvement from predictor-guided rewriting

### 1. Introduction (1.5 pages)
- Motivation: Production LLM deployments need reliability estimates
- Gap: Must run N samples to measure consistency (expensive)
- Insight: Prompt ambiguity and task clarity predict consistency
- Contribution: Multi-level feature framework + predictor + rewriting tool

### 2. Related Work (1 page)
- LLM consistency measurement (cite our ICML)
- Linguistic pragmatics (modal semantics, politeness theory)
- Ambiguity in NLP (lexical, referential, syntactic)
- Prompt engineering and optimization

### 3. Feature Framework (2.5 pages)
- 3.1 Surface Linguistic Features (25): modals, hedging, politeness
- 3.2 Semantic Features (19): ambiguity, underspecification, complexity
- 3.3 Pragmatic Features (17): task clarity, presupposition, context
- 3.4 Schema Features (6): tool/parameter complexity
- 3.5 Detection methodology and implementation

### 4. Consistency Predictor (1.5 pages)
- Two-stage architecture (feature extraction → prediction)
- Model comparison (XGBoost vs MLP vs DistilBERT)
- Feature group ablation study

### 5. Experiments (3 pages)
- Exp 1: Feature correlations (semantic > surface)
- Exp 2: Causal interventions (ambiguity resolution has largest effect)
- Exp 3: Predictor performance (0.78 correlation)
- Exp 4: Cross-model generalization
- Exp 5: Predictor-guided prompt rewriting (+38%)

### 6. Discussion (1 page)
- Why semantic features dominate: multiple valid interpretations
- Implications for prompt engineering: clarity > politeness
- Limitations: domain specificity, feature engineering effort

### 7. Conclusion (0.5 page)
- Key finding: Task clarity and ambiguity are strongest predictors
- Practical tool: Lightweight predictor for production use
- Future: End-to-end learned features, domain adaptation

---

## 11. Timeline

| Week | Dates | Tasks |
|------|-------|-------|
| 1 | Feb 10-16 | Implement 67-feature extraction pipeline |
| 2 | Feb 17-23 | Exp 1 (correlations), Exp 3 (model training) |
| 3 | Feb 24-Mar 2 | Exp 2 (causal interventions), Exp 4 (cross-model) |
| 4 | Mar 3-9 | Exp 5 (rewriting), figures, tables |
| 5 | Mar 10-16 | Paper writing (intro, feature framework) |
| 6 | Mar 17-23 | Paper writing (experiments, discussion) |
| 7 | Mar 24-26 | Abstract submission (Mar 26 deadline) |
| 8 | Mar 27-31 | Final revisions, full paper submission |

---

## 12. Cost Estimate

| Item | Quantity | Cost |
|------|----------|------|
| Existing data analysis | - | $0 |
| Causal interventions (Exp 2) | 32K API calls | ~$150 |
| Prompt rewriting (Exp 5) | 4K API calls | ~$15 |
| EC2 compute | 5 hours | ~$5 |
| **Total** | | **~$170** |

---

## 13. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Semantic features hard to extract | Use established NLP tools (spaCy, Stanza, WordNet) |
| Low correlation with semantic features | Include all 67 features; ablation will show best subset |
| Coreference resolution errors | Use ensemble of coref models |
| Poor predictor generalization | Test on held-out models; fine-tuning option |
| Causal interventions inconclusive | Focus on correlational analysis as primary |
| Time constraints | Prioritize Exp 1, 3, 5; Exp 2, 4 as stretch goals |

---

## 14. Expected Contributions

1. **Theoretical:** First systematic analysis showing that **semantic/pragmatic features** (ambiguity, task clarity, underspecification) predict LLM structured output consistency more strongly than surface linguistic features (modal verbs, hedging, politeness)

2. **Methodological:** 67-feature extraction framework spanning 4 levels:
   - Surface linguistic (25 features)
   - Semantic (19 features)
   - Pragmatic (17 features)
   - Schema complexity (6 features)

3. **Practical:** Lightweight predictor model (~30K params) achieving:
   - 0.78 Pearson correlation with actual consistency
   - 0.82 AUC for detecting low-consistency prompts
   - ~5ms inference time (no GPU required)

4. **Actionable:** Predictor-guided prompt rewriting system achieving:
   - 38% average consistency improvement
   - 73% reduction in low-consistency prompts
   - Prioritized fixes: constraints > ambiguity resolution > underspec fixes > surface changes

5. **Empirical:** Analysis of 67 features across 2.1M LLM outputs from 19 models, revealing:
   - Task clarity (r=-0.38) strongest predictor
   - Semantic features provide +0.20 correlation gain over surface features alone
   - Causal evidence that ambiguity resolution improves consistency

---

## 15. Key Takeaway

> **Prompt clarity matters more than prompt politeness.** To improve LLM structured output consistency, focus on resolving ambiguity, adding constraints, and specifying missing information—not on modal verb choice or hedging patterns.
