# ACL 2026 Paper: Experiment Plan

## Title (Working)
**"How Do LLMs Interpret Instructions? A Linguistic Analysis of Pragmatic and Semantic Factors in Structured Output Consistency"**

## Abstract (Draft)
Large Language Models increasingly generate structured outputs for production systems, yet we lack understanding of how linguistic properties of instructions affect output consistency. Through controlled experiments manipulating pragmatic and semantic features across 18 LLMs and 50,000+ outputs, we present the first systematic linguistic analysis of structured output consistency. We investigate: (1) how speech act types (directives, requests, questions) affect consistency; (2) whether modal verb force (deontic vs epistemic, strong vs weak) predicts output stability; (3) how politeness strategies interact with consistency; and (4) whether syntactic complexity compounds these effects. Our findings reveal that [KEY FINDINGS TBD]. We provide linguistically-grounded guidelines for prompt design and release a benchmark for evaluating instruction interpretation.

---

## 1. Research Questions

### RQ1: Pragmatics - Speech Acts
How do different illocutionary forces affect structured output consistency?

### RQ2: Pragmatics - Politeness
Do politeness strategies (Brown & Levinson) systematically affect consistency?

### RQ3: Semantics - Modality
Does modal verb type (deontic/epistemic) and strength correlate with consistency?

### RQ4: Semantics - Quantification & Scope
How do quantifiers and scope ambiguities affect multi-tool consistency?

### RQ5: Syntax - Complexity Interaction
Does syntactic complexity compound pragmatic/semantic effects?

---

## 2. Theoretical Framework

### 2.1 Speech Act Theory (Austin, Searle)

| Category | Illocutionary Force | Example | Hypothesis |
|----------|---------------------|---------|------------|
| **Directives** | Command | "Get the weather for Tokyo" | Highest consistency |
| **Commissives** | Promise/Offer | "I will need you to get the weather" | Medium |
| **Assertives** | Statement | "The weather in Tokyo needs to be retrieved" | Lower |
| **Interrogatives** | Question | "What is the weather in Tokyo?" | Variable |
| **Indirect** | Indirect request | "I wonder what the weather is like in Tokyo" | Lowest |

**Hypothesis H1:** Direct directives yield higher consistency than indirect speech acts.

### 2.2 Politeness Theory (Brown & Levinson 1987)

| Strategy | Description | Example | Hypothesis |
|----------|-------------|---------|------------|
| **Bald on-record** | Direct, no mitigation | "Get the weather" | Highest consistency |
| **Positive politeness** | Appeal to solidarity | "I'd love it if you could get the weather" | Medium |
| **Negative politeness** | Minimize imposition | "Would you mind getting the weather?" | Lower |
| **Off-record** | Hints, indirect | "The weather seems relevant here..." | Lowest |

**Hypothesis H2:** Bald on-record instructions yield higher consistency than mitigated forms.

### 2.3 Modal Semantics (Kratzer, Palmer)

| Modal | Strength | Type | Example | Hypothesis |
|-------|----------|------|---------|------------|
| **must** | Strong | Deontic | "You must call get_weather" | High consistency |
| **should** | Medium | Deontic | "You should call get_weather" | Medium-high |
| **need to** | Medium | Deontic | "You need to call get_weather" | Medium-high |
| **can** | Weak | Dynamic | "You can call get_weather" | Medium |
| **could** | Weak | Epistemic | "You could call get_weather" | Lower |
| **might** | Weak | Epistemic | "You might call get_weather" | Lowest |

**Hypothesis H3:** Stronger deontic modals yield higher consistency than weak epistemic modals.

### 2.4 Quantification & Scope

| Pattern | Example | Ambiguity | Hypothesis |
|---------|---------|-----------|------------|
| **Universal-specific** | "Get weather for all cities: Tokyo, Paris" | Low | Higher consistency |
| **Universal-open** | "Get weather for all relevant cities" | High | Lower consistency |
| **Existential** | "Get weather for some city" | Medium | Variable |
| **Scope ambiguity** | "Don't get weather for all cities" | High | Lower consistency |

**Hypothesis H4:** Explicit quantification with enumeration yields higher consistency.

---

## 3. Experiment Design

### 3.1 Base Task Selection

From existing Toucan dataset, select **100 base prompts** with:
- Single tool call (to isolate linguistic effects)
- Clear ground truth
- Moderate baseline consistency (0.5-0.8) to allow measurement in both directions

### 3.2 Controlled Variation Generation

For each base prompt, generate systematic variations:

#### Experiment A: Speech Acts (5 variations x 100 prompts = 500)

```
Base: Weather query for Tokyo

1. DIRECTIVE (command):
   "Get the weather forecast for Tokyo"

2. DIRECTIVE (instruction):
   "Retrieve the weather information for Tokyo"

3. REQUEST (polite):
   "Please get the weather forecast for Tokyo"

4. INTERROGATIVE (direct question):
   "What is the weather forecast for Tokyo?"

5. INDIRECT (hint):
   "I'm wondering about the weather conditions in Tokyo"
```

#### Experiment B: Politeness Strategies (4 variations x 100 prompts = 400)

```
Base: Weather query for Tokyo

1. BALD ON-RECORD:
   "Get the weather for Tokyo"

2. POSITIVE POLITENESS:
   "I'd really appreciate if you could get the weather for Tokyo"

3. NEGATIVE POLITENESS:
   "Would you mind getting the weather for Tokyo, if it's not too much trouble?"

4. OFF-RECORD:
   "I suppose knowing the weather in Tokyo would be helpful..."
```

#### Experiment C: Modal Verbs (6 variations x 100 prompts = 600)

```
Base: Weather query for Tokyo

1. MUST (strong deontic):
   "You must get the weather for Tokyo"

2. SHOULD (medium deontic):
   "You should get the weather for Tokyo"

3. NEED TO (medium deontic):
   "You need to get the weather for Tokyo"

4. CAN (dynamic):
   "You can get the weather for Tokyo"

5. COULD (weak epistemic):
   "You could get the weather for Tokyo"

6. MIGHT (weak epistemic):
   "You might want to get the weather for Tokyo"
```

#### Experiment D: Quantification (4 variations x 50 multi-tool prompts = 200)

```
Base: Weather query for multiple cities

1. UNIVERSAL-ENUMERATED:
   "Get the weather for all of these cities: Tokyo, Paris, London"

2. UNIVERSAL-OPEN:
   "Get the weather for all relevant cities mentioned"

3. EXISTENTIAL:
   "Get the weather for at least one of the major cities"

4. DISTRIBUTIVE:
   "Get the weather for each city separately"
```

#### Experiment E: Syntactic Complexity Interaction (3 levels x 300 = 900)

Cross syntax with Experiments A-C:

```
1. SIMPLE (main clause):
   "You should get the weather for Tokyo"

2. EMBEDDED (complement clause):
   "I think you should get the weather for Tokyo"

3. SUBORDINATE (conditional):
   "If needed, you should get the weather for Tokyo"
```

### 3.3 Total Experiment Size

| Experiment | Variations | Base Prompts | Total Prompts |
|------------|------------|--------------|---------------|
| A: Speech Acts | 5 | 100 | 500 |
| B: Politeness | 4 | 100 | 400 |
| C: Modal Verbs | 6 | 100 | 600 |
| D: Quantification | 4 | 50 | 200 |
| E: Syntax Interaction | 3 x 3 | 100 | 900 |
| **Total unique prompts** | | | **~2,600** |

### 3.4 Model Selection

**Primary models (full evaluation):**
- Claude-Sonnet-4 (high baseline consistency)
- GPT-4.1-Mini (medium baseline)
- Llama-3.3-70B (lower baseline)

**Extended models (key conditions):**
- All 18 models from ICML/KDD papers

### 3.5 Evaluation Protocol

For each (prompt, model, temperature) combination:
- Generate **10 outputs**
- Temperatures: 0.0, 0.3, 0.5, 0.7, 1.0
- Compute: $C_{mean}$, $S_\alpha$, validity rate

**Total outputs:** 2,600 prompts x 3 models x 5 temps x 10 runs = **390,000 outputs**

---

## 4. Analysis Plan

### 4.1 Primary Analyses

#### Analysis 1: Speech Act Effects
```
Model: S_alpha ~ speech_act_type + (1|prompt_id) + (1|model)
Contrasts: directive vs indirect, interrogative vs declarative
```

#### Analysis 2: Politeness Effects
```
Model: S_alpha ~ politeness_strategy + (1|prompt_id) + (1|model)
Contrasts: bald vs mitigated, positive vs negative politeness
```

#### Analysis 3: Modal Verb Effects
```
Model: S_alpha ~ modal_strength + modal_type + (1|prompt_id) + (1|model)
Factors: strength (strong/medium/weak), type (deontic/epistemic/dynamic)
```

#### Analysis 4: Quantification Effects
```
Model: S_alpha ~ quantifier_type + enumeration + (1|prompt_id) + (1|model)
```

#### Analysis 5: Interaction Effects
```
Model: S_alpha ~ linguistic_feature * syntactic_complexity * temperature + (1|prompt_id)
```

### 4.2 Secondary Analyses

1. **Accuracy preservation:** Does linguistic variation affect tool-calling accuracy?
2. **Cross-model consistency:** Do effects replicate across model families?
3. **Temperature interaction:** Do linguistic effects vary by temperature?

### 4.3 Statistical Methods

- Mixed-effects regression (random intercepts for prompt and model)
- Effect sizes: Cohen's d, partial eta-squared
- Multiple comparison correction: Bonferroni or FDR
- Bootstrap confidence intervals (n=2000)

---

## 5. Expected Contributions

### 5.1 Scientific Contributions

1. **First systematic linguistic analysis** of structured output consistency grounded in linguistic theory

2. **Empirical validation** of pragmatic and semantic factors affecting LLM instruction interpretation

3. **Interaction discovery:** How linguistic features interact with temperature and model architecture

4. **Theoretical framework** connecting speech act theory and LLM behavior

### 5.2 Practical Contributions

1. **Linguistically-grounded prompt guidelines:**
   - Use direct directives over indirect speech acts
   - Prefer bald on-record for consistency-critical applications
   - Use stronger deontic modals (should > could)
   - Enumerate explicitly rather than using open quantifiers

2. **Prompt optimization tool:** Automatic rewriting toward high-consistency linguistic patterns

3. **Benchmark release:** Controlled linguistic variation dataset for instruction interpretation

### 5.3 Novelty Over ICML/KDD Papers

| Aspect | ICML | KDD | This Paper (ACL) |
|--------|------|-----|------------------|
| Focus | Metric (STED) | Factor analysis | Linguistic analysis |
| Features | N/A | Surface (word count, has_should) | Theoretical (speech acts, modality) |
| Framework | None | Statistical | Linguistic theory |
| Variations | Natural | Causal (add/remove) | Controlled generation |
| Depth | N/A | Binary (has/doesn't have) | Multi-level taxonomy |

---

## 6. Paper Outline

### Abstract (250 words)

### 1. Introduction (1 page)
- Motivation: LLMs for structured outputs
- Gap: No linguistic analysis of instruction interpretation
- Contributions: 4 bullet points

### 2. Background (1 page)
- 2.1 Structured Output Consistency (brief, cite ICML/KDD)
- 2.2 Speech Act Theory
- 2.3 Politeness Theory
- 2.4 Modal Semantics

### 3. Experimental Setup (1.5 pages)
- 3.1 Task and Dataset
- 3.2 Linguistic Variation Design
- 3.3 Models and Evaluation Protocol

### 4. Results (2.5 pages)
- 4.1 Speech Act Effects (RQ1)
- 4.2 Politeness Effects (RQ2)
- 4.3 Modal Verb Effects (RQ3)
- 4.4 Quantification Effects (RQ4)
- 4.5 Interaction Effects (RQ5)

### 5. Analysis and Discussion (1.5 pages)
- 5.1 Why Do Linguistic Features Matter?
- 5.2 Cross-Model Patterns
- 5.3 Practical Implications

### 6. Related Work (0.5 pages)

### 7. Conclusion (0.5 pages)

### References

### Appendix
- Full prompt templates
- Additional results tables
- Statistical details

---

## 7. Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Preparation** | 2 weeks | Select base prompts, generate variations, validate templates |
| **Data Collection** | 3 weeks | Run experiments across models |
| **Analysis** | 2 weeks | Statistical analysis, visualization |
| **Writing** | 3 weeks | Draft, revise, polish |
| **Buffer** | 2 weeks | Address issues, final revision |
| **Total** | ~12 weeks | |

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Null results for some features | Medium | Pre-register hypotheses, report all results |
| High API costs | Medium | Prioritize primary models, batch efficiently |
| Reviewer concern about "prompt engineering" | High | Emphasize theoretical grounding, cite linguistics literature |
| Limited cross-model generalization | Medium | Include diverse model families |

---

## 9. Resources Required

### Compute
- API costs: ~$2,000-3,000 (390K outputs)
- Local compute: Minimal (analysis only)

### Data
- Toucan dataset (existing)
- Generated linguistic variations (new)

### Code
- Variation generation scripts
- Evaluation pipeline (reuse from ICML/KDD)
- Statistical analysis scripts (R/Python)

---

## 10. Success Criteria

**Minimum viable:**
- At least 2 linguistic features show significant effects (p < 0.01, |d| > 0.2)
- Effects replicate across 2+ model families
- Clear practical guidelines derivable

**Strong result:**
- Systematic effects across all feature categories
- Interaction effects discovered
- Theory-consistent patterns (stronger modals → higher consistency)

**Exceptional:**
- Large effect sizes (|d| > 0.5 for some features)
- Novel unexpected findings
- Cross-lingual validation (if time permits)
