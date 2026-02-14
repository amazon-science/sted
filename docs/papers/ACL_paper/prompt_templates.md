# ACL Paper: Prompt Variation Templates

## Overview

This document provides concrete examples of how each linguistic variation is generated from base prompts.

---

## Template 1: Speech Act Variations

### Base Prompt (from Toucan)
```
Original: "I need to check the weather forecast for my trip to Tokyo next week"
Tool: get_weather(location="Tokyo")
```

### Generated Variations

| ID | Speech Act | Variation |
|----|------------|-----------|
| SA1 | **Directive (imperative)** | "Get the weather forecast for Tokyo for next week." |
| SA2 | **Directive (infinitive)** | "Retrieve the weather information for Tokyo." |
| SA3 | **Request (please)** | "Please get the weather forecast for Tokyo for next week." |
| SA4 | **Interrogative (direct)** | "What is the weather forecast for Tokyo next week?" |
| SA5 | **Interrogative (indirect)** | "Can you tell me what the weather will be like in Tokyo?" |
| SA6 | **Declarative (need)** | "I need the weather forecast for Tokyo for next week." |
| SA7 | **Hint (indirect)** | "I'm planning a trip to Tokyo and wondering about the weather..." |

### Transformation Rules

```python
def generate_speech_act_variations(base_action, base_args):
    """
    base_action: "get the weather"
    base_args: {"location": "Tokyo", "time": "next week"}
    """
    templates = {
        "directive_imperative": "{action} for {location}.",
        "directive_infinitive": "Retrieve {object} for {location}.",
        "request_please": "Please {action} for {location}.",
        "interrogative_direct": "What is {object} for {location}?",
        "interrogative_indirect": "Can you tell me {object} for {location}?",
        "declarative_need": "I need {object} for {location}.",
        "hint_indirect": "I'm wondering about {object} in {location}..."
    }
    return {k: v.format(...) for k, v in templates.items()}
```

---

## Template 2: Politeness Variations

### Base Prompt
```
Original: "Search for flights from NYC to London"
Tool: search_flights(origin="NYC", destination="London")
```

### Generated Variations (Brown & Levinson Framework)

| ID | Strategy | Variation |
|----|----------|-----------|
| PL1 | **Bald on-record** | "Search for flights from NYC to London." |
| PL2 | **Positive politeness (solidarity)** | "Hey, could you search for flights from NYC to London for me?" |
| PL3 | **Positive politeness (appreciation)** | "I'd really appreciate it if you could search for flights from NYC to London." |
| PL4 | **Negative politeness (minimize)** | "Would you mind searching for flights from NYC to London?" |
| PL5 | **Negative politeness (apologetic)** | "Sorry to bother you, but could you search for flights from NYC to London?" |
| PL6 | **Off-record (hint)** | "I'm thinking about flying from NYC to London..." |
| PL7 | **Off-record (association)** | "Flight prices between NYC and London seem interesting..." |

### Politeness Markers

```python
POLITENESS_MARKERS = {
    "bald": [],
    "positive": ["hey", "thanks", "appreciate", "love it if"],
    "negative": ["would you mind", "if it's not too much trouble",
                 "sorry to bother", "I was wondering if"],
    "off_record": ["I'm thinking about", "seems like", "I suppose"]
}
```

---

## Template 3: Modal Verb Variations

### Base Prompt
```
Original: "Calculate the distance between two coordinates"
Tool: calculate_distance(point1=[lat, lon], point2=[lat, lon])
```

### Generated Variations (Modal Strength Hierarchy)

| ID | Modal | Strength | Type | Variation |
|----|-------|----------|------|-----------|
| MV1 | **must** | Strong | Deontic | "You must calculate the distance between the coordinates." |
| MV2 | **have to** | Strong | Deontic | "You have to calculate the distance between the coordinates." |
| MV3 | **should** | Medium | Deontic | "You should calculate the distance between the coordinates." |
| MV4 | **ought to** | Medium | Deontic | "You ought to calculate the distance between the coordinates." |
| MV5 | **need to** | Medium | Deontic | "You need to calculate the distance between the coordinates." |
| MV6 | **can** | Weak | Dynamic | "You can calculate the distance between the coordinates." |
| MV7 | **could** | Weak | Epistemic | "You could calculate the distance between the coordinates." |
| MV8 | **may** | Weak | Epistemic | "You may calculate the distance between the coordinates." |
| MV9 | **might** | Weak | Epistemic | "You might want to calculate the distance between the coordinates." |
| MV10 | **No modal** | Baseline | - | "Calculate the distance between the coordinates." |

### Modal Hierarchy (Kratzer)

```
STRONG DEONTIC: must > have to > should > ought to > need to
WEAK EPISTEMIC: can > could > may > might

Expected consistency: must > should > can > might
```

---

## Template 4: Quantification Variations

### Base Prompt (Multi-tool)
```
Original: "Get weather for multiple cities for my travel planning"
Tools: get_weather(location=X) for X in [Tokyo, Paris, London]
```

### Generated Variations

| ID | Quantifier | Variation |
|----|------------|-----------|
| QT1 | **Universal + enumerated** | "Get the weather for all of these cities: Tokyo, Paris, and London." |
| QT2 | **Universal + open** | "Get the weather for all the cities I might visit." |
| QT3 | **Distributive** | "Get the weather for each city: Tokyo, Paris, and London." |
| QT4 | **Existential** | "Get the weather for at least one major city." |
| QT5 | **Partitive** | "Get the weather for some of the cities on my list." |
| QT6 | **Negated universal** | "Don't skip any cities - get weather for Tokyo, Paris, and London." |

### Scope Ambiguity Cases

```
Ambiguous: "Don't get weather for all cities"
Reading 1: NOT(for all cities): Get weather for some, not all
Reading 2: For all cities, NOT(get weather): Get weather for none

Hypothesis: Ambiguous quantification → lower consistency
```

---

## Template 5: Syntactic Complexity Variations

### Base Prompt
```
Original: "You should search for restaurants nearby"
Tool: search_restaurants(location="nearby")
```

### Generated Variations (Crossed with Modal)

| ID | Structure | Modal | Variation |
|----|-----------|-------|-----------|
| SX1 | **Simple (main clause)** | should | "You should search for restaurants nearby." |
| SX2 | **Embedded (complement)** | should | "I think you should search for restaurants nearby." |
| SX3 | **Embedded (relative)** | should | "The action you should take is searching for restaurants nearby." |
| SX4 | **Subordinate (conditional)** | should | "If I'm hungry, you should search for restaurants nearby." |
| SX5 | **Subordinate (temporal)** | should | "When I ask, you should search for restaurants nearby." |
| SX6 | **Coordinated** | should | "You should search for restaurants and filter by rating." |
| SX7 | **Complex (multiple embedding)** | should | "I believe that you should search for restaurants that are nearby." |

### Syntactic Complexity Metrics

```python
def compute_syntactic_complexity(sentence):
    """
    Metrics:
    - Embedding depth (number of clauses)
    - Dependency tree depth
    - Number of subordinate clauses
    - Words per clause
    """
    return {
        "embedding_depth": count_embedded_clauses(sentence),
        "dep_tree_depth": get_dependency_depth(sentence),
        "subordinate_count": count_subordinates(sentence),
        "words_per_clause": len(words) / num_clauses
    }
```

---

## Template 6: Combined Variations (Interaction Study)

### Full Factorial Design Example

Cross: Modal (3 levels) x Syntax (3 levels) x Politeness (2 levels) = 18 combinations

| ID | Modal | Syntax | Politeness | Variation |
|----|-------|--------|------------|-----------|
| C1 | must | simple | bald | "You must get the weather." |
| C2 | must | simple | polite | "Please, you must get the weather." |
| C3 | must | embedded | bald | "I think you must get the weather." |
| C4 | must | embedded | polite | "I think you really must get the weather, if you don't mind." |
| C5 | should | simple | bald | "You should get the weather." |
| C6 | should | simple | polite | "You should get the weather, please." |
| C7 | should | embedded | bald | "I think you should get the weather." |
| C8 | should | embedded | polite | "I believe you should get the weather, if possible." |
| C9 | might | simple | bald | "You might get the weather." |
| C10 | might | simple | polite | "You might want to get the weather, please." |
| C11 | might | embedded | bald | "I think you might get the weather." |
| C12 | might | embedded | polite | "I was wondering if you might get the weather..." |
| ... | ... | ... | ... | ... |

---

## Variation Generation Script

```python
"""
generate_linguistic_variations.py

Generates controlled linguistic variations from base Toucan prompts.
"""

import json
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class LinguisticVariation:
    base_id: str
    variation_type: str  # speech_act, politeness, modal, quantifier, syntax
    variation_id: str    # SA1, PL1, MV1, etc.
    original_prompt: str
    varied_prompt: str
    linguistic_features: Dict[str, str]

class VariationGenerator:

    SPEECH_ACT_TEMPLATES = {
        "directive_imperative": "{verb} {object}.",
        "directive_infinitive": "Retrieve {object}.",
        "request_please": "Please {verb} {object}.",
        "interrogative_direct": "What is {object}?",
        "interrogative_indirect": "Can you tell me {object}?",
        "declarative_need": "I need {object}.",
        "hint_indirect": "I'm wondering about {object}..."
    }

    MODAL_TEMPLATES = {
        "must": "You must {action}.",
        "have_to": "You have to {action}.",
        "should": "You should {action}.",
        "ought_to": "You ought to {action}.",
        "need_to": "You need to {action}.",
        "can": "You can {action}.",
        "could": "You could {action}.",
        "may": "You may {action}.",
        "might": "You might want to {action}.",
        "no_modal": "{Action}."  # Capitalized
    }

    POLITENESS_TEMPLATES = {
        "bald": "{request}",
        "positive_solidarity": "Hey, could you {request}?",
        "positive_appreciation": "I'd really appreciate if you could {request}.",
        "negative_minimize": "Would you mind {gerund}?",
        "negative_apologetic": "Sorry to bother you, but could you {request}?",
        "off_record_hint": "I'm thinking about {topic}...",
        "off_record_association": "{Topic} seems relevant here..."
    }

    SYNTAX_TEMPLATES = {
        "simple": "{sentence}",
        "embedded_complement": "I think {sentence_lower}",
        "embedded_relative": "The action you should take is {gerund}.",
        "subordinate_conditional": "If needed, {sentence_lower}",
        "subordinate_temporal": "When appropriate, {sentence_lower}",
        "coordinated": "{sentence} and report the results."
    }

    def generate_all_variations(self, base_prompts: List[Dict]) -> List[LinguisticVariation]:
        """Generate all variations for a list of base prompts."""
        variations = []
        for prompt in base_prompts:
            variations.extend(self.generate_speech_act_variations(prompt))
            variations.extend(self.generate_modal_variations(prompt))
            variations.extend(self.generate_politeness_variations(prompt))
            variations.extend(self.generate_syntax_variations(prompt))
        return variations

# Example usage
if __name__ == "__main__":
    generator = VariationGenerator()

    # Load base prompts
    with open("data/toucan/toucan_tool_calls_1006.json") as f:
        base_prompts = json.load(f)[:100]  # Select 100 base prompts

    # Generate variations
    variations = generator.generate_all_variations(base_prompts)

    # Save
    with open("data/acl_linguistic_variations.json", "w") as f:
        json.dump([v.__dict__ for v in variations], f, indent=2)

    print(f"Generated {len(variations)} variations from {len(base_prompts)} base prompts")
```

---

## Quality Control

### Validation Checklist

For each variation:
- [ ] Preserves original semantic intent (same tool should be called)
- [ ] Grammatically correct
- [ ] Natural sounding (not artificially constructed)
- [ ] Clearly distinct from other variations
- [ ] Linguistic feature correctly instantiated

### Human Validation Sample

Randomly sample 50 variations per category, have 2 annotators verify:
1. Intent preservation (same tool expected?)
2. Grammaticality (1-5 scale)
3. Naturalness (1-5 scale)
4. Feature correctness (yes/no)

---

## Expected Dataset Statistics

| Category | Variations/Prompt | Base Prompts | Total |
|----------|-------------------|--------------|-------|
| Speech Acts | 7 | 100 | 700 |
| Politeness | 7 | 100 | 700 |
| Modal Verbs | 10 | 100 | 1,000 |
| Quantification | 6 | 50 | 300 |
| Syntax | 6 | 100 | 600 |
| **Total** | | | **3,300** |

With 3 models x 5 temperatures x 10 runs = **495,000 outputs**
