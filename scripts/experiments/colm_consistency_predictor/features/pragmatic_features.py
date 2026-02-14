#!/usr/bin/env python3
"""
Pragmatic Features Extraction (17 features)

Categories:
- Task Clarity (8 features): question type, answer cardinality, success criteria, etc.
- Pragmatic Load (5 features): presuppositions, implicature, context dependency
- Additional Structural (4 features): list markers, conjunctions, negations, specificity

Based on:
- Speech act theory
- Pragmatic inference
- Task specification theory
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class PragmaticFeatures:
    """Container for 17 pragmatic features."""
    # Task Clarity (8)
    question_type: float = 0.5  # 0=closed, 1=open
    answer_cardinality: float = 0.5  # 0=single, 1=unbounded
    success_criteria_explicit: float = 0.5
    task_steps: int = 1
    constraint_count: int = 0
    goal_explicitness: float = 0.5
    output_format_specified: float = 0.0
    task_clarity_score: float = 0.5

    # Pragmatic Load (5)
    presupposition_count: int = 0
    implicature_strength: float = 0.0
    context_dependency: float = 0.0
    speech_act_indirectness: float = 0.0
    pragmatic_load: float = 0.0

    # Additional structural (4)
    list_markers: int = 0
    conjunction_count: int = 0
    negation_count: int = 0
    specificity_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Task clarity
            'question_type': self.question_type,
            'answer_cardinality': self.answer_cardinality,
            'success_criteria_explicit': self.success_criteria_explicit,
            'task_steps': self.task_steps,
            'constraint_count': self.constraint_count,
            'goal_explicitness': self.goal_explicitness,
            'output_format_specified': self.output_format_specified,
            'task_clarity_score': self.task_clarity_score,
            # Pragmatic
            'presupposition_count': self.presupposition_count,
            'implicature_strength': self.implicature_strength,
            'context_dependency': self.context_dependency,
            'speech_act_indirectness': self.speech_act_indirectness,
            'pragmatic_load': self.pragmatic_load,
            # Structural
            'list_markers': self.list_markers,
            'conjunction_count': self.conjunction_count,
            'negation_count': self.negation_count,
            'specificity_score': self.specificity_score,
        }


# Open vs closed question patterns
OPEN_QUESTION_PATTERNS = [
    r'\b(what|how|why|describe|explain|discuss|tell me about)\b',
    r'\b(thoughts|opinion|ideas|suggestions|recommendations)\b',
    r'\b(could you|can you).*(help|assist|suggest)',
    r'\b(any|some).*(way|method|approach|idea)',
    r'\bwhat.*think\b',
    r'\bhow would you\b',
]

CLOSED_QUESTION_PATTERNS = [
    r'^(is|are|do|does|did|will|can|should|would|has|have)\b.*\?$',
    r'\b(true|false|yes|no)\b.*\?',
    r'\b(which one|select|choose|pick)\b',
    r'\bhow many\b',
    r'\bhow much\b',
]

# Cardinality indicators
SINGLE_ANSWER_PATTERNS = [
    r'\bthe\s+(?:one|single|only|best|first|main)\b',
    r'\bexactly\s+(?:one|1)\b',
    r'\bwhich\s+one\b',
    r'\bthe\s+\w+\s+that\b',  # "the file that..."
]

MULTIPLE_ANSWER_PATTERNS = [
    r'\b(?:all|every|each|any|some|many|several|multiple|various)\b',
    r'\blist\b',
    r'\b(?:top|first)\s+\d+\b',
    r'\bas many as\b',
]

# Success criteria patterns
SUCCESS_CRITERIA_PATTERNS = [
    r'\bmust\s+(?:be|have|contain|include|return)\b',
    r'\bshould\s+(?:be|have|contain|include|return)\b',
    r'\brequired?\b',
    r'\bexpected?\b',
    r'\bvalid\b',
    r'\bcorrect\b',
    r'\bexact(?:ly)?\b',
]

# Output format patterns
FORMAT_PATTERNS = [
    r'\bjson\b',
    r'\bxml\b',
    r'\bcsv\b',
    r'\bformat(?:ted)?\b',
    r'\breturn\s+(?:a|the)?\s*\{',
    r'\bschema\b',
    r'\bstructure[d]?\b',
    r'\barray\b',
    r'\blist\b.*\bof\b',
    r'\bdictionary\b',
    r'\bobject\b',
]

# Presupposition triggers
PRESUPPOSITION_TRIGGERS = {
    'change_of_state': ['stop', 'start', 'begin', 'continue', 'resume', 'finish', 'end', 'complete'],
    'factive': ['know', 'realize', 'regret', 'notice', 'remember', 'forget', 'discover'],
    'iterative': ['again', 'another', 'more', 'still', 'anymore', 'yet'],
    'temporal': ['before', 'after', 'since', 'while', 'when', 'until'],
}

# Context dependency patterns
CONTEXT_DEPENDENCY_PATTERNS = [
    r'\b(the|this|that|these|those)\s+\w+\b',  # Definite references
    r'\b(last|previous|current|next|same)\b',
    r'\b(continue|resume|proceed)\b',
    r'\b(as\s+before|like\s+before)\b',
    r'\b(mentioned|above|below|earlier)\b',
    r'\bfrom\s+(the\s+)?(last|previous)\b',
]

# Indirect speech act patterns
INDIRECT_PATTERNS = [
    r'\bit would be nice\b',
    r'\bi was wondering\b',
    r'\bif you could\b',
    r'\bwould you mind\b',
    r'\bi\'d appreciate\b',
    r'\bany chance\b',
    r'\bperhaps you could\b',
    r'\bmight it be possible\b',
]

# Task step indicators
STEP_INDICATORS = [
    r'\b(?:first|then|next|after|finally|lastly)\b',
    r'\b(?:step\s*\d|1\.|2\.|3\.)\b',
    r'\b(?:and\s+then|and\s+also)\b',
    r'\,\s+then\b',
]

# Constraint patterns
CONSTRAINT_PATTERNS = [
    r'\b(?:only|just|exactly|at\s+(?:least|most))\b',
    r'\b(?:must|should|need\s+to|have\s+to)\b',
    r'\b(?:no\s+more\s+than|up\s+to|between)\b',
    r'\b(?:within|by|before|after)\s+\d',
    r'\b(?:limited\s+to|restricted\s+to)\b',
    r'\b(?:excluding?|except|without)\b',
]


def count_patterns(text: str, patterns: List[str]) -> int:
    """Count pattern matches in text."""
    count = 0
    text_lower = text.lower()
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower, re.IGNORECASE))
    return count


def compute_question_type(text: str) -> float:
    """
    Compute question type: 0=closed, 1=open.
    """
    open_count = count_patterns(text, OPEN_QUESTION_PATTERNS)
    closed_count = count_patterns(text, CLOSED_QUESTION_PATTERNS)

    # No questions detected
    if open_count == 0 and closed_count == 0:
        # Check if it's a command (default to semi-open)
        if re.search(r'^[A-Z][a-z]+\s', text):  # Starts with capitalized word
            return 0.6  # Commands are somewhat open
        return 0.5  # Neutral

    total = open_count + closed_count
    return open_count / total if total > 0 else 0.5


def compute_answer_cardinality(text: str) -> float:
    """
    Compute expected answer cardinality: 0=single, 1=multiple/unbounded.
    """
    single_count = count_patterns(text, SINGLE_ANSWER_PATTERNS)
    multiple_count = count_patterns(text, MULTIPLE_ANSWER_PATTERNS)

    if single_count == 0 and multiple_count == 0:
        return 0.5  # Neutral

    total = single_count + multiple_count
    return multiple_count / total if total > 0 else 0.5


def compute_success_criteria(text: str) -> float:
    """
    Compute how explicitly success criteria are defined.
    """
    criteria_count = count_patterns(text, SUCCESS_CRITERIA_PATTERNS)
    word_count = len(text.split())

    # Normalize by word count
    return min(1.0, criteria_count / max(word_count / 10, 1))


def compute_output_format(text: str) -> float:
    """
    Compute whether output format is specified.
    """
    format_count = count_patterns(text, FORMAT_PATTERNS)
    return min(1.0, format_count / 2)  # 2+ format mentions = 1.0


def compute_task_steps(text: str) -> int:
    """
    Estimate number of task steps.
    """
    step_count = count_patterns(text, STEP_INDICATORS)
    # Also count "and" between verbs as potential steps
    and_count = len(re.findall(r'\b(?:and|then)\b', text.lower()))

    return max(1, step_count + and_count // 2)


def compute_constraint_count(text: str) -> int:
    """
    Count explicit constraints in text.
    """
    return count_patterns(text, CONSTRAINT_PATTERNS)


def compute_goal_explicitness(text: str) -> float:
    """
    Compute how explicitly the goal is stated.
    """
    goal_patterns = [
        r'\bi\s+(?:want|need|would\s+like)\b',
        r'\bgoal\s+is\b',
        r'\bobjective\s+is\b',
        r'\bplease\s+(?:find|get|create|search)\b',
        r'\breturn\s+(?:the|a|all)\b',
    ]

    goal_count = count_patterns(text, goal_patterns)
    return min(1.0, goal_count / 2)


def compute_presuppositions(text: str) -> int:
    """
    Count presupposition triggers.
    """
    text_lower = text.lower()
    count = 0

    for trigger_type, triggers in PRESUPPOSITION_TRIGGERS.items():
        for trigger in triggers:
            count += len(re.findall(rf'\b{trigger}\b', text_lower))

    return count


def compute_context_dependency(text: str) -> float:
    """
    Compute how much the prompt depends on external context.
    """
    dep_count = count_patterns(text, CONTEXT_DEPENDENCY_PATTERNS)
    word_count = len(text.split())

    return min(1.0, dep_count / max(word_count / 5, 1))


def compute_speech_act_indirectness(text: str) -> float:
    """
    Compute indirectness of speech acts.
    """
    indirect_count = count_patterns(text, INDIRECT_PATTERNS)
    return min(1.0, indirect_count / 2)


def compute_list_markers(text: str) -> int:
    """
    Count list markers (numbered, bulleted).
    """
    patterns = [
        r'^\s*\d+[\.\)]\s',  # 1. or 1)
        r'^\s*[\-\*\•]\s',   # - or * or bullet
        r'\n\s*\d+[\.\)]\s', # numbered in text
        r'\n\s*[\-\*\•]\s',  # bulleted in text
    ]

    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.MULTILINE))

    return count


def compute_conjunction_count(text: str) -> int:
    """Count coordinating and subordinating conjunctions."""
    conjunctions = ['and', 'or', 'but', 'so', 'yet', 'for', 'nor',
                   'because', 'although', 'while', 'if', 'when', 'unless']
    text_lower = text.lower()

    count = 0
    for conj in conjunctions:
        count += len(re.findall(rf'\b{conj}\b', text_lower))

    return count


def compute_negation_count(text: str) -> int:
    """Count negations."""
    negations = ['not', 'no', 'never', "don't", "won't", "can't", "shouldn't",
                "wouldn't", "couldn't", 'none', 'nothing', 'nowhere', 'neither']
    text_lower = text.lower()

    count = 0
    for neg in negations:
        count += len(re.findall(rf'\b{re.escape(neg)}\b', text_lower))

    return count


def compute_specificity(text: str) -> float:
    """
    Compute specificity based on named entities, numbers, and concrete terms.
    """
    # Count numbers
    numbers = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))

    # Count quoted strings
    quotes = len(re.findall(r'["\'][^"\']+["\']', text))

    # Count capitalized words (potential names/entities)
    words = text.split()
    caps = sum(1 for i, w in enumerate(words)
               if w[0].isupper() and i > 0 and len(w) > 1)

    # Count specific file extensions, paths
    specific = len(re.findall(r'\.\w{2,4}\b', text))  # .json, .csv, etc.
    specific += len(re.findall(r'[/\\]\w+', text))    # paths

    word_count = len(words)
    specificity = (numbers + quotes + caps + specific) / max(word_count / 3, 1)

    return min(1.0, specificity)


def extract_pragmatic_features(text: str) -> PragmaticFeatures:
    """Extract all 17 pragmatic features from text."""
    features = PragmaticFeatures()

    # Task clarity features
    features.question_type = compute_question_type(text)
    features.answer_cardinality = compute_answer_cardinality(text)
    features.success_criteria_explicit = compute_success_criteria(text)
    features.task_steps = compute_task_steps(text)
    features.constraint_count = compute_constraint_count(text)
    features.goal_explicitness = compute_goal_explicitness(text)
    features.output_format_specified = compute_output_format(text)

    # Compute task clarity score (higher = more clear, lower = more ambiguous)
    # Invert so that LOW clarity = HIGH consistency risk
    features.task_clarity_score = (
        (1 - features.question_type) * 0.25 +  # Closed questions are clearer
        (1 - features.answer_cardinality) * 0.20 +  # Single answer is clearer
        features.success_criteria_explicit * 0.15 +
        features.goal_explicitness * 0.15 +
        features.output_format_specified * 0.15 +
        min(features.constraint_count / 3, 1) * 0.10  # More constraints = clearer
    )

    # Pragmatic load features
    features.presupposition_count = compute_presuppositions(text)
    features.context_dependency = compute_context_dependency(text)
    features.speech_act_indirectness = compute_speech_act_indirectness(text)

    # Implicature strength (how much is implied vs stated)
    features.implicature_strength = (
        features.speech_act_indirectness * 0.5 +
        (1 - features.goal_explicitness) * 0.3 +
        features.context_dependency * 0.2
    )

    # Pragmatic load (overall)
    features.pragmatic_load = (
        min(features.presupposition_count / 3, 1) * 0.3 +
        features.context_dependency * 0.3 +
        features.implicature_strength * 0.4
    )

    # Additional structural
    features.list_markers = compute_list_markers(text)
    features.conjunction_count = compute_conjunction_count(text)
    features.negation_count = compute_negation_count(text)
    features.specificity_score = compute_specificity(text)

    return features


# Test
if __name__ == '__main__':
    test_prompts = [
        "What are your thoughts on how to improve this?",  # Open, vague
        "Is the file ready?",  # Closed, simple
        "Find exactly 5 JSON files that contain 'user' and return them as a list.",  # Clear, constrained
        "Continue processing the data from before.",  # Context dependent
        "It would be nice if you could perhaps help me with something.",  # Indirect
    ]

    print("Pragmatic Feature Extraction Test")
    print("=" * 60)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:60]}...")
        features = extract_pragmatic_features(prompt)
        feat_dict = features.to_dict()

        # Show key features
        print("Key features:")
        for k in ['question_type', 'answer_cardinality', 'task_clarity_score',
                  'context_dependency', 'pragmatic_load', 'specificity_score']:
            v = feat_dict[k]
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
