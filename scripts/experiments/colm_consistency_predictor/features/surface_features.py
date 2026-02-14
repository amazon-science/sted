#!/usr/bin/env python3
"""
Surface Linguistic Features Extraction (25 features)

Categories:
- Modal verbs (6 features)
- Hedging (5 features)
- Politeness (6 features)
- Speech acts (4 features)
- Structural (8 features)

Based on:
- Kratzer's modal semantics
- Palmer's modal typology
- Brown & Levinson's politeness theory
- Searle's speech act taxonomy
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class SurfaceFeatures:
    """Container for 25 surface linguistic features."""
    # Modal verbs (6)
    modal_deontic_strong: float = 0.0
    modal_deontic_weak: float = 0.0
    modal_epistemic_strong: float = 0.0
    modal_epistemic_weak: float = 0.0
    modal_dynamic: float = 0.0
    modal_count: int = 0

    # Hedging (5)
    hedge_epistemic: float = 0.0
    hedge_plausibility: float = 0.0
    hedge_conditional: float = 0.0
    hedge_approximator: float = 0.0
    hedge_count: int = 0

    # Politeness (6)
    polite_bald: float = 0.0
    polite_positive: float = 0.0
    polite_negative: float = 0.0
    polite_indirect: float = 0.0
    polite_impersonal: float = 0.0
    politeness_score: float = 0.5

    # Speech acts (4)
    speech_directive: float = 0.0
    speech_interrogative: float = 0.0
    speech_declarative: float = 0.0
    speech_indirect: float = 0.0

    # Structural (4 - reduced from 8, others moved to pragmatic)
    prompt_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    question_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            # Modal verbs
            'modal_deontic_strong': self.modal_deontic_strong,
            'modal_deontic_weak': self.modal_deontic_weak,
            'modal_epistemic_strong': self.modal_epistemic_strong,
            'modal_epistemic_weak': self.modal_epistemic_weak,
            'modal_dynamic': self.modal_dynamic,
            'modal_count': self.modal_count,
            # Hedging
            'hedge_epistemic': self.hedge_epistemic,
            'hedge_plausibility': self.hedge_plausibility,
            'hedge_conditional': self.hedge_conditional,
            'hedge_approximator': self.hedge_approximator,
            'hedge_count': self.hedge_count,
            # Politeness
            'polite_bald': self.polite_bald,
            'polite_positive': self.polite_positive,
            'polite_negative': self.polite_negative,
            'polite_indirect': self.polite_indirect,
            'polite_impersonal': self.polite_impersonal,
            'politeness_score': self.politeness_score,
            # Speech acts
            'speech_directive': self.speech_directive,
            'speech_interrogative': self.speech_interrogative,
            'speech_declarative': self.speech_declarative,
            'speech_indirect': self.speech_indirect,
            # Structural
            'prompt_length': self.prompt_length,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'question_count': self.question_count,
        }


# Modal verb patterns (based on Kratzer and Palmer)
MODAL_PATTERNS = {
    'deontic_strong': [
        r'\bmust\b', r'\bneed to\b', r'\bhave to\b', r'\brequired\b',
        r'\bshall\b', r'\bmandatory\b', r'\bnecessary\b'
    ],
    'deontic_weak': [
        r'\bshould\b', r'\bought to\b', r'\bsupposed to\b',
        r'\brecommended\b', r'\badvisable\b'
    ],
    'epistemic_strong': [
        r'\bwill\b', r'\bwould\b', r'\bdefinitely\b', r'\bcertainly\b'
    ],
    'epistemic_weak': [
        r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bpossibly\b',
        r'\bperhaps\b', r'\bprobably\b'
    ],
    'dynamic': [
        r'\bcan\b', r'\bable to\b', r'\bcapable\b', r'\bmanage to\b'
    ]
}

# Hedging patterns
HEDGE_PATTERNS = {
    'epistemic': [
        r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b', r'\bpotentially\b',
        r'\bi think\b', r'\bi believe\b', r'\bi guess\b'
    ],
    'plausibility': [
        r'\bseems?\b', r'\bappears?\b', r'\blooks? like\b',
        r'\bapparently\b', r'\bpresumably\b'
    ],
    'conditional': [
        r'\bif\b', r'\bunless\b', r'\bprovided\b', r'\bassuming\b',
        r'\bin case\b', r'\bwhen\b.*\bthen\b'
    ],
    'approximator': [
        r'\babout\b', r'\baround\b', r'\broughly\b', r'\bapproximately\b',
        r'\bmore or less\b', r'\bkind of\b', r'\bsort of\b'
    ]
}

# Politeness patterns (based on Brown & Levinson)
POLITENESS_PATTERNS = {
    'bald': [
        # Imperative without softener (detected by structure, not pattern)
    ],
    'positive': [
        r'\bplease\b', r'\bthanks?\b', r'\bthank you\b',
        r'\bappreciate\b', r'\bgrateful\b', r'\bkindly\b'
    ],
    'negative': [
        r'\bwould you mind\b', r'\bcould you possibly\b',
        r'\bif you don\'t mind\b', r'\bif it\'s not too much\b',
        r'\bsorry to\b', r'\bi hate to\b'
    ],
    'indirect': [
        r'\bi was wondering\b', r'\bi\'d like\b', r'\bit would be great\b',
        r'\bit would be helpful\b', r'\bwould it be possible\b'
    ],
    'impersonal': [
        r'\bone might\b', r'\bit is suggested\b', r'\bit would seem\b',
        r'\bit appears that\b', r'\bit is recommended\b'
    ]
}

# Speech act patterns (based on Searle)
SPEECH_ACT_PATTERNS = {
    'directive': [
        # Imperatives: verb at start of sentence (detected structurally)
        r'^[A-Z][a-z]+\s', r'\bdo\b', r'\bdon\'t\b', r'\bget\b',
        r'\bmake\b', r'\bfind\b', r'\bcreate\b', r'\blist\b'
    ],
    'interrogative': [
        r'\?$', r'^(what|who|where|when|why|how|which|is|are|do|does|did|can|could|will|would)\b'
    ],
    'declarative': [
        r'\bi want\b', r'\bi need\b', r'\bi\'m looking for\b',
        r'\bi\'m trying to\b', r'\bmy goal is\b'
    ],
    'indirect': [
        r'\bit would be nice\b', r'\bif you could\b',
        r'\bi\'d appreciate\b', r'\bany chance\b'
    ]
}


def count_pattern_matches(text: str, patterns: List[str]) -> int:
    """Count how many patterns match in text."""
    text_lower = text.lower()
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower, re.IGNORECASE))
    return count


def extract_modal_features(text: str) -> Dict[str, Any]:
    """Extract modal verb features."""
    text_lower = text.lower()
    word_count = len(text.split())

    features = {}
    total_modals = 0

    for modal_type, patterns in MODAL_PATTERNS.items():
        count = count_pattern_matches(text, patterns)
        # Normalize by word count
        features[f'modal_{modal_type}'] = count / max(word_count, 1)
        total_modals += count

    features['modal_count'] = total_modals
    return features


def extract_hedge_features(text: str) -> Dict[str, Any]:
    """Extract hedging features."""
    word_count = len(text.split())

    features = {}
    total_hedges = 0

    for hedge_type, patterns in HEDGE_PATTERNS.items():
        count = count_pattern_matches(text, patterns)
        features[f'hedge_{hedge_type}'] = count / max(word_count, 1)
        total_hedges += count

    features['hedge_count'] = total_hedges
    return features


def extract_politeness_features(text: str) -> Dict[str, Any]:
    """Extract politeness features."""
    text_lower = text.lower()

    features = {
        'polite_bald': 0.0,
        'polite_positive': 0.0,
        'polite_negative': 0.0,
        'polite_indirect': 0.0,
        'polite_impersonal': 0.0,
    }

    # Check for positive politeness markers
    positive_count = count_pattern_matches(text, POLITENESS_PATTERNS['positive'])
    features['polite_positive'] = min(positive_count / 3, 1.0)  # Normalize to [0,1]

    # Check for negative politeness
    negative_count = count_pattern_matches(text, POLITENESS_PATTERNS['negative'])
    features['polite_negative'] = min(negative_count / 2, 1.0)

    # Check for indirect
    indirect_count = count_pattern_matches(text, POLITENESS_PATTERNS['indirect'])
    features['polite_indirect'] = min(indirect_count / 2, 1.0)

    # Check for impersonal
    impersonal_count = count_pattern_matches(text, POLITENESS_PATTERNS['impersonal'])
    features['polite_impersonal'] = min(impersonal_count / 2, 1.0)

    # Check for bald on-record (imperative without softeners)
    sentences = re.split(r'[.!?]', text)
    bald_count = 0
    for sent in sentences:
        sent = sent.strip()
        if sent and not any(re.search(p, sent, re.IGNORECASE) for p in
                          POLITENESS_PATTERNS['positive'] +
                          POLITENESS_PATTERNS['negative'] +
                          POLITENESS_PATTERNS['indirect']):
            # Check if starts with verb (imperative)
            if re.match(r'^[A-Z][a-z]+\s', sent) and not re.match(r'^(I|You|We|They|He|She|It|The|A|An)\s', sent):
                bald_count += 1

    features['polite_bald'] = min(bald_count / max(len(sentences), 1), 1.0)

    # Compute overall politeness score (0 = bald, 1 = very polite)
    politeness_score = 0.5  # neutral baseline
    politeness_score += features['polite_positive'] * 0.2
    politeness_score += features['polite_negative'] * 0.3
    politeness_score += features['polite_indirect'] * 0.2
    politeness_score += features['polite_impersonal'] * 0.1
    politeness_score -= features['polite_bald'] * 0.3

    features['politeness_score'] = max(0, min(1, politeness_score))

    return features


def extract_speech_act_features(text: str) -> Dict[str, Any]:
    """Extract speech act features."""
    features = {
        'speech_directive': 0.0,
        'speech_interrogative': 0.0,
        'speech_declarative': 0.0,
        'speech_indirect': 0.0,
    }

    sentences = re.split(r'[.!?]', text)
    num_sentences = max(len([s for s in sentences if s.strip()]), 1)

    # Count question marks
    question_count = text.count('?')
    features['speech_interrogative'] = min(question_count / num_sentences, 1.0)

    # Count directive patterns
    directive_count = count_pattern_matches(text, SPEECH_ACT_PATTERNS['directive'])
    features['speech_directive'] = min(directive_count / (num_sentences * 2), 1.0)

    # Count declarative patterns
    declarative_count = count_pattern_matches(text, SPEECH_ACT_PATTERNS['declarative'])
    features['speech_declarative'] = min(declarative_count / num_sentences, 1.0)

    # Count indirect patterns
    indirect_count = count_pattern_matches(text, SPEECH_ACT_PATTERNS['indirect'])
    features['speech_indirect'] = min(indirect_count / num_sentences, 1.0)

    return features


def extract_structural_features(text: str) -> Dict[str, Any]:
    """Extract structural features."""
    features = {
        'prompt_length': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'question_count': text.count('?'),
    }
    return features


def extract_surface_features(text: str) -> SurfaceFeatures:
    """Extract all 25 surface linguistic features from text."""
    features = SurfaceFeatures()

    # Extract each category
    modal = extract_modal_features(text)
    hedge = extract_hedge_features(text)
    polite = extract_politeness_features(text)
    speech = extract_speech_act_features(text)
    struct = extract_structural_features(text)

    # Populate dataclass
    # Modal
    features.modal_deontic_strong = modal.get('modal_deontic_strong', 0)
    features.modal_deontic_weak = modal.get('modal_deontic_weak', 0)
    features.modal_epistemic_strong = modal.get('modal_epistemic_strong', 0)
    features.modal_epistemic_weak = modal.get('modal_epistemic_weak', 0)
    features.modal_dynamic = modal.get('modal_dynamic', 0)
    features.modal_count = modal.get('modal_count', 0)

    # Hedge
    features.hedge_epistemic = hedge.get('hedge_epistemic', 0)
    features.hedge_plausibility = hedge.get('hedge_plausibility', 0)
    features.hedge_conditional = hedge.get('hedge_conditional', 0)
    features.hedge_approximator = hedge.get('hedge_approximator', 0)
    features.hedge_count = hedge.get('hedge_count', 0)

    # Politeness
    features.polite_bald = polite.get('polite_bald', 0)
    features.polite_positive = polite.get('polite_positive', 0)
    features.polite_negative = polite.get('polite_negative', 0)
    features.polite_indirect = polite.get('polite_indirect', 0)
    features.polite_impersonal = polite.get('polite_impersonal', 0)
    features.politeness_score = polite.get('politeness_score', 0.5)

    # Speech acts
    features.speech_directive = speech.get('speech_directive', 0)
    features.speech_interrogative = speech.get('speech_interrogative', 0)
    features.speech_declarative = speech.get('speech_declarative', 0)
    features.speech_indirect = speech.get('speech_indirect', 0)

    # Structural
    features.prompt_length = struct.get('prompt_length', 0)
    features.word_count = struct.get('word_count', 0)
    features.sentence_count = struct.get('sentence_count', 0)
    features.question_count = struct.get('question_count', 0)

    return features


# Test
if __name__ == '__main__':
    test_prompts = [
        "You must find all users and return their emails immediately.",
        "Could you please help me search for files? I would appreciate it.",
        "If possible, maybe try to get the data when you have a chance.",
        "I'm looking for a way to process this. Would it be possible to help?",
        "List the top 5 results.",
    ]

    print("Surface Linguistic Feature Extraction Test")
    print("=" * 60)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:60]}...")
        features = extract_surface_features(prompt)
        feat_dict = features.to_dict()

        # Show non-zero features
        print("Non-zero features:")
        for k, v in feat_dict.items():
            if v != 0 and v != 0.5:  # Skip zeros and neutral politeness
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
