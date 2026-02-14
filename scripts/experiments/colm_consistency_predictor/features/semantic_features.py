#!/usr/bin/env python3
"""
Semantic Features Extraction (19 features)

Categories:
- Ambiguity (7 features): lexical, syntactic, referential, scope, attachment, ellipsis
- Underspecification (6 features): missing args, vague quantifiers/temporals/qualifiers
- Semantic complexity (6 features): entities, relations, logical operators, negation

Dependencies:
- spacy (for NER, dependency parsing)
- nltk (for WordNet polysemy)
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import warnings

# Try to import NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("spacy not available, using fallback methods")

try:
    from nltk.corpus import wordnet as wn
    import nltk
    try:
        wn.synsets('test')
        WORDNET_AVAILABLE = True
    except:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    warnings.warn("nltk/wordnet not available, using fallback methods")


@dataclass
class SemanticFeatures:
    """Container for 19 semantic features."""
    # Ambiguity (7)
    lexical_ambiguity: float = 0.0
    syntactic_ambiguity: float = 0.0
    referential_ambiguity: float = 0.0
    scope_ambiguity: float = 0.0
    attachment_ambiguity: float = 0.0
    ellipsis_count: int = 0
    ambiguity_score: float = 0.0

    # Underspecification (6)
    missing_arguments: float = 0.0
    vague_quantifiers: float = 0.0
    vague_temporals: float = 0.0
    implicit_constraints: float = 0.0
    undefined_terms: float = 0.0
    underspec_score: float = 0.0

    # Semantic complexity (6)
    entity_count: int = 0
    relation_count: int = 0
    logical_operators: int = 0
    coreference_chains: int = 0
    negation_complexity: float = 0.0
    semantic_density: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Ambiguity
            'lexical_ambiguity': self.lexical_ambiguity,
            'syntactic_ambiguity': self.syntactic_ambiguity,
            'referential_ambiguity': self.referential_ambiguity,
            'scope_ambiguity': self.scope_ambiguity,
            'attachment_ambiguity': self.attachment_ambiguity,
            'ellipsis_count': self.ellipsis_count,
            'ambiguity_score': self.ambiguity_score,
            # Underspecification
            'missing_arguments': self.missing_arguments,
            'vague_quantifiers': self.vague_quantifiers,
            'vague_temporals': self.vague_temporals,
            'implicit_constraints': self.implicit_constraints,
            'undefined_terms': self.undefined_terms,
            'underspec_score': self.underspec_score,
            # Semantic complexity
            'entity_count': self.entity_count,
            'relation_count': self.relation_count,
            'logical_operators': self.logical_operators,
            'coreference_chains': self.coreference_chains,
            'negation_complexity': self.negation_complexity,
            'semantic_density': self.semantic_density,
        }


# Vague term dictionaries
VAGUE_QUANTIFIERS = {
    'few', 'some', 'many', 'several', 'various', 'multiple', 'numerous',
    'couple', 'handful', 'bunch', 'lots', 'plenty', 'enough', 'any'
}

VAGUE_TEMPORALS = {
    'soon', 'later', 'recently', 'eventually', 'shortly', 'quickly',
    'immediately', 'promptly', 'asap', 'whenever', 'sometime', 'always', 'never'
}

VAGUE_QUALIFIERS = {
    'good', 'bad', 'nice', 'proper', 'appropriate', 'suitable', 'best',
    'better', 'correct', 'right', 'wrong', 'optimal', 'ideal', 'adequate'
}

# Logical operators
LOGICAL_OPERATORS = {
    'conjunction': ['and', 'also', 'as well as', 'both', 'plus', 'along with'],
    'disjunction': ['or', 'either', 'alternatively', 'otherwise'],
    'negation': ['not', 'no', 'never', "don't", "won't", "can't", "shouldn't", 'without', 'none'],
    'conditional': ['if', 'when', 'unless', 'provided', 'assuming', 'in case'],
    'causal': ['because', 'since', 'therefore', 'so', 'thus', 'hence', 'as a result'],
}

# Pronouns that may indicate referential ambiguity
AMBIGUOUS_PRONOUNS = {'it', 'this', 'that', 'they', 'them', 'these', 'those', 'one'}

# Scope operators
SCOPE_OPERATORS = {
    'quantifiers': ['all', 'every', 'each', 'some', 'any', 'no', 'none', 'most'],
    'negation': ['not', "don't", "won't", 'never', 'no'],
}

# Ellipsis indicators
ELLIPSIS_PATTERNS = [
    r'\band\s+(?:also\s+)?(?:the\s+)?(?:same|similar)\b',
    r'\bas\s+well\b',
    r'\btoo\b$',
    r'\beither\b',
    r',\s*(?:and|or)\s+',  # coordination that might have ellipsis
]

# Missing argument verbs (verbs that typically require objects)
TRANSITIVE_VERBS = {
    'send', 'email', 'search', 'find', 'get', 'create', 'delete', 'update',
    'process', 'analyze', 'call', 'fetch', 'retrieve', 'save', 'load',
    'read', 'write', 'open', 'close', 'start', 'stop', 'run', 'execute'
}


# Initialize spaCy if available
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None and SPACY_AVAILABLE:
        try:
            _nlp = spacy.load('en_core_web_sm')
        except OSError:
            try:
                _nlp = spacy.load('en_core_web_lg')
            except OSError:
                warnings.warn("No spaCy model found. Run: python -m spacy download en_core_web_sm")
                _nlp = False
    return _nlp if _nlp else None


def compute_lexical_ambiguity(text: str) -> float:
    """
    Compute lexical ambiguity based on WordNet polysemy.
    Higher value = more ambiguous words.
    """
    if not WORDNET_AVAILABLE:
        # Fallback: use word length as proxy (longer words tend to be less ambiguous)
        words = text.lower().split()
        if not words:
            return 0.0
        avg_len = sum(len(w) for w in words) / len(words)
        return max(0, 1 - avg_len / 10)  # Normalize: shorter words = more ambiguous

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    if not words:
        return 0.0

    total_synsets = 0
    valid_words = 0

    for word in words:
        synsets = wn.synsets(word)
        if synsets:
            total_synsets += len(synsets)
            valid_words += 1

    if valid_words == 0:
        return 0.0

    avg_synsets = total_synsets / valid_words
    # Normalize: 1 synset = 0 ambiguity, 10+ synsets = 1.0 ambiguity
    return min(1.0, (avg_synsets - 1) / 9)


def compute_referential_ambiguity(text: str) -> float:
    """
    Compute referential ambiguity based on pronoun usage without clear antecedents.
    """
    words = text.lower().split()
    if not words:
        return 0.0

    # Count ambiguous pronouns
    pronoun_count = sum(1 for w in words if w.strip('.,!?') in AMBIGUOUS_PRONOUNS)

    # Check if pronouns appear early (before potential antecedents)
    early_pronouns = 0
    for i, word in enumerate(words[:5]):  # First 5 words
        if word.strip('.,!?') in AMBIGUOUS_PRONOUNS:
            early_pronouns += 1

    # Compute score
    pronoun_ratio = pronoun_count / len(words)
    early_penalty = early_pronouns * 0.2

    return min(1.0, pronoun_ratio * 2 + early_penalty)


def compute_scope_ambiguity(text: str) -> float:
    """
    Compute scope ambiguity from quantifiers and negation combinations.
    """
    text_lower = text.lower()

    quantifier_count = sum(
        1 for q in SCOPE_OPERATORS['quantifiers']
        if re.search(rf'\b{q}\b', text_lower)
    )

    negation_count = sum(
        1 for n in SCOPE_OPERATORS['negation']
        if re.search(rf'\b{n}\b', text_lower)
    )

    # Scope ambiguity increases when quantifiers and negation co-occur
    if quantifier_count > 0 and negation_count > 0:
        return min(1.0, (quantifier_count * negation_count) / 4)

    return 0.0


def compute_vague_terms(text: str) -> Dict[str, float]:
    """Count vague quantifiers, temporals, and qualifiers."""
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    word_count = len(text.split())

    vague_quant = len(words & VAGUE_QUANTIFIERS)
    vague_temp = len(words & VAGUE_TEMPORALS)
    vague_qual = len(words & VAGUE_QUALIFIERS)

    return {
        'vague_quantifiers': min(1.0, vague_quant / max(word_count / 10, 1)),
        'vague_temporals': min(1.0, vague_temp / max(word_count / 10, 1)),
        'implicit_constraints': min(1.0, vague_qual / max(word_count / 10, 1)),
    }


def compute_missing_arguments(text: str) -> float:
    """
    Detect potentially missing arguments for transitive verbs.
    """
    nlp = get_nlp()
    text_lower = text.lower()

    if nlp:
        doc = nlp(text)
        missing_count = 0
        verb_count = 0

        for token in doc:
            if token.pos_ == 'VERB' and token.lemma_ in TRANSITIVE_VERBS:
                verb_count += 1
                # Check if verb has a direct object
                has_obj = any(child.dep_ in ('dobj', 'obj', 'pobj') for child in token.children)
                if not has_obj:
                    missing_count += 1

        if verb_count == 0:
            return 0.0
        return missing_count / verb_count
    else:
        # Fallback: look for transitive verbs followed by punctuation or end
        missing = 0
        for verb in TRANSITIVE_VERBS:
            # Verb at end of sentence or followed by punctuation
            if re.search(rf'\b{verb}\b\s*[.!?,]', text_lower) or re.search(rf'\b{verb}\s*$', text_lower):
                missing += 1
        return min(1.0, missing / 3)


def compute_logical_operators(text: str) -> int:
    """Count logical operators in text."""
    text_lower = text.lower()
    count = 0

    for op_type, operators in LOGICAL_OPERATORS.items():
        for op in operators:
            count += len(re.findall(rf'\b{re.escape(op)}\b', text_lower))

    return count


def compute_negation_complexity(text: str) -> float:
    """
    Compute negation complexity (nested/multiple negations).
    """
    text_lower = text.lower()

    negations = LOGICAL_OPERATORS['negation']
    neg_count = sum(len(re.findall(rf'\b{re.escape(n)}\b', text_lower)) for n in negations)

    # Check for double negation
    double_neg_patterns = [
        r"not\s+\w+\s+no\b",
        r"never\s+\w+\s+not\b",
        r"don't\s+\w+\s+nothing\b",
    ]

    double_neg = sum(1 for p in double_neg_patterns if re.search(p, text_lower))

    return min(1.0, (neg_count / 5) + (double_neg * 0.3))


def compute_entity_count(text: str) -> int:
    """Count named entities in text."""
    nlp = get_nlp()

    if nlp:
        doc = nlp(text)
        return len(doc.ents)
    else:
        # Fallback: count capitalized words (excluding sentence starts)
        words = text.split()
        caps = sum(1 for i, w in enumerate(words)
                   if w[0].isupper() and i > 0 and words[i-1][-1] not in '.!?')
        return caps


def compute_relation_count(text: str) -> int:
    """Count relations (verbs connecting entities)."""
    nlp = get_nlp()

    if nlp:
        doc = nlp(text)
        # Count verbs with both subject and object
        relations = 0
        for token in doc:
            if token.pos_ == 'VERB':
                has_subj = any(child.dep_ in ('nsubj', 'nsubjpass') for child in token.children)
                has_obj = any(child.dep_ in ('dobj', 'obj', 'pobj', 'attr') for child in token.children)
                if has_subj or has_obj:
                    relations += 1
        return relations
    else:
        # Fallback: count verbs
        verb_patterns = r'\b(is|are|was|were|has|have|had|do|does|did|get|gets|make|makes)\b'
        return len(re.findall(verb_patterns, text.lower()))


def compute_ellipsis(text: str) -> int:
    """Detect potential ellipsis sites."""
    count = 0
    for pattern in ELLIPSIS_PATTERNS:
        count += len(re.findall(pattern, text.lower()))
    return count


def extract_semantic_features(text: str) -> SemanticFeatures:
    """Extract all 19 semantic features from text."""
    features = SemanticFeatures()

    word_count = len(text.split())

    # Ambiguity features
    features.lexical_ambiguity = compute_lexical_ambiguity(text)
    features.referential_ambiguity = compute_referential_ambiguity(text)
    features.scope_ambiguity = compute_scope_ambiguity(text)
    features.ellipsis_count = compute_ellipsis(text)

    # Syntactic and attachment ambiguity (simplified without dual parser)
    # Use sentence complexity as proxy
    sentences = re.split(r'[.!?]', text)
    avg_sent_len = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
    features.syntactic_ambiguity = min(1.0, avg_sent_len / 30)
    features.attachment_ambiguity = min(1.0, text.count(',') / max(word_count / 5, 1))

    # Compute ambiguity score (weighted average)
    features.ambiguity_score = (
        features.lexical_ambiguity * 0.2 +
        features.referential_ambiguity * 0.3 +
        features.scope_ambiguity * 0.2 +
        features.syntactic_ambiguity * 0.15 +
        features.attachment_ambiguity * 0.15
    )

    # Underspecification features
    vague = compute_vague_terms(text)
    features.vague_quantifiers = vague['vague_quantifiers']
    features.vague_temporals = vague['vague_temporals']
    features.implicit_constraints = vague['implicit_constraints']
    features.missing_arguments = compute_missing_arguments(text)
    features.undefined_terms = 0.0  # Would need domain knowledge

    # Compute underspec score
    features.underspec_score = (
        features.missing_arguments * 0.3 +
        features.vague_quantifiers * 0.25 +
        features.vague_temporals * 0.2 +
        features.implicit_constraints * 0.25
    )

    # Semantic complexity features
    features.entity_count = compute_entity_count(text)
    features.relation_count = compute_relation_count(text)
    features.logical_operators = compute_logical_operators(text)
    features.negation_complexity = compute_negation_complexity(text)
    features.coreference_chains = 0  # Would need coreference resolution

    # Semantic density
    if word_count > 0:
        features.semantic_density = (features.entity_count + features.relation_count) / word_count
    else:
        features.semantic_density = 0.0

    return features


# Test
if __name__ == '__main__':
    test_prompts = [
        "Send email.",  # Missing arguments
        "Process it and save the results somewhere.",  # Referential ambiguity
        "Don't notify all users about any changes.",  # Scope ambiguity
        "Find a few good files from the recent data.",  # Vague terms
        "Search for John's documents in the reports folder if they exist and the server is running.",  # Complex
    ]

    print("Semantic Feature Extraction Test")
    print("=" * 60)
    print(f"spaCy available: {SPACY_AVAILABLE}")
    print(f"WordNet available: {WORDNET_AVAILABLE}")

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:60]}...")
        features = extract_semantic_features(prompt)
        feat_dict = features.to_dict()

        # Show non-zero features
        print("Non-zero features:")
        for k, v in feat_dict.items():
            if v != 0:
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
