"""
ACL Paper: Linguistic Variation Generator

Generates controlled linguistic variations from Toucan prompts to study
the effect of pragmatic and semantic features on structured output consistency.

Usage:
    python generate_variations.py --input data/toucan/toucan_tool_calls_1006.json \
                                  --output data/acl_variations/variations.json \
                                  --num_base 100
"""

import json
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random

# Try to import NLP libraries
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    HAS_SPACY = False
    print("Warning: spaCy not available. Using regex-based extraction.")


@dataclass
class LinguisticVariation:
    """Represents a single linguistic variation of a base prompt."""
    base_id: str
    variation_id: str
    variation_type: str  # speech_act, politeness, modal, quantifier, syntax
    variation_subtype: str  # e.g., "directive_imperative", "must", "bald"
    original_prompt: str
    varied_prompt: str
    target_tools: str
    tools: List[Dict]
    tool_calls: List[Dict]
    linguistic_features: Dict[str, str]


class ActionExtractor:
    """Extracts the core action from a prompt for transformation."""

    # Common action verbs in tool-calling contexts
    ACTION_VERBS = [
        'get', 'find', 'search', 'retrieve', 'fetch', 'look up', 'lookup',
        'calculate', 'compute', 'convert', 'check', 'verify', 'validate',
        'create', 'generate', 'make', 'build', 'send', 'post', 'submit',
        'update', 'modify', 'change', 'delete', 'remove', 'list', 'show',
        'analyze', 'compare', 'measure', 'count', 'sum', 'average'
    ]

    def extract_action(self, prompt: str, tool_name: str) -> Tuple[str, str, str]:
        """
        Extract action verb, object, and context from prompt.
        Returns: (verb, object, full_action)
        """
        # Try to infer from tool name first
        tool_action = self._action_from_tool_name(tool_name)

        # Simple heuristic: find action verb in prompt
        prompt_lower = prompt.lower()
        for verb in self.ACTION_VERBS:
            if verb in prompt_lower:
                # Extract the phrase after the verb
                pattern = rf'{verb}\s+(?:the\s+)?(.+?)(?:\.|,|$|\?)'
                match = re.search(pattern, prompt_lower)
                if match:
                    obj = match.group(1).strip()
                    # Clean up the object
                    obj = re.sub(r'\s+(?:for|to|from|in|on|at)\s+.*$', '', obj)
                    return verb, obj[:50], f"{verb} {obj[:50]}"

        # Fallback to tool-based action
        if tool_action:
            return tool_action, "the requested information", tool_action

        return "perform the action", "as requested", "perform the requested action"

    def _action_from_tool_name(self, tool_name: str) -> Optional[str]:
        """Infer action from tool name."""
        # Remove prefixes like "lyrical-mcp-"
        clean_name = re.sub(r'^[a-z]+-[a-z]+-', '', tool_name)
        # Convert snake_case to spaces
        clean_name = clean_name.replace('_', ' ').replace('-', ' ')
        return clean_name if clean_name else None


class SpeechActGenerator:
    """Generates speech act variations."""

    TEMPLATES = {
        'directive_imperative': "{verb} {object}.",
        'directive_imperative_full': "{Verb} {object} {context}.",
        'directive_infinitive': "Retrieve {object}.",
        'request_please': "Please {verb} {object}.",
        'request_kindly': "Kindly {verb} {object}.",
        'interrogative_what': "What is {object}?",
        'interrogative_can': "Can you {verb} {object}?",
        'interrogative_could': "Could you {verb} {object}?",
        'declarative_need': "I need {object}.",
        'declarative_want': "I want to know {object}.",
        'hint_wondering': "I'm wondering about {object}...",
        'hint_interested': "I'm interested in {object}.",
    }

    def generate(self, verb: str, obj: str, context: str = "") -> List[Dict]:
        """Generate all speech act variations."""
        variations = []
        for subtype, template in self.TEMPLATES.items():
            try:
                varied = template.format(
                    verb=verb,
                    Verb=verb.capitalize(),
                    object=obj,
                    context=context
                )
                variations.append({
                    'subtype': subtype,
                    'prompt': varied,
                    'features': {
                        'speech_act_type': subtype.split('_')[0],
                        'illocutionary_force': self._get_force(subtype)
                    }
                })
            except KeyError:
                continue
        return variations

    def _get_force(self, subtype: str) -> str:
        """Get illocutionary force category."""
        if subtype.startswith('directive'):
            return 'directive'
        elif subtype.startswith('request'):
            return 'request'
        elif subtype.startswith('interrogative'):
            return 'interrogative'
        elif subtype.startswith('declarative'):
            return 'declarative'
        elif subtype.startswith('hint'):
            return 'indirect'
        return 'unknown'


class ModalVerbGenerator:
    """Generates modal verb variations with different strengths."""

    MODALS = {
        # Strong deontic
        'must': {'strength': 'strong', 'type': 'deontic', 'template': "You must {action}."},
        'have_to': {'strength': 'strong', 'type': 'deontic', 'template': "You have to {action}."},

        # Medium deontic
        'should': {'strength': 'medium', 'type': 'deontic', 'template': "You should {action}."},
        'ought_to': {'strength': 'medium', 'type': 'deontic', 'template': "You ought to {action}."},
        'need_to': {'strength': 'medium', 'type': 'deontic', 'template': "You need to {action}."},

        # Weak dynamic/epistemic
        'can': {'strength': 'weak', 'type': 'dynamic', 'template': "You can {action}."},
        'could': {'strength': 'weak', 'type': 'epistemic', 'template': "You could {action}."},
        'may': {'strength': 'weak', 'type': 'epistemic', 'template': "You may {action}."},
        'might': {'strength': 'weak', 'type': 'epistemic', 'template': "You might want to {action}."},

        # No modal (baseline)
        'no_modal': {'strength': 'none', 'type': 'none', 'template': "{Action}."},
    }

    def generate(self, action: str) -> List[Dict]:
        """Generate all modal verb variations."""
        variations = []
        for modal, info in self.MODALS.items():
            try:
                if modal == 'no_modal':
                    varied = info['template'].format(Action=action.capitalize())
                else:
                    varied = info['template'].format(action=action)

                variations.append({
                    'subtype': modal,
                    'prompt': varied,
                    'features': {
                        'modal': modal,
                        'modal_strength': info['strength'],
                        'modal_type': info['type']
                    }
                })
            except KeyError:
                continue
        return variations


class PolitenessGenerator:
    """Generates politeness strategy variations (Brown & Levinson)."""

    STRATEGIES = {
        # Bald on-record
        'bald': {
            'strategy': 'bald_on_record',
            'template': "{action}."
        },
        'bald_imperative': {
            'strategy': 'bald_on_record',
            'template': "{Action} now."
        },

        # Positive politeness
        'positive_hey': {
            'strategy': 'positive_politeness',
            'template': "Hey, could you {action}?"
        },
        'positive_appreciate': {
            'strategy': 'positive_politeness',
            'template': "I'd really appreciate it if you could {action}."
        },
        'positive_love': {
            'strategy': 'positive_politeness',
            'template': "I'd love it if you could {action}."
        },

        # Negative politeness
        'negative_mind': {
            'strategy': 'negative_politeness',
            'template': "Would you mind {gerund}?"
        },
        'negative_possible': {
            'strategy': 'negative_politeness',
            'template': "If possible, could you {action}?"
        },
        'negative_trouble': {
            'strategy': 'negative_politeness',
            'template': "If it's not too much trouble, please {action}."
        },
        'negative_wondering': {
            'strategy': 'negative_politeness',
            'template': "I was wondering if you could {action}."
        },

        # Off-record
        'offrecord_thinking': {
            'strategy': 'off_record',
            'template': "I'm thinking about {topic}..."
        },
        'offrecord_suppose': {
            'strategy': 'off_record',
            'template': "I suppose {topic} would be useful..."
        },
    }

    def generate(self, action: str, topic: str) -> List[Dict]:
        """Generate all politeness variations."""
        variations = []

        # Create gerund form
        gerund = self._to_gerund(action)

        for subtype, info in self.STRATEGIES.items():
            try:
                varied = info['template'].format(
                    action=action,
                    Action=action.capitalize(),
                    gerund=gerund,
                    topic=topic
                )

                variations.append({
                    'subtype': subtype,
                    'prompt': varied,
                    'features': {
                        'politeness_strategy': info['strategy'],
                        'face_threat': self._get_face_threat(info['strategy'])
                    }
                })
            except KeyError:
                continue
        return variations

    def _to_gerund(self, action: str) -> str:
        """Convert action to gerund form."""
        words = action.split()
        if words:
            verb = words[0]
            if verb.endswith('e'):
                verb = verb[:-1] + 'ing'
            elif verb.endswith('ie'):
                verb = verb[:-2] + 'ying'
            elif len(verb) > 2 and verb[-1] not in 'aeiou' and verb[-2] in 'aeiou':
                verb = verb + verb[-1] + 'ing'
            else:
                verb = verb + 'ing'
            return ' '.join([verb] + words[1:])
        return action + 'ing'

    def _get_face_threat(self, strategy: str) -> str:
        """Get face threat level."""
        levels = {
            'bald_on_record': 'high',
            'positive_politeness': 'medium',
            'negative_politeness': 'low',
            'off_record': 'minimal'
        }
        return levels.get(strategy, 'unknown')


class SyntaxGenerator:
    """Generates syntactic complexity variations."""

    STRUCTURES = {
        # Simple structures
        'simple_main': {
            'complexity': 'simple',
            'template': "{sentence}"
        },

        # Embedded clauses
        'embedded_think': {
            'complexity': 'embedded',
            'template': "I think {sentence_lower}"
        },
        'embedded_believe': {
            'complexity': 'embedded',
            'template': "I believe {sentence_lower}"
        },
        'embedded_expect': {
            'complexity': 'embedded',
            'template': "I expect that {sentence_lower}"
        },

        # Subordinate clauses
        'subordinate_if': {
            'complexity': 'subordinate',
            'template': "If needed, {sentence_lower}"
        },
        'subordinate_when': {
            'complexity': 'subordinate',
            'template': "When appropriate, {sentence_lower}"
        },
        'subordinate_after': {
            'complexity': 'subordinate',
            'template': "After considering the request, {sentence_lower}"
        },

        # Relative clauses
        'relative_action': {
            'complexity': 'relative',
            'template': "The action you should take is to {action}."
        },

        # Coordinated
        'coordinated_and': {
            'complexity': 'coordinated',
            'template': "{sentence} and return the results."
        },
    }

    def generate(self, base_sentence: str, action: str) -> List[Dict]:
        """Generate all syntactic variations."""
        variations = []

        # Create lowercase version (for embedding)
        sentence_lower = base_sentence[0].lower() + base_sentence[1:] if base_sentence else base_sentence
        # Remove trailing period for embedding
        sentence_lower = sentence_lower.rstrip('.')

        for subtype, info in self.STRUCTURES.items():
            try:
                varied = info['template'].format(
                    sentence=base_sentence,
                    sentence_lower=sentence_lower,
                    action=action
                )
                # Ensure proper capitalization and punctuation
                varied = varied[0].upper() + varied[1:]
                if not varied.endswith(('.', '?', '!')):
                    varied += '.'

                variations.append({
                    'subtype': subtype,
                    'prompt': varied,
                    'features': {
                        'syntactic_complexity': info['complexity'],
                        'clause_type': subtype.split('_')[0]
                    }
                })
            except KeyError:
                continue
        return variations


class QuantificationGenerator:
    """Generates quantification variations for multi-tool prompts."""

    PATTERNS = {
        'universal_enumerated': {
            'quantifier': 'universal',
            'specificity': 'enumerated',
            'template': "{action} for all of these: {items}."
        },
        'universal_open': {
            'quantifier': 'universal',
            'specificity': 'open',
            'template': "{action} for all relevant items."
        },
        'distributive_each': {
            'quantifier': 'distributive',
            'specificity': 'enumerated',
            'template': "{action} for each item: {items}."
        },
        'existential_some': {
            'quantifier': 'existential',
            'specificity': 'open',
            'template': "{action} for some of the items."
        },
        'existential_one': {
            'quantifier': 'existential',
            'specificity': 'open',
            'template': "{action} for at least one item."
        },
        'partitive': {
            'quantifier': 'partitive',
            'specificity': 'open',
            'template': "{action} for part of the list."
        },
    }

    def generate(self, action: str, items: List[str] = None) -> List[Dict]:
        """Generate quantification variations."""
        variations = []
        items_str = ", ".join(items) if items else "item1, item2, item3"

        for subtype, info in self.PATTERNS.items():
            try:
                varied = info['template'].format(
                    action=action.capitalize(),
                    items=items_str
                )

                variations.append({
                    'subtype': subtype,
                    'prompt': varied,
                    'features': {
                        'quantifier_type': info['quantifier'],
                        'specificity': info['specificity']
                    }
                })
            except KeyError:
                continue
        return variations


class LinguisticVariationPipeline:
    """Main pipeline for generating all linguistic variations."""

    def __init__(self):
        self.extractor = ActionExtractor()
        self.speech_act_gen = SpeechActGenerator()
        self.modal_gen = ModalVerbGenerator()
        self.politeness_gen = PolitenessGenerator()
        self.syntax_gen = SyntaxGenerator()
        self.quantification_gen = QuantificationGenerator()

    def filter_suitable_prompts(self, data: List[Dict], max_prompts: int = 100) -> List[Dict]:
        """Filter prompts suitable for variation (English, single tool, moderate length)."""
        suitable = []

        for item in data:
            question = item.get('question', '')
            num_tools = item.get('num_tool_calls', 0)

            # Filter criteria
            if not question:
                continue
            # Prefer English (simple ASCII check)
            if not question.isascii():
                continue
            # Prefer single tool calls for cleaner experiments
            if num_tools != 1:
                continue
            # Moderate length (not too short or too long)
            if len(question) < 30 or len(question) > 500:
                continue

            suitable.append(item)

        # Shuffle and select
        random.shuffle(suitable)
        return suitable[:max_prompts]

    def generate_variations_for_prompt(self, item: Dict) -> List[LinguisticVariation]:
        """Generate all linguistic variations for a single prompt."""
        variations = []

        base_id = item['id']
        original = item['question']
        target_tools = item.get('target_tools', '')
        tools = item.get('tools', [])
        tool_calls = item.get('tool_calls', [])

        # Extract primary tool name
        tool_name = tool_calls[0]['name'] if tool_calls else 'unknown'

        # Extract action components
        verb, obj, full_action = self.extractor.extract_action(original, tool_name)
        topic = obj  # For politeness generator

        # Generate Speech Act variations
        for var in self.speech_act_gen.generate(verb, obj):
            variations.append(LinguisticVariation(
                base_id=base_id,
                variation_id=f"{base_id}_SA_{var['subtype']}",
                variation_type='speech_act',
                variation_subtype=var['subtype'],
                original_prompt=original,
                varied_prompt=var['prompt'],
                target_tools=target_tools,
                tools=tools,
                tool_calls=tool_calls,
                linguistic_features=var['features']
            ))

        # Generate Modal Verb variations
        for var in self.modal_gen.generate(full_action):
            variations.append(LinguisticVariation(
                base_id=base_id,
                variation_id=f"{base_id}_MV_{var['subtype']}",
                variation_type='modal',
                variation_subtype=var['subtype'],
                original_prompt=original,
                varied_prompt=var['prompt'],
                target_tools=target_tools,
                tools=tools,
                tool_calls=tool_calls,
                linguistic_features=var['features']
            ))

        # Generate Politeness variations
        for var in self.politeness_gen.generate(full_action, topic):
            variations.append(LinguisticVariation(
                base_id=base_id,
                variation_id=f"{base_id}_PL_{var['subtype']}",
                variation_type='politeness',
                variation_subtype=var['subtype'],
                original_prompt=original,
                varied_prompt=var['prompt'],
                target_tools=target_tools,
                tools=tools,
                tool_calls=tool_calls,
                linguistic_features=var['features']
            ))

        # Generate Syntax variations (using modal "should" as base)
        base_sentence = f"You should {full_action}."
        for var in self.syntax_gen.generate(base_sentence, full_action):
            variations.append(LinguisticVariation(
                base_id=base_id,
                variation_id=f"{base_id}_SX_{var['subtype']}",
                variation_type='syntax',
                variation_subtype=var['subtype'],
                original_prompt=original,
                varied_prompt=var['prompt'],
                target_tools=target_tools,
                tools=tools,
                tool_calls=tool_calls,
                linguistic_features=var['features']
            ))

        return variations

    def generate_all(self, data: List[Dict], max_prompts: int = 100) -> Tuple[List[LinguisticVariation], Dict]:
        """Generate variations for all suitable prompts."""
        # Filter suitable prompts
        suitable = self.filter_suitable_prompts(data, max_prompts)
        print(f"Selected {len(suitable)} suitable prompts from {len(data)} total")

        all_variations = []
        stats = defaultdict(int)

        for item in suitable:
            variations = self.generate_variations_for_prompt(item)
            all_variations.extend(variations)

            # Track stats
            for var in variations:
                stats[var.variation_type] += 1

        return all_variations, dict(stats)


def main():
    parser = argparse.ArgumentParser(description='Generate linguistic variations for ACL paper')
    parser.add_argument('--input', type=str,
                        default='data/toucan/toucan_tool_calls_1006.json',
                        help='Input Toucan dataset')
    parser.add_argument('--output', type=str,
                        default='data/acl_variations/linguistic_variations.json',
                        help='Output variations file')
    parser.add_argument('--num_base', type=int, default=100,
                        help='Number of base prompts to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    with open(input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} prompts from {input_path}")

    # Generate variations
    pipeline = LinguisticVariationPipeline()
    variations, stats = pipeline.generate_all(data, args.num_base)

    # Print stats
    print(f"\nGeneration Statistics:")
    print(f"  Total variations: {len(variations)}")
    for var_type, count in sorted(stats.items()):
        print(f"  {var_type}: {count}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump([asdict(v) for v in variations], f, indent=2)
    print(f"\nSaved {len(variations)} variations to {output_path}")

    # Print sample variations
    print(f"\n{'='*60}")
    print("Sample Variations (first base prompt):")
    print('='*60)

    if variations:
        first_base = variations[0].base_id
        samples = [v for v in variations if v.base_id == first_base][:10]

        print(f"\nOriginal: {samples[0].original_prompt[:100]}...")
        print(f"\nVariations:")
        for v in samples:
            print(f"  [{v.variation_type}:{v.variation_subtype}]")
            print(f"    {v.varied_prompt}")


if __name__ == '__main__':
    main()
