"""
ACL Paper: Linguistic Variation Experiments

This module provides tools for generating and evaluating linguistic variations
to study the effect of pragmatic and semantic features on LLM structured output consistency.

Modules:
- generate_variations: Generate controlled linguistic variations from base prompts
- run_evaluation: Run tool-calling experiments on variations
- analyze_results: Analyze results and generate tables/figures

Usage:
    # 1. Generate variations
    python -m scripts.experiments.acl_linguistic_variations.generate_variations \
        --input data/toucan/toucan_tool_calls_1006.json \
        --output data/acl_variations/variations.json \
        --num_base 100

    # 2. Run evaluation
    python -m scripts.experiments.acl_linguistic_variations.run_evaluation \
        --input data/acl_variations/variations.json \
        --output results/acl_linguistic/eval_results.json \
        --model Claude-Sonnet-4 \
        --runs 10

    # 3. Analyze results
    python -m scripts.experiments.acl_linguistic_variations.analyze_results \
        --input results/acl_linguistic/eval_results.json \
        --output results/acl_linguistic/analysis/
"""

from .generate_variations import (
    LinguisticVariation,
    LinguisticVariationPipeline,
    SpeechActGenerator,
    ModalVerbGenerator,
    PolitenessGenerator,
    SyntaxGenerator,
    QuantificationGenerator,
)

__all__ = [
    'LinguisticVariation',
    'LinguisticVariationPipeline',
    'SpeechActGenerator',
    'ModalVerbGenerator',
    'PolitenessGenerator',
    'SyntaxGenerator',
    'QuantificationGenerator',
]
