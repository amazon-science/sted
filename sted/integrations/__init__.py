"""Adapters that plug STED into evaluation frameworks.

    sted.integrations.langsmith     sted_evaluator, sted_consistency_evaluator
    sted.integrations.inspect_ai    sted_match (scorer), sted_agreement (epoch reducer)

Each module imports its framework lazily, so `import sted` never requires them.
"""
