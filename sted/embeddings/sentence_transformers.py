"""SentenceTransformer batch-encoder for STED.

Extracted from semantic_json_tree_consistency.py during the v0.2.0 refactor.
The original method is a thin wrapper that delegates to this module-level
helper, so existing call sites and import paths keep working.
"""
from __future__ import annotations

import warnings
from typing import List

from tqdm import tqdm


def batch_encode_sentence_transformer(
    evaluator,
    strings: List[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> None:
    """Batch encode strings using SentenceTransformer."""
    # Preprocess all strings
    processed_strings = [evaluator._preprocess_key_name(s) for s in strings]

    # Encode in batches
    iterator = range(0, len(processed_strings), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding batches", total=len(processed_strings) // batch_size + 1)

    for i in iterator:
        batch = processed_strings[i:i + batch_size]
        original_batch = strings[i:i + batch_size]

        try:
            embeddings = evaluator.embedding_model.encode(batch, show_progress_bar=False, batch_size=batch_size)
            for orig_str, emb in zip(original_batch, embeddings):
                evaluator._store_embedding(orig_str, emb)
        except Exception as e:
            warnings.warn(f"Batch encoding failed: {e}")
