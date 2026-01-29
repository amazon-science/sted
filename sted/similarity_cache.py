"""
Caching system for string similarity computations.
"""

from typing import Optional, List, Tuple


class StringSimilarityCache:
    """Cache for storing string similarity scores to avoid recomputation."""

    def __init__(self):
        self.cache = {}

    def get_key(self, s1: str, s2: str) -> str:
        """Generate a consistent cache key for two strings."""
        return "|".join(sorted([s1, s2]))

    def get(self, s1: str, s2: str) -> Optional[float]:
        """Get cached similarity score for two strings."""
        return self.cache.get(self.get_key(s1, s2))

    def set(self, s1: str, s2: str, score: float):
        """Cache similarity score for two strings."""
        self.cache[self.get_key(s1, s2)] = score

    def batch_set(self, pairs: List[Tuple[str, str]], scores: List[float]):
        """Cache multiple similarity scores at once."""
        for (s1, s2), score in zip(pairs, scores):
            self.set(s1, s2, score)

    def clear(self):
        """Clear all cached values."""
        self.cache.clear()

    def size(self) -> int:
        """Get the number of cached entries."""
        return len(self.cache)
