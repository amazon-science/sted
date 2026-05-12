"""Memory-bounded LRU cache for STED's subtree comparison results.

Extracted from semantic_json_tree_consistency.py during the v0.2.0 refactor.
The original class is re-exported there for backward compatibility.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple


class LRUCache:
    """Memory-bounded LRU cache for subtree comparison results.

    When ``maxsize == 0`` the cache is fully disabled: ``get`` always returns
    ``None`` (counted as a miss) and ``set`` is a no-op. This is used in
    production where Hungarian-on-children rarely produces repeated subtree
    pairs, so the cache adds memory without speed.
    """

    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple) -> Optional[float]:
        """Get value from cache, moving to end (most recently used)."""
        if self.maxsize == 0:
            self.misses += 1
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, key: Tuple, value: float) -> None:
        """Set value in cache, evicting LRU if at capacity."""
        if self.maxsize == 0:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        """Clear the cache and reset hit/miss counters."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Tuple) -> bool:
        return key in self._cache

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
