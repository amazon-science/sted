"""Lossless terminal paths for parsed JSON.

Object keys and array indices occupy different namespaces. Empty containers are
terminals, so their presence and kind survive flattening. Nonempty containers
are represented by the typed steps to their descendants.
"""
from enum import Enum
from typing import Any, Dict, Tuple

Path = Tuple[Tuple[str, Any], ...]


class EmptyContainer(Enum):
    OBJECT = "object"
    ARRAY = "array"


def terminal_paths(obj: Any, prefix: Path = (), *, skip: Any = None,
                   skip_enabled: bool = False) -> Dict[Path, Any]:
    if skip_enabled and obj is skip:
        return {}
    if isinstance(obj, dict):
        if not obj:
            return {prefix: EmptyContainer.OBJECT}
        out = {}
        for key, value in obj.items():
            out.update(terminal_paths(value, prefix + (("key", key),),
                                      skip=skip, skip_enabled=skip_enabled))
        return out
    if isinstance(obj, list):
        if not obj:
            return {prefix: EmptyContainer.ARRAY}
        out = {}
        for index, value in enumerate(obj):
            out.update(terminal_paths(value, prefix + (("index", index),),
                                      skip=skip, skip_enabled=skip_enabled))
        return out
    return {prefix: obj}
