"""Centralized logging for the sted library.

Users can configure verbosity via:
    import logging
    logging.getLogger("sted").setLevel(logging.DEBUG)

Or programmatically via sted.set_log_level("DEBUG").
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_LOGGER_NAME = "sted"
_DEFAULT_LEVEL = logging.WARNING

_logger = logging.getLogger(_LOGGER_NAME)
_logger.setLevel(_DEFAULT_LEVEL)
# Avoid double-emission when the user also configures the root logger.
_logger.propagate = False
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [sted %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    _logger.addHandler(_handler)


def get_logger(submodule: Optional[str] = None) -> logging.Logger:
    """Return the sted logger (or a named submodule logger).

    Args:
        submodule: Optional dotted suffix, e.g. "agent_evaluator".
    """
    if submodule:
        return logging.getLogger(f"{_LOGGER_NAME}.{submodule}")
    return _logger


def set_log_level(level: str | int) -> None:
    """Set the sted package log level.

    Args:
        level: Either a string ("DEBUG", "INFO", "WARNING", "ERROR") or an
            integer log level.
    """
    if isinstance(level, str):
        level = level.upper()
        level = getattr(logging, level, _DEFAULT_LEVEL)
    _logger.setLevel(level)
