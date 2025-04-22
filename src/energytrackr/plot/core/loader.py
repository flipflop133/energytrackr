"""Utility for dynamic *module:attr* loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

from energytrackr.utils.exceptions import (
    AttributeNotFoundError,
    InvalidDottedPathError,
)


def load_callable(dotted: str) -> Callable[..., Any]:
    """Return the attribute referenced by *module:attr* string.

    Args:
        dotted : str
            The dotted path to the callable, e.g. "module:attr".

    Returns:
        Callable[..., Any]
            The callable object referenced by the dotted path.

    Raises:
        InvalidDottedPathError : If the dotted path is not in the expected format.
        AttributeNotFoundError : If the attribute is not found in the specified module.

    Examples:
        >>> fn = load_callable("math:sqrt")
        >>> fn(9)
        3.0
    """
    try:
        mod_path, attr = dotted.split(":", 1)
    except ValueError as exc:
        raise InvalidDottedPathError(dotted) from exc
    module: ModuleType = importlib.import_module(mod_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeNotFoundError(mod_path, attr) from exc
