"""Compatibility module for optional dependencies.

This module provides lazy loading utilities for optional dependencies
(pandas, polars) and clear error messages when they are not installed.
"""

from typing import Any

# Cached availability flags
_PANDAS_AVAILABLE: bool | None = None
_POLARS_AVAILABLE: bool | None = None


class MissingDependencyError(ImportError):
    """Raised when a required optional dependency is not installed."""

    def __init__(self, library: str, format_name: str):
        self.library = library
        self.format_name = format_name
        msg = (
            f"The '{library}' library is required for return_format='{format_name}'. "
            f"Install it with: uv pip install gridstatusio[{library}"
        )
        super().__init__(msg)


def pandas_available() -> bool:
    """Check if pandas is available.

    Returns:
        bool: True if pandas can be imported, False otherwise.
    """
    global _PANDAS_AVAILABLE
    if _PANDAS_AVAILABLE is None:
        try:
            import pandas  # noqa: F401

            _PANDAS_AVAILABLE = True
        except ImportError:
            _PANDAS_AVAILABLE = False
    return _PANDAS_AVAILABLE


def polars_available() -> bool:
    """Check if polars is available.

    Returns:
        bool: True if polars can be imported, False otherwise.
    """
    global _POLARS_AVAILABLE
    if _POLARS_AVAILABLE is None:
        try:
            import polars  # noqa: F401

            _POLARS_AVAILABLE = True
        except ImportError:
            _POLARS_AVAILABLE = False
    return _POLARS_AVAILABLE


def import_pandas() -> Any:
    """Import and return pandas module.

    Returns:
        The pandas module.

    Raises:
        MissingDependencyError: If pandas is not installed.
    """
    if not pandas_available():
        raise MissingDependencyError("pandas", "pandas")
    import pandas as pd

    return pd


def import_polars() -> Any:
    """Import and return polars module.

    Returns:
        The polars module.

    Raises:
        MissingDependencyError: If polars is not installed.
    """
    if not polars_available():
        raise MissingDependencyError("polars", "polars")
    import polars as pl

    return pl
