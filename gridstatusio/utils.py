import logging
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from gridstatusio._compat import import_pandas


def setup_gsio_logger(level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger("gridstatusio")

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ),
        )
        logger.addHandler(handler)

    return logger


logger = setup_gsio_logger()


def handle_date(
    date: str | datetime | Any | None,
    tz: str | None = None,
    use_pandas: bool = True,
) -> str | datetime | Any | None:
    """Handle date parsing for API requests.

    Parameters:
        date: The date to parse (string, datetime, pd.Timestamp, or None)
        tz: Optional timezone for localization
        use_pandas: If True and pandas available, use pandas for parsing.
                   If False, return string/datetime without pandas processing.

    Returns:
        Processed date suitable for API request parameters
    """
    if date is None:
        return None

    # For non-pandas mode, just validate and return the string/datetime
    if not use_pandas:
        if date == "today":
            return datetime.now().strftime("%Y-%m-%d")
        if isinstance(date, datetime):
            return date.isoformat()
        # Assume string is already in valid format
        return date

    # Pandas mode
    pd = import_pandas()

    if pd.isna(date):
        return None

    if date == "today":
        date = pd.Timestamp.now(tz=tz).normalize()

    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)

    if tz:
        # After pd.to_datetime(), date is a pandas Timestamp
        if date.tzinfo is None:  # type: ignore[union-attr]
            date = date.tz_localize(tz)  # type: ignore[union-attr]
        else:
            date = date.tz_convert(tz)  # type: ignore[union-attr]

    return date


@contextmanager
def silence_deprecation_warnings():
    """Context manager to temporarily silence deprecation warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield
