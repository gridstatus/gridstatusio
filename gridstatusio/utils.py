import logging
import pandas as pd

def setup_gs_logger(level: int = logging.DEBUG) -> logging.Logger:
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

logger = setup_gs_logger()

def handle_date(date: str | pd.Timestamp | None, tz: str | None = None) -> pd.Timestamp | None:
    if date is None:
        return date

    if date == "today":
        date = pd.Timestamp.now(tz=tz).normalize()

    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)

    if tz:
        if date.tzinfo is None:
            date = date.tz_localize(tz)
        else:
            date = date.tz_convert(tz)

    return date
