import pandas as pd


def handle_date(date, tz=None):
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
            # todo see if this triggers in tests
            date = date.tz_convert(tz)

    return date
