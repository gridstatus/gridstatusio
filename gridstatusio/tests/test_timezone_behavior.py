import os
from typing import cast

import pytest

import gridstatusio as gs
from gridstatusio.utils import silence_deprecation_warnings

client = gs.GridStatusClient(
    api_key=os.getenv("GRIDSTATUS_API_KEY_TEST"),
    host=os.getenv("GRIDSTATUS_HOST_TEST", "https://api.gridstatus.io/v1"),
)


class TestTimezoneBehavior:
    dataset = "caiso_fuel_mix"
    start = "2024-12-01T00:00:00Z"
    end = "2024-12-01T01:00:00Z"
    tz = "US/Pacific"
    timezone = "US/Pacific"
    columns = ["wind", "solar"]

    expected_columns_with_tz = [
        "interval_start_local",
        "interval_end_local",
        "wind",
        "solar",
    ]

    expected_columns_with_timezone = [
        "interval_start_local",
        "interval_start_utc",
        "interval_end_local",
        "interval_end_utc",
        "wind",
        "solar",
    ]

    expected_columns_no_tz_no_timezone = [
        "interval_start_utc",
        "interval_end_utc",
        "wind",
        "solar",
    ]

    def test_tz_only(self):
        with silence_deprecation_warnings():
            df = client.get_dataset(
                self.dataset,
                start=self.start,
                end=self.end,
                tz=self.tz,
                columns=self.columns,
            )
        assert df.columns.tolist() == self.expected_columns_with_tz

        assert str(df["interval_start_local"].min()) == "2024-11-30 16:00:00-08:00"
        assert str(df["interval_end_local"].max()) == "2024-11-30 17:00:00-08:00"

    def test_timezone_only(self):
        df = client.get_dataset(
            self.dataset,
            start=self.start,
            end=self.end,
            timezone=self.timezone,
            columns=self.columns,
        )
        assert df.columns.tolist() == self.expected_columns_with_timezone

        assert str(df["interval_start_local"].min()) == "2024-11-30 16:00:00-08:00"
        assert str(df["interval_start_utc"].min()) == "2024-12-01 00:00:00+00:00"
        assert str(df["interval_end_local"].max()) == "2024-11-30 17:00:00-08:00"
        assert str(df["interval_end_utc"].max()) == "2024-12-01 01:00:00+00:00"

    def test_tz_and_timezone(self):
        with silence_deprecation_warnings():
            with pytest.raises(ValueError):
                client.get_dataset(
                    self.dataset,
                    start=self.start,
                    end=self.end,
                    tz=self.tz,
                    timezone=self.timezone,
                    columns=self.columns,
                )

    def test_no_tz_and_no_timezone(self):
        df = client.get_dataset(
            self.dataset,
            start=self.start,
            end=self.end,
            columns=self.columns,
        )

        assert df.columns.tolist() == self.expected_columns_no_tz_no_timezone

        assert str(df["interval_start_utc"].min()) == "2024-12-01 00:00:00+00:00"
        assert str(df["interval_end_utc"].max()) == "2024-12-01 01:00:00+00:00"

    def test_tz_on_dst_start(self):
        with silence_deprecation_warnings():
            self.start = "2024-03-10T08:00:00Z"
            self.end = "2024-03-10T12:00:00Z"

            df = client.get_dataset(
                self.dataset,
                start=self.start,
                end=self.end,
                tz=self.tz,
                columns=self.columns,
            )

        assert df.columns.tolist() == self.expected_columns_with_tz

        # First offset should be 8 hours, then 7 hours after DST starts
        assert str(df["interval_start_local"].min()) == "2024-03-10 00:00:00-08:00"
        assert str(df["interval_end_local"].max()) == "2024-03-10 05:00:00-07:00"

    def test_timezone_on_dst_start(self):
        self.start = "2024-03-10T08:00:00Z"
        self.end = "2024-03-10T12:00:00Z"

        df = client.get_dataset(
            self.dataset,
            start=self.start,
            end=self.end,
            timezone=self.timezone,
            columns=self.columns,
        )

        assert df.columns.tolist() == self.expected_columns_with_timezone

        # First offset should be 8 hours, then 7 hours after DST starts
        assert str(df["interval_start_local"].min()) == "2024-03-10 00:00:00-08:00"
        assert str(df["interval_end_local"].max()) == "2024-03-10 05:00:00-07:00"

    def test_tz_on_dst_end(self):
        with silence_deprecation_warnings():
            self.start = "2024-11-03T08:00:00Z"
            self.end = "2024-11-03T12:00:00Z"

            df = client.get_dataset(
                self.dataset,
                start=self.start,
                end=self.end,
                tz=self.tz,
                columns=self.columns,
            )

        assert df.columns.tolist() == self.expected_columns_with_tz

        # First offset should be 7 hours, then 8 hours after DST ends
        assert str(df["interval_start_local"].min()) == "2024-11-03 01:00:00-07:00"
        assert str(df["interval_end_local"].max()) == "2024-11-03 04:00:00-08:00"

    def test_timezone_on_dst_end(self):
        self.start = "2024-11-03T08:00:00Z"
        self.end = "2024-11-03T12:00:00Z"

        df = client.get_dataset(
            self.dataset,
            start=self.start,
            end=self.end,
            timezone=self.timezone,
            columns=self.columns,
        )

        assert df.columns.tolist() == self.expected_columns_with_timezone

        # First offset should be 7 hours, then 8 hours after DST ends
        assert str(df["interval_start_local"].min()) == "2024-11-03 01:00:00-07:00"
        assert str(df["interval_end_local"].max()) == "2024-11-03 04:00:00-08:00"

    def test_tz_with_naive_start_and_end(self):
        with silence_deprecation_warnings():
            self.start = "2024-12-01 00:00:00"
            self.end = "2024-12-01 01:00:00"

            df = client.get_dataset(
                self.dataset,
                start=self.start,
                end=self.end,
                tz=self.tz,
                columns=self.columns,
            )

        assert df.columns.tolist() == self.expected_columns_with_tz

        assert str(df["interval_start_local"].min()) == "2024-12-01 00:00:00-08:00"
        assert str(df["interval_end_local"].max()) == "2024-12-01 01:00:00-08:00"

    def test_timezone_with_naive_start_and_end(self):
        self.start = "2024-12-01 00:00:00"
        self.end = "2024-12-01 01:00:00"

        df = client.get_dataset(
            self.dataset,
            start=self.start,
            end=self.end,
            timezone=self.timezone,
            columns=self.columns,
        )

        assert df.columns.tolist() == self.expected_columns_with_timezone

        assert str(df["interval_start_local"].min()) == "2024-12-01 00:00:00-08:00"
        assert str(df["interval_start_utc"].min()) == "2024-12-01 08:00:00+00:00"
        assert str(df["interval_end_local"].max()) == "2024-12-01 01:00:00-08:00"
        assert str(df["interval_end_utc"].max()) == "2024-12-01 09:00:00+00:00"

    def test_market_timezone(self):
        df = client.get_dataset(
            self.dataset,
            start=self.start,
            end=self.end,
            timezone="market",
            columns=self.columns,
        )
        assert df.columns.tolist() == self.expected_columns_with_timezone

        assert str(df["interval_start_local"].min()) == "2024-11-30 16:00:00-08:00"
        assert str(df["interval_start_utc"].min()) == "2024-12-01 00:00:00+00:00"
        assert str(df["interval_end_local"].max()) == "2024-11-30 17:00:00-08:00"
        assert str(df["interval_end_utc"].max()) == "2024-12-01 01:00:00+00:00"

    def test_handle_date_tz_convert(self):
        """Test the tz_convert case in utils.handle_date function"""
        import pandas as pd

        from gridstatusio.utils import handle_date

        # Test with timezone-aware timestamp that needs conversion
        timestamp = cast(pd.Timestamp, pd.Timestamp("2024-01-01 12:00:00", tz="UTC"))
        result = handle_date(timestamp, tz="US/Pacific")

        assert isinstance(result, pd.Timestamp)
        assert result.tzinfo is not None
        assert str(result.tzinfo) == "US/Pacific"

        # UTC 12:00:00 should be 04:00:00 in US/Pacific (8 hours behind)
        expected = pd.Timestamp("2024-01-01 04:00:00", tz="US/Pacific")
        assert result == expected
