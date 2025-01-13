import os

import pytest

import gridstatusio as gs

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

    def test_tz_only(self):
        df = client.get_dataset(
            self.dataset,
            start=self.start,
            end=self.end,
            tz=self.tz,
            columns=self.columns,
        )
        assert df.columns.tolist() == [
            "interval_start_local",
            "interval_end_local",
            "wind",
            "solar",
        ]

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
        assert df.columns.tolist() == [
            "interval_start_local",
            "interval_start_utc",
            "interval_end_local",
            "interval_end_utc",
            "wind",
            "solar",
        ]

        assert str(df["interval_start_local"].min()) == "2024-11-30 16:00:00-08:00"
        assert str(df["interval_start_utc"].min()) == "2024-12-01 00:00:00+00:00"
        assert str(df["interval_end_local"].max()) == "2024-11-30 17:00:00-08:00"
        assert str(df["interval_end_utc"].max()) == "2024-12-01 01:00:00+00:00"

    def test_tz_and_timezone(self):
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

        assert df.columns.tolist() == [
            "interval_start_utc",
            "interval_end_utc",
            "wind",
            "solar",
        ]

        assert str(df["interval_start_utc"].min()) == "2024-12-01 00:00:00+00:00"
        assert str(df["interval_end_utc"].max()) == "2024-12-01 01:00:00+00:00"
