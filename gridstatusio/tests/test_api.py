import os
from datetime import datetime
from typing import Any, cast

import pandas as pd
import polars as pl
import pytest

import gridstatusio as gs
from gridstatusio.utils import silence_deprecation_warnings

# Get API key
API_KEY = os.getenv("GRIDSTATUS_API_KEY_TEST") or os.getenv("GRIDSTATUS_API_KEY")
HOST = os.getenv("GRIDSTATUS_HOST_TEST", "https://api.gridstatus.io/v1")


# Helper functions for format-agnostic checks
def get_length(data: Any, return_format: str) -> int:
    """Get length of data regardless of format."""
    return len(data)


def get_columns(data: Any, return_format: str) -> list[str]:
    """Get column names regardless of format."""
    if return_format == "python":
        if len(data) == 0:
            return []
        return list(data[0].keys())
    elif return_format == "pandas":
        return data.columns.to_list()
    elif return_format == "polars":
        return data.columns
    raise ValueError(f"Unknown format: {return_format}")


def get_column_values(data: Any, column: str, return_format: str) -> list:
    """Get values from a column regardless of format."""
    if return_format == "python":
        return [row[column] for row in data]
    elif return_format == "pandas":
        return data[column].tolist()
    elif return_format == "polars":
        return data[column].to_list()
    raise ValueError(f"Unknown format: {return_format}")


def get_unique_values(data: Any, column: str, return_format: str) -> set:
    """Get unique values from a column regardless of format."""
    if return_format == "python":
        return set(row[column] for row in data)
    elif return_format == "pandas":
        return set(data[column].unique())
    elif return_format == "polars":
        return set(data[column].unique().to_list())
    raise ValueError(f"Unknown format: {return_format}")


def get_min_value(data: Any, column: str, return_format: str) -> Any:
    """Get min value from a column regardless of format."""
    if return_format == "python":
        values = [row[column] for row in data]
        return min(values)
    elif return_format == "pandas":
        return data[column].min()
    elif return_format == "polars":
        return data[column].min()
    raise ValueError(f"Unknown format: {return_format}")


def get_max_value(data: Any, column: str, return_format: str) -> Any:
    """Get max value from a column regardless of format."""
    if return_format == "python":
        values = [row[column] for row in data]
        return max(values)
    elif return_format == "pandas":
        return data[column].max()
    elif return_format == "polars":
        return data[column].max()
    raise ValueError(f"Unknown format: {return_format}")


def check_result(
    data: Any,
    return_format: str,
    length: int | None = None,
    columns: list[str] | None = None,
    min_length: int | None = None,
):
    """Check result data regardless of format."""
    # Check type
    if return_format == "python":
        assert isinstance(data, list)
        if len(data) > 0:
            assert all(isinstance(row, dict) for row in data)
    elif return_format == "pandas":
        assert isinstance(data, pd.DataFrame)
    elif return_format == "polars":
        assert isinstance(data, pl.DataFrame)

    # Check length
    actual_length = get_length(data, return_format)
    if length is not None:
        assert actual_length == length, f"Expected length {length}, got {actual_length}"
    if min_length is not None:
        assert (
            actual_length >= min_length
        ), f"Expected at least {min_length} rows, got {actual_length}"

    # Check columns
    if columns is not None and actual_length > 0:
        actual_columns = get_columns(data, return_format)
        assert (
            actual_columns == columns
        ), f"Expected columns {columns}, got {actual_columns}"


def check_datetime_column(data: Any, column: str, return_format: str):
    """Check that a column contains datetime data (format-specific)."""
    if return_format == "python":
        # In python format, datetimes are ISO8601 strings
        values = get_column_values(data, column, return_format)
        if len(values) > 0 and values[0] is not None:
            assert isinstance(
                values[0],
                str,
            ), f"Expected string in python format, got {type(values[0])}"
    elif return_format == "pandas":
        assert pd.api.types.is_datetime64_any_dtype(data[column])
    elif return_format == "polars":
        assert data[column].dtype == pl.Datetime or data[column].dtype.is_temporal()


def format_date_for_comparison(value: Any, return_format: str) -> str:
    """Format a date value for comparison."""
    if return_format == "python":
        # Python format returns ISO strings, extract date part
        if isinstance(value, str):
            return value[:10]
        return str(value)[:10]
    elif return_format == "pandas":
        return value.strftime("%Y-%m-%d")
    elif return_format == "polars":
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)[:10]
    return str(value)[:10]


# Fixtures
@pytest.fixture(params=["pandas", "polars", "python"])
def return_format(request):
    """Parametrize tests across all return formats."""
    return request.param


@pytest.fixture
def client(return_format):
    """Create a client with the specified return format."""
    return gs.GridStatusClient(api_key=API_KEY, host=HOST, return_format=return_format)


@pytest.fixture
def pandas_client():
    """Create a pandas-only client for tests that specifically need pandas."""
    return gs.GridStatusClient(api_key=API_KEY, host=HOST, return_format="pandas")


# Tests
def test_invalid_api_key():
    client = gs.GridStatusClient(api_key="invalid")
    try:
        client.get_dataset(dataset="isone_fuel_mix", verbose=True)
    except Exception as e:
        assert "Invalid API key" in str(e)


def test_uses_columns(client, return_format):
    dataset = "ercot_sced_gen_resource_60_day"
    one_column = "resource_name"
    columns = ["interval_start_utc", "interval_end_utc", one_column]
    limit = 100

    data = client.get_dataset(
        dataset=dataset,
        columns=columns,
        verbose=False,
        limit=limit,
    )
    check_result(data, return_format, columns=columns, length=limit)

    # time columns always included even if not specified
    data = client.get_dataset(
        dataset=dataset,
        columns=[one_column],
        verbose=False,
        limit=limit,
    )
    check_result(data, return_format, columns=columns, length=limit)


def test_handles_unknown_columns(client, return_format):
    dataset = "ercot_fuel_mix"

    with pytest.raises(Exception) as e:
        client.get_dataset(
            dataset=dataset,
            columns=["invalid_column"],
            verbose=True,
        )

    assert "Column invalid_column not found in dataset" in str(e.value)


def test_list_datasets(client, return_format):
    datasets = client.list_datasets(return_list=True)
    assert isinstance(datasets, list), "Expected a list of datasets"
    assert len(datasets) > 0, "Expected at least one dataset"


def test_list_datasets_filter(client, return_format):
    filter_term = "fuel_mix"
    min_results = 7
    # run once without printing things out
    client.list_datasets(filter_term=filter_term, return_list=False)
    datasets = client.list_datasets(filter_term=filter_term, return_list=True)
    assert datasets is not None, f"No datasets returned for filter term '{filter_term}'"
    assert (
        len(datasets) >= min_results
    ), f"Expected at least {min_results} results with filter term '{filter_term}'"


def test_set_api_works():
    client = gs.GridStatusClient(api_key="test")
    assert client.api_key == "test"


def test_get_dataset_date_range(client, return_format):
    start = "2023-01-01"
    end = "2023-01-05"
    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start=start,
        end=end,
        verbose=False,
    )

    check_result(data, return_format, min_length=1)

    # Check date range
    min_date = format_date_for_comparison(
        get_min_value(data, "interval_start_utc", return_format),
        return_format,
    )
    max_date = format_date_for_comparison(
        get_max_value(data, "interval_end_utc", return_format),
        return_format,
    )
    assert min_date == start
    assert max_date == end


def test_index_unique_multiple_pages(client, return_format):
    start = "2023-01-01"
    end = "2023-01-02"
    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start=start,
        end=end,
        verbose=False,
        limit=100,
    )

    check_result(data, return_format, min_length=1)


def test_filter_operator(client, return_format):
    dataset = "caiso_curtailment"
    limit = 1000
    category_column = "curtailment_type"
    category_value = "Economic"
    numeric_column = "curtailment_mw"
    numeric_value = 100

    # Test = operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=category_column,
        filter_value=category_value,
        filter_operator="=",
        limit=limit,
        verbose=False,
    )
    check_result(data, return_format, min_length=1)
    unique_values = get_unique_values(data, category_column, return_format)
    assert unique_values == {category_value}

    # Test < operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator="<",
        limit=limit,
        verbose=False,
    )
    check_result(data, return_format, min_length=1)
    max_val = get_max_value(data, numeric_column, return_format)
    assert max_val < numeric_value

    # Test > operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator=">",
        limit=limit,
        verbose=False,
    )
    check_result(data, return_format, min_length=1)
    min_val = get_min_value(data, numeric_column, return_format)
    assert min_val > numeric_value


def test_filter_operator_in(client, return_format):
    locations = ["LZ_WEST", "LZ_HOUSTON"]
    data = client.get_dataset(
        dataset="ercot_spp_day_ahead_hourly",
        filter_column="location",
        filter_value=locations,
        filter_operator="in",
        start="2023-09-07",
        limit=10,
        verbose=False,
    )
    unique_locs = get_unique_values(data, "location", return_format)
    assert unique_locs == set(locations)
    check_result(data, return_format)


def test_get_dataset_verbose_false(client, return_format, caplog):
    caplog.set_level("INFO")
    client.get_dataset(
        dataset="isone_fuel_mix",
        start="2023-01-01",
        end="2023-01-05",
        limit=1,
        verbose=False,
    )
    caplog.clear()


def test_get_dataset_verbose_true(client, return_format, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="gridstatusio")
    client.get_dataset(
        dataset="isone_fuel_mix",
        start="2023-01-01",
        end="2023-01-05",
        limit=1,
        verbose=True,
    )
    log_messages = [record.message for record in caplog.records]
    assert len(log_messages) > 0
    assert any("Done in" in msg for msg in log_messages)
    caplog.clear()


def test_resample_frequency(client, return_format):
    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start="2023-01-01",
        end="2023-01-02",
        resample="1 hour",
        verbose=False,
    )

    check_result(data, return_format, length=24)


def test_resample_function(client, return_format):
    data_max = client.get_dataset(
        dataset="caiso_load",
        start="2023-09-01",
        end="2023-09-03",
        resample="1 day",
        resample_function="max",
        verbose=False,
    )

    data_min = client.get_dataset(
        dataset="caiso_load",
        start="2023-09-01",
        end="2023-09-03",
        resample="1 day",
        resample_function="min",
        verbose=False,
    )

    check_result(data_max, return_format, length=2)
    check_result(data_min, return_format, length=2)

    # max load should be higher than min load
    max_loads = get_column_values(data_max, "load", return_format)
    min_loads = get_column_values(data_min, "load", return_format)
    for max_val, min_val in zip(max_loads, min_loads):
        assert max_val > min_val


def test_resampling_across_days(client, return_format):
    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start="2023-01-01",
        end="2023-01-03",
        resample="1 day",
        verbose=False,
    )

    check_result(data, return_format, length=2)


def test_pagination(client, return_format):
    dataset = "isone_fuel_mix"

    # return 100 rows
    data = client.get_dataset(dataset=dataset, limit=100, verbose=False)
    check_result(data, return_format, length=100)

    # test multiple pages
    data = client.get_dataset(dataset=dataset, limit=100, page_size=25, verbose=False)
    check_result(data, return_format, length=100)

    # test limit less than page size
    data = client.get_dataset(dataset=dataset, limit=25, page_size=100, verbose=False)
    check_result(data, return_format, length=25)


@pytest.mark.parametrize(
    "iso,market_date,expected_date",
    [
        ("ERCOT", None, datetime.now().strftime("%Y-%m-%d")),
        ("CAISO", "2024-07-01", "2024-07-01"),
        ("spp", datetime(2024, 7, 10), "2024-07-10"),
    ],
)
def test_reports_api(iso, market_date, expected_date):
    # Reports API always returns dict, independent of return_format
    client = gs.GridStatusClient(api_key=API_KEY, host=HOST)
    resp = client.get_daily_peak_report(iso=iso, market_date=market_date)
    assert isinstance(resp, dict)
    assert resp["ISO"] == iso.upper()
    assert resp["market_date"] == expected_date


def test_get_api_usage():
    # API usage always returns dict, independent of return_format
    client = gs.GridStatusClient(api_key=API_KEY, host=HOST)
    usage = client.get_api_usage()

    assert list(usage.keys()) == [
        "plan_name",
        "limits",
        "current_usage_period_start",
        "current_usage_period_end",
        "current_period_usage",
    ]


# Tests that require pandas-specific functionality
class TestPandasSpecific:
    """Tests that specifically require pandas DataFrame functionality."""

    def test_cursor_pagination_equals_offset_pagination(self, pandas_client):
        common_args = {
            "dataset": "ercot_lmp_by_bus",
            "start": "2023-01-01",
            "end": "2023-01-02",
            "limit": 500,
            "page_size": 100,
            "verbose": False,
        }

        cursor = pandas_client.get_dataset(**common_args, use_cursor_pagination=True)
        offset = pandas_client.get_dataset(**common_args, use_cursor_pagination=False)

        assert cursor.equals(offset)

    def test_resample_and_paginated(self, pandas_client):
        common_args = {
            "dataset": "isone_fuel_mix",
            "start": "2023-01-01",
            "end": "2023-01-02",
            "limit": 1000,
            "resample": "1 hour",
            "verbose": False,
        }

        paginated = pandas_client.get_dataset(**common_args, page_size=100)
        non_paginated = pandas_client.get_dataset(**common_args, page_size=1000)

        assert paginated.equals(non_paginated)
        assert len(paginated) == 24

    def test_publish_time_latest(self, pandas_client):
        today = pd.Timestamp.now(tz="UTC").floor("D")

        df = pandas_client.get_dataset(
            dataset="miso_wind_forecast_hourly",
            start=cast(pd.Timestamp, today - pd.Timedelta(days=2)),
            end=today,
            publish_time="latest",
            verbose=False,
        )

        assert df["publish_time_utc"].nunique() > 1, "Expected multiple publish times"
        assert (
            df["interval_start_utc"].value_counts() == 1
        ).all(), "Expected each interval to only occur once"

    def test_publish_time_latest_report(self, pandas_client):
        df = pandas_client.get_dataset(
            dataset="miso_wind_forecast_hourly",
            publish_time="latest_report",
            verbose=False,
        )

        assert df["publish_time_utc"].nunique() == 1, "Expected one publish time"

    def test_handles_all_nan_columns(self, pandas_client):
        start = "2020-01-01"
        end = "2020-01-02"
        btm_col = "btm_solar.capitl"
        time_columns = ["interval_start_utc", "interval_end_utc"]

        df = pandas_client.get_dataset(
            "nyiso_standardized_5_min",
            start=start,
            end=end,
            columns=time_columns + [btm_col],
            verbose=False,
        )

        assert set(time_columns + [btm_col]) == set(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df[time_columns[0]])
        assert df[btm_col].dtype == "object"

    def test_handles_no_results(self, pandas_client):
        btm_col = "capitl"
        time_columns = ["interval_start_utc", "interval_end_utc"]

        df = pandas_client.get_dataset(
            "nyiso_btm_solar",
            start="2020-01-01",
            end="2020-01-02",
            columns=time_columns + [btm_col],
            verbose=False,
        )

        assert set(time_columns + [btm_col]) == set(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df[time_columns[0]])

    def test_market_day_data_downsampling(self, pandas_client):
        df = pandas_client.get_dataset(
            "pjm_outages_daily",
            start="2024-01-01",
            end="2024-05-01",
            resample="1 month",
            verbose=False,
        )

        assert df["interval_start_utc"].min() == pd.Timestamp("2024-01-01", tz="UTC")
        assert df["interval_end_utc"].max() == pd.Timestamp("2024-05-01", tz="UTC")

    def test_market_day_data_upsampling(self, pandas_client):
        df = pandas_client.get_dataset(
            "pjm_outages_daily",
            start="2024-01-01",
            end="2024-01-05",
            resample="1 hour",
            verbose=False,
        )

        assert df["interval_start_utc"].min() == pd.Timestamp(
            "2024-01-01 00:00:00",
            tz="UTC",
        )

    def test_publish_time_filtering(self, pandas_client):
        publish_time_filter = "2023-09-30T12:00:00Z"

        df = pandas_client.get_dataset(
            dataset="ercot_load_forecast_by_forecast_zone",
            start="2023-10-01",
            end="2023-10-02",
            verbose=False,
        )

        assert df["publish_time_utc"].min() < pd.Timestamp(
            publish_time_filter,
            tz="UTC",
        )

        df_filtered = pandas_client.get_dataset(
            dataset="ercot_load_forecast_by_forecast_zone",
            start="2023-10-01",
            end="2023-10-02",
            publish_time_start=publish_time_filter,
            verbose=False,
        )

        assert df_filtered["publish_time_utc"].min() >= pd.Timestamp(
            publish_time_filter,
            tz="UTC",
        )

    def test_invalid_resampling_frequency(self, pandas_client):
        with pytest.raises(Exception):
            pandas_client.get_dataset(
                "pjm_load",
                resample="1 hour market",
                start="2024-01-01",
                end="2024-01-02",
                verbose=False,
            )

    def test_tz_parameter_deprecated(self, pandas_client):
        with silence_deprecation_warnings():
            df = pandas_client.get_dataset(
                dataset="isone_fuel_mix",
                start="2023-01-01",
                end="2023-01-02",
                tz="America/New_York",
                limit=10,
                verbose=False,
            )
        assert "interval_start_local" in df.columns
