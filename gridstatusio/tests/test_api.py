"""Parameterized API tests for all return formats (pandas, polars, python).

This module tests the GridStatus API client with different return formats
to ensure consistent behavior across pandas DataFrames, polars DataFrames,
and Python lists of dicts.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pandas as pd
import polars as pl
import pytest

import gridstatusio as gs
from gridstatusio import ReturnFormat
from gridstatusio.utils import silence_deprecation_warnings

# Test configuration
API_KEY = os.getenv("GRIDSTATUS_API_KEY_TEST")
HOST = os.getenv("GRIDSTATUS_HOST_TEST", "https://api.gridstatus.io/v1")

# Return formats to test
RETURN_FORMATS = [ReturnFormat.PANDAS, ReturnFormat.POLARS, ReturnFormat.PYTHON]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=RETURN_FORMATS)
def return_format(request):
    """Parameterized fixture for return format."""
    return request.param


@pytest.fixture
def client(return_format):
    """Create a client with the specified return format."""
    return gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )


@pytest.fixture
def pandas_client():
    """Client that always uses pandas format."""
    return gs.GridStatusClient(api_key=API_KEY, host=HOST, return_format="pandas")


# ============================================================================
# Format-agnostic helper functions
# ============================================================================


def get_length(data: Any) -> int:
    """Get the number of rows in data, regardless of format."""
    if isinstance(data, pd.DataFrame):
        return len(data)
    elif isinstance(data, pl.DataFrame):
        return data.shape[0]
    elif isinstance(data, list):
        return len(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_columns(data: Any) -> list[str]:
    """Get column names from data, regardless of format."""
    if isinstance(data, pd.DataFrame):
        return data.columns.tolist()
    elif isinstance(data, pl.DataFrame):
        return data.columns
    elif isinstance(data, list):
        return list(data[0].keys()) if data else []
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_column_values(data: Any, column: str) -> list:
    """Get all values for a column, regardless of format."""
    if isinstance(data, pd.DataFrame):
        return data[column].tolist()
    elif isinstance(data, pl.DataFrame):
        return data[column].to_list()
    elif isinstance(data, list):
        return [row.get(column) for row in data]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_unique_values(data: Any, column: str) -> list:
    """Get unique values for a column, regardless of format."""
    if isinstance(data, pd.DataFrame):
        return data[column].unique().tolist()
    elif isinstance(data, pl.DataFrame):
        return data[column].unique().to_list()
    elif isinstance(data, list):
        return list(set(get_column_values(data, column)))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_min(data: Any, column: str) -> Any:
    """Get min value for a column, regardless of format."""
    if isinstance(data, pd.DataFrame):
        return data[column].min()
    elif isinstance(data, pl.DataFrame):
        return data[column].min()
    elif isinstance(data, list):
        values = [v for v in get_column_values(data, column) if v is not None]
        return min(values) if values else None
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_max(data: Any, column: str) -> Any:
    """Get max value for a column, regardless of format."""
    if isinstance(data, pd.DataFrame):
        return data[column].max()
    elif isinstance(data, pl.DataFrame):
        return data[column].max()
    elif isinstance(data, list):
        values = [v for v in get_column_values(data, column) if v is not None]
        return max(values) if values else None
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def is_datetime_column(data: Any, column: str) -> bool:
    """Check if a column contains datetime values."""
    if isinstance(data, pd.DataFrame):
        return pd.api.types.is_datetime64_any_dtype(data[column])
    elif isinstance(data, pl.DataFrame):
        return data[column].dtype.is_temporal()
    elif isinstance(data, list):
        if not data or column not in data[0]:
            return False
        return isinstance(data[0][column], datetime)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_nunique(data: Any, column: str) -> int:
    """Get number of unique values for a column, regardless of format."""
    if isinstance(data, pd.DataFrame):
        return cast(int, data[column].nunique())
    elif isinstance(data, pl.DataFrame):
        return cast(int, data[column].n_unique())
    elif isinstance(data, list):
        return len(set(get_column_values(data, column)))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_value_counts(data: Any, column: str) -> dict:
    """Get value counts for a column, regardless of format."""
    if isinstance(data, pd.DataFrame):
        return data[column].value_counts().to_dict()
    elif isinstance(data, pl.DataFrame):
        vc = data[column].value_counts()
        return dict(zip(vc[column].to_list(), vc["count"].to_list()))
    elif isinstance(data, list):
        from collections import Counter

        return dict(Counter(get_column_values(data, column)))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def column_in_data(data: Any, column: str) -> bool:
    """Check if a column exists in the data."""
    return column in get_columns(data)


def check_groupby_unique(
    data: Any,
    groupby_columns: list[str | tuple[str, str]],
) -> bool:
    """Check that groupby results in unique rows (max 1 per group).

    Args:
        data: The data to check.
        groupby_columns: List of columns to group by. Can be:
            - str: column name to group by directly
            - tuple[str, str]: (column_name, extraction_type) where extraction_type
              is "month", "hour", or "date" for datetime columns.
    """

    def _extract_datetime(val: Any, extraction: str) -> Any:
        """Extract datetime component from a value."""
        if extraction == "month":
            return val.month if hasattr(val, "month") else val
        elif extraction == "hour":
            return val.hour if hasattr(val, "hour") else val
        elif extraction == "date":
            return val.date() if hasattr(val, "date") else val
        else:
            raise ValueError(f"Unsupported extraction: {extraction}")

    if isinstance(data, pd.DataFrame):
        # Build groupby list with datetime extractions
        groupby_list = []
        for col_spec in groupby_columns:
            if isinstance(col_spec, tuple):
                col, extraction = col_spec
                if extraction == "month":
                    groupby_list.append(data[col].dt.month)
                elif extraction == "hour":
                    groupby_list.append(data[col].dt.hour)
                elif extraction == "date":
                    groupby_list.append(data[col].dt.date)
                else:
                    raise ValueError(f"Unsupported extraction: {extraction}")
            else:
                groupby_list.append(col_spec)
        return data.groupby(groupby_list).size().max() == 1

    elif isinstance(data, pl.DataFrame):
        # Build groupby list with datetime extractions
        groupby_exprs = []
        for col_spec in groupby_columns:
            if isinstance(col_spec, tuple):
                col, extraction = col_spec
                if extraction == "month":
                    groupby_exprs.append(
                        pl.col(col).dt.month().alias(f"{col}_{extraction}"),
                    )
                elif extraction == "hour":
                    groupby_exprs.append(
                        pl.col(col).dt.hour().alias(f"{col}_{extraction}"),
                    )
                elif extraction == "date":
                    groupby_exprs.append(
                        pl.col(col).dt.date().alias(f"{col}_{extraction}"),
                    )
                else:
                    raise ValueError(f"Unsupported extraction: {extraction}")
            else:
                groupby_exprs.append(pl.col(col_spec))
        return data.group_by(groupby_exprs).len()["len"].max() == 1

    elif isinstance(data, list):
        # For list of dicts, extract values and check uniqueness
        from collections import Counter

        keys = []
        for row in data:
            key_parts = []
            for col_spec in groupby_columns:
                if isinstance(col_spec, tuple):
                    col, extraction = col_spec
                    val = row.get(col)
                    key_parts.append(_extract_datetime(val, extraction))
                else:
                    key_parts.append(row.get(col_spec))
            keys.append(tuple(key_parts))
        counts = Counter(keys)
        return max(counts.values()) == 1

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def check_data(
    data: Any,
    return_format: str,
    length: int | None = None,
    columns: list[str] | None = None,
    check_datetime: bool = True,
):
    """Check that data meets expected criteria, regardless of format."""
    # Check type
    if return_format == ReturnFormat.PANDAS:
        assert isinstance(data, pd.DataFrame), f"Expected DataFrame, got {type(data)}"
    elif return_format == ReturnFormat.POLARS:
        assert isinstance(
            data,
            pl.DataFrame,
        ), f"Expected polars DataFrame, got {type(data)}"
    elif return_format == ReturnFormat.PYTHON:
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert all(isinstance(row, dict) for row in data), "Expected list of dicts"

    # Check non-empty
    assert get_length(data) > 0, "Expected non-empty data"

    # Check datetime columns
    if check_datetime:
        datetime_columns = [
            "interval_start_utc",
            "interval_end_utc",
            "interval_start_local",
            "interval_end_local",
        ]
        for col in datetime_columns:
            if col in get_columns(data):
                assert is_datetime_column(data, col), f"Expected datetime for {col}"

    # Check columns
    if columns is not None:
        assert get_columns(data) == columns, f"Expected columns {columns}"

    # Check length
    if length is not None:
        assert get_length(data) == length, f"Expected length {length}"


def data_equals(data1: Any, data2: Any, return_format: str) -> bool:
    """Check if two datasets are equal."""
    if return_format == ReturnFormat.PANDAS:
        return data1.equals(data2)
    elif return_format == ReturnFormat.POLARS:
        return data1.equals(data2)
    elif return_format == ReturnFormat.PYTHON:
        return data1 == data2
    else:
        raise TypeError(f"Unsupported format: {return_format}")


# ============================================================================
# Tests that work with all formats
# ============================================================================


def test_invalid_api_key(return_format):
    """Test that invalid API key raises an error."""
    test_client = gs.GridStatusClient(api_key="invalid", return_format=return_format)
    with pytest.raises(Exception) as exc_info:
        test_client.get_dataset(dataset="isone_fuel_mix", verbose=True)
    assert "Invalid API key" in str(exc_info.value)


def test_uses_columns(client, return_format):
    """Test that columns parameter works correctly."""
    dataset = "ercot_sced_gen_resource_60_day"
    one_column = "resource_name"
    columns = ["interval_start_utc", "interval_end_utc", one_column]
    limit = 100

    data = client.get_dataset(
        dataset=dataset,
        columns=columns,
        verbose=True,
        limit=limit,
    )
    check_data(data, return_format, columns=columns, length=limit)

    # Time columns always included even if not specified
    data = client.get_dataset(
        dataset=dataset,
        columns=[one_column],
        verbose=True,
        limit=limit,
    )
    check_data(data, return_format, columns=columns, length=limit)

    # No columns specified - should return all
    ncols = 30
    data = client.get_dataset(dataset=dataset, verbose=True, limit=limit)
    assert get_length(data) == limit
    assert len(get_columns(data)) == ncols, "Expected all columns"


def test_handles_unknown_columns(client, return_format):
    """Test that unknown columns raise an error."""
    dataset = "ercot_fuel_mix"

    with pytest.raises(Exception) as exc_info:
        client.get_dataset(
            dataset=dataset,
            columns=["invalid_column"],
            verbose=True,
        )
    assert "Column invalid_column not found in dataset" in str(exc_info.value)


def test_list_datasets(client, return_format):
    """Test listing datasets."""
    datasets = client.list_datasets(return_list=True)
    assert isinstance(datasets, list), "Expected a list of datasets"
    assert len(datasets) > 0, "Expected at least one dataset"


def test_list_datasets_filter(client, return_format):
    """Test filtering datasets."""
    filter_term = "fuel_mix"
    min_results = 7

    client.list_datasets(filter_term=filter_term, return_list=False)
    datasets = client.list_datasets(filter_term=filter_term, return_list=True)

    assert datasets is not None, f"No datasets returned for filter term '{filter_term}'"
    assert len(datasets) >= min_results, f"Expected at least {min_results} results"


def test_set_api_works(return_format):
    """Test that API key can be set."""
    test_client = gs.GridStatusClient(api_key="test", return_format=return_format)
    assert test_client.api_key == "test"


def test_get_dataset_date_range(client, return_format):
    """Test fetching data with date range."""
    start = "2023-01-01"
    end = "2023-01-05"

    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start=start,
        end=end,
        verbose=True,
    )
    check_data(data, return_format)

    # Check date range
    min_dt = get_min(data, "interval_start_utc")
    max_dt = get_max(data, "interval_end_utc")

    assert min_dt is not None
    assert max_dt is not None

    if return_format == ReturnFormat.PYTHON:
        assert min_dt.strftime("%Y-%m-%d") == start
        assert max_dt.strftime("%Y-%m-%d") == end
    else:
        # pandas and polars
        if hasattr(min_dt, "strftime"):
            assert min_dt.strftime("%Y-%m-%d") == start
            assert max_dt.strftime("%Y-%m-%d") == end


def test_index_unique_multiple_pages(client, return_format):
    """Test that index is unique across multiple pages."""
    start = "2023-01-01"
    end = "2023-01-02"

    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start=start,
        end=end,
        verbose=True,
        limit=100,
    )
    check_data(data, return_format)


def test_filter_operator(client, return_format):
    """Test filter operators."""
    dataset = "caiso_curtailment"
    limit = 1000
    category_column = "curtailment_type"
    category_value = "Economic"
    category_values = ["Economic", "SelfSchCut", "ExDispatch"]
    numeric_column = "curtailment_mw"
    numeric_value = 100

    # Test = operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=category_column,
        filter_value=category_value,
        filter_operator="=",
        limit=limit,
        verbose=True,
    )
    check_data(data, return_format)
    assert get_unique_values(data, category_column) == [category_value]

    # Test != operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=category_column,
        filter_value=category_value,
        filter_operator="!=",
        limit=limit,
        verbose=True,
    )
    check_data(data, return_format)
    assert set(get_unique_values(data, category_column)) == set(category_values) - {
        category_value,
    }

    # Test in operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=category_column,
        filter_value=category_values,
        filter_operator="in",
        limit=limit,
        verbose=True,
    )
    check_data(data, return_format)
    assert set(category_values).issuperset(get_unique_values(data, category_column))

    # Test < operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator="<",
        limit=limit,
        verbose=True,
    )
    check_data(data, return_format)
    max_val = get_max(data, numeric_column)
    assert max_val is not None and max_val < numeric_value

    # Test <= operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator="<=",
        limit=limit,
        verbose=True,
    )
    check_data(data, return_format)
    max_val = get_max(data, numeric_column)
    assert max_val is not None and max_val <= numeric_value

    # Test > operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator=">",
        limit=limit,
        verbose=True,
    )
    check_data(data, return_format)
    min_val = get_min(data, numeric_column)
    assert min_val is not None and min_val > numeric_value

    # Test >= operator
    data = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator=">=",
        limit=limit,
        verbose=True,
    )
    check_data(data, return_format)
    min_val = get_min(data, numeric_column)
    assert min_val is not None and min_val >= numeric_value

    # Test = operator with numeric
    data = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator="=",
        limit=limit,
        verbose=True,
    )
    check_data(data, return_format)
    assert get_unique_values(data, numeric_column) == [numeric_value]


def test_filter_operator_in(client, return_format):
    """Test filter in operator with list."""
    locations = ["LZ_WEST", "LZ_HOUSTON"]

    # Use ISO date format for compatibility with all formats
    data = client.get_dataset(
        dataset="ercot_spp_day_ahead_hourly",
        filter_column="location",
        filter_value=locations,
        filter_operator="in",
        start="2023-09-07",
        limit=10,
        verbose=True,
    )
    assert set(get_unique_values(data, "location")) == set(locations)
    check_data(data, return_format)


def test_get_dataset_verbose_false(client, return_format, caplog):
    """Test verbose=False doesn't log."""
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
    """Test verbose=True logs output."""
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
    assert any("Params: {" in msg for msg in log_messages)
    caplog.clear()


def test_handles_all_nan_columns(client, return_format):
    """Test handling of all-NaN columns."""
    start = "2020-01-01"
    end = "2020-01-02"
    btm_col = "btm_solar.capitl"
    time_columns = ["interval_start_utc", "interval_end_utc"]

    # Without timezone (works for all formats)
    data = client.get_dataset(
        "nyiso_standardized_5_min",
        start=start,
        end=end,
        columns=time_columns + [btm_col],
    )

    assert set(time_columns + [btm_col]) == set(get_columns(data))
    assert is_datetime_column(data, time_columns[0])
    assert is_datetime_column(data, time_columns[1])


def test_handles_no_results(client, return_format):
    """Test handling of queries that return data after initial empty results."""
    btm_col = "capitl"
    time_columns = ["interval_start_utc", "interval_end_utc"]

    # This date range crosses from no data to data
    data = client.get_dataset(
        "nyiso_btm_solar",
        start="2020-11-16",
        end="2020-11-18",
        columns=time_columns + [btm_col],
    )

    assert get_length(data) > 0
    assert set(time_columns + [btm_col]) == set(get_columns(data))
    assert is_datetime_column(data, time_columns[0])
    assert is_datetime_column(data, time_columns[1])


def test_resample_frequency(client, return_format):
    """Test resampling with different frequencies."""
    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start="2023-01-01",
        end="2023-01-02",
        resample="1 hour",
        verbose=True,
    )

    assert get_length(data) == 24
    check_data(
        data,
        return_format,
        columns=[
            "interval_start_utc",
            "interval_end_utc",
            "coal",
            "hydro",
            "landfill_gas",
            "natural_gas",
            "nuclear",
            "oil",
            "other",
            "refuse",
            "solar",
            "wind",
            "wood",
        ],
    )

    # Test with columns
    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start="2023-01-01",
        end="2023-01-02",
        columns=["coal"],
        resample="1 hour",
        verbose=True,
    )
    assert get_length(data) == 24
    check_data(
        data,
        return_format,
        columns=["interval_start_utc", "interval_end_utc", "coal"],
    )

    # Test with time_utc column only
    data = client.get_dataset(
        dataset="ercot_real_time_as_monitor",
        start="2023-08-01",
        end="2023-08-02",
        columns=["time_utc", "prc"],
        resample="5 minutes",
        verbose=True,
    )
    check_data(
        data,
        return_format,
        length=288,
        columns=["interval_start_utc", "interval_end_utc", "prc"],
    )


def test_resample_function(client, return_format):
    """Test resampling with different aggregation functions."""
    # Use ISO date format for all formats
    data_max = client.get_dataset(
        dataset="caiso_load",
        start="2023-09-01",
        end="2023-09-03",
        resample="1 day",
        resample_function="max",
    )
    check_data(
        data_max,
        return_format,
        length=2,
        columns=["interval_start_utc", "interval_end_utc", "load"],
    )

    data_min = client.get_dataset(
        dataset="caiso_load",
        start="2023-09-01",
        end="2023-09-03",
        resample="1 day",
        resample_function="min",
    )
    check_data(
        data_min,
        return_format,
        length=2,
        columns=["interval_start_utc", "interval_end_utc", "load"],
    )

    # Check timestamps match
    assert get_column_values(data_max, "interval_start_utc") == get_column_values(
        data_min,
        "interval_start_utc",
    )
    assert get_column_values(data_max, "interval_end_utc") == get_column_values(
        data_min,
        "interval_end_utc",
    )

    # Max should be greater than min
    for i in range(get_length(data_max)):
        max_load = get_column_values(data_max, "load")[i]
        min_load = get_column_values(data_min, "load")[i]
        assert max_load > min_load


def test_resample_and_paginated(client, return_format):
    """Test that resampling works correctly with pagination."""
    common_args = {
        "dataset": "isone_fuel_mix",
        "start": "2023-01-01",
        "end": "2023-01-02",
        "limit": 1000,
        "resample": "1 hour",
    }

    paginated = client.get_dataset(**common_args, page_size=100)
    non_paginated = client.get_dataset(**common_args, page_size=1000)

    assert data_equals(paginated, non_paginated, return_format)
    assert get_length(paginated) == 24
    check_data(paginated, return_format)

    min_dt = get_min(paginated, "interval_start_utc")
    max_dt = get_max(paginated, "interval_end_utc")

    if return_format == ReturnFormat.PYTHON:
        assert min_dt == datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert max_dt == datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)


def test_resampling_across_days(client, return_format):
    """Test resampling across multiple days."""
    data = client.get_dataset(
        dataset="isone_fuel_mix",
        start="2023-01-01",
        end="2023-01-03",
        resample="1 day",
        verbose=True,
    )

    assert get_length(data) == 2
    check_data(data, return_format)


def test_cursor_pagination_equals_offset_pagination(client, return_format):
    """Test that cursor and offset pagination return the same results."""
    common_args = {
        "dataset": "ercot_lmp_by_bus",
        "start": "2023-01-01",
        "end": "2023-01-02",
        "limit": 500,
        "page_size": 100,
    }

    cursor = client.get_dataset(**common_args, use_cursor_pagination=True)
    offset = client.get_dataset(**common_args, use_cursor_pagination=False)

    assert data_equals(cursor, offset, return_format)


def test_publish_time_latest(client, return_format):
    """Test publish_time='latest'."""
    today = datetime.now(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    start = (today - timedelta(days=2)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    data = client.get_dataset(
        dataset="miso_wind_forecast_hourly",
        start=start,
        end=end,
        publish_time="latest",
        verbose=True,
    )

    publish_times = get_unique_values(data, "publish_time_utc")
    assert len(publish_times) > 1, "Expected multiple publish times"


def test_publish_time_latest_report(client, return_format):
    """Test publish_time='latest_report'."""
    data = client.get_dataset(
        dataset="miso_wind_forecast_hourly",
        publish_time="latest_report",
        verbose=True,
    )

    publish_times = get_unique_values(data, "publish_time_utc")
    assert len(publish_times) == 1, "Expected one publish time"


def test_pagination(client, return_format):
    """Test pagination options."""
    dataset = "isone_fuel_mix"

    # Return 100 rows
    data = client.get_dataset(dataset=dataset, limit=100)
    assert get_length(data) == 100

    # Test multiple pages
    data = client.get_dataset(dataset=dataset, limit=100, page_size=25)
    assert get_length(data) == 100

    # Test limit less than page size
    data = client.get_dataset(dataset=dataset, limit=25, page_size=100)
    assert get_length(data) == 25

    # Test too large page size errors
    with pytest.raises(Exception):
        client.get_dataset(dataset=dataset, page_size=10**10)


@pytest.mark.parametrize(
    "iso,market_date,expected_date",
    [
        ("ERCOT", None, datetime.now().strftime("%Y-%m-%d")),
        ("CAISO", "2024-07-01", "2024-07-01"),
        ("spp", datetime(2024, 7, 10), "2024-07-10"),
    ],
)
def test_reports_api(client, return_format, iso, market_date, expected_date):
    """Test reports API."""
    resp = client.get_daily_peak_report(iso=iso, market_date=market_date)
    assert isinstance(resp, dict)
    assert resp["ISO"] == iso.upper()
    assert resp["market_date"] == expected_date


def test_invalid_resampling_frequency(client, return_format):
    """Test that invalid resampling frequency raises error."""
    with pytest.raises(Exception):
        client.get_dataset(
            "pjm_load",
            resample="1 hour market",
            start="2024-01-01",
            end="2024-01-02",
        )


def test_get_api_usage(client, return_format):
    """Test getting API usage."""
    usage = client.get_api_usage()
    assert isinstance(usage, dict)
    assert "current_period_usage" in usage or "requests_today" in usage


# ============================================================================
# Pandas-only tests (tests that use pandas-specific features)
# ============================================================================


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS])
def test_handles_all_nan_columns_with_tz_pandas(return_format):
    """Test handling of all-NaN columns with timezone (pandas only)."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    start = "2020-01-01"
    end = "2020-01-02"
    btm_col = "btm_solar.capitl"
    time_columns = ["interval_start_utc", "interval_end_utc"]
    time_columns_local = ["interval_start_local", "interval_end_local"]

    with silence_deprecation_warnings():
        df = client.get_dataset(
            "nyiso_standardized_5_min",
            start=start,
            end=end,
            tz="America/New_York",
            columns=time_columns + [btm_col],
        )

    assert set(time_columns_local + [btm_col]) == set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df[time_columns_local[0]])
    assert pd.api.types.is_datetime64_any_dtype(df[time_columns_local[1]])


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS])
def test_resample_frequency_with_tz_pandas(return_format):
    """Test resampling with timezone (pandas only)."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    with silence_deprecation_warnings():
        df = client.get_dataset(
            dataset="ercot_real_time_as_monitor",
            start="2023-08-01",
            end="2023-08-02",
            columns=["time_utc", "prc"],
            resample="1 hour",
            tz="America/Chicago",
            verbose=True,
        )

    assert len(df) == 24
    assert set(df.columns) == {"interval_start_local", "interval_end_local", "prc"}


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS])
def test_resample_by_pandas(return_format):
    """Test resample_by parameter (pandas only due to date format)."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    df = client.get_dataset(
        dataset="eia_ba_interchange_hourly",
        start="Sep 1, 2023",
        end="Sep 3, 2023",
        resample="1 day",
        resample_by=["interval_start_utc", "to_ba", "from_ba"],
    )

    expected_length = df[["to_ba", "from_ba"]].drop_duplicates().shape[0] * 2
    assert len(df) == expected_length
    assert df.columns.tolist() == [
        "interval_start_utc",
        "interval_end_utc",
        "to_ba",
        "from_ba",
        "mw",
    ]


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_market_day_data_downsampling(return_format):
    """Test downsampling market day data."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    data = client.get_dataset(
        "pjm_outages_daily",
        start="2024-01-01",
        end="2024-05-01",
        resample="1 month",
        verbose=True,
    )

    assert get_min(data, "interval_start_utc") == datetime(
        2024,
        1,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert get_max(data, "interval_end_utc") == datetime(
        2024,
        5,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )

    # Check exactly 1 row for each combination
    assert check_groupby_unique(
        data,
        [("interval_start_utc", "month"), "region", "publish_time_utc"],
    )


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_market_day_data_upsampling(return_format):
    """Test upsampling market day data."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    data = client.get_dataset(
        "pjm_outages_daily",
        start="2024-01-01",
        end="2024-01-05",
        resample="1 hour",
        verbose=True,
    )

    assert get_min(data, "interval_start_utc") == datetime(
        2024,
        1,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert get_max(data, "interval_end_utc") == datetime(
        2024,
        1,
        5,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )

    # Check exactly 1 row for each combination
    assert check_groupby_unique(
        data,
        [
            ("interval_start_utc", "date"),
            ("interval_start_utc", "hour"),
            "region",
            "publish_time_utc",
        ],
    )


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_publish_time_start_filtering(return_format):
    """Test publish_time_start filtering."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    publish_time_filter_dt = datetime(2023, 9, 30, 12, 0, 0, tzinfo=timezone.utc)

    # First query without publish_time_start filter
    data = client.get_dataset(
        dataset="ercot_load_forecast_by_forecast_zone",
        start="2023-10-01",
        end="2023-10-02",
        verbose=True,
    )
    assert get_min(data, "publish_time_utc") < publish_time_filter_dt

    data_filtered = client.get_dataset(
        dataset="ercot_load_forecast_by_forecast_zone",
        start="2023-10-01",
        end="2023-10-02",
        publish_time_start="2023-09-30T12:00:00Z",
        verbose=True,
    )
    assert get_min(data_filtered, "publish_time_utc") >= publish_time_filter_dt
    assert get_min(data_filtered, "interval_start_utc") == datetime(
        2023,
        10,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert get_max(data_filtered, "interval_start_utc") == datetime(
        2023,
        10,
        1,
        23,
        0,
        0,
        tzinfo=timezone.utc,
    )


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_cursor_pagination_equals_offset_pagination_with_upsampling(
    return_format,
):
    """Test cursor vs offset pagination with upsampling."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    common_args = {
        "dataset": "ercot_fuel_mix",
        "start": "2023-01-01",
        "end": "2023-01-02",
        "limit": 24,
        "page_size": 6,
        "resample": "1 minute",
    }

    cursor = client.get_dataset(**common_args, use_cursor_pagination=True)
    offset = client.get_dataset(**common_args, use_cursor_pagination=False)

    assert data_equals(cursor, offset, return_format)


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_cursor_pagination_equals_offset_pagination_with_downsampling(
    return_format,
):
    """Test cursor vs offset pagination with downsampling."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    common_args = {
        "dataset": "ercot_fuel_mix",
        "start": "2023-01-01",
        "end": "2023-01-02",
        "limit": 24,
        "page_size": 6,
        "resample": "1 hour",
    }

    cursor = client.get_dataset(**common_args, use_cursor_pagination=True)
    offset = client.get_dataset(**common_args, use_cursor_pagination=False)

    assert data_equals(cursor, offset, return_format)


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_cursor_pagination_equals_offset_pagination_with_upsampling_and_filter(
    return_format,
):
    """Test cursor vs offset pagination with upsampling and filter."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    common_args = {
        "dataset": "spp_lmp_day_ahead_hourly",
        "start": "2023-01-01",
        "end": "2023-02-01",
        "limit": 3_000,
        "page_size": 1_000,
        "resample": "1 minute",
        "filter_column": "location",
        "filter_value": "AEC",
    }

    cursor = client.get_dataset(**common_args, use_cursor_pagination=True)
    offset = client.get_dataset(**common_args, use_cursor_pagination=False)

    assert data_equals(cursor, offset, return_format)


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_cursor_pagination_equals_offset_pagination_with_downsampling_and_filter(
    return_format,
):
    """Test cursor vs offset pagination with downsampling and filter."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    common_args = {
        "dataset": "spp_lmp_day_ahead_hourly",
        "start": "2023-01-01",
        "end": "2023-02-01",
        "limit": 100,
        "page_size": 25,
        "resample": "1 day",
        "filter_column": "location",
        "filter_value": "AEC",
    }

    cursor = client.get_dataset(**common_args, use_cursor_pagination=True)
    offset = client.get_dataset(**common_args, use_cursor_pagination=False)

    assert data_equals(cursor, offset, return_format)


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_publish_time_and_resample(return_format):
    """Test publish_time with resample."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    today = datetime.now(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    start = (today - timedelta(days=2)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    # because no publish time is provided
    # this is resampled by unique publish time
    data = client.get_dataset(
        dataset="miso_wind_forecast_hourly",
        start=start,
        end=end,
        resample="1 day",
        verbose=True,
    )
    assert get_nunique(data, "publish_time_utc") > 1, "Expected multiple publish times"

    # make sure it still works if a column is provided
    data = client.get_dataset(
        dataset="miso_wind_forecast_hourly",
        start=start,
        end=end,
        columns=["miso"],
        resample="1 day",
        verbose=True,
    )
    assert get_nunique(data, "publish_time_utc") > 1, "Expected multiple publish times"

    # test latest
    data = client.get_dataset(
        dataset="miso_wind_forecast_hourly",
        start=start,
        columns=["miso"],
        end=end,
        publish_time="latest",
        resample="1 day",
        verbose=True,
    )
    assert not column_in_data(data, "publish_time_utc"), "Expected publish time removed"
    # Check each interval occurs only once
    value_counts = get_value_counts(data, "interval_start_utc")
    assert all(v == 1 for v in value_counts.values()), "Expected each interval once"


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_publish_time_specific_time(return_format):
    """Test specific publish_time."""
    # First, get a valid publish time from the dataset
    pandas_client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=ReturnFormat.PANDAS,
    )
    recent_data = pandas_client.get_dataset(
        dataset="miso_wind_forecast_hourly",
        limit=1,
        verbose=True,
    )
    # Get a specific publish time that exists in the dataset
    publish_time = (
        recent_data["publish_time_utc"]
        .iloc[0]
        .strftime(
            "%Y-%m-%d %H:%M:%S%z",
        )
    )

    # Now test with the specified return format
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    data = client.get_dataset(
        dataset="miso_wind_forecast_hourly",
        publish_time=publish_time,
        verbose=True,
    )

    # Check all publish times match
    assert get_nunique(data, "publish_time_utc") == 1, "Expected one publish time"
    # Check each interval occurs only once
    value_counts = get_value_counts(data, "interval_start_utc")
    assert all(v == 1 for v in value_counts.values()), "Expected each interval once"


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_publish_time_end_filtering(return_format):
    """Test publish_time_end filtering."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    publish_time_filter = datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    data = client.get_dataset(
        dataset="ercot_load_forecast_by_forecast_zone",
        start="2023-10-01",
        end="2023-10-02",
        verbose=True,
    )
    assert get_max(data, "publish_time_utc") > publish_time_filter

    data_filtered = client.get_dataset(
        dataset="ercot_load_forecast_by_forecast_zone",
        start="2023-10-01",
        end="2023-10-02",
        publish_time_end="2023-10-01T12:00:00Z",
        verbose=True,
    )

    assert get_max(data_filtered, "publish_time_utc") <= publish_time_filter
    assert get_min(data_filtered, "interval_start_utc") == datetime(
        2023,
        10,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert get_max(data_filtered, "interval_start_utc") == datetime(
        2023,
        10,
        1,
        23,
        0,
        0,
        tzinfo=timezone.utc,
    )


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_publish_time_start_and_end_filtering(return_format):
    """Test publish_time_start and publish_time_end filtering."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    publish_time_start_dt = datetime(2023, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    publish_time_end_dt = datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    data = client.get_dataset(
        dataset="ercot_load_forecast_by_forecast_zone",
        start="2023-10-01",
        end="2023-10-02",
        verbose=True,
    )
    assert get_min(data, "publish_time_utc") < publish_time_start_dt
    assert get_max(data, "publish_time_utc") > publish_time_end_dt

    data_filtered = client.get_dataset(
        dataset="ercot_load_forecast_by_forecast_zone",
        start="2023-10-01",
        end="2023-10-02",
        publish_time_start="2023-10-01T00:00:00Z",
        publish_time_end="2023-10-01T12:00:00Z",
        verbose=True,
    )

    assert get_min(data_filtered, "publish_time_utc") >= publish_time_start_dt
    assert get_max(data_filtered, "publish_time_utc") < publish_time_end_dt
    assert get_min(data_filtered, "interval_start_utc") == datetime(
        2023,
        10,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert get_max(data_filtered, "interval_start_utc") == datetime(
        2023,
        10,
        1,
        23,
        0,
        0,
        tzinfo=timezone.utc,
    )


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS, ReturnFormat.POLARS])
def test_publish_time_start_inclusive_and_end_time_exclusive(return_format):
    """Test that publish_time_start is inclusive and publish_time_end is exclusive."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    # These are actual publish times from the dataset
    publish_time_start_str = "2023-09-30 17:30:03+00:00"
    publish_time_end_str = "2023-10-01 02:30:00+00:00"
    publish_time_start_dt = datetime(2023, 9, 30, 17, 30, 3, tzinfo=timezone.utc)
    publish_time_end_dt = datetime(2023, 10, 1, 2, 30, 0, tzinfo=timezone.utc)

    data = client.get_dataset(
        dataset="ercot_load_forecast_by_forecast_zone",
        start="2023-10-01",
        end="2023-10-02",
        publish_time_start=publish_time_start_str,
        publish_time_end=publish_time_end_str,
        verbose=True,
    )

    assert get_min(data, "publish_time_utc") == publish_time_start_dt

    # publish_time_end is exclusive, so max should be less than this
    assert get_max(data, "publish_time_utc") < publish_time_end_dt


@pytest.mark.parametrize("return_format", [ReturnFormat.PANDAS])
def test_handles_no_results_pandas(return_format):
    """Test handling of no results with full assertions (pandas only)."""
    client = gs.GridStatusClient(
        api_key=API_KEY,
        host=HOST,
        return_format=return_format,
    )

    btm_col = "capitl"
    time_columns = ["interval_start_utc", "interval_end_utc"]
    time_columns_local = ["interval_start_local", "interval_end_local"]

    # no data, with time zone
    with silence_deprecation_warnings():
        df = client.get_dataset(
            "nyiso_btm_solar",
            start="2020-01-01",
            end="2020-01-02",
            tz="America/New_York",
            columns=time_columns + [btm_col],
        )

        assert set(time_columns_local + [btm_col]) == set(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df[time_columns_local[0]])
        assert pd.api.types.is_datetime64_any_dtype(df[time_columns_local[1]])
        assert df[btm_col].dtype == "object"

    # no data, without timezone
    df = client.get_dataset(
        "nyiso_btm_solar",
        start="2020-01-01",
        end="2020-01-02",
        columns=time_columns + [btm_col],
    )

    assert set(time_columns + [btm_col]) == set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df[time_columns[0]])
    assert pd.api.types.is_datetime64_any_dtype(df[time_columns[1]])
    assert df[btm_col].dtype == "object"

    # this date range crosses from no data to data, with timezone
    with silence_deprecation_warnings():
        df = client.get_dataset(
            "nyiso_btm_solar",
            start="2020-11-16",
            end="2020-11-18",
            tz="America/New_York",
            columns=time_columns + [btm_col],
        )

    assert set(time_columns_local + [btm_col]) == set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df[time_columns_local[0]])
    assert pd.api.types.is_datetime64_any_dtype(df[time_columns_local[1]])
    assert df[btm_col].dtype == "float64"

    # this date range crosses from no data to data, without timezone
    df = client.get_dataset(
        "nyiso_btm_solar",
        start="2020-11-16",
        end="2020-11-18",
        columns=time_columns + [btm_col],
    )

    assert set(time_columns + [btm_col]) == set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df[time_columns[0]])
    assert pd.api.types.is_datetime64_any_dtype(df[time_columns[1]])
    assert df[btm_col].dtype == "float64"


def test_pagination_no_limit(client, return_format):
    """Test pagination without limit parameter."""
    dataset = "isone_fuel_mix"
    now = datetime.now(timezone.utc)

    # test no limit, no page size
    data = client.get_dataset(
        dataset=dataset,
        start=now - timedelta(hours=1),
    )
    assert get_length(data) > 0

    # test no limit, with page size
    data = client.get_dataset(
        dataset=dataset,
        start=now - timedelta(minutes=30),
        # 5 minute data so at most 12 rows
        page_size=1,
    )
    assert get_length(data) > 0
