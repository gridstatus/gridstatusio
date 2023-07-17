import os

import pandas as pd

import gridstatusio as gs

client = gs.GridStatusClient(api_key=os.getenv("GRIDSTATUS_API_KEY_TEST"))


def test_list_datasets():
    datasets = client.list_datasets(return_list=True)
    assert isinstance(datasets, list), "Expected a list of datasets"
    assert len(datasets) > 0, "Expected at least one dataset"


def test_list_datasets_filter():
    filter_term = "fuel_mix"
    min_results = 7
    datasets = client.list_datasets(filter_term=filter_term, return_list=True)
    assert (
        len(datasets) >= min_results
    ), f"Expected at least {min_results} results with filter term '{filter_term}'"


def _check_dataframe(df):
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.index.is_unique


def test_set_api_works():
    client = gs.GridStatusClient(api_key="test")
    assert client.api_key == "test"


# todo test require_only_kwargs
def test_get_dataset_date_range():
    start = "2023-01-01"
    end = "2023-01-05"
    df = client.get_dataset(
        dataset="isone_fuel_mix",
        start=start,
        end=end,
        verbose=True,
    )

    _check_dataframe(df)

    # make sure min of interval_start_utc equals start
    assert df["interval_start_utc"].min().strftime("%Y-%m-%d") == start
    assert df["interval_end_utc"].max().strftime("%Y-%m-%d") == end


def test_index_unique_multiple_pages():
    start = "2023-01-01"
    end = "2023-01-02"
    df = client.get_dataset(
        dataset="isone_fuel_mix",
        start=start,
        end=end,
        verbose=True,
        limit=100,
    )

    _check_dataframe(df)
