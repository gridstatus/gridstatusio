import os

import pandas as pd

import gridstatusio as gs

client = gs.GridStatusClient(
    api_key=os.getenv("GRIDSTATUS_API_KEY_TEST"),
    host="http://localhost:8000/v1",
)


def test_uses_columns():
    dataset = "ercot_sced_gen_resource_60_day"
    columns = ["interval_start_utc", "interval_end_utc", "resource_name"]
    max_rows = 100
    df = client.get_dataset(
        dataset=dataset,
        columns=columns,
        verbose=True,
        max_rows=max_rows,
    )
    assert set(columns).issubset(df.columns), "Expected only the specified columns"
    assert len(df) == max_rows, "Expected max_rows to be respected"

    # no columns specified
    ncols = 28
    df = client.get_dataset(dataset=dataset, verbose=True, max_rows=max_rows)
    assert df.shape == (max_rows, ncols), "Expected all columns"


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


def test_filter_operator():
    dataset = "caiso_curtailment"
    max_rows = 100
    category_column = "curtailment_type"
    category_value = "Economic"
    category_values = ["Economic", "SelfSchCut", "ExDispatch"]
    numeric_column = "curtailment_mw"
    numeric_value = 100

    df = client.get_dataset(
        dataset=dataset,
        filter_column=category_column,
        filter_value=category_value,
        filter_operator="=",
        max_rows=max_rows,
        verbose=True,
    )
    _check_dataframe(df)
    assert df["curtailment_type"].unique() == [category_value]

    df = client.get_dataset(
        dataset=dataset,
        filter_column=category_column,
        filter_value=category_value,
        filter_operator="!=",
        max_rows=max_rows,
        verbose=True,
    )
    _check_dataframe(df)
    assert set(df["curtailment_type"].unique()) == set(category_values) - {
        category_value,
    }

    df = client.get_dataset(
        dataset=dataset,
        filter_column=category_column,
        filter_value=category_values,
        filter_operator="in",
        max_rows=max_rows,
        verbose=True,
    )
    _check_dataframe(df)
    assert set(df["curtailment_type"].unique()) == set(category_values)

    # test numeric operators = ["<", "<=", ">", ">=", "="]

    df = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator="<",
        max_rows=max_rows,
        verbose=True,
    )

    _check_dataframe(df)
    assert df["curtailment_mw"].max() < numeric_value

    df = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator="<=",
        max_rows=max_rows,
        verbose=True,
    )

    _check_dataframe(df)
    assert df["curtailment_mw"].max() <= numeric_value

    df = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator=">",
        max_rows=max_rows,
        verbose=True,
    )

    _check_dataframe(df)
    assert df["curtailment_mw"].min() > numeric_value

    df = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator=">=",
        max_rows=max_rows,
        verbose=True,
    )

    _check_dataframe(df)
    assert df["curtailment_mw"].min() >= numeric_value

    df = client.get_dataset(
        dataset=dataset,
        filter_column=numeric_column,
        filter_value=numeric_value,
        filter_operator="=",
        max_rows=max_rows,
        verbose=True,
    )

    _check_dataframe(df)
    assert df["curtailment_mw"].unique() == [numeric_value]
