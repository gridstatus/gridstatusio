import pandas as pd
import pytest

import gridstatusio as gs


def _check_dataframe(df):
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_api_key_must_be_set():
    with pytest.raises(AttributeError):
        gs.get_api_key()


def test_set_api_works():
    gs.api_key = "test"
    assert gs.get_api_key() == "test"


# todo test require_only_kwargs


def test_get_dataset_date():
    gs.api_key = "TQfPZ3zOQ65Z7AbI1jMsMaLJvjcwAuyG85WjKNJI"

    df = gs.get_dataset(
        dataset="isone/fuel_mix_clean",
        date="2023-01-01",
    )

    _check_dataframe(df)


def test_get_dataset_date_range():
    gs.api_key = "TQfPZ3zOQ65Z7AbI1jMsMaLJvjcwAuyG85WjKNJI"

    df = gs.get_dataset(
        dataset="isone/fuel_mix_clean",
        start="2023-01-01",
        end="2023-01-05",
        verbose=True,
    )

    _check_dataframe(df)


def test_list_datasets():
    gs.api_key = "TQfPZ3zOQ65Z7AbI1jMsMaLJvjcwAuyG85WjKNJI"

    df = gs.list_datasets()

    cols = [
        "dataset",
        "earliest",
        "latest",
        "num_missing",
        "earliest_no_missing",
        "missing",
    ]

    assert df.columns.tolist() == cols
