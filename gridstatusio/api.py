from io import StringIO

import pandas as pd
import requests
import tqdm

API_BASE = "https://data.gridstatus.io"

# decorator that requieres kwargs


def require_only_kwargs(func):
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            raise TypeError("Please use only keyword arguments.")
        return func(**kwargs)

    return wrapper


def get_api_key():
    import gridstatusio as gs

    try:
        return gs.api_key
    except AttributeError:
        raise AttributeError(
            "API key not set. Please set the api_key attribute.",
        )


def _make_request(url, verbose=False):
    """Make a request to the GridStatus.io API."""
    if verbose:
        print("Requesting: {}".format(url))

    headers = {"x-api-key": get_api_key()}
    response = requests.get(url, headers=headers)

    return response


@require_only_kwargs
def _get_single_csv_dataset(dataset: str, date, verbose=False):
    """Get a single dataset from the GridStatus.io API."""
    if date:
        date_str = date.strftime("%Y-%m-%d")
        url = "{}/{}/archive/{}.csv".format(API_BASE, dataset, date_str)
    else:
        url = "{}/{}/latest.csv".format(API_BASE, dataset)

    response = _make_request(url, verbose=verbose)
    df = pd.read_csv(StringIO(response.text))

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time")

    return df


@require_only_kwargs
def get_dataset(dataset: str, date=None, start=None, end=None, verbose=False):
    """Get a dataset from the GridStatus.io API.

    Args:
        dataset (str): The dataset to get. For example, "isone/fuel_mix_clean".
        date (str): The date to get. For example, "2020-01-01".
        start (str): The start date to get. For example, "2020-01-01".
        end (str): The end date to get. For example, "2020-01-05".
        verbose (bool): Whether to print verbose output.

    Returns:
        pandas.DataFrame: The dataset.
    """

    # require either date or start and end
    if date is None and (start is None or end is None):
        raise TypeError("Please provide either date or start and end.")

    if date is not None:
        date = pd.to_datetime(date)
        return _get_single_csv_dataset(
            dataset=dataset,
            date=date,
            verbose=verbose,
        )

    dfs = []
    dates = pd.date_range(start, end).tolist()
    for date in tqdm.tqdm(dates):
        df = _get_single_csv_dataset(
            dataset=dataset,
            date=date,
            verbose=verbose,
        )
        dfs.append(df)

    return pd.concat(dfs)


def list_datasets(verbose=False):
    """List all datasets available on the GridStatus.io API."""
    dataset_availability = "datasets/historical_data_availability"

    df = _get_single_csv_dataset(
        dataset=dataset_availability,
        date=None,
        verbose=verbose,
    )

    df["dataset"] = df["iso"] + "/" + df["dataset"]
    df = df[~df["hidden"]]
    df.drop(["iso", "hidden"], axis=1, inplace=True)

    return df
