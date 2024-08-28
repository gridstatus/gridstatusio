import io
import time
from datetime import datetime
from typing import Dict, Union

import pandas as pd
import requests
from tabulate import tabulate
from termcolor import colored

from gridstatusio import __version__, utils


def log(msg, verbose, level="info", end="\n"):
    """Print a message if verbose matches the level"""
    if verbose is True:
        verbose = "info"

    # if verbose is debug, print everything
    if verbose == "debug":
        print(msg, end=end)
    elif verbose == "info" and level == "info":
        print(msg, end=end)


class GridStatusClient:
    def __init__(
        self,
        api_key=None,
        host="https://api.gridstatus.io/v1",
        request_format="json",
    ):
        """Create a GridStatus.io API client

        Parameters:
            api_key (str): The API key to use for authentication.
            If not provided, the GRIDSTATUS_API_KEY environment variable will be used.

            host (str): The host to use for the API.
                Defaults to https://api.gridstatus.io

            request_format (str): The format to use for requests. Options are "json"
                or "csv". Defaults to "json".
        """

        if api_key is None:
            import os

            api_key = os.environ.get("GRIDSTATUS_API_KEY")

        if api_key is None:
            raise Exception(
                "No API key provided. Either pass an api_key to the \
                GridStatusClient constructor or set the \
                GRIDSTATUS_API_KEY environment variable.",
            )

        self.api_key = api_key
        self.host = host
        self.request_format = request_format

        assert self.request_format in [
            "json",
            "csv",
        ], "request_format must be 'json' or 'csv'"

    def __repr__(self) -> str:
        return f"GridStatusClient(host={self.host})"

    def get(self, url, params=None, verbose=False, return_raw_response_json=False):
        if params is None:
            params = {}

        headers = {
            "x-api-key": self.api_key,
            # set client and version
            "x-client": "gridstatusio-python",
            "x-client-version": __version__,
        }

        # note
        # parameter name different for API
        # than for python client
        if "return_format" not in params:
            params["return_format"] = self.request_format

            if self.request_format == "json":
                params["json_schema"] = "array-of-arrays"

        log(f"\nGET {url}", verbose=verbose, level="debug")
        log(f"Params: {params}", verbose=verbose, level="debug")

        response = requests.get(url, params=params, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        if return_raw_response_json:
            return response.json()

        meta = None
        dataset_metadata = None

        if self.request_format == "json":
            data = response.json()
            df = pd.DataFrame(data["data"][1:], columns=data["data"][0])
            meta = data["meta"]
            dataset_metadata = data["dataset_metadata"]
        elif self.request_format == "csv":
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)

        return df, meta, dataset_metadata

    def list_datasets(self, filter_term=None, return_list=False):
        """List available datasets from the API,
        with optional filter and return list option.

        Parameters:
            filter_term (str, optional): The term to match against
                dataset id, name, or description, case-insensitive.
            return_list (bool, optional): If set to True, returns
                the filtered datasets as a list. Defaults to False.

        Returns:
            list, None: The filtered datasets as a list if
                return_list is set to True, otherwise None.
        """
        url = f"{self.host}/datasets/"

        df, _meta, _dataset_metadata = self.get(url)

        matched_datasets = []

        for dataset in df.to_dict("records"):
            dataset_description = dataset.get("description", "")
            if dataset_description is None:
                dataset_description = ""
            if filter_term is None or (
                filter_term.lower() in dataset["id"].lower()
                or filter_term.lower() in dataset["name"].lower()
                or filter_term.lower() in dataset_description.lower()
            ):
                matched_datasets.append(dataset)

                if not return_list:
                    headers = ["Key", "Value"]
                    dataset_table = [
                        ["Name", colored(dataset["name"], "cyan", attrs=["bold"])],
                        ["ID", colored(dataset["id"], "yellow")],
                        ["Description", colored(dataset_description, "green")],
                        [
                            "Earliest available time",
                            colored(dataset["earliest_available_time"], "blue"),
                        ],
                        [
                            "Latest available time",
                            colored(dataset["latest_available_time"], "blue"),
                        ],
                    ]

                    dataset_metadata = dataset.get("dataset_metadata", {})
                    num_rows = dataset_metadata.get("num_rows")
                    available_cols = [
                        col["name"]
                        for col in dataset_metadata.get("available_cols", [])
                    ]

                    if num_rows is not None:
                        dataset_table.append(
                            ["Number of rows", colored(num_rows, "red")],
                        )
                    if available_cols:
                        dataset_table.append(
                            ["Available columns", ", ".join(available_cols)],
                        )
                    else:
                        dataset_table.append(
                            ["Available columns", "No available columns information."],
                        )

                    more_info_url = (
                        f"https://www.gridstatus.io/datasets/{dataset['id']}"
                    )
                    dataset_table.append(["More Info", more_info_url])

                    log(
                        tabulate(dataset_table, headers=headers, tablefmt="pretty"),
                        True,
                    )
                    log("\n", True)

        if return_list:
            return matched_datasets

    def get_dataset(
        self,
        dataset,
        start=None,
        end=None,
        columns=None,
        filter_column=None,
        filter_value=None,
        filter_operator="=",
        publish_time=None,
        resample=None,
        resample_by=None,
        resample_function="mean",
        limit=None,
        page_size=None,
        tz=None,
        verbose=True,
        use_cursor_pagination=True,
    ):
        """Get a dataset from GridStatus.io API

        Parameters:
            dataset (str): The name of the dataset to fetch

            start (str): The start time of the data to fetch. If not provided,
                defaults to the earliest available time for the dataset.

            end (str): The end time of the data to fetch. If not provided,
                defaults to the latest available time for the dataset.

            columns (list): The columns to fetch. If not provided,
                defaults to all available columns.

            filter_column (str): The column to filter on

            filter_value (str): The value to filter on. If filter operator is "in",
                this should be a list of values.

            filter_operator (str): The operator to use for the filter.
                Defaults to "=". Possible values are "=",
                "!=", ">", "<", ">=", "<=", "in".

            publish_time (str): Controls the filtering based on the dataset's
                publish time. Possible values:
                - 'latest_report': Returns records only from the most
                    recently published report.
                - 'latest': For any given timestamp, fetches the most recently
                    reported data point associated with it.
                - timestamp str: Returns records that were published
                    at the provided timestamp.
                - None: No filtering based on publish time.

            resample (str): The frequency to resample the data to. For example,
                "30 minutes", "1 hour", "7 days", "1 year". Can be any integer
                followed by a time unit. Defaults to None, which does not resample

            resample_by (str or list): A column or list of columns to resample by.
                By default resamples by the time index column. If resample
                is None, this is ignored.

            resample_function (str): The function to use for resampling. Defaults to
                "mean". Possible values are "mean", "sum", "min", "max", "stddev",
                "count", "variance". If resample is None, this is ignored.

            limit (int): The maximum number of rows to return across entire result set.
                Defaults to None, which fetches all rows that match the request.

            page_size (int): The maximum number of rows to fetch per page.
                Defaults to None, which uses maximum allowed by subscription.

            tz (str): The timezone to convert utc timestamps to. Defaults to UTC.

            verbose (bool): If set to True or "info", prints additional information.
                If set to "debug", prints more additional debug information. If
                set to False, no additional information is printed. Defaults to True.

            use_cursor_pagination (bool): If set to True, uses cursor pagination on
                the server side to fetch data. Defaults to False. When False, the
                server will use page-based pagination which is generally slower
                for large datasets.

        Returns:
            pd.DataFrame: The dataset as a pandas dataframe
        """
        if tz is None:
            tz = "UTC"

        if start is not None:
            start = utils.handle_date(start, tz)

        if end is not None:
            end = utils.handle_date(end, tz)

        # handle pagination
        page = 1
        has_next_page = True
        dfs = []
        total_time = 0
        total_rows = 0

        # Initialize cursor to an empty string. If we are using cursor pagination, an
        # empty string indicates the first page.
        cursor = ""

        # This will be the same for all pages
        dataset_metadata = None

        while has_next_page:
            start_time = time.time()

            params = {
                "start_time": start,
                "end_time": end,
                "limit": limit,
                "page": page,
                "page_size": page_size,
                "resample_frequency": resample,
                "resample_by": (
                    ",".join(resample_by) if resample_by is not None else None
                ),
                "resample_function": resample_function if resample else None,
                "publish_time": publish_time,
            }

            # Setting the cursor value in the parameters tells the server to use
            # cursor pagination.
            if use_cursor_pagination:
                params["cursor"] = cursor

            url = f"{self.host}/datasets/{dataset}/query"
            # todo test this conditional
            if filter_column is not None or filter_value != "":
                if isinstance(filter_value, list) and filter_operator == "in":
                    filter_value = ",".join(filter_value)

                params["filter_column"] = filter_column
                params["filter_value"] = filter_value
                params["filter_operator"] = filter_operator

            if columns is not None:
                params["columns"] = ",".join(columns)

            # Log the fetching message
            log(f"Fetching Page {page}...", verbose, end="")

            df, meta, dataset_metadata = self.get(url, params=params, verbose=verbose)
            has_next_page = meta.get("hasNextPage", False)
            # Extract the cursor to send in the next request for cursor pagination
            cursor = meta.get("cursor")

            total_rows += len(df)

            dfs.append(df)

            response_time = time.time() - start_time
            total_time += response_time
            avg_time_per_page = total_time / page

            # Update the fetching message with the done message
            if page == 1:
                log(f"Done in {round(response_time, 2)} seconds. ", verbose)

            else:
                log(
                    f"Done in {round(response_time, 2)} seconds. "
                    f"Total time: {round(total_time, 2)}s. "
                    f"Avg per page: {round(avg_time_per_page, 2)}s",
                    verbose,
                )

            if limit:
                # Calculate percentage of rows fetched
                pct = round((total_rows / limit) * 100, 2)
                log(f"Total rows: {total_rows:,}/{limit:,} ({pct}% of limit)", verbose)

            page += 1

        log("", verbose=verbose)  # Add a newline for cleaner output

        df = pd.concat(dfs).reset_index(drop=True)

        # Print the additional information
        log(f"Total number of rows: {len(df)}", verbose=verbose)

        all_columns = dataset_metadata.get("all_columns", [])

        # These are columns that are always datetimes. In some situations, we will
        # add these columns to a dataset even if they are not in the dataset metadata,
        # for example, when resampling.
        always_datetime_columns = [
            "interval_start_utc",
            "interval_end_utc",
        ]

        for col_name in df.columns:
            col_metadata = next(
                (col for col in all_columns if col["name"] == col_name),
                None,
            )

            if (col_metadata and col_metadata["is_datetime"]) or (
                col_name in always_datetime_columns
            ):
                df[col_name] = pd.to_datetime(df[col_name], utc=True)

                if tz != "UTC":
                    df[col_name] = df[col_name].dt.tz_convert(tz)
                    # rename with _local suffix
                    df = df.rename(
                        columns={col_name: col_name.replace("_utc", "") + "_local"},
                    )

        return df

    def get_daily_peak_report(
        self,
        iso: str,
        market_date: Union[str, datetime, None] = None,
    ) -> Dict:
        """Get a daily peak report from the GridStatus.io API for the specified
        ISO on the specified date.

        Note: This report can only be generated by users who are on a Grid Status
        paid plan.

        Parameters:
            iso (str): The name of the iso for which to generate the report. Must
            be one of: CAISO, ERCOT, ISONE, MISO, NYISO, PJM, SPP

            market_date (str or date, optional): The market date for which to generate
                the report. If provided as a string, specify date as YYYY-MM-DD format.
                If not provided, defaults to the current date.

        Returns:
            dict: The daily peak report as a dict.
        """
        if market_date is None:
            market_date = datetime.today()

        if isinstance(market_date, datetime):
            market_date = market_date.strftime("%Y-%m-%d")

        url = f"{self.host}/reports/daily_peak/{iso}?date={market_date}"
        return self.get(url, return_raw_response_json=True)


if __name__ == "__main__":
    import os

    client = GridStatusClient(api_key=os.getenv("GRIDSTATUS_API_KEY_TEST"))

    client.list_datasets(filter_term="lmp")
