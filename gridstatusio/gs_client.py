import io
import time

import gridstatus
import pandas as pd
import requests
from tabulate import tabulate
from termcolor import colored

from gridstatusio import __version__


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

    def get(self, url, params=None, verbose=False):
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

        if self.request_format == "json":
            data = response.json()
            df = pd.DataFrame(data["data"][1:], columns=data["data"][0])
        elif self.request_format == "csv":
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)

        has_next_page = response.headers["x-has-next-page"] == "true"

        return df, has_next_page

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

        df, has_next_page = self.get(url)

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
                    available_cols = dataset_metadata.get("available_cols")

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
        resample=None,
        limit=None,
        max_rows=None,
        tz=None,
        verbose=True,
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
            limit (int): The maximum number of rows to fetch at time.
                Defaults maximum allowed by the API.
            max_rows (int): The maximum number of rows to fetch.
                Defaults to None, which fetches all rows that match the request.
            tz (str): The timezone to convert utc timestamps to. Defaults to UTC.
            verbose (bool): If set to True or "info", prints additional information.
                If set to "debug", prints more additional debug information. If
                set to False, no additional information is printed. Defaults to True.

        Returns:
            pd.DataFrame: The dataset as a pandas dataframe
        """
        if tz is None:
            tz = "UTC"

        if start is not None:
            start = gridstatus.utils._handle_date(start, tz)

        if end is not None:
            end = gridstatus.utils._handle_date(end, tz)

        # handle pagination
        page = 1
        has_next_page = True
        dfs = []
        total_time = 0
        while has_next_page:
            start_time = time.time()

            params = {
                "start_time": start,
                "end_time": end,
                "limit": limit,
                "page": page,
                "max_rows": max_rows,
                "resample_frequency": resample,
            }
            url = f"{self.host}/datasets/{dataset}/query/"
            if filter_column is not None:
                if isinstance(filter_value, list) and filter_operator == "in":
                    filter_value = ",".join(filter_value)

                params["filter_column"] = filter_column
                params["filter_value"] = filter_value
                params["filter_operator"] = filter_operator

            if columns is not None:
                params["columns"] = ",".join(columns)

            # Log the fetching message
            log(f"Fetching Page {page}...", verbose, end="")

            df, has_next_page = self.get(url, params=params, verbose=verbose)

            dfs.append(df)

            response_time = time.time() - start_time
            total_time += response_time
            avg_time_per_page = total_time / page

            # Update the fetching message with the done message
            if page == 1:
                log(
                    f"Done in {round(response_time, 2)} seconds. ",
                    verbose,
                )

            else:
                log(
                    f"Done in {round(response_time, 2)} seconds. "
                    f"Total time: {round(total_time, 2)}s. "
                    f"Avg per page: {round(avg_time_per_page, 2)}s",
                    verbose,
                )

            page += 1

            # time.sleep(
            #     0.1
            # )  # Add a small delay to ensure the output is updated correctly

        log("", verbose=verbose)  # Add a newline for cleaner output

        df = pd.concat(dfs).reset_index(drop=True)

        # Print the additional information
        log(
            f"Total number of rows: {len(df)}",
            verbose=verbose,
        )

        # convert to datetime for any columns that end in _utc
        # or are of type object
        for col in df.columns:
            if df[col].dtype == "object" or col.endswith("_utc"):
                # if ends with _utc we assume it's a datetime

                is_definitely_datetime = False
                if col.endswith("_utc"):
                    is_definitely_datetime = True

                try:
                    # if it's definitely a datetime we try without a format
                    # otherwise we try but it must match the format
                    # to avoid converting non-datetime columns
                    date_format = "%Y-%m-%dT%H:%M:%S%z"
                    if is_definitely_datetime:
                        date_format = None

                    df[col] = pd.to_datetime(
                        df[col],
                        format=date_format,
                        utc=True,
                    )

                    # if all values are NaT and its not
                    # assume its not a datetime column
                    if not is_definitely_datetime and df[col].isnull().all():
                        df[col] = pd.NA
                        continue

                    if tz != "UTC":
                        df[col] = df[col].dt.tz_convert(tz)
                        # rename with _utc suffix
                        df = df.rename(
                            columns={col: col.replace("_utc", "") + "_local"},
                        )

                except ValueError:
                    pass

        return df


if __name__ == "__main__":
    import os

    client = GridStatusClient(api_key=os.getenv("GRIDSTATUS_API_KEY_TEST"))

    client.list_datasets(filter_term="lmp")
