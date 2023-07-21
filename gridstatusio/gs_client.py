import io
import time

import gridstatus
import pandas as pd
import requests
from tabulate import tabulate
from termcolor import colored, cprint

from gridstatusio import __version__


class GridStatusClient:
    def __init__(self, api_key=None, host="https://api.gridstatus.io/v1"):
        """Create a GridStatus.io API client

        Parameters:
            api_key (str): The API key to use for authentication.
            If not provided, the GRIDSTATUS_API_KEY environment variable will be used.

            host (str): The host to use for the API.
                Defaults to https://api.gridstatus.io
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

    def __repr__(self) -> str:
        return f"GridStatusClient(host={self.host})"

    def get(self, url, params=None, verbose=False):
        headers = {
            "x-api-key": self.api_key,
            # set client and version
            "x-client": "gridstatusio-python",
            "x-client-version": __version__,
        }

        if verbose:
            print(f"GET {url}")
            print(f"Params: {params}")
        response = requests.get(url, params=params, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        return response

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

        response = self.get(url)

        data = response.json()

        matched_datasets = []

        for dataset in data["data"]:
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

                    print(tabulate(dataset_table, headers=headers, tablefmt="pretty"))
                    print("\n")

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
        limit=10000,
        max_rows=None,
        tz=None,
        verbose=False,
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
                Defaults to 10000.
            max_rows (int): The maximum number of rows to fetch.
                Defaults to None, which fetches all rows that match the request.
            tz (str): The timezone to convert utc timestamps to. Defaults to UTC.
            verbose (bool): If set to True, prints out the number
                of rows fetched and the time taken to fetch them.

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
                "return_format": "csv",
                "page": page,
                "max_rows": max_rows,
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

            response = self.get(url, params=params, verbose=verbose)

            df = pd.read_csv(io.StringIO(response.text))
            dfs.append(df)

            has_next_page = response.headers["x-has-next-page"] == "true"
            response_time = time.time() - start_time
            total_time += response_time
            avg_time_per_page = total_time / page

            print(
                f"\rFetching page {page}: Time for last page: \
                    {round(response_time, 2)} seconds | "
                f"Average time per page: {round(avg_time_per_page, 2)} seconds",
                end="",
            )
            page += 1
            # time.sleep(
            #     0.1
            # )  # Add a small delay to ensure the output is updated correctly

        print()  # Add a newline for cleaner output

        df = pd.concat(dfs).reset_index(drop=True)

        # Print the additional information
        cprint(f"\nTotal number of rows: {len(df)}", "cyan")
        cprint(f"Total Time: {round(total_time, 2)} seconds", "cyan")
        cprint(
            f"Average time per page: {round(total_time / (page - 1), 2)} seconds",
            "cyan",
        )

        # convert to datetime for any columns that end in _utc
        # or are of type object
        for col in df.columns:
            if df[col].dtype == "object" or col.endswith("_utc"):
                try:
                    df[col] = pd.to_datetime(df[col], utc=True)
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
