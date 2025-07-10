import io
import time
import warnings
from datetime import datetime
from typing import cast

import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout
from tabulate import tabulate
from termcolor import colored

from gridstatusio import __version__, utils
from gridstatusio.utils import logger

# Define retriable HTTP status codes
RETRIABLE_STATUS_CODES = {
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# Define retriable exception types
RETRIABLE_EXCEPTIONS = (
    ConnectionError,
    Timeout,
)


class GridStatusClient:
    def __init__(
        self,
        api_key: str | None = None,
        host: str = "https://api.gridstatus.io/v1",
        request_format: str = "json",
        max_retries: int = 5,
        base_delay: float = 2.0,
        exponential_base: float = 2.0,
    ):
        """Create a GridStatus.io API client

        Parameters:
            api_key (str): The API key to use for authentication.
            If not provided, the GRIDSTATUS_API_KEY environment variable will be used.

            host (str): The host to use for the API.
                Defaults to https://api.gridstatus.io

            request_format (str): The format to use for requests. Options are "json"
                or "csv". Defaults to "json".

            max_retries (int): The maximum number of retries to attempt for retriable
                errors (rate limits, server errors, network timeouts). Defaults to 4.

            base_delay (float): Base delay in seconds for exponential backoff.
                Defaults to 2.0.

            exponential_base (float): Base for exponential backoff calculation.
                Defaults to 2.0.
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
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_base = exponential_base

        assert self.request_format in [
            "json",
            "csv",
        ], "request_format must be 'json' or 'csv'"

    def __repr__(self) -> str:
        return f"GridStatusClient(host={self.host})"

    def _get_with_retry(
        self,
        url: str,
        params: dict,
        headers: dict,
        verbose: bool | str = False,
    ) -> requests.Response:
        """Execute GET request with retry logic for retriable errors"""

        if verbose:
            logger.info(f"GET {url}")
            logger.info(f"Params: {params}")

        retries = 0
        response = None

        def _retry_delay_and_log(reason: str):
            delay = self.base_delay * (self.exponential_base**retries)
            delay = int(delay)

            logger.info(
                f"{reason}. Retrying in {delay} seconds. "
                f"Retry {retries + 1} of {self.max_retries}.",
            )
            time.sleep(delay)

        while retries <= self.max_retries:
            try:
                response = requests.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    break
                elif response.status_code in RETRIABLE_STATUS_CODES:
                    # sometimes there is special detail in the response
                    # for example, rate limited errors have a detail field
                    # that includes the rate limit details
                    detail = response.json().get("detail", response.text)
                    if retries >= self.max_retries:
                        raise Exception(
                            f"HTTP {response.status_code}: {detail}. "
                            f"Exceeded maximum number of retries",
                        )
                    _retry_delay_and_log(detail)
                    retries += 1
                else:
                    raise Exception(f"Error {response.status_code}: {response.text}")

            except RETRIABLE_EXCEPTIONS as e:
                if retries >= self.max_retries:
                    raise Exception(
                        f"Network error: {str(e)}. Exceeded maximum number of retries",
                    )
                _retry_delay_and_log(f"Network error ({type(e).__name__})")
                retries += 1

            except Exception:
                raise

        if response is None:
            raise RuntimeError("Response is None")

        return response

    def get(
        self,
        url: str,
        params: dict | None = None,
        verbose: bool | str = False,
        return_raw_response_json: bool = False,
    ) -> pd.DataFrame | dict | tuple[pd.DataFrame, dict | None, dict | None]:
        if params is None:
            params = {}

        headers = {
            "x-api-key": self.api_key,
            # set client and version
            "x-client": "gridstatusio-python",
            "x-client-version": __version__,
        }

        # NOTE: parameter name different for API
        # than for python client
        if "return_format" not in params:
            params["return_format"] = self.request_format

            if self.request_format == "json":
                params["json_schema"] = "array-of-arrays"

        response = self._get_with_retry(url, params, headers, verbose)

        if return_raw_response_json:
            return response.json()

        meta: dict | None = None
        dataset_metadata: dict | None = None
        df: pd.DataFrame

        if self.request_format == "json":
            data = response.json()
            df = pd.DataFrame(data["data"][1:], columns=data["data"][0])
            meta = data["meta"]
            dataset_metadata = data["dataset_metadata"]
        elif self.request_format == "csv":
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        else:
            raise ValueError(f"Unsupported request_format: {self.request_format}")

        return df, meta, dataset_metadata

    def list_datasets(
        self,
        filter_term: str | None = None,
        return_list: bool = False,
    ) -> list[dict] | None:
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
                            "Earliest available time (UTC)",
                            colored(dataset["earliest_available_time_utc"], "blue"),
                        ],
                        [
                            "Latest available time (UTC)",
                            colored(dataset["latest_available_time_utc"], "blue"),
                        ],
                    ]

                    num_rows = dataset.get("num_rows")
                    all_columns = [
                        col["name"] for col in dataset.get("all_columns", [])
                    ]

                    if num_rows is not None:
                        dataset_table.append(
                            ["Number of rows", colored(num_rows, "red")],
                        )
                    if all_columns:
                        dataset_table.append(
                            ["Available columns", ", ".join(all_columns)],
                        )
                    else:
                        dataset_table.append(
                            ["Available columns", "No available columns information."],
                        )

                    more_info_url = (
                        f"https://www.gridstatus.io/datasets/{dataset['id']}"
                    )
                    dataset_table.append(["More Info", more_info_url])

                    logger.info(
                        tabulate(dataset_table, headers=headers, tablefmt="pretty"),
                    )
                    logger.info("")

        if return_list:
            return matched_datasets

    def get_dataset(
        self,
        dataset: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        publish_time_start: str | pd.Timestamp | None = None,
        publish_time_end: str | pd.Timestamp | None = None,
        columns: list[str] | None = None,
        filter_column: str | None = None,
        filter_value: str | int | list[str] | None = None,
        filter_operator: str = "=",
        publish_time: str | None = None,
        resample: str | None = None,
        resample_by: str | list[str] | None = None,
        resample_function: str = "mean",
        limit: int | None = None,
        page_size: int | None = None,
        tz: str | None = None,
        timezone: str | None = None,
        verbose: bool | str = True,
        use_cursor_pagination: bool = True,
        sleep_time: int = 0,
    ):
        """Get a dataset from GridStatus.io API

        Parameters:
            dataset (str): The name of the dataset to fetch

            start (str): The start time of the data to fetch based on the dataset's
            time_index_column. If not provided, defaults to the earliest available time
            for the dataset.

            end (str): The end time of the data to fetch based on the dataset's
            time_index_column. If not provided, defaults to the latest available time
            for the dataset.

            publish_time_start (str): The start time of the data to fetch based on the
            dataset's publish_time_column. Data where
            publish_time_start >= publish_time_column will be returned.

            publish_time_end (str): The end time of the data to fetch based on the
            dataset's publish_time_column. Data where
            publish_time_end < publish_time_column will be returned.

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

            tz (str): DEPRECATED: please use 'timezone' parameter instead.
                The timezone to convert utc timestamps to.

            timezone (str): The timezone to use for returning results and resampling to
                frequencies one day or lower. When provided, the returned data will
                have both UTC and local time columns. If not provided, the returned data
                will have only UTC time columns. Defaults to None.

            verbose (bool): If set to True, prints additional information.
                If set to False, no additional information is printed. Defaults to True.

            use_cursor_pagination (bool): If set to True, uses cursor pagination on
                the server side to fetch data. Defaults to False. When False, the
                server will use page-based pagination which is generally slower
                for large datasets.

            sleep_time (int): The amount of time, in seconds, to wait between requests
                when requesting multiple pages of data. Can be used to slow request
                frequency to help avoid hitting API rate limits. Defaults to 0.

        Returns:
            pd.DataFrame: The dataset as a pandas dataframe
        """
        if tz:
            warnings.warn(
                "The 'tz' parameter is deprecated. Please use 'timezone' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if timezone:
                raise ValueError("'tz' and 'timezone' parameters cannot both be set.")

        if start is not None:
            start = utils.handle_date(start, tz)

        if end is not None:
            end = utils.handle_date(end, tz)

        if publish_time_start is not None:
            publish_time_start = utils.handle_date(publish_time_start, tz)

        if publish_time_end is not None:
            publish_time_end = utils.handle_date(publish_time_end, tz)

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
                "publish_time_start": publish_time_start,
                "publish_time_end": publish_time_end,
                "limit": limit,
                "page": page,
                "page_size": page_size,
                "resample_frequency": resample,
                "resample_by": (
                    ",".join(resample_by) if resample_by is not None else None
                ),
                "resample_function": resample_function if resample else None,
                "publish_time": publish_time,
                "timezone": timezone,
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

            logger.info(f"Fetching Page {page}...")

            df, meta, dataset_metadata = self.get(url, params=params, verbose=verbose)
            has_next_page = (
                meta.get("hasNextPage", False) if meta is not None else False
            )
            # Extract the cursor to send in the next request for cursor pagination
            cursor = meta.get("cursor") if meta is not None else None

            total_rows += len(df)

            dfs.append(df)

            response_time = time.time() - start_time
            total_time += response_time
            avg_time_per_page = total_time / page

            # Update the fetching message with the done message
            if page == 1:
                logger.info(f"Done in {round(response_time, 2)} seconds. ")

            else:
                logger.info(
                    f"Done in {round(response_time, 2)} seconds. "
                    f"Total time: {round(total_time, 2)}s. "
                    f"Avg per page: {round(avg_time_per_page, 2)}s",
                )

            if limit:
                # Calculate percentage of rows fetched
                pct = round((total_rows / limit) * 100, 2)
                logger.info(f"Total rows: {total_rows:,}/{limit:,} ({pct}% of limit)")

            page += 1
            time.sleep(sleep_time)

        df = pd.concat(dfs).reset_index(drop=True)

        logger.info(f"Total number of rows: {len(df)}")
        all_columns = (
            dataset_metadata.get("all_columns", [])
            if dataset_metadata is not None
            else []
        )

        # These are columns that are always datetimes. In some situations, we will
        # add these columns to a dataset even if they are not in the dataset metadata,
        # for example, when resampling.
        always_datetime_columns = [
            "interval_start_utc",
            "interval_end_utc",
        ]

        data_timezone = (
            dataset_metadata["data_timezone"] if dataset_metadata is not None else "UTC"
        )

        for col_name in df.columns:
            col_metadata = next(
                (col for col in all_columns if col["name"] == col_name),
                None,
            )

            if (col_metadata and col_metadata["is_datetime"]) or (
                col_name in always_datetime_columns
            ):
                # We need to parse all datetime columns in UTC before converting to
                # local columns because only UTC can handle DST changes.
                df[col_name] = pd.to_datetime(df[col_name], utc=True)

                # TODO: remove old behavior
                # If timezone is provided, returned data will have both local columns
                # and _utc columns. We will leave the _utc columns as is.
                if (tz and tz != "UTC") or (
                    timezone
                    and data_timezone != "UTC"
                    and not col_name.endswith("_utc")
                ):
                    df[col_name] = df[col_name].dt.tz_convert(tz or data_timezone)

                    if tz:
                        df = df.rename(
                            columns={col_name: col_name.replace("_utc", "") + "_local"},
                        )

        return df

    def get_daily_peak_report(
        self,
        iso: str,
        market_date: str | datetime | None = None,
    ) -> dict[str, object]:
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
        return cast(dict[str, object], self.get(url, return_raw_response_json=True))


if __name__ == "__main__":
    import os

    client = GridStatusClient(api_key=os.getenv("GRIDSTATUS_API_KEY_TEST"))

    client.list_datasets(filter_term="lmp")
