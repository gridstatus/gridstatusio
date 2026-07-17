<p align="center">
<img width=75% src="/gridstatus-header.png" alt="gridstatus logo" />
</p>
<p align="center">
  <a href="https://github.com/gridstatus/gridstatusio/actions/workflows/tests.yaml" target="_blank">
        <img src="https://github.com/gridstatus/gridstatusio/actions/workflows/tests.yaml/badge.svg" alt="Tests" />
    </a>
   <a href="https://badge.fury.io/py/gridstatusio" target="_blank">
        <img src="https://badge.fury.io/py/gridstatusio.svg?maxAge=2592000" alt="PyPI version">
    </a>
</p>

# GridStatus.io Hosted API — Python Client

`gridstatusio` is a Python client for the [GridStatus.io Hosted API](https://www.gridstatus.io/api), which provides historical and real-time electricity market data from North American ISOs through a single REST API.

Available datasets cover load and demand, fuel and generation mix, forecasts, locational marginal prices (LMPs), interchange, ancillary services, and more.

Browse 500+ datasets in the [Data Catalog](https://www.gridstatus.io/datasets).

Contents: [Installation](#installation) · [Why the hosted API?](#why-the-hosted-api) · [Quick example](#quick-example) · [Getting started](#getting-started) · [Return formats](#return-formats) · [API usage](#checking-your-api-usage) · [More examples](#more-examples)

## Installation

`gridstatusio` supports Python 3.10+. Install with uv or pip.

```bash
# Standard installation (includes pandas)
uv pip install gridstatusio

# With polars support (for polars DataFrames)
uv pip install gridstatusio[polars]

# With notebook support (for running example notebooks)
uv pip install gridstatusio[notebooks]

# With all optional dependencies
uv pip install gridstatusio[all]
```

## Why the hosted API?

The hosted API differs from the open-source [`gridstatus`](https://github.com/gridstatus/gridstatus) library in several ways:

| Hosted API | Open-source `gridstatus` |
|------------|--------------------------|
| Normalized data with consistent column names, timestamp formats, and DST handling where the underlying data supports it | Raw data directly from ISO sources |
| Single REST API | Source-specific integrations |
| Historical data queryable immediately | You build and maintain your own archive |
| Server-side filtering by time, columns, and supported field values, including `=` and `in` filters | Filtering depends on each source |

Use the open-source library when you want raw data directly from the ISOs with no account. Use this client when you want normalized, hosted data through a single API. See [pricing](https://www.gridstatus.io/pricing).

## Quick example

```python
from gridstatusio import GridStatusClient

client = GridStatusClient(api_key="<your_api_key>")
df = client.get_dataset(
    "ercot_fuel_mix",
    start="2024-06-01",
    end="2024-06-02",
    limit=1000,
)
```

## Getting started

1. **Get an API key.** Sign up for a Grid Status account and copy your key from the [Settings page](https://www.gridstatus.io/settings/api).
2. **Provide the key.** Set `export GRIDSTATUS_API_KEY=your_api_key`, or pass it directly: `GridStatusClient(api_key="<your_api_key>")`.
3. **Find a dataset.** Call `client.list_datasets()` or browse the [Data Catalog](https://www.gridstatus.io/datasets).

```python
from gridstatusio import GridStatusClient

client = GridStatusClient()  # reads GRIDSTATUS_API_KEY from the environment

client.list_datasets()
```

### Dataset metadata

Use `client.get_dataset_metadata(dataset_id)` to get a dataset's description, available time range, columns, and more. It always returns a dictionary, with timestamp fields parsed into timezone-aware datetimes:

```python
metadata = client.get_dataset_metadata("ercot_fuel_mix")

# {
#     "id": "ercot_fuel_mix",
#     "name": "ERCOT Fuel Mix",
#     "earliest_available_time_utc": datetime(2017, 1, 1, 6, 0, tzinfo=timezone.utc),
#     "all_columns": [{"name": "interval_start_utc", ...}, ...],
#     ...
# }
```

## Return formats

`get_dataset(...)` supports pandas, polars, or Python objects. Set the return format at the client level or per-call. Dataset metadata is always returned as a dictionary.

```python
from gridstatusio import GridStatusClient

# Set default format when creating the client
client = GridStatusClient(return_format="pandas")  # or "polars" or "python"

# Override format for a specific call
data = client.get_dataset("ercot_fuel_mix", limit=100, return_format="python")
```

| Format | Return Type | Description |
|--------|------------|-------------|
| `"pandas"` | `pd.DataFrame` | Pandas DataFrame with parsed datetime columns |
| `"polars"` | `pl.DataFrame` | Polars DataFrame with parsed datetime columns |
| `"python"` | `list[dict]` | List of dictionaries with parsed datetime columns |

If `return_format` is not specified, the client returns **pandas DataFrames** by default.

```python
# Python format → list of dicts:
# [
#     {"interval_start_utc": "2025-01-01T00:00:00+00:00", "coal": 1234.5, ...},
#     {"interval_start_utc": "2025-01-01T00:05:00+00:00", "coal": 1235.2, ...},
# ]
```

### Using without pandas (advanced)

Pandas is installed by default, but the library uses lazy loading so it's only imported when needed. In minimal environments, you can install without dependencies and use `return_format="python"` to avoid pandas entirely:

```bash
uv pip install gridstatusio --no-deps
uv pip install requests termcolor tabulate
```

```python
from gridstatusio import GridStatusClient

# Must explicitly set return_format="python" to avoid the pandas import
client = GridStatusClient(api_key="your_key", return_format="python")
data = client.get_dataset("ercot_fuel_mix", limit=100)
```

If you don't set `return_format="python"`, the client attempts to use pandas and raises an error if it isn't installed.

## Checking your API usage

```python
usage = client.get_api_usage()
```

Shows the limits for your API key, the start/end of the current usage period, and usage in the current period. A limit of `-1` means no limit.

The free plan allows 500,000 rows per month. You can view detailed usage stats in [Settings](https://www.gridstatus.io/settings/usage).

## Retry configuration

The API enforces per-second/minute/hour rate limits. The client retries rate limits (429), server errors (5xx), and network issues with exponential backoff. See the [Pricing Page](https://www.gridstatus.io/pricing) for specific limits.

```python
client = GridStatusClient(
    max_retries=3,        # Maximum retries (default: 5)
    base_delay=1.0,       # Base delay in seconds (default: 2.0)
    exponential_base=1.5, # Exponential backoff multiplier (default: 2.0)
)
```

Set `max_retries=0` to disable retries.

## Version check

The client checks PyPI for library updates on import. To disable this (e.g. in restricted environments), set:

```bash
export GSIO_SKIP_VERSION_CHECK=true
```

## More examples

- [Getting Started](Examples/1.%20Getting%20Started.ipynb)
- [Finding Hubs and Zones in Pricing Data](Examples/2.%20ISO%20Hubs.ipynb)
- [ERCOT Pricing Data](Examples/3.%20ERCOT%20Pricing%20Data.ipynb)
- [CAISO April Net Load Analysis](Examples/4.%20CAISO%20April%20Net%20Load.ipynb)
- [Stacked Net Load Visualization](Examples/5.%20Stacked%20Net%20Load%20Visualization.ipynb)
- [Resample Data to Different Frequencies](Examples/6.%20Resampling%20Data.ipynb)

## Resources

- [OpenAPI spec](https://api.gridstatus.io/openapi.json)
- [LLMs manifest](https://gridstatus.io/llms.txt)
- [Docs assistant](https://docs.gridstatus.io/)

## Get help

For usage or data-access questions, email contact@gridstatus.io.
