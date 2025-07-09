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

# GridStatus.io Hosted API

Python client for accessing the [GridStatus.io Hosted API](https://www.gridstatus.io/api). 

Browse all available datasets in our [Data Catalog](https://www.gridstatus.io/datasets).


## Installation

`gridstatusio` supports Python 3.10+. Install with pip:

```bash
pip install gridstatusio
```

Upgrade using the following command:

```bash
pip install --upgrade gridstatusio
```

## Getting Started

* Sign up for a Grid Status account and get your API key from the [Settings page](https://www.gridstatus.io/settings/api)
* Set your API key as an environment variable: `export GRIDSTATUS_API_KEY=your_api_key`
* **NOTE**: the Grid Status API has a 1 million rows per month limit on the free plan. This limit is _very_ easy to exceed when querying data, especially real time prices.
  * Make sure to add `limit` to all of your `get_dataset` calls to avoid quickly exceeding the limit.
* The Grid Status API has rate limits that restrict the number of requests that are allowed each second, minute and hour. If rate limits are hit the client will automatically retry the request after a delay. You can configure the maximum number of retries using the `max_retries` parameter when initializing the client. If you find yourself hitting rate limits, you may need to add a delay between your requests. The [Grid Status Pricing Page](https://www.gridstatus.io/pricing) contains more details on specific rate limits.

Check out this example notebook: [Getting Started](/Examples/Getting%20Started.ipynb)

Other Examples:

- [ERCOT Pricing Data](/Examples/ERCOT%20Pricing%20Data.ipynb)
- [Finding Hubs and Zones in Pricing Data](/Examples/ISO%20Hubs.ipynb)
- [Stacked Net Load Visualization](/Examples/Stacked%20Net%20Load%20Visualization.ipynb)
- [CAISO April Net Load Analysis](/Examples/CAISO%20April%20Net%20Load.ipynb)
- [Resample Data to Different Frequencies](/Examples/Resample%20Data.ipynb)

## Retry Configuration

The client retries failed requests due to rate limits (429), server errors (5xx), and network issues using exponential backoff. You can customize retry behavior:

```python
client = GridStatusClient(
    max_retries=3,        # Maximum retries (default: 5)
    base_delay=1.0,       # Base delay in seconds (default: 2.0)
    exponential_base=1.5, # Exponential backoff multiplier (default: 2.0)
)
```

The retry delay follows the formula `delay = base_delay * (exponential_base ** retry_count)`.

Retries are useful when:

* You're making pagination-heavy requests and risk hitting short-term rate limits
* A request fails due to a temporary server error
* A network issue or timeout interrupts the request

To disable retries entirely, set `max_retries=0`.

## Open Source

If you prefer to use an open source library that fetches data directly from the source, you can check out this [github repo](https://github.com/gridstatus/gridstatus). 

## Get Help

We'd love to answer any usage or data access questions! Please let us know by emailing us at contact@gridstatus.io
