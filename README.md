# GridStatus.io Hosted API

Python client for accessing the [GridStatusIO API](https://www.gridstatus.io/api).

## Installation

You can install GridStatus.io Hosted API by running the following command in your terminal or command prompt:

```bash
pip install gridstatusio
```

## Usage

### Set API Key

To use the library, you need to set your API key. You can do this by importing the gridstatusio module and setting the api_key attribute to your API key as shown below:

```python
import gridstatusio as gs

gs.api_key  = '<YOUR-API-KEY>'
```

If you don't have an API key, you can request one [here](https://www.gridstatus.io/api). 

### List available datasets

You can use the list_datasets() function to retrieve a list of all available datasets:

```python
gs.list_datasets()
```

### Retrieving a Specific Date

To retrieve data for a specific dataset on a specific date, you can use the get_dataset() function as shown below. This function returns data as a Pandas DataFrame

```python
gs.get_dataset(
    dataset="isone/fuel_mix_clean",
    date="2023-01-01",
)
```

### Retrieving Data for a Date Range

To retrieve data for a dataset over a range of dates, you can use the get_dataset() function as shown below:

```python
df = gs.get_dataset(
    dataset="isone/fuel_mix_clean",
    start="2023-01-01",
    end="2023-01-05",
)
```

## Open Source

If you prefer to use an open source library that fetches data directly from the source, you can check out the [github repo](https://github.com/kmax12/gridstatus). For more information on Hosted API vs Open Source API, please see this [guide](https://www.gridstatus.io/docs#section/Hosted-API-vs-Open-Source-API)

## Get Help

We'd love to answer any usage or data access questions! Please let us know by emailing us at contact@gridstatus.io