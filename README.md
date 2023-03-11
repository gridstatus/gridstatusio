# GridStatus.io Hosted API

Python client for accessing the [GridStatusIO API](https://www.gridstatus.io/api).

## Installation

Install the library using pip:

```bash
pip install gridstatusio
```


## Usage

### Set API Key
To use the library, first set your API key:

```python
import gridstatusio as gs

gs.api_key  = '<YOUR-API-KEY>'
```

if you don't have an API, request one [here](https://www.gridstatus.io/api). 

### List available datasets

```python
gs.list_datasets()
```

### Get specific date

Request a dataset at a specific date

```python
gs.get_dataset(
    dataset="isone/fuel_mix_clean",
    date="2023-01-01",
)
```

### Get data range

```python
df = gs.get_dataset(
    dataset="isone/fuel_mix_clean",
    start="2023-01-01",
    end="2023-01-05",
    verbose=True,
)
```

## Get Help

We'd love to answer any usage or data access questions! Please let us know by emailing us at contact@gridstatus.io