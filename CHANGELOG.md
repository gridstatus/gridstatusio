# Changelog

## 0.13.0 - July 10, 2025

 - **Enhanced retry logic with exponential backoff**: 
   - The client now retries on additional HTTP status codes (500, 502, 503, 504) and network exceptions (ConnectionError, Timeout), not just 429 rate limit errors
   - Added `base_delay` (default: 2.0 seconds) and `exponential_base` (default: 2.0) for customizable exponential backoff timing
   - Increased default `max_retries` from 3 to 5 for better resilience
   - Enhanced logging shows specific error details and retry timing information

- ISONE LMP Real Time now defaults to `_prelim` dataset
- Bump `setuptools` to latest version


## 0.12.0 - July 3, 2025

- Add support for Python 3.13
- Drop support for Python 3.9
- Bump minimum pandas version to 2.3.0
- Add support for pyright type checking in CI and pre-commit
- Add function type hints
- Update README links

## 0.11.0 - May 30, 2025

- Add support for filtering data by publish time
- Add support for specifying `timezone=market` when querying datasets to return local timestamps in the ISO timezone

## 0.10.1 - May 9, 2025

- Switch to `uv` for package management

## 0.10.0 - March 11, 2025

- Increases version to get new dataset column namings from API
- Update to this version if you want to get the new column names early
- After 2025-03-36T15:00:00Z, the new column namings will be the default behavior
- See the [docs changelog](https://docs.gridstatus.io/changelog/march-2025/11-march-add-_utc-suffix-to-dataset-columns-where-it-is-missing) for more information

### Breaking Changes

- Changes in datasets

| dataset_id                                       | old_column_name         | new_column_name             |
|--------------------------------------------------|-------------------------|-----------------------------|
| caiso_curtailed_non_operational_generator_report | curtailment_end_time    | curtailment_end_time_utc    |
| caiso_curtailed_non_operational_generator_report | curtailment_start_time  | curtailment_start_time_utc  |
| ercot_indicative_lmp_by_settlement_point         | rtd_timestamp           | rtd_timestamp_utc           |
| ercot_lmp_by_bus                                 | sced_timestamp          | sced_timestamp_utc          |
| ercot_lmp_by_settlement_point                    | sced_timestamp          | sced_timestamp_utc          |
| ercot_lmp_with_adders_by_settlement_point        | sced_timestamp          | sced_timestamp_utc          |
| ercot_real_time_adders_and_reserves              | sced_timestamp          | sced_timestamp_utc          |
| ercot_sced_gen_resource_60_day                   | sced_time_stamp         | sced_timestamp_utc          |
| ercot_sced_load_resource_60_day                  | sced_time_stamp         | sced_timestamp_utc          |
| ercot_sced_smne_60_day                           | interval_time           | interval_time_utc           |
| ercot_sced_system_lambda                         | sced_time_stamp         | sced_timestamp_utc          |
| ercot_shadow_prices_sced                         | sced_timestamp          | sced_timestamp_utc          |
| ercot_spp_day_ahead_price_corrections            | price_correction_time   | price_correction_time_utc   |
| ercot_spp_real_time_price_corrections            | price_correction_time   | price_correction_time_utc   |
| ercot_unplanned_resource_outages                 | actual_end_date         | actual_end_date_utc         |
| ercot_unplanned_resource_outages                 | actual_outage_start     | actual_outage_start_utc     |
| ercot_unplanned_resource_outages                 | current_as_of           | current_as_of_utc           |
| ercot_unplanned_resource_outages                 | planned_end_date        | planned_end_date_utc        |
| ieso_adequacy_report_forecast                    | last_modified           | last_modified_utc           |
| pjm_lmp_it_sced_5_min                            | case_approval_time      | case_approval_time_utc      |

- Dataset metadata changes

| old_column_name         | new_column_name             |
|-------------------------|-----------------------------|
| earliest_available_time | earliest_available_time_utc |
| last_checked_time       | last_checked_time_utc       |
| latest_available_time   | latest_available_time_utc   |


## 0.9.0 - January 22, 2025

- Adds a `timezone` parameter used for returning results in local time and resampling results to frequencies one day or lower
  - With this addition, the `tz` parameter is deprecated and will be removed in a future release
  - When using the `timezone` parameter, the `_local` columns will be in the timezone and `_utc` columns will be in UTC

__With `tz` (deprecated)__:

```python
client.get_dataset(
    dataset='ercot_load',
    start=start,
    end=end,
    tz='America/New_York'
)
```

|    | interval_start_local      | interval_end_local        |   load |
|---:|:--------------------------|:--------------------------|-------:|
|  0 | 2025-01-01 00:00:00-05:00 | 2025-01-01 00:05:00-05:00 |  44859 |
|  1 | 2025-01-01 00:05:00-05:00 | 2025-01-01 00:10:00-05:00 |  44993 |
|  2 | 2025-01-01 00:10:00-05:00 | 2025-01-01 00:15:00-05:00 |  44894 |
|  3 | 2025-01-01 00:15:00-05:00 | 2025-01-01 00:20:00-05:00 |  44863 |
|  4 | 2025-01-01 00:20:00-05:00 | 2025-01-01 00:25:00-05:00 |  44845 |

__With `timezone`__:

```python
client.get_dataset(
    dataset='ercot_load',
    start=start,
    end=end,
    timezone='America/New_York'
)
```

|    | interval_start_local      | interval_start_utc        | interval_end_local        | interval_end_utc          |   load |
|---:|:--------------------------|:--------------------------|:--------------------------|:--------------------------|-------:|
|  0 | 2025-01-01 00:00:00-05:00 | 2025-01-01 05:00:00+00:00 | 2025-01-01 00:05:00-05:00 | 2025-01-01 05:05:00+00:00 |  44859 |
|  1 | 2025-01-01 00:05:00-05:00 | 2025-01-01 05:05:00+00:00 | 2025-01-01 00:10:00-05:00 | 2025-01-01 05:10:00+00:00 |  44993 |
|  2 | 2025-01-01 00:10:00-05:00 | 2025-01-01 05:10:00+00:00 | 2025-01-01 00:15:00-05:00 | 2025-01-01 05:15:00+00:00 |  44894 |
|  3 | 2025-01-01 00:15:00-05:00 | 2025-01-01 05:15:00+00:00 | 2025-01-01 00:20:00-05:00 | 2025-01-01 05:20:00+00:00 |  44863 |
|  4 | 2025-01-01 00:20:00-05:00 | 2025-01-01 05:20:00+00:00 | 2025-01-01 00:25:00-05:00 | 2025-01-01 05:25:00+00:00 |  44845 |

## 0.8.0 - September 25, 2024

- Adds automatic retries using an exponential backoff when fetching data if an API rate limit is hit. Also adds parameters for configuring `max_retries` on `GridStatusClient` and an optional `sleep_time` parameter on `GridStatusClient.get_dataset`.

## 0.7.0 - August 28, 2024

- Updates for resampling changes on server. Upsampling is now supported.

## 0.6.5 - August 6, 2024

- Remove `gridstatus` and add `tabulate` as dependencies.

## 0.6.4 - August 2, 2024

- Updates pandas version to allow for pandas 2.0 compatibility

## 0.6.3 - July 15, 2024

- Adds a new `client.get_daily_peak_report()` method for accessing daily peak LMP/load data for a specific ISO and date.

## 0.6.2 - June 18, 2024

- Sets capped version of Python to <4
- Upgrades numpy to a version that supports Python 3.12

## 0.6.1 - June 17, 2024

- Adds support for Python 3.12

## v0.6.0 - June 17, 2024

- Add ability to use cursor-based pagination instead of offset-based pagination.
  - Cursor-based pagination can be 30-50% faster than offset-based pagination for large datasets.
  - Cursor-based pagination is the default for `client.get_dataset()`.
- Drops support for Python 3.8

## v0.5.9 - March 26, 2024

- Fix warning to upgrade when already on latest version

## v0.5.8 - March 14, 2024

- [Does not affect usage]. Switch to `poetry` for package management

## v0.5.7 - March 14, 2024

- Fix bug where `GridStatusClient.list_datasets` would error when `return_list=False`

## v0.5.6 - Dec 22, 2023

- Rename `max_rows` to `page_size`

## v0.5.5 - Dec 2, 2023

- Add support for querying based on `publish_time` to `client.get_dataset`

## v0.5.4 - Sept 17, 2023

- Add `resample_by` and `resample_function` parameters to further specify resampling behavior. Example notebook: [Resample Data to Different Frequencies](/Examples/Resample%20Data.ipynb)

## v0.5.3 - Sept 15, 2023

- Support data resampling with `resample` parameter

## v0.5.2 - Aug 31, 2023

### Bug Fixes

- Fix date parsing in cases where there is no data or all missing values

## v0.5.1 - Aug 22, 2023

### Enhancements

- Checks for an updated version of gridstatusio library at import and alerts user if a newer version is available.

## v0.5.0 - Aug 22, 2023

### Examples

- Add stacked net load visualization example notebook

### Enhancements

- When using `client.get_dataset()`, the number of rows per request now defaults to maximum allowed by your API key. You can specify a lower limit using the `limit` parameter. We recommend using the default for maximum performance.

### Bug Fixes

- Fix date parsing in older versions of pandas

## v0.4.0 - July 31, 2023

### Enhancements

- Improve dataset download times by switching to json return format from API

### Bug Fixes

- Fix mixed dtype warning

## v0.3.1 - July 27, 2023

### Enhancements

- Revised the functionality of the verbose flag. Now, when it's set to False, no output is generated. If set to True or "info", it provides a moderate level of information. For the most detailed output, set it to "debug".

## v0.3.0 - July 21, 2023

### Enhancements

- Specifiy a filter operator when using filtering. Supports "=", "!=", ">", "<", ">=", "<=", "in".
- Specify subset of columns to query

## v0.2.3 - July 17th, 2023

### Bug Fixes

- Fix parsing of Forecast Time column in forecast datasets
- Fix duplicated dataframe index when querying across multiple pages
