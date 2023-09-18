# Changelog

## v0.5.4 - Next

- Add `resample_by` and `resample_function` parameters to further specify resampling behavior

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
