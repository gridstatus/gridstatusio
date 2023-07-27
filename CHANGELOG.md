# Changelog

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
