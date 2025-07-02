import os
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest
import requests

from gridstatusio.gs_client import GridStatusClient


class TestGridStatusClient:
    """Test the GridStatusClient class"""

    def test_init_with_api_key(self):
        """Test client initialization with explicit API key"""
        client = GridStatusClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.host == "https://api.gridstatus.io/v1"
        assert client.request_format == "json"
        assert client.max_retries == 3

    def test_init_with_environment_variable(self):
        """Test client initialization with environment variable"""
        with patch.dict(os.environ, {"GRIDSTATUS_API_KEY": "env_key"}):
            client = GridStatusClient()
            assert client.api_key == "env_key"

    def test_init_with_custom_parameters(self):
        """Test client initialization with custom parameters"""
        client = GridStatusClient(
            api_key="test_key",
            host="https://test.api.com",
            request_format="csv",
            max_retries=5,
        )
        assert client.api_key == "test_key"
        assert client.host == "https://test.api.com"
        assert client.request_format == "csv"
        assert client.max_retries == 5

    def test_init_no_api_key(self):
        """Test client initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                GridStatusClient()
            assert "No API key provided" in str(exc_info.value)

    def test_init_invalid_request_format(self):
        """Test client initialization with invalid request format"""
        with pytest.raises(AssertionError) as exc_info:
            GridStatusClient(api_key="test_key", request_format="invalid")
        assert "request_format must be 'json' or 'csv'" in str(exc_info.value)

    def test_repr(self):
        """Test client string representation"""
        client = GridStatusClient(api_key="test_key", host="https://test.api.com")
        assert repr(client) == "GridStatusClient(host=https://test.api.com)"

    @patch("requests.get")
    def test_get_success_json(self, mock_get):
        """Test successful GET request with JSON format"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["col1", "col2"],
                ["val1", "val2"],
                ["val3", "val4"],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {"test": "metadata"},
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df, meta, dataset_metadata = client.get("https://test.api.com/test")

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://test.api.com/test"
        assert call_args[1]["headers"]["x-api-key"] == "test_key"
        assert call_args[1]["headers"]["x-client"] == "gridstatusio-python"

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]
        assert meta == {"hasNextPage": False}
        assert dataset_metadata == {"test": "metadata"}

    @patch("requests.get")
    def test_get_success_csv(self, mock_get):
        """Test successful GET request with CSV format"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "col1,col2\nval1,val2\nval3,val4"
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key", request_format="csv")
        df, meta, dataset_metadata = client.get("https://test.api.com/test")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]

    @patch("requests.get")
    def test_get_return_raw_response_json(self, mock_get):
        """Test GET request with return_raw_response_json=True"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        result = client.get("https://test.api.com/test", return_raw_response_json=True)

        assert result == {"test": "data"}

    @patch("requests.get")
    def test_get_rate_limit_retry_success(self, mock_get):
        """Test GET request with rate limit retry"""
        mock_response_429 = Mock()
        mock_response_429.status_code = 429

        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            "data": [["col1"], ["val1"]],
            "meta": {},
            "dataset_metadata": {},
        }

        mock_get.side_effect = [mock_response_429, mock_response_200]

        client = GridStatusClient(api_key="test_key", max_retries=3)
        df, meta, dataset_metadata = client.get("https://test.api.com/test", verbose=True)

        assert mock_get.call_count == 2
        assert isinstance(df, pd.DataFrame)

    @patch("requests.get")
    def test_get_rate_limit_max_retries_exceeded(self, mock_get):
        """Test GET request with rate limit exceeding max retries"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key", max_retries=2)
        with pytest.raises(Exception) as exc_info:
            client.get("https://test.api.com/test")
        assert "Rate limited. Exceeded maximum number of retries" in str(exc_info.value)

    @patch("requests.get")
    def test_get_http_error(self, mock_get):
        """Test GET request with HTTP error"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        with pytest.raises(Exception) as exc_info:
            client.get("https://test.api.com/test")
        assert "Error 500: Internal Server Error" in str(exc_info.value)

    @patch("requests.get")
    def test_list_datasets_success(self, mock_get):
        """Test successful list_datasets call"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["id", "name", "description", "earliest_available_time_utc", "latest_available_time_utc", "num_rows", "all_columns"],
                ["test_dataset", "Test Dataset", "A test dataset", "2023-01-01", "2023-12-31", 1000, '[{"name": "col1"}]'],
            ],
            "meta": {},
            "dataset_metadata": {},
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        result = client.list_datasets(return_list=True)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "test_dataset"

    @patch("requests.get")
    def test_list_datasets_with_filter(self, mock_get):
        """Test list_datasets with filter term"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["id", "name", "description", "earliest_available_time_utc", "latest_available_time_utc", "num_rows", "all_columns"],
                ["lmp_dataset", "LMP Dataset", "Load data", "2023-01-01", "2023-12-31", 1000, '[{"name": "col1"}]'],
                ["fuel_mix", "Fuel Mix", "Fuel mix data", "2023-01-01", "2023-12-31", 1000, '[{"name": "col1"}]'],
            ],
            "meta": {},
            "dataset_metadata": {},
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        result = client.list_datasets(filter_term="lmp", return_list=True)

        assert len(result) == 1
        assert result[0]["id"] == "lmp_dataset"

    @patch("requests.get")
    def test_list_datasets_with_none_description(self, mock_get):
        """Test list_datasets with None description"""  
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["id", "name", "description", "earliest_available_time_utc", "latest_available_time_utc", "num_rows", "all_columns"],
                ["test_dataset", "Test Dataset", None, "2023-01-01", "2023-12-31", 1000, '[{"name": "col1"}]'],
            ],
            "meta": {},
            "dataset_metadata": {},
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        result = client.list_datasets(return_list=True)

        assert len(result) == 1
        assert result[0]["description"] == ""

    @patch("requests.get")
    def test_get_dataset_single_page(self, mock_get):
        """Test get_dataset with single page response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
                ["2023-01-01T01:00:00Z", "2023-01-01T02:00:00Z", 200],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset("test_dataset", limit=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "interval_start_utc" in df.columns
        assert "interval_end_utc" in df.columns
        assert "value" in df.columns

    @patch("requests.get")
    def test_get_dataset_multiple_pages(self, mock_get):
        """Test get_dataset with multiple pages"""
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
            ],
            "meta": {"hasNextPage": True, "cursor": "next_cursor"},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }

        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T01:00:00Z", "2023-01-01T02:00:00Z", 200],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }

        mock_get.side_effect = [mock_response_1, mock_response_2]

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset("test_dataset", limit=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_get_dataset_with_filters(self, mock_get):
        """Test get_dataset with column and value filters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "category", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", "A", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "category", "is_datetime": False},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset(
            "test_dataset",
            columns=["interval_start_utc", "category"],
            filter_column="category",
            filter_value="A",
            filter_operator="=",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "category" in df.columns

    @patch("requests.get")
    def test_get_dataset_with_list_filter_value(self, mock_get):
        """Test get_dataset with list filter value for 'in' operator"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "category", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", "A", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "category", "is_datetime": False},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset(
            "test_dataset",
            filter_column="category",
            filter_value=["A", "B"],
            filter_operator="in",
        )

        assert isinstance(df, pd.DataFrame)
        call_args = mock_get.call_args
        assert call_args[1]["params"]["filter_value"] == "A,B"

    @patch("requests.get")
    def test_get_dataset_with_resampling(self, mock_get):
        """Test get_dataset with resampling parameters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T02:00:00Z", 150],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset(
            "test_dataset",
            resample="1 hour",
            resample_function="mean",
        )

        assert isinstance(df, pd.DataFrame)
        call_args = mock_get.call_args
        assert call_args[1]["params"]["resample_frequency"] == "1 hour"
        assert call_args[1]["params"]["resample_function"] == "mean"

    @patch("requests.get")
    def test_get_dataset_with_timezone(self, mock_get):
        """Test get_dataset with timezone parameter"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "America/New_York",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset("test_dataset", timezone="America/New_York")

        assert isinstance(df, pd.DataFrame)
        call_args = mock_get.call_args
        assert call_args[1]["params"]["timezone"] == "America/New_York"

    def test_get_dataset_deprecated_tz_parameter(self):
        """Test get_dataset with deprecated tz parameter"""
        client = GridStatusClient(api_key="test_key")
        
        with pytest.warns(DeprecationWarning):
            with patch.object(client, 'get') as mock_get:
                client.get_dataset("test_dataset", tz="America/New_York")
        
        call_args = mock_get.call_args
        assert call_args[1]["params"]["timezone"] == "America/New_York"

    def test_get_dataset_tz_and_timezone_both_set(self):
        """Test get_dataset with both tz and timezone parameters"""
        client = GridStatusClient(api_key="test_key")
        
        with pytest.raises(ValueError) as exc_info:
            client.get_dataset("test_dataset", tz="America/New_York", timezone="America/Chicago")
        assert "'tz' and 'timezone' parameters cannot both be set" in str(exc_info.value)

    @patch("requests.get")
    def test_get_dataset_with_sleep_time(self, mock_get):
        """Test get_dataset with sleep_time parameter"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        
        with patch('time.sleep') as mock_sleep:
            df = client.get_dataset("test_dataset", sleep_time=1)
            
            mock_sleep.assert_called_once_with(1)

    @patch("requests.get")
    def test_get_daily_peak_report(self, mock_get):
        """Test get_daily_peak_report method"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "iso": "ERCOT",
            "date": "2024-01-01",
            "peak_load": 75000,
            "peak_time": "2024-01-01T18:00:00Z",
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        result = client.get_daily_peak_report("ERCOT", "2024-01-01")

        assert result["iso"] == "ERCOT"
        assert result["date"] == "2024-01-01"
        assert result["peak_load"] == 75000

        call_args = mock_get.call_args
        assert "reports/daily_peak/ERCOT" in call_args[0][0]
        assert "date=2024-01-01" in call_args[0][0]

    def test_get_daily_peak_report_with_datetime(self):
        """Test get_daily_peak_report with datetime object"""
        client = GridStatusClient(api_key="test_key")
        
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"test": "data"}
            
            test_date = datetime(2024, 1, 1)
            result = client.get_daily_peak_report("ERCOT", test_date)
            
            call_args = mock_get.call_args
            assert "date=2024-01-01" in call_args[0][0]

    def test_get_daily_peak_report_with_none_date(self):
        """Test get_daily_peak_report with None date"""
        client = GridStatusClient(api_key="test_key")
        
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"test": "data"}
            
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.today.return_value = datetime(2024, 1, 1)
                result = client.get_daily_peak_report("ERCOT", None)
                
                call_args = mock_get.call_args
                assert "date=2024-01-01" in call_args[0][0]

    @patch("requests.get")
    def test_get_dataset_with_cursor_pagination(self, mock_get):
        """Test get_dataset with cursor pagination"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset("test_dataset", use_cursor_pagination=True)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["cursor"] == ""

    @patch("requests.get")
    def test_get_dataset_with_publish_time_filters(self, mock_get):
        """Test get_dataset with publish time filters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset(
            "test_dataset",
            publish_time_start="2023-01-01T00:00:00Z",
            publish_time_end="2023-01-02T00:00:00Z",
            publish_time="latest",
        )

        call_args = mock_get.call_args
        assert call_args[1]["params"]["publish_time_start"] == "2023-01-01T00:00:00Z"
        assert call_args[1]["params"]["publish_time_end"] == "2023-01-02T00:00:00Z"
        assert call_args[1]["params"]["publish_time"] == "latest"

    @patch("requests.get")
    def test_get_dataset_with_limit_and_page_size(self, mock_get):
        """Test get_dataset with limit and page_size parameters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset("test_dataset", limit=1000, page_size=100)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["limit"] == 1000
        assert call_args[1]["params"]["page_size"] == 100

    @patch("requests.get")
    def test_get_dataset_verbose_output(self, mock_get, capsys):
        """Test get_dataset verbose output"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset("test_dataset", verbose=True)

        captured = capsys.readouterr()
        assert "Fetching Page 1..." in captured.out
        assert "Done in" in captured.out
        assert "Total number of rows: 1" in captured.out

    @patch("requests.get")
    def test_get_dataset_debug_verbose_output(self, mock_get, capsys):
        """Test get_dataset with debug verbose level"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                ["interval_start_utc", "interval_end_utc", "value"],
                ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", 100],
            ],
            "meta": {"hasNextPage": False},
            "dataset_metadata": {
                "all_columns": [
                    {"name": "interval_start_utc", "is_datetime": True},
                    {"name": "interval_end_utc", "is_datetime": True},
                    {"name": "value", "is_datetime": False},
                ],
                "data_timezone": "UTC",
            },
        }
        mock_get.return_value = mock_response

        client = GridStatusClient(api_key="test_key")
        df = client.get_dataset("test_dataset", verbose="debug")

        captured = capsys.readouterr()
        assert "GET" in captured.out
        assert "Params:" in captured.out
