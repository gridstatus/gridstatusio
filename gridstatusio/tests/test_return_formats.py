"""Tests for optional dependency handling and return formats."""

import os

import pandas as pd
import polars as pl
import pytest

import gridstatusio as gs
from gridstatusio._compat import (
    MissingDependencyError,
    pandas_available,
    polars_available,
)

# Get API key - check both possible environment variable names
API_KEY = os.getenv("GRIDSTATUS_API_KEY_TEST") or os.getenv("GRIDSTATUS_API_KEY")


class TestCompatModule:
    """Test the _compat module utilities."""

    def test_pandas_available_when_installed(self):
        """Test that pandas_available returns True when pandas is installed."""
        # In test environment, pandas should be available
        assert pandas_available() is True

    def test_polars_available_when_installed(self):
        """Test that polars_available returns True when polars is installed."""
        # In test environment, polars should be available
        assert polars_available() is True

    def test_missing_dependency_error_message(self):
        """Test that MissingDependencyError has correct message format."""
        err = MissingDependencyError("pandas", "pandas")
        assert "pandas" in str(err)
        assert "pip install" in str(err)
        assert "gridstatusio[pandas]" in str(err)

    def test_missing_dependency_error_attributes(self):
        """Test that MissingDependencyError has correct attributes."""
        err = MissingDependencyError("polars", "polars")
        assert err.library == "polars"
        assert err.format_name == "polars"


class TestReturnFormatValidation:
    """Test return format validation."""

    @pytest.fixture
    def client(self):
        return gs.GridStatusClient(api_key=API_KEY)

    def test_valid_pandas_format(self, client):
        """Test that pandas format is valid when pandas is available."""
        # Should not raise
        client._validate_return_format("pandas")

    def test_valid_python_format(self, client):
        """Test that python format is always valid."""
        # Should not raise
        client._validate_return_format("python")

    def test_valid_polars_format(self, client):
        """Test that polars format is valid when polars is available."""
        # Should not raise
        client._validate_return_format("polars")

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="'invalid_format' is not a valid"):
            gs.GridStatusClient(api_key=API_KEY, return_format="invalid_format")


class TestClientReturnFormatDefault:
    """Test client default return format behavior."""

    def test_default_return_format_is_pandas_when_available(self):
        """Test that default return_format is pandas when pandas is installed."""
        client = gs.GridStatusClient(api_key=API_KEY)
        # Since pandas is available in test env, should default to pandas
        assert client.return_format == "pandas"

    def test_explicit_python_return_format(self):
        """Test that return_format can be explicitly set to python."""
        client = gs.GridStatusClient(api_key=API_KEY, return_format="python")
        assert client.return_format == "python"

    def test_explicit_pandas_return_format(self):
        """Test that return_format can be explicitly set to pandas."""
        client = gs.GridStatusClient(api_key=API_KEY, return_format="pandas")
        assert client.return_format == "pandas"

    def test_explicit_polars_return_format(self):
        """Test that return_format can be explicitly set to polars."""
        client = gs.GridStatusClient(api_key=API_KEY, return_format="polars")
        assert client.return_format == "polars"


class TestReturnFormatsIntegration:
    """Integration tests for different return formats."""

    @pytest.fixture
    def client(self):
        return gs.GridStatusClient(api_key=API_KEY)

    def test_pandas_format_returns_dataframe(self, client):
        """Test that pandas format returns a pandas DataFrame."""
        result = client.get_dataset(
            "isone_fuel_mix",
            limit=10,
            return_format="pandas",
            verbose=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_python_format_returns_list_of_dicts(self, client):
        """Test that python format returns a list of dictionaries."""
        result = client.get_dataset(
            "isone_fuel_mix",
            limit=10,
            return_format="python",
            verbose=False,
        )
        assert isinstance(result, list)
        assert len(result) == 10
        assert all(isinstance(row, dict) for row in result)

    def test_python_format_datetime_as_string(self, client):
        """Test that datetime columns are ISO8601 strings in python format."""
        result = client.get_dataset(
            "isone_fuel_mix",
            limit=10,
            return_format="python",
            verbose=False,
        )
        first_row = result[0]
        # Check that datetime columns are strings
        assert isinstance(first_row["interval_start_utc"], str)

    def test_client_level_default_used(self, client):
        """Test that client-level return_format is used when not specified."""
        # Client defaults to pandas
        result = client.get_dataset(
            "isone_fuel_mix",
            limit=5,
            verbose=False,
        )
        assert isinstance(result, pd.DataFrame)

    def test_per_call_override(self):
        """Test that per-call return_format overrides client default."""
        client = gs.GridStatusClient(
            api_key=API_KEY,
            return_format="pandas",  # Client default
        )
        # Override to python for this call
        result = client.get_dataset(
            "isone_fuel_mix",
            limit=5,
            return_format="python",
            verbose=False,
        )
        assert isinstance(result, list)

    def test_list_datasets_works(self, client):
        """Test that list_datasets works with any return_format setting."""
        # list_datasets internally uses python format
        datasets = client.list_datasets(filter_term="fuel_mix", return_list=True)
        assert isinstance(datasets, list)
        assert len(datasets) > 0


class TestPolarsFormat:
    """Tests for polars return format."""

    @pytest.fixture
    def client(self):
        return gs.GridStatusClient(api_key=API_KEY)

    def test_polars_format_returns_dataframe(self, client):
        """Test that polars format returns a polars DataFrame."""
        result = client.get_dataset(
            "isone_fuel_mix",
            limit=10,
            return_format="polars",
            verbose=False,
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10

    def test_polars_client_level_default(self):
        """Test that client can default to polars format."""
        client = gs.GridStatusClient(api_key=API_KEY, return_format="polars")
        result = client.get_dataset(
            "isone_fuel_mix",
            limit=5,
            verbose=False,
        )
        assert isinstance(result, pl.DataFrame)


class TestMissingDependencyErrorExport:
    """Test that MissingDependencyError is properly exported."""

    def test_can_import_from_gridstatusio(self):
        """Test that MissingDependencyError can be imported from gridstatusio."""
        from gridstatusio import MissingDependencyError

        assert MissingDependencyError is not None

    def test_is_importerror_subclass(self):
        """Test that MissingDependencyError is an ImportError subclass."""
        from gridstatusio import MissingDependencyError

        assert issubclass(MissingDependencyError, ImportError)


class TestReturnFormatEnum:
    """Test that ReturnFormat enum is properly exported and usable."""

    def test_can_import_from_gridstatusio(self):
        """Test that ReturnFormat can be imported from gridstatusio."""
        from gridstatusio import ReturnFormat

        assert ReturnFormat is not None

    def test_enum_values(self):
        """Test that enum has expected values."""
        from gridstatusio import ReturnFormat

        assert ReturnFormat.PANDAS == "pandas"
        assert ReturnFormat.POLARS == "polars"
        assert ReturnFormat.PYTHON == "python"

    def test_client_accepts_enum_value(self):
        """Test that client accepts enum values for return_format."""
        from gridstatusio import ReturnFormat

        client = gs.GridStatusClient(api_key=API_KEY, return_format=ReturnFormat.PANDAS)
        assert client.return_format == ReturnFormat.PANDAS

    def test_get_dataset_accepts_enum_value(self):
        """Test that get_dataset accepts enum values for return_format."""
        from gridstatusio import ReturnFormat

        client = gs.GridStatusClient(api_key=API_KEY)
        result = client.get_dataset(
            "isone_fuel_mix",
            limit=5,
            return_format=ReturnFormat.PYTHON,
            verbose=False,
        )
        assert isinstance(result, list)
