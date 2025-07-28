import os
from unittest.mock import MagicMock, patch

import pytest

from gridstatusio.version import check_for_update, version_is_higher


@pytest.mark.parametrize(
    "latest, current, expected",
    [
        ("1.0.0", "0.9.9", True),
        ("0.9.9", "1.0.0", False),
        ("1.0.0", "1.0.0", False),
        ("1.0.1", "1.0.0", True),
        ("1.0.0", "1.0.1", False),
        ("1.1.0", "1.0.9", True),
    ],
)
def test_version_is_higher(latest, current, expected):
    assert version_is_higher(latest, current) == expected


def test_version_check_disabled():
    """Test that version check is skipped when environment variable is set"""
    os.environ["GSIO_SKIP_VERSION_CHECK"] = "true"
    with patch("requests.get") as mock_get:
        check_for_update()

        mock_get.assert_not_called()


def test_version_check_enabled():
    """Test that version check runs when environment variable is not set"""
    os.environ.pop("GSIO_SKIP_VERSION_CHECK", None)

    mock_response = MagicMock()
    mock_response.json.return_value = {"info": {"version": "0.14.0"}}

    with patch("requests.get", return_value=mock_response) as mock_get:
        check_for_update()

        mock_get.assert_called_once_with("https://pypi.org/pypi/gridstatusio/json")
