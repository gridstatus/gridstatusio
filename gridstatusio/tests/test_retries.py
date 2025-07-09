import os
from unittest.mock import patch

import pytest
from requests.exceptions import ConnectionError

import gridstatusio as gs

client = gs.GridStatusClient(
    api_key=os.getenv("GRIDSTATUS_API_KEY_TEST"),
    host=os.getenv("GRIDSTATUS_HOST_TEST", "https://api.gridstatus.io/v1"),
    max_retries=3,
    base_delay=1.0,
    exponential_base=2,
)


@patch("requests.get")
def test_rate_limit_hit_backoff(mock_get_request, caplog):
    caplog.set_level("INFO")
    mock_get_request.return_value.status_code = 429
    mock_get_request.return_value.json.return_value = {
        "detail": "Too Many Requests. Limit: 6 per 1 second",
    }
    with pytest.raises(
        Exception,
        match="HTTP 429: Too Many Requests. Limit: 6 per 1 second",
    ):
        client.get_dataset(
            "pjm_load",
            start="2024-01-01",
            end="2024-01-02",
        )

    log_messages = [record.message for record in caplog.records]
    for i in range(0, client.max_retries):
        expected_text = (
            f"Too Many Requests. Limit: 6 per 1 second. "
            f"Retrying in {1 * 2**i} seconds. "
            f"Retry {i + 1} of {client.max_retries}."
        )
        assert expected_text in log_messages


@patch("requests.get")
def test_connection_error_backoff(mock_get_request, caplog):
    caplog.set_level("INFO")
    mock_get_request.side_effect = ConnectionError("Connection failed")

    with pytest.raises(
        Exception,
        match="Network error: Connection failed. Exceeded maximum number of retries",
    ):
        client.get_dataset(
            "pjm_load",
            start="2024-01-01",
            end="2024-01-02",
        )

    log_messages = [record.message for record in caplog.records]
    for i in range(0, client.max_retries):
        expected_text = (
            f"Network error (ConnectionError). "
            f"Retrying in {1 * 2**i} seconds. "
            f"Retry {i + 1} of {client.max_retries}."
        )
        assert expected_text in log_messages
