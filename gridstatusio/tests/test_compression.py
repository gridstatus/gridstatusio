"""Regression tests for brotli response compression.

The client never sets ``Accept-Encoding`` itself -- it relies on urllib3's
capability-derived default. urllib3 only advertises ``br`` (and registers the
decoder) when a brotli library is installed, so these tests guard that the brotli
dependency is present and wired into the request path. If the dependency is ever
dropped, ``br`` silently disappears from the negotiated encodings and the large
data payloads fall back to gzip -- these tests fail loudly when that happens.
"""

import importlib.util

import requests
from urllib3.util.request import ACCEPT_ENCODING


def test_brotli_decoder_is_installed() -> None:
    """A brotli decoder must be importable for urllib3 to negotiate ``br``.

    ``brotli`` is the CPython library; ``brotlicffi`` is its PyPy equivalent.
    urllib3 accepts either, so the presence of one is enough.
    """
    has_brotli_decoder = (
        importlib.util.find_spec("brotli") is not None
        or importlib.util.find_spec("brotlicffi") is not None
    )
    assert (
        has_brotli_decoder
    ), "a brotli decoder (brotli or brotlicffi) must be installed"


def test_default_accept_encoding_advertises_brotli() -> None:
    """requests/urllib3 advertise ``br`` by default once brotli is installed.

    The client sends no explicit Accept-Encoding, so this default is what reaches
    the server on every request.
    """
    assert "br" in ACCEPT_ENCODING
    assert "br" in requests.utils.default_headers()["Accept-Encoding"]
