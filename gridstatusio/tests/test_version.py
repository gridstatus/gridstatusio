import pytest

from gridstatusio.version import version_is_higher


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
