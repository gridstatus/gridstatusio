import requests
from termcolor import colored

__version__ = "0.5.0"


def get_latest_version():
    """Get the latest version of gridstatusio from PyPI"""

    response = requests.get("https://pypi.org/pypi/gridstatusio/json")  # noqa: E501
    latest_version = response.json()["info"]["version"]
    return latest_version


def check_for_update():
    latest = get_latest_version()
    if latest != __version__:
        print(
            # make bold
            colored(
                f"There is a newer version of the gridstatusio library available ({latest}). You are using version {__version__}.",  # noqa: E501
                "red",
                attrs=["bold"],
            ),
        )
        print(
            colored(
                "We recommend upgrading via the 'pip install --upgrade gridstatusio' command.",  # noqa: E501
                "red",
            ),
        )
