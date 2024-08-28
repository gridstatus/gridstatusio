import requests
from termcolor import colored

__version__ = "0.7.0"


def get_latest_version():
    """Get the latest version of gridstatusio from PyPI"""

    response = requests.get("https://pypi.org/pypi/gridstatusio/json")  # noqa: E501
    latest_version = response.json()["info"]["version"]
    return latest_version


def version_is_higher(latest, current):
    latest_parts = [int(x) for x in latest.split(".")]
    current_parts = [int(x) for x in current.split(".")]

    for latest_part, current_part in zip(latest_parts, current_parts):
        if latest_part > current_part:
            return True
        elif latest_part < current_part:
            return False
    return False


def check_for_update():
    latest = get_latest_version()
    if version_is_higher(latest, __version__):
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
                "\nWe recommend upgrading via the 'pip install --upgrade gridstatusio' command.",  # noqa: E501
                "red",
            ),
        )
        print(
            colored(
                "\nSee the changelog here: https://github.com/gridstatus/gridstatusio/blob/main/CHANGELOG.md",  # noqa: E501
                "red",
            ),
        )
