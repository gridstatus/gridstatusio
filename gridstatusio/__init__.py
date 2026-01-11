from gridstatusio.version import __version__, check_for_update

from gridstatusio._compat import MissingDependencyError
from gridstatusio.gs_client import GridStatusClient, ReturnFormat


check_for_update()
