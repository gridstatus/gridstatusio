# How to release

1. Bump version in `gridstatusio/version.py`, `pyproject.toml`, and the `CHANGELOG.md`
2. Record changes since the last release in the `CHANGELOG.md`
3. Make release on GitHub and tag it with a matching version number. The tag must start with v. For example, `v0.1.0` if the version is `0.1.0`
4. The package should be automatically published to PyPI when the release is created.
5. Confirm package was uploaded to [PyPi](https://pypi.org/project/gridstatusio/)
