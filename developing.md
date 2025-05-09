# Setting up a Development Environment
* Install Python 3.9 or higher and make sure java is installed on your system
* Fork the repository and clone it to your local machine. For a PR, create a new branch in your fork.
* Install the project dependencies by running:

```shell
# Install uv if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
uv pip install --all-extras -e .

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```
* Activate the virtual environment by running `source .venv/bin/activate` (Unix/macOS) or `.venv\Scripts\activate` (Windows). This will run in a virtual environment with all the dependencies installed.
* Installing the dev dependencies enables a pre-commit hook that ensures linting has been run before committing

## Environment Variables
* Set the `GRIDSTATUS_API_KEY` environment variable to your GridStatus API key. This is required to run the tests and examples.
* The best way to ensure everything is installed correctly by running running the tests with `make test`. They should all pass.

## Running Tests and Linting
To ensure that your changes are correct and follow our style guide, we ask you to run the tests and linting before submitting a pull request. You can use the following commands to do so:

```shell
# Run all tests
make test
# Lint the code
make lint
# Fix linting errors
make lint-fix
```

We use `pytest` for testing, so you can also run the test directly with the `pytest` command from within the activated virtual environment.
