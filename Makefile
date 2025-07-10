.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

PYTEST_CMD := uv run pytest -s -vv gridstatusio/
NOT_SLOW := -m "not slow"

.PHONY: test
test:
	$(PYTEST_CMD) $(NOT_SLOW) --reruns 5 --reruns-delay 3

# Running tests in parallel will generally result in rate limiting from the API
.PHONY: test-parallel
test-parallel:
	$(PYTEST_CMD) $(NOT_SLOW) -n auto

.PHONY: test-slow
test-slow:
	$(PYTEST_CMD) -m slow

.PHONY: installdeps-dev
installdeps-dev:
	uv sync
	uv run pre-commit install

.PHONY: installdeps-test
installdeps-test:
	uv sync

.PHONY: installdeps-docs
installdeps-docs:
	uv sync

.PHONY: lint
lint:
	uv run ruff check gridstatusio/
	uv run ruff format gridstatusio/ --check

.PHONY: lint-fix
lint-fix:
	uv run ruff check gridstatusio/ --fix
	uv run ruff format gridstatusio/

.PHONY: upgradepip
upgradepip:
	uv pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	uv pip install --upgrade build

.PHONY: upgradesetuptools
upgradesetuptools:
	uv pip install --upgrade setuptools

.PHONY: package
package: upgradepip upgradebuild upgradesetuptools
	uv build

.PHONY: type-check
type-check:
	uv run pyright --project pyproject.toml
