.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

PYTEST_CMD := poetry run pytest -s -vv gridstatusio/ -n auto
NOT_SLOW := -m "not slow"

.PHONY: test
test:
	$(PYTEST_CMD) $(NOT_SLOW) --reruns 5 --reruns-delay 3

.PHONY: test-slow
test-slow:
	$(PYTEST_CMD) -m slow

.PHONY: installdeps-dev
installdeps-dev:
	poetry install --all-extras
	poetry run pre-commit install

.PHONY: installdeps-test
installdeps-test:
	poetry install --all-extras

.PHONY: installdeps-docs
installdeps-docs:
	poetry install --all-extras

.PHONY: lint
lint:
	poetry run ruff check gridstatusio/
	poetry run ruff format gridstatusio/ --check

.PHONY: lint-fix
lint-fix:
	poetry run ruff check gridstatusio/ --fix
	poetry run ruff format gridstatusio/

.PHONY: package
package: upgradepip upgradebuild upgradesetuptools
	poetry build
