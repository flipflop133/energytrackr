# Variables
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
DOCS_BUILD := $(DOCS_DIR)/_build/html

# Ensure venv and install all dependencies
.PHONY: venv update-venv clean-venv install install-docs install-tests install-dev

venv:
	python -m venv $(VENV)

update-venv: venv
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r docs/requirements.txt
	$(PIP) install --upgrade -r tests/requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt
	$(PIP) install --upgrade pre-commit coverage pylint pyright ruff pytest

clean-venv:
	rm -rf $(VENV)
	$(MAKE) venv
	$(MAKE) install-dev

install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install -r docs/requirements.txt
	$(PIP) install -r tests/requirements.txt

install-docs:
	$(PIP) install -r docs/requirements.txt

install-tests:
	$(PIP) install -r tests/requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r docs/requirements.txt
	$(PIP) install -r tests/requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install pre-commit coverage pylint pyright ruff pytest

# Format using Ruff
.PHONY: format
format:
	$(PYTHON) -m ruff format --force-exclude

# Lint using Ruff
.PHONY: ruff
ruff:
	$(PYTHON) -m ruff check --force-exclude

# Lint using Pylint
.PHONY: pylint
pylint:
	$(PYTHON) -m pylint -v $(SRC_DIR)

# Type checking using Pyright
.PHONY: typecheck
typecheck:
	$(PYTHON) -m pyright

# Run unit tests
.PHONY: test
test:
	$(PYTHON) -m pytest

# Run test coverage with threshold
.PHONY: coverage
coverage:
	$(PYTHON) -m coverage run -m pytest
	$(PYTHON) -m coverage report --fail-under=80

# Run all linters and type checks
.PHONY: lint
lint: ruff pylint typecheck

# Run format, lint, tests, coverage
.PHONY: check
check: format lint coverage

# Run all pre-commit hooks
.PHONY: precommit
precommit:
	$(PYTHON) -m pre_commit run --all-files

# Build documentation using Sphinx
.PHONY: docs
docs:
	$(MAKE) -C $(DOCS_DIR) html

# Clean generated documentation
.PHONY: clean-docs
clean-docs:
	rm -rf $(DOCS_BUILD)
