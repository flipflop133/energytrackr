# Variables
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
DOCS_BUILD := $(DOCS_DIR)/build/html

# Files
REQ := requirements.txt
REQ_DEV := requirements-dev.txt
REQ_DOCS := docs/requirements.txt
REQ_TEST := tests/requirements.txt

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

.PHONY: venv update-venv install install-dev clean-venv

venv:
	@test -d $(VENV) || python -m venv $(VENV)

update-venv: venv
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r $(REQ)
	$(PIP) install --upgrade -r $(REQ_DEV)
	$(PIP) install --upgrade -r $(REQ_DOCS)
	$(PIP) install --upgrade -r $(REQ_TEST)
	$(PIP) install -e .

clean-venv:
	rm -rf $(VENV)

install: venv
	@echo "Installing main app (editable)..."
	$(PIP) install -e .

install-dev: install
	@echo "Installing dev/test/docs dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQ)
	$(PIP) install -r $(REQ_DEV)
	$(PIP) install -r $(REQ_DOCS)
	$(PIP) install -r $(REQ_TEST)

# ----------------------------------------------------------------------
# Lint / Check
# ----------------------------------------------------------------------

.PHONY: format ruff pylint typecheck lint

format:
	$(PYTHON) -m ruff format --force-exclude

ruff:
	$(PYTHON) -m ruff check --force-exclude

pylint:
	$(PYTHON) -m pylint -v $(SRC_DIR)

typecheck:
	$(PYTHON) -m pyright

lint: ruff pylint typecheck

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

.PHONY: test coverage

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m coverage run -m pytest
	$(PYTHON) -m coverage report --fail-under=80

# ----------------------------------------------------------------------
# Meta
# ----------------------------------------------------------------------

.PHONY: check precommit

check: format lint coverage

precommit:
	$(PYTHON) -m pre_commit run --all-files

# ----------------------------------------------------------------------
# Docs
# ----------------------------------------------------------------------

.PHONY: docs clean-docs

docs:
	$(VENV)/bin/sphinx-build \
	  -W -b html \
	  $(DOCS_DIR)/source \
	  $(DOCS_BUILD)

clean-docs:
	rm -rf $(DOCS_BUILD)

apidoc:
	$(VENV)/bin/sphinx-apidoc -o $(DOCS_DIR)/source/api src/energytrackr

apidoc-force:
	$(VENV)/bin/sphinx-apidoc -f -o $(DOCS_DIR)/source/api src/energytrackr