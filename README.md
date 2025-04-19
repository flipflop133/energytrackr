
# ⚡ energytrackr - Energy Measurement Pipeline

[![build-deploy workflow status badge](https://github.com/flipflop133/energytrackr/actions/workflows/check.yml/badge.svg)](https://github.com/flipflop133/energytrackr/actions/workflows/check.yml/badge.svg)

![Logo](logo.svg)

A modular, pluggable pipeline to **detect energy regressions** across Git commits, branches, or tags. Ideal for research and diagnostics in performance-aware software engineering.

---

## 📑 Index

- [⚡ energytrackr - Energy Measurement Pipeline](#-energytrackr---energy-measurement-pipeline)
  - [📑 Index](#-index)
  - [🚀 Features](#-features)
  - [🏗️ Pipeline Overview](#️-pipeline-overview)
  - [📄 Example Configuration](#-example-configuration)
  - [📦 Installation](#-installation)
  - [🧪 Usage](#-usage)
  - [🧩 Write Your Own Stage](#-write-your-own-stage)
  - [📊 Output](#-output)
  - [🛠️ Development Setup](#️-development-setup)
    - [📦 Environment Setup](#-environment-setup)
    - [🔄 Pre-Commit Hooks](#-pre-commit-hooks)
    - [🧪 Testing and Quality Checks](#-testing-and-quality-checks)
    - [📖 Documentation](#-documentation)
    - [💡 Recommended VSCode Extensions](#-recommended-vscode-extensions)
    - [⚠️ Project Standards](#️-project-standards)
    - [🧰 Summary of `Makefile` Commands](#-summary-of-makefile-commands)
  - [📚 Documentation](#-documentation-1)
  - [✅ TODO - sorted by priority](#-todo---sorted-by-priority)
  - [🧠 Acknowledgements](#-acknowledgements)
  - [📄 License](#-license)

---

## 🚀 Features

- 🔌 **Modular architecture** — add/remove stages easily
- 🔁 **Batch & repeat execution** — ensures statistical significance
- 🔍 **Energy regression detection** — based on Intel RAPL or `perf`
- 📦 **Multi-language support** — via custom build/test stages
- 📊 **Automated plots** — violin charts + change point detection
- 🛠️ **CLI-based** — easy to use and integrate into scripts

---

## 🏗️ Pipeline Overview

```text
[main.py]
   ↓
[Load Config & Repo]
   ↓
[Pre-Stages]       → Check setup
[Pre-Test Stages]  → Checkout, Build, Prepare
[Batch Stages]     → Measure energy across N repetitions
   ↓
[Results: CSV + PNG]
```

---

## 📄 Example Configuration

Your pipeline is controlled by a `config.json` file:

```json
{
  "repo": {
    "url": "https://github.com/example/project.git",
    "branch": "main"
  },
  "execution_plan": {
    "granularity": "commits",
    "num_commits": 10,
    "num_runs": 1,
    "num_repeats": 30,
    "randomize_tasks": true
  },
  "test_command": "pytest",
  "setup_commands": ["pip install -r requirements.txt"]
}
```

📘 See the [docs on configuration](https://yourdocs.readthedocs.io/en/latest/usage.html#configuration) for full schema.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/energy-pipeline.git
./setup.sh
```

Prepare your system for accurate measurements:

```bash
sudo ./system_setup.sh first-setup
reboot
sudo ./system_setup.sh setup
```

For more details: [Installation Guide](https://yourdocs.readthedocs.io/en/latest/installation.html)

---

## 🧪 Usage

Run a stability check (recommended before measurement):

```bash
energytrackr stability-test
```

Measure energy across commits:

```bash
energytrackr measure --config path/to/config.json
```

Sort CSV by Git history:

```bash
energytrackr sort unsorted.csv /repo/path sorted.csv
```

Generate plots:

```bash
energytrackr plot sorted.csv
```

---

## 🧩 Write Your Own Stage

Want to support another language or measurement tool? Just add a Python file to `modules/`, e.g.:

```python
class MyStage(PipelineStage):
    def run(self, context):
        print("Running custom stage")
```

Expose it via `get_stage()` and list it in your config:

```json
"modules_enabled": ["my_stage.py"]
```

---

## 📊 Output

- CSV: `[commit, energy-pkg, energy-core, energy-gpu]`
- PNG plots with:
  - Violin distribution per commit
  - Median & error bars
  - Normality testing
  - Change point markers

---

Absolutely! Here’s a revised and **cleaned-up `README.md` section for developers** with your current tooling, workflow, and best practices in mind:

---

## 🛠️ Development Setup

### 📦 Environment Setup

- Create and initialize a virtual environment with all necessary dependencies:

  ```bash
  make install-dev
  ```

This will:
- Create the virtual environment in `.venv/`
- Install runtime, test, and documentation dependencies
- Install developer tools like `pre-commit`, `coverage`, `pylint`, `pyright`, etc.

> ℹ️ Requires `make` and `python>=3.13`.

---

### 🔄 Pre-Commit Hooks

- Install Git hooks (done once):

  ```bash
  pre-commit install
  ```

- Run all hooks manually:

  ```bash
  make precommit
  ```

> Hooks include formatting (`ruff format`), linting (`ruff`, `pylint`), YAML and whitespace checks, and test+coverage validation.

---

### 🧪 Testing and Quality Checks

Run tests:

```bash
make test
```

Run tests with coverage (fails if coverage < 80%):

```bash
make coverage
```

Run all linters (Ruff, Pylint, Pyright):

```bash
make lint
```

Run full quality gate (format + lint + tests + coverage):

```bash
make check
```

---

### 📖 Documentation

We use **Sphinx** for documentation, and it's hosted on [ReadTheDocs](https://energy-analyzer.readthedocs.io).

To build the docs locally:

```bash
make docs
```

The HTML output will be in `docs/_build/html`.

To clean generated files:

```bash
make clean-docs
```

---

### 💡 Recommended VSCode Extensions

To maximize developer experience:

- **Python** (core support)
- **Ruff** (lint + format)
- **Pylance** (type checking)
- **Pylint**
- **Python Debugger**
- **Docker**
- **Git Extension Pack**

> Configure VSCode to use `.venv/` as the Python interpreter.

---

### ⚠️ Project Standards

- We require **at least 80% test coverage**.
- All code must pass **`ruff` formatting and linting**.
- All **tests must pass** before pushing.
- All code must have **strict typing** (`pyright`).
- Use `pre-commit` to catch issues before they reach CI.

---

### 🧰 Summary of `Makefile` Commands

```bash
make install-dev     # Full dev environment setup
make format          # Auto-format code with Ruff
make lint            # Run Ruff, Pylint, Pyright
make test            # Run pytest
make coverage        # Run tests + enforce coverage threshold
make check           # Full pipeline: format + lint + tests
make precommit       # Manually run pre-commit hooks
make docs            # Build documentation with Sphinx
make clean-docs      # Remove generated doc files
```

---

## 📚 Documentation

- 📘 [Full Documentation (Sphinx)](https://energy-analyzer.readthedocs.io)
- 🧱 [Pipeline Architecture](https://energy-analyzer.readthedocs.io/en/latest/architecture.html)
- ⚙️ [Usage Guide](https://energy-analyzer.readthedocs.io/en/latest/usage.html)
- 🧩 [Writing Custom Stages](https://energy-analyzer.readthedocs.io/en/latest/stages.html)

---

## ✅ TODO - sorted by priority

- [x] Don't erase produced CSV files, use a timestamp with project name
- [x] Automatically detect java version in pom and use export this one, so tests don't fail
- [x] Build project one time for each commit, for this copy the project x batch times and checkout in each one and compile in each one than for the 30 runs for each commit we just need to run the tests, copying and compiling can be done in parallel and with unlocked frequencies
- [x] Add documentation probably with sphinx
- [ ] Save run conditions (temperature, CPU governor, number of CPU cycles, etc.), perf could be use for part of this and fastfetch for the rest. Also save config file. Place all this metadata either in the CSV or in a separate file
- [ ] Display run conditions on the graph (e.g. temperature)
- [ ] From measured CPU cycles, use CPU power usage provided by manufacturer to display a second line on the graph to compare energy consumption from RAPL with theoretical estimations.
- [ ] Run only the same tests between commits
- [ ] Do a warm-up run before the actual measurement
- [ ] Add a cooldown between measurements, 1 second by default
- [ ] Check that pc won't go into sleep mode during the test
- [ ] Check that most background processes are disabled
- [ ] Unload drivers/modules that could interfere with the measurement
- [ ] Add tests with code coverage
- [ ] Add GitHub actions

---

## 🧠 Acknowledgements

- Inspired by energy-efficient software engineering research
- Powered by: `GitPython`, `perf`, `tqdm`, `matplotlib`, `ruptures`, `pydantic`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
