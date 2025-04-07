# ⚡ Energy Measurement Pipeline

A modular, pluggable pipeline to **detect energy regressions** across Git commits, branches, or tags. Ideal for research and diagnostics in performance-aware software engineering.

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
cd energy-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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
python main.py stability-test
```

Measure energy across commits:

```bash
python main.py measure --config path/to/config.json
```

Sort CSV by Git history:

```bash
python main.py sort unsorted.csv /repo/path sorted.csv
```

Generate plots:

```bash
python main.py plot sorted.csv
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

## 📚 Documentation

- 📘 [Full Documentation (Sphinx)](https://yourdocs.readthedocs.io)
- 🧱 [Pipeline Architecture](https://yourdocs.readthedocs.io/en/latest/architecture.html)
- ⚙️ [Usage Guide](https://yourdocs.readthedocs.io/en/latest/usage.html)
- 🧩 [Writing Custom Stages](https://yourdocs.readthedocs.io/en/latest/stages.html)

---

## 🧠 Acknowledgements

- Inspired by energy-efficient software engineering research
- Powered by: `GitPython`, `perf`, `tqdm`, `matplotlib`, `ruptures`, `pydantic`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).