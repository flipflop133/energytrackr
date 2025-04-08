# ⚡ Energy Measurement Pipeline

A modular, pluggable pipeline to **detect energy regressions** across Git commits, branches, or tags. Ideal for research and diagnostics in performance-aware software engineering.

---

## TODO - sorted by priority

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
- [ ] Add github actions

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

- 📘 [Full Documentation (Sphinx)](https://energy-analyzer.readthedocs.io)
- 🧱 [Pipeline Architecture](https://energy-analyzer.readthedocs.io/en/latest/architecture.html)
- ⚙️ [Usage Guide](https://energy-analyzer.readthedocs.io/en/latest/usage.html)
- 🧩 [Writing Custom Stages](https://energy-analyzer.readthedocs.io/en/latest/stages.html)

---

## 🧠 Acknowledgements

- Inspired by energy-efficient software engineering research
- Powered by: `GitPython`, `perf`, `tqdm`, `matplotlib`, `ruptures`, `pydantic`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
