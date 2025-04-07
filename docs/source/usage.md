# Usage Guide

This guide walks you through the steps to use the Energy Pipeline for detecting energy regressions in your codebase.

---

## 🧩 Overview

The pipeline automates:
- Cloning and checking out each commit
- Running your tests N times per commit
- Measuring energy with Intel RAPL
- Plotting energy consumption trends and detecting regressions

---

## 📁 1. Prepare a Configuration File

You need to create a config file in JSON format that follows your `config.schema.json` structure. This includes:

- Repository URL
- Branch or tags to analyze
- Number of commits, runs, and repetitions
- Test command to run
- Output paths

Example:
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
    "batch_size": 5,
    "randomize_tasks": true
  },
  "test_command": "./run-tests.sh"
}
```

---

## 🛠️ 2. System Preparation

Ensure:
- You're in a TTY session (not desktop environment)
- Minimal background services are running
- Battery is full and charging
- Auto-brightness, CPU autoscaling, Wi-Fi etc. are disabled

Run the system setup:

```bash
sudo ./system_setup.sh first-setup
reboot
sudo ./system_setup.sh setup
```

You can later revert changes:

```bash
sudo ./system_setup.sh revert-setup
```

---

## 🚀 3. Run the Pipeline

### Run a Stability Test

Before real measurements:

```bash
python main.py stability-test
```

This checks system temperature and performance counter stability.

---

### Run Measurement

```bash
python main.py measure --config path/to/config.json
```

- This runs the entire pipeline
- Output is a `.csv` file in the same folder as the config

---

## 📊 4. Sort the Results (Optional)

To ensure results follow Git commit order:

```bash
python main.py sort path/to/results.csv /path/to/repo sorted_results.csv
```

---

## 📈 5. Generate Plots

To visualize energy trends and regressions:

```bash
python main.py plot path/to/sorted_results.csv
```

This creates one PNG per energy metric (`energy-pkg`, `energy-core`, `energy-gpu`), with:

- Error bars and medians
- Violin plots (distribution)
- Change point detection
- Breaking commit markers
- Normality tests per commit

---

## 🐳 Optional: Run in Docker

You can use Docker to isolate the environment:

```bash
docker buildx build -t pipeline .
docker run -it --privileged pipeline:latest
```

Make sure to mount required volumes and copy your config + repo path into the container.

---

## 🧠 Tips

- Run in `tmux` if on a server:

```bash
tmux new -s energy
# [run your pipeline]
# Detach with Ctrl+B D
# Reattach with: tmux attach -t energy
```

- Log files will be written to the working directory

---

## ✅ Output

- CSV with energy data: `[commit, energy-pkg, energy-core, energy-gpu]`
- PNG plots in same folder as CSV
- Plots include short commit hashes and markers for regressions

---

## 📌 Recap

| Step           | Command                                                    |
| -------------- | ---------------------------------------------------------- |
| Stability test | `python main.py stability-test`                            |
| Run pipeline   | `python main.py measure --config config.json`              |
| Sort results   | `python main.py sort results.csv /path/to/repo sorted.csv` |
| Plot results   | `python main.py plot sorted.csv`                           |