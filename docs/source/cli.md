# 💻 CLI Reference

The Energy Pipeline is controlled via the `main.py` command-line interface (CLI).

This CLI allows you to:
- Run energy measurements over Git commits
- Check system stability
- Sort result files by commit order
- Generate plots from CSV data

---

## 🧾 Usage

```bash
python main.py <command> [options]
```

---

## 📌 Available Commands

| Command          | Description                                         |
| ---------------- | --------------------------------------------------- |
| `measure`        | Runs the full energy measurement pipeline           |
| `stability-test` | Verifies that your system is ready for measurement  |
| `sort`           | Reorders a result CSV file using Git commit history |
| `plot`           | Generates energy plots from a CSV file              |

---

## 🔍 `measure`

Runs the pipeline and saves results to a CSV file.

```bash
python main.py measure --config path/to/config.json
```

### Options:

| Option     | Description                  | Default       |
| ---------- | ---------------------------- | ------------- |
| `--config` | Path to the config JSON file | `config.json` |

### Example:

```bash
python main.py measure --config configs/myproject.json
```

---

## 🧪 `stability-test`

Checks if the system is in a stable condition for running measurements.

It verifies:
- CPU temperature
- Permissions to read performance counters
- Access to Intel RAPL interface

```bash
python main.py stability-test
```

If any condition fails, the pipeline should **not** be run.

---

## 📊 `plot`

Creates visual graphs from a CSV result file.

```bash
python main.py plot path/to/results.csv
```

This will generate one plot per energy metric (e.g., `energy-pkg`, `energy-core`, `energy-gpu`) and save them as PNGs.

### Output

- Saved as PNG files in the same folder as the CSV
- Includes:
  - Median with error bars
  - Violin plots for distribution
  - Change point detection
  - Breaking commit markers
  - Normality analysis

### Example:

```bash
python main.py plot results/sorted_results.csv
```

---

## 🔃 `sort`

Sorts a CSV file by Git history to align results with chronological commits.

```bash
python main.py sort <input_csv> <repo_path> <output_csv>
```

### Parameters:

| Name         | Description                                |
| ------------ | ------------------------------------------ |
| `input_csv`  | Path to the unsorted CSV file              |
| `repo_path`  | Path to the Git repository                 |
| `output_csv` | Path where the sorted file will be written |

### Example:

```bash
python main.py sort results/raw.csv /path/to/project results/sorted.csv
```

This ensures the plot will reflect correct commit chronology.

---

## 🧠 Internals Summary

- The CLI uses Python's `argparse` module with `subparsers`
- If an unknown command is provided, it raises a custom `UnknownCommandError`
- All commands log progress to the console and use `tqdm` for visual feedback

---

## 🧪 Quick Test Run

```bash
# Run a dry test on system
python main.py stability-test

# Run the pipeline on a config file
python main.py measure --config configs/sample.json

# Sort and analyze results
python main.py sort results.csv ./projects/sample/ sorted.csv
python main.py plot sorted.csv
```

---

Need help writing a config file? See [`usage.md`](usage.md)