# Energy Consumption Testing Tool

## Overview

This project provides a tool for measuring energy consumption across multiple commits in a Git repository. It automates the process of:

- Cloning the specified repository.
- Checking out each commit sequentially.
- Running a given test command multiple times.
- Measuring energy consumption using a Bash script.
- Detecting significant energy regressions.

## Prerequisites

- Python 3
- Git
- Bash
- Pandas (install via `pip install pandas`)
- GitPython (install via `pip install gitpython`)
- Intel RAPL enabled (for power measurement)

## Installation

Clone this repository and ensure dependencies are installed:

```sh
pip install pandas gitpython
```

## Usage

Run the script with the following arguments:

```sh
python git_energy_tester.py <repo_url> <branch> "<test_command>" <num_commits> <num_runs>
```

### Example

```sh
python main.py https://github.com/mpv-player/mpv.git master "mpv --end=3 sample.mp4" 10 100
```

This command will:

- Clone the `mpv` repository.
- Test the last 10 commits.
- Run `mpv --end=3 sample.mp4` 100 times per commit.
- Log energy consumption data.

## Results

- Energy measurement results are stored in `projects/<project_name>/energy_results.csv`.
- The script analyzes results and flags regressions (increases > 20%).

## Docker

docker buildx build -t pipeline .
docker run -it --privileged pipeline:latest
