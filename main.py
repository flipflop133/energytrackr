#!/usr/bin/env python3
import os
import subprocess
import git
import argparse
import pandas as pd
from time import sleep
import statistics
import json
import time
import random
import shutil
from pyparsing import Any


def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as file:
        return json.load(file)


def run_single_energy_test(repo_path: str, output_file: str, config: dict) -> None:
    """
    Runs a single instance of the energy measurement test.
    """
    script_path = os.path.join(os.getcwd(), "measure_energy.sh")
    # Ensure CPU temperature is within safe limits
    while not is_temperature_safe(config):
        print("⚠️ CPU temperature is too high. Waiting for it to cool down...")
        sleep(1)
    # Run pre-command if provided
    if config["test"]["pre_command"]:
        subprocess.run(
            config["test"]["pre_command"],
            shell=True,
            check=True,
        )
    # Run the energy measurement script
    try:
        subprocess.run(
            [
                "sudo",
                "sh",
                script_path,
                repo_path,
                config["test"]["command"],
                output_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
    # Run post-command if provided
    if config["test"]["post_command"]:
        try:
            subprocess.run(
                config["test"]["post_command"],
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: Command failed with exit code {e.returncode}")
            print(f"Standard Output:\n{e.stdout}")
            print(f"Standard Error:\n{e.stderr}")


def is_temperature_safe(config: dict) -> bool:
    """Check if temperature is within safe limits (CPU not throttling)."""
    temperature = int(
        subprocess.run(
            ["cat", config["cpu_themal_file_path"]],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return temperature < config["thresholds"]["temperature_safe_limit"]


def analyze_results(output_file: str, config: dict) -> None:
    """
    Analyzes energy usage trends and detects significant regressions.
    Args:
        output_file (str): CSV file containing the energy measurements.
        config (dict): The configuration dictionary.
    """
    df = pd.read_csv(
        output_file,
        header=None,
        names=["commit", "energy_used", "exit_code"],
    )
    df["energy_used"] = df["energy_used"].astype(int)
    print("\nEnergy Consumption Trend:")
    print(df)
    df["prev_energy"] = df["energy_used"].shift(1)
    df["increase"] = (df["energy_used"] - df["prev_energy"]) / df["prev_energy"] * 100
    regressions = df[df["increase"] > config["thresholds"]["energy_regression_percent"]]
    if not regressions.empty:
        print("\n⚠️ Energy Regressions Detected:")
        print(regressions)
    else:
        print("\n✅ No significant regressions detected.")


def is_system_stable(k: float = 3.5, warmup_time: int = 5, duration: int = 30) -> bool:
    """
    Checks if the system's power consumption is stable using the Modified Z-Score.
    Args:
        k (float): Threshold for the Modified Z-score (default = 3.5).
        warmup_time (int): Time in seconds for initial power observation.
        duration (int): Total monitoring duration in seconds.
    Returns:
        bool: True if system is stable, False otherwise.
    """

    def get_energy_uj() -> int:
        return int(
            subprocess.run(
                ["sudo", "cat", "/sys/class/powercap/intel-rapl:0/energy_uj"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )

    power_samples = []
    prev_energy = get_energy_uj()
    print(f"Gathering baseline power data for {warmup_time} seconds...")
    for _ in range(warmup_time):
        sleep(1)
        current_energy = get_energy_uj()
        power = current_energy - prev_energy
        power_samples.append(power)
        prev_energy = current_energy
    median_power = statistics.median(power_samples)
    mad_power = statistics.median([abs(x - median_power) for x in power_samples]) or 1
    print(f"Baseline median power: {median_power / 1_000_000} W")
    print(f"Baseline MAD: {mad_power / 1_000_000} W")
    for _ in range(duration - warmup_time):
        sleep(1)
        current_energy = get_energy_uj()
        power = current_energy - prev_energy
        prev_energy = current_energy
        mz_score = 0.6745 * (power - median_power) / mad_power
        print(f"Power consumption: {power / 1_000_000} W")
        print(f"Modified Z-Score: {mz_score}")
        if abs(mz_score) > k:
            print("⚠️ System is NOT stable!")
            return False
        print("✅ System is stable.\n")
    return True


def commit_contains_c_code(commit: git.Commit, config: dict) -> bool:
    """Determine if a commit modifies any files with a C-related extension."""
    files = commit.stats.files
    for file in files.keys():
        if file.endswith(tuple(config["file_extensions"])):
            return True
    return False


def setup_repo(repo_path: str, repo_url: str) -> git.Repo:
    """
    Clones the repository if not present; otherwise opens the existing one.
    """
    if not os.path.exists(repo_path):
        print(f"Cloning {repo_url} into {repo_path}...")
        return git.Repo.clone_from(repo_url, repo_path)
    else:
        print(f"Using existing repo at {repo_path}...")
        return git.Repo(repo_path)


def main(config_path: str) -> None:
    start_time = time.time()
    config: dict = load_config(config_path)
    project_name = os.path.basename(config["repository"]["url"]).replace(".git", "")
    project_dir = os.path.join("projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    repo_path = os.path.join(project_dir, ".cache" + project_name)
    output_file = os.path.join(project_dir, config["output"]["file"])

    # Read parameters from config
    num_commits = config["test"]["num_commits"]
    batch_size = config["test"].get(
        "batch_size", 100
    )  # default is 100 commits per batch
    randomize = config["test"].get("randomize_commits", False)
    num_runs = config["test"]["num_runs"]
    num_repeats = config["test"].get("num_repeats", 1)

    # Clear previous output file if exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Clone the repository to get the list of commits
    repo = setup_repo(repo_path, config["repository"]["url"])
    commits = list(
        repo.iter_commits(config["repository"]["branch"], max_count=num_commits)
    )
    total_batches = (len(commits) + batch_size - 1) // batch_size
    print(
        f"Total commits: {len(commits)} in {total_batches} batches (batch size: {batch_size})"
    )

    current_commit = None
    global_task_counter = 0

    # Process commits batch by batch
    for batch_index in range(total_batches):
        batch_commits = commits[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]
        # Build list of tasks for this batch (each task is one energy measurement run)
        tasks = []
        for commit in batch_commits:
            if not commit_contains_c_code(commit, config):
                print(f"Skipping commit {commit.hexsha} as it does not contain C code.")
                continue
            for _ in range(num_runs * num_repeats):
                tasks.append(commit)
        if randomize:
            random.shuffle(tasks)
        print(
            f"\nProcessing batch {batch_index + 1}/{total_batches} with {len(tasks)} tasks..."
        )
        for i, commit in enumerate(tasks):
            # If the commit changes, checkout and build the project
            if current_commit != commit.hexsha:
                print(f"\n🔄 Checking out commit {commit.hexsha}...")
                repo.git.checkout(commit.hexsha)
                print("Building the project...")
                for command in config["compile_commands"]:
                    try:
                        subprocess.run(
                            command,
                            shell=True,
                            cwd=repo_path,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError as e:
                        print(
                            f"Error: Command '{command}' failed with exit code {e.returncode}"
                        )
                        print(f"Stdout:\n{e.stdout}")
                        print(f"Stderr:\n{e.stderr}")
                current_commit = commit.hexsha
            # Run a single energy measurement test for this task
            run_single_energy_test(repo_path, output_file, config)
            global_task_counter += 1
            print(
                f"Global progress: {global_task_counter} tasks completed (batch progress: {i + 1}/{len(tasks)})"
            )
            elapsed_time = int(time.time() - start_time)
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            print(f"Elapsed time: {formatted_time}")
        # Delete the repository cache to free up space after processing a batch
        print(
            f"\nBatch {batch_index + 1} completed. Deleting repository cache to free up space..."
        )
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        # Re-clone repository for the next batch (if any)
        if batch_index < total_batches - 1:
            repo = setup_repo(repo_path, config["repository"]["url"])
            current_commit = None

    # Final step: checkout back to the latest commit on the branch
    repo.git.checkout(config["repository"]["branch"])
    print("\n✅ Restored to latest commit.")
    # Analyze energy consumption trends
    analyze_results(output_file, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test energy consumption across Git commits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stability_parser = subparsers.add_parser(
        "stability-test", help="Run the system stability test"
    )

    measure_parser = subparsers.add_parser(
        "measure", help="Run the energy measurement test"
    )
    measure_parser.add_argument(
        "config_path", help="Path to the configuration JSON file"
    )

    args = parser.parse_args()

    if args.command == "measure":
        main(args.config_path)
    elif args.command == "stability-test":
        is_system_stable()
