"""Test energy consumption across Git commits."""

#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import time
from time import sleep

import git
import git.types
from git.types import Files_TD
from pyparsing import Any
from tqdm import tqdm  # Import tqdm for progress reporting


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path) as file:
        return dict(json.load(file))


def measure_energy(repo_path: str, test_command: str, output_file: str) -> None:
    """Runs a test command using `perf` to measure energy consumption and logs the results.

    Args:
        repo_path (str): Path to the Git repository.
        test_command (str): Command to execute for testing.
        output_file (str): Path to the output CSV file.

    """
    try:
        # Run the test command with `perf` to measure energy consumption
        tqdm.write(f"Running test command: {test_command}")
        perf_command = f"perf stat -e power/energy-pkg/ {test_command}"

        result: subprocess.CompletedProcess[str] | None = run_command(perf_command, repo_path)

        if result is None:
            tqdm.write("Failed to run perf command. Skipping energy measurement.")
            return

        if result.returncode != 0:
            tqdm.write(f"Error running perf command: {result.stderr}")
            return

        perf_output = result.stderr  # `perf stat` outputs to stderr by default

        # Extract energy values from perf output
        energy_pkg = extract_energy_value(perf_output, "power/energy-pkg/")

        if energy_pkg is None:
            tqdm.write("Failed to extract energy measurement from perf output.")
            return

        tqdm.write(f"ENERGY_PKG: {energy_pkg}")

        # Get the current Git commit hash
        repo = git.Repo(repo_path)
        commit_hash = repo.head.object.hexsha

        # Append results to file
        with open(output_file, "a") as file:
            file.write(f"{commit_hash},{energy_pkg}\n")

        tqdm.write(f"Results appended to {output_file}")
    except Exception as e:
        tqdm.write(f"Error: {e}")


def extract_energy_value(perf_output: str, event_name: str) -> str | None:
    """Extracts the energy measurement value from the perf output."""
    for line in perf_output.split("\n"):
        if event_name in line:
            parts = line.split()
            if parts:
                return parts[0]  # First column is the energy value
    return None


def run_single_energy_test(repo_path: str, output_file: str, config: dict[str, Any]) -> None:
    """Runs a single instance of the energy measurement test."""
    script_path = os.path.join(os.getcwd(), "measure_energy.sh")
    # Ensure CPU temperature is within safe limits
    while not is_temperature_safe(config):
        tqdm.write("⚠️ CPU temperature is too high. Waiting for it to cool down...")
        sleep(1)
    # Run pre-command if provided
    if config.get("test", {}).get("pre_command"):
        run_command(config["test"]["pre_command"])
    # Run the energy measurement
    measure_energy(repo_path, config["test"]["command"], output_file)
    # Run post-command if provided
    if config.get("test", {}).get("post_command"):
        run_command(config["test"]["post_command"])


def run_command(arg: str, cwd: str | None = None) -> subprocess.CompletedProcess[str] | None:
    """Executes a shell command and captures its output.

    Args:
        arg (str): The command to run.
        cwd (str | None): The working directory path for the command.

    Raises:
        CalledProcessError: If the command exits with a non-zero status.

    Logs:
        Error messages, standard output, and standard error if the command fails.

    """
    try:
        tqdm.write(f"Running command: {arg}")
        result = subprocess.run(
            args=arg,
            cwd=cwd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        tqdm.write(f"Error: Command {arg} failed with exit code {e.returncode}")
        if e.stdout:
            tqdm.write(f"Standard Output:\n{e.stdout}")
        if e.stderr:
            tqdm.write(f"Standard Error:\n{e.stderr}")
        return None


def is_temperature_safe(config: dict[str, Any]) -> bool:
    """Check if temperature is within safe limits (CPU not throttling)."""
    tqdm.write("Checking CPU temperature...")
    command_result = run_command(f"cat {config['cpu_themal_file_path']}")
    if command_result is None:
        tqdm.write("Failed to get CPU temperature. Continuing with the test...")
        return True
    temperature = int(command_result.stdout.strip())
    tqdm.write(f"CPU temperature: {temperature}°C")
    assert isinstance(config["thresholds"]["temperature_safe_limit"], int)
    return temperature < config["thresholds"]["temperature_safe_limit"]


def is_system_stable(k: float = 3.5, warmup_time: int = 5, duration: int = 30) -> bool:
    """Checks if the system's power consumption is stable using the Modified Z-Score.

    Args:
        k (float): Threshold for the Modified Z-score (default = 3.5).
        warmup_time (int): Time in seconds for initial power observation.
        duration (int): Total monitoring duration in seconds.

    Returns:
        bool: True if system is stable, False otherwise.

    """

    def get_energy_uj() -> int:
        command_result = run_command("sudo cat /sys/class/powercap/intel-rapl:0/energy_uj")
        if command_result is None:
            return 0
        return int(command_result.stdout.strip())

    power_samples: list[int] = []
    prev_energy = get_energy_uj()
    tqdm.write(f"Gathering baseline power data for {warmup_time} seconds...")
    for _ in range(warmup_time):
        sleep(1)
        current_energy = get_energy_uj()
        power = current_energy - prev_energy
        power_samples.append(power)
        prev_energy = current_energy
    median_power = statistics.median(power_samples)
    mad_power = statistics.median([abs(x - median_power) for x in power_samples]) or 1
    tqdm.write(f"Baseline median power: {median_power / 1_000_000} W")
    tqdm.write(f"Baseline MAD: {mad_power / 1_000_000} W")
    for _ in range(duration - warmup_time):
        sleep(1)
        current_energy = get_energy_uj()
        power = current_energy - prev_energy
        prev_energy = current_energy
        mz_score = 0.6745 * (power - median_power) / mad_power
        tqdm.write(f"Power consumption: {power / 1_000_000} W")
        tqdm.write(f"Modified Z-Score: {mz_score}")
        if abs(mz_score) > k:
            tqdm.write("⚠️ System is NOT stable!")
            return False
        tqdm.write("✅ System is stable.\n")
    return True


def commit_contains_c_code(commit: git.Commit, config: dict[str, Any]) -> bool:
    """Determine if a commit modifies any files with a C-related extension."""
    files: dict[git.types.PathLike, Files_TD] = commit.stats.files
    return any(str(file).endswith(tuple(config["file_extensions"])) for file in files)


def setup_repo(repo_path: str, repo_url: str) -> git.Repo:
    """Clones the repository if not present; otherwise opens the existing one."""
    if not os.path.exists(repo_path):
        tqdm.write(f"Cloning {repo_url} into {repo_path}...")
        return git.Repo.clone_from(repo_url, repo_path)
    else:
        tqdm.write(f"Using existing repo at {repo_path}...")
        return git.Repo(repo_path)


def generate_tasks(batch_commits: list[git.Commit], config: dict[str, Any]) -> list[git.Commit]:
    """Generate a list of energy measurement tasks for the provided commits."""
    tasks: list[git.Commit] = []
    num_runs = config["test"]["num_runs"]
    num_repeats = config["test"].get("num_repeats", 1)
    randomize = config["test"].get("randomize_tasks", False)
    for commit in batch_commits:
        if config["test"].get("granularity", "commit") == "commit" and not commit_contains_c_code(commit, config):
            tqdm.write(f"Skipping commit {commit.hexsha} as it does not contain C code.")
            continue
        # Each commit is scheduled for num_runs * num_repeats tests
        tasks.extend([commit] * (num_runs * num_repeats))
    if randomize:
        random.shuffle(tasks)
    return tasks


def main(config_path: str) -> None:
    """Main function for the energy consumption testing tool.

    Loads the configuration from a JSON file, clones the repository, and processes
    commits in batches. For each batch, it checks out the commit, builds the project,
    runs the energy measurement test, and saves the results to a CSV file.

    After processing each batch, it deletes the repository cache to free up space.
    Finally, it checks out back to the latest commit on the branch.
    """
    start_time = time.time()
    config: dict[str, Any] = load_config(config_path)
    project_name = os.path.basename(config["repository"]["url"]).replace(".git", "").strip().lower()
    project_dir = os.path.join("projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    repo_path = os.path.join(project_dir, ".cache" + project_name)
    output_file = os.path.join(project_dir, config["output"]["file"])

    # Read parameters from config
    num_commits = config["test"]["num_commits"]
    batch_size = config["test"].get("batch_size", 100)  # default is 100 commits per batch

    # Clear previous output file if exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Clone the repository to get the list of commits
    repo = setup_repo(repo_path, config["repository"]["url"])
    if config["test"]["granularity"] == "branches":
        branches = list(repo.remotes.origin.refs)
        commits = [branch.commit for branch in branches]  # Get only the latest commit of each branch
        tqdm.write(f"Branches: {branches}")
        tqdm.write(f"Commits: {commits}")
    elif config["test"]["granularity"] == "tags":
        tags = list(repo.tags)
        commits = [commit for tag in tags for commit in repo.iter_commits(tag, max_count=num_commits)]
    else:
        commits = list(repo.iter_commits(config["repository"]["branch"], max_count=num_commits))
    total_batches = (len(commits) + batch_size - 1) // batch_size
    tqdm.write(f"Total commits: {len(commits)} in {total_batches} batches (batch size: {batch_size})")

    # Run project setup commands
    if "setup_commands" in config:
        for command in config["setup_commands"]:
            tqdm.write(f"Running setup command: {command}")
            subprocess.run(command, shell=True, check=True)

    current_commit: str = ""
    global_task_counter = 0

    # Process commits batch by batch
    for batch_index in range(total_batches):
        batch_commits = commits[batch_index * batch_size : (batch_index + 1) * batch_size]
        # Build list of tasks for this batch (each task is one energy measurement run)
        tasks: list[git.Commit] = generate_tasks(batch_commits, config)

        tqdm.write(f"\nProcessing batch {batch_index + 1}/{total_batches} with {len(tasks)} tasks...")

        # Use a tqdm progress bar for tasks in this batch
        with tqdm(total=len(tasks), desc=f"Batch {batch_index + 1}/{total_batches}", leave=False) as pbar:
            for _, commit in enumerate(tasks):
                # If the commit changes, checkout and build the project
                if current_commit != commit.hexsha:
                    tqdm.write(f"\n🔄 Checking out commit {commit.hexsha}...")
                    repo.git.checkout(commit.hexsha)
                    tqdm.write("Building the project...")
                    if config.get("test", {}).get("mode", "run") == "run":
                        build_failed = False
                        for command in config["compile_commands"]:
                            if run_command(command, repo_path) is None:
                                tqdm.write("Failed to build the project. Skipping this commit...")
                                build_failed = True
                                break
                        if build_failed:
                            continue
                    current_commit = commit.hexsha
                # Run a single energy measurement test for this task
                run_single_energy_test(repo_path, output_file, config)
                global_task_counter += 1
                elapsed_time = int(time.time() - start_time)
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                pbar.set_postfix(global_tasks=global_task_counter, elapsed=formatted_time)
                pbar.update(1)

        # Delete the repository cache to free up space after processing a batch
        tqdm.write(f"\nBatch {batch_index + 1} completed. Deleting repository cache to free up space...")
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        # Re-clone repository for the next batch (if any)
        if batch_index < total_batches - 1:
            repo = setup_repo(repo_path, config["repository"]["url"])
            current_commit = ""

    # Final step: checkout back to the latest commit on the branch
    repo.git.checkout(config["repository"]["branch"])
    tqdm.write("\n✅ Restored to latest commit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test energy consumption across Git commits.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stability_parser = subparsers.add_parser("stability-test", help="Run the system stability test")

    measure_parser = subparsers.add_parser("measure", help="Run the energy measurement test")
    measure_parser.add_argument("config_path", help="Path to the configuration JSON file")

    args = parser.parse_args()

    if args.command == "measure":
        main(args.config_path)
    elif args.command == "stability-test":
        is_system_stable()
