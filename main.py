"""Test energy consumption across Git commits."""

#!/usr/bin/env python3
import argparse
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from time import sleep

import git
import git.types
from tqdm import tqdm

from config_model import PipelineConfig


def load_config(config_path: str) -> PipelineConfig:
    """Load configuration from a JSON file and validate it against the config model."""
    config = json.loads(Path(config_path).read_text())
    config.model_validate_json(config)
    return PipelineConfig(**config)


def measure_energy(repo_path: str, test_command: str, output_file: str, config: PipelineConfig) -> None:
    """Runs a test command using `perf` to measure energy consumption and logs the results.

    Args:
        repo_path (str): Path to the Git repository.
        test_command (str): Command to execute for testing.
        output_file (str): Path to the output CSV file.
        config (dict[str, Any]): Configuration parameters.

    """
    try:
        # Run the test command with `perf` to measure energy consumption
        tqdm.write(f"Running test command: {test_command}")
        perf_command = f"sudo perf stat -e power/energy-pkg/ {test_command}"

        path: str = repo_path + config.execution_plan.test_command_path
        result: subprocess.CompletedProcess[str] | None = run_command(perf_command, path)

        if result is None:
            tqdm.write("Failed to run perf command. Skipping energy measurement.")
            return

        if result.returncode != 0:
            tqdm.write(f"Error running perf command: {result.stderr}")
            return

        perf_output: str = result.stdout  # `perf stat` outputs to stderr by default

        # Extract energy values from perf output
        energy_pkg = extract_energy_value(perf_output, "power/energy-pkg/")

        if energy_pkg is None:
            tqdm.write("Failed to extract energy measurement from perf output.")
            return

        tqdm.write(f"ENERGY_PKG: {energy_pkg}")

        # Get the current Git commit hash
        repo = git.Repo(repo_path)
        commit_hash: str = repo.head.object.hexsha

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
                if "<not" in parts[0]:
                    # Handling the unsupported measurement case
                    return None
                return parts[0]  # First column is the energy value
    return None


def run_single_energy_test(
    repo_path: str,
    output_file: str,
    config: PipelineConfig,
    commit: git.Commit,
    global_task_counter: int,
) -> None:
    """Runs a single instance of the energy measurement test."""
    # Ensure CPU temperature is within safe limits
    while not is_temperature_safe(config):
        tqdm.write("⚠️ CPU temperature is too high. Waiting for it to cool down...")
        sleep(1)
    # Run pre-command if provided
    if config.execution_plan.pre_command:
        if global_task_counter != 0 and config.execution_plan.pre_command_condition_files:
            files: set[str] = config.execution_plan.pre_command_condition_files
            if commit_contains_patterns(commit, files):
                run_command(config.execution_plan.pre_command, repo_path)
        else:
            run_command(config.execution_plan.pre_command, repo_path)
    # Run the energy measurement
    measure_energy(repo_path, config.execution_plan.test_command, output_file, config)
    # Run post-command if provided
    if config.execution_plan.post_command:
        run_command(config.execution_plan.post_command, repo_path)


def run_command(arg: str, cwd: str | None = None) -> subprocess.CompletedProcess[str] | None:
    """Executes a shell command, streaming and capturing its output in real time.

    Args:
        arg (str): The command to run.
        cwd (str | None): The working directory for the command.

    Returns:
        subprocess.CompletedProcess[str]: The completed process with captured output on success.
        None: If the command fails.

    Raises:
        CalledProcessError: If the command exits with a non-zero status.

    """
    tqdm.write(f"Running command: {arg}")
    process = subprocess.Popen(
        arg,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []

    # Read and stream the output in real time
    if process.stdout is not None:
        for line in process.stdout:
            clean_line = line.rstrip()
            if clean_line:  # avoid empty lines
                tqdm.write(clean_line)
                output_lines.append(clean_line)

    retcode = process.wait()
    output = "\n".join(output_lines)

    if retcode != 0:
        tqdm.write(f"Error: Command '{arg}' failed with exit code {retcode}")
        return None

    return subprocess.CompletedProcess(args=arg, returncode=retcode, stdout=output)


def is_temperature_safe(config: PipelineConfig) -> bool:
    """Check if temperature is within safe limits (CPU not throttling)."""
    tqdm.write("Checking CPU temperature...")
    command_result = run_command(f"cat {config.cpu_thermal_file}")
    if command_result is None:
        tqdm.write("Failed to get CPU temperature. Continuing with the test...")
        return True
    temperature = int(command_result.stdout.strip())
    tqdm.write(f"CPU temperature: {temperature}°C")
    return temperature < config.limits.temperature_safe_limit


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


def commit_contains_patterns(commit: git.Commit, patterns: set[str]) -> bool:
    """Determine if a commit modifies any files that end with any of the given patterns."""
    return any(str(file).endswith(tuple(patterns)) for file in commit.stats.files)


def setup_repo(repo_path: str, repo_url: str, config: PipelineConfig) -> git.Repo:
    """Clones the repository if not present; otherwise opens the existing one."""
    if not os.path.exists(repo_path):
        tqdm.write(f"Cloning {repo_url} into {repo_path}...")
        return git.Repo.clone_from(repo_url, repo_path, multi_options=config.repo.clone_options)
    else:
        tqdm.write(f"Using existing repo at {repo_path}...")
        return git.Repo(repo_path)


def generate_tasks(batch_commits: list[git.Commit], config: PipelineConfig) -> list[git.Commit]:
    """Generate a list of energy measurement tasks for the provided commits."""
    tasks: list[git.Commit] = []
    num_runs: int = config.execution_plan.num_runs
    num_repeats: int = config.execution_plan.num_repeats
    randomize: bool = config.execution_plan.randomize_tasks
    for commit in batch_commits:
        if config.execution_plan.granularity == "commits" and not commit_contains_patterns(
            commit,
            config.tracked_file_extensions,
        ):
            tqdm.write(f"Skipping commit {commit.hexsha} as it does not contain C code.")
            continue
        else:
            tqdm.write(f"Adding commit {commit.hexsha} to the task list.")
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
    global_task_counter = 0
    start_time = time.time()
    config = load_config(config_path)
    project_name: str = os.path.basename(config.repo.url).replace(".git", "").strip().lower()
    project_dir: str = os.path.join("projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    repo_path: str = os.path.join(project_dir, ".cache" + project_name)
    output_file: str = os.path.join(project_dir, config.results.file)
    num_commits: int = config.execution_plan.num_commits
    batch_size: int = config.execution_plan.batch_size

    # Clear previous output file if exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Clone the repository to get the list of commits
    repo = setup_repo(repo_path, config.repo.url, config)
    if config.execution_plan.granularity == "branches":
        branches = list(repo.remotes.origin.refs)
        commits = [branch.commit for branch in branches]  # Get only the latest commit of each branch
        tqdm.write(f"Branches: {branches}")
        tqdm.write(f"Commits: {commits}")
    elif config.execution_plan.granularity == "tags":
        tags = list(repo.tags)
        commits = [commit for tag in tags for commit in repo.iter_commits(tag, max_count=num_commits)]
    else:
        commits = list(repo.iter_commits(config.repo.branch, max_count=num_commits))
    total_batches = (len(commits) + batch_size - 1) // batch_size
    tqdm.write(f"Total commits: {len(commits)} in {total_batches} batches (batch size: {batch_size})")

    # Run project setup commands
    if config.setup_commands:
        tqdm.write("\nRunning setup commands...")
        for command in config.setup_commands:
            run_command(command, repo_path)

    current_commit: str = ""

    # Process commits batch by batch
    for batch_index in range(total_batches):
        batch_commits = commits[batch_index * batch_size : (batch_index + 1) * batch_size]
        # Build list of tasks for this batch (each task is one energy measurement run)
        tasks: list[git.Commit] = generate_tasks(batch_commits, config)

        tqdm.write(f"\nProcessing batch {batch_index + 1}/{total_batches} with {len(tasks)} tasks...")

        # Use a tqdm progress bar for tasks in this batch
        with tqdm(total=len(tasks), desc=f"Batch {batch_index + 1}/{total_batches}", leave=False) as pbar:
            for _, commit in enumerate(tasks):
                build_failed = False
                # If the commit changes, checkout and build the project
                if current_commit != commit.hexsha:
                    tqdm.write(f"\n🔄 Checking out commit {commit.hexsha}...")
                    repo.git.checkout(commit.hexsha)
                    tqdm.write("Building the project...")
                    if config.execution_plan.mode == "run":
                        for command in config.compile_commands:
                            if run_command(command, repo_path) is None:
                                tqdm.write("Failed to build the project. Skipping this commit...")
                                build_failed = True
                                break
                        if build_failed:
                            continue
                    current_commit = commit.hexsha
                # Run a single energy measurement test for this task
                run_single_energy_test(repo_path, output_file, config, commit=commit, global_task_counter=global_task_counter)
                global_task_counter += 1
                elapsed_time = int(time.time() - start_time)
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                pbar.set_postfix(global_tasks=global_task_counter, elapsed=formatted_time)
                pbar.update(1)

    # Final step: checkout back to the latest commit on the branch
    repo.git.checkout(config.repo.branch)
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
