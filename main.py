"""Test energy consumption across Git commits."""

#!/usr/bin/env python3
import argparse
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
from config_store import Config


def load_config(config_path: str) -> None:
    """Load configuration from a JSON file and validate it against the config model."""
    config_str = Path(config_path).read_text()
    Config.set_config(PipelineConfig.model_validate_json(config_str))


def measure_energy(repo_path: str, test_command: str, output_file: str) -> bool:
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

        path: str = repo_path + Config.get_config().execution_plan.test_command_path
        result: subprocess.CompletedProcess[str] | None = run_command(perf_command, path)

        # If the command itself couldn't launch
        if result is None:
            tqdm.write("Failed to run perf command. Skipping energy measurement.")
            # Return False if ignoring failures is off; else True
            return Config.get_config().execution_plan.ignore_failures

        # If `perf` reported an error
        if result.returncode != 0:
            if not Config.get_config().execution_plan.ignore_failures:
                tqdm.write(f"Error running perf command: {result.stdout}")
                return False  # Must skip the rest of this commit
            else:
                # If ignoring failures, keep going
                tqdm.write(f"Ignore-failures=True; continuing despite error:\n{result.stdout}")

        perf_output: str = result.stdout  # `perf stat` often uses stderr, but we captured all

        # Extract energy values from perf output
        energy_pkg = extract_energy_value(perf_output, "power/energy-pkg/")
        if energy_pkg is None:
            tqdm.write("Failed to extract energy measurement from perf output.")
            # If ignoring failures is off, signal skip
            return Config.get_config().execution_plan.ignore_failures

        tqdm.write(f"ENERGY_PKG: {energy_pkg}")

        # Get the current Git commit hash
        repo = git.Repo(repo_path)
        commit_hash: str = repo.head.object.hexsha

        # Append results to file
        with open(output_file, "a") as file:
            file.write(f"{commit_hash},{energy_pkg}\n")

        tqdm.write(f"Results appended to {output_file}")
        return True

    except Exception as e:
        tqdm.write(f"Error: {e}")
        # If ignoring is off, we skip
        return Config.get_config().execution_plan.ignore_failures


def extract_energy_value(perf_output: str, event_name: str) -> str | None:
    """Extracts the energy measurement value from the perf output."""
    for line in perf_output.split("\n"):
        if event_name in line:
            parts = line.split()
            if parts:
                if "<not" in parts[0]:
                    # Handling the unsupported measurement case
                    return None
                value = parts[0]  # First column is the energy value
                # Remove commas as they can interfere with CSV formatting
                return value.replace(",", "")
    return None


def run_single_energy_test(
    repo_path: str,
    output_file: str,
    commit: git.Commit,
    global_task_counter: int,
) -> bool:
    """Runs a single instance of the energy measurement test. Returns True if success, else False."""
    # Ensure CPU temperature is within safe limits
    while not is_temperature_safe():
        tqdm.write("⚠️ CPU temperature is too high. Waiting for it to cool down...")
        sleep(1)

    success = True
    # Run pre-command if provided
    pre_command = Config.get_config().execution_plan.pre_command
    if pre_command:
        if global_task_counter != 0 and Config.get_config().execution_plan.pre_command_condition_files:
            files: set[str] = Config.get_config().execution_plan.pre_command_condition_files
            if commit_contains_patterns(commit, files):
                return_code = run_command(pre_command, repo_path)
        else:
            return_code = run_command(pre_command, repo_path)
        if return_code is None or return_code.returncode != 0:
            tqdm.write(f"Error: Pre-command '{pre_command}' failed.")
            success = False

    # Run the energy measurement
    if not success and not Config.get_config().execution_plan.ignore_failures:
        return success
    success = measure_energy(repo_path, Config.get_config().execution_plan.test_command, output_file)

    # Run post-command if provided
    post_command = Config.get_config().execution_plan.post_command
    if post_command:
        run_command(post_command, repo_path)

    return success


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
        if Config.get_config().execution_plan.ignore_failures:
            tqdm.write(f"Error: Command '{arg}' failed with exit code {retcode}")
            tqdm.write("But ignore mode enabled, continuing with the test...")
            tqdm.write(f"Output: {output}")
            return subprocess.CompletedProcess(args=arg, returncode=retcode, stdout=output)
        else:
            tqdm.write(f"Error: Command '{arg}' failed with exit code {retcode}")
            return None

    return subprocess.CompletedProcess(args=arg, returncode=retcode, stdout=output)


def is_temperature_safe() -> bool:
    """Check if temperature is within safe limits (CPU not throttling)."""
    tqdm.write("Checking CPU temperature...")
    command_result = run_command(f"cat {Config.get_config().cpu_thermal_file}")
    if command_result is None:
        tqdm.write("Failed to get CPU temperature. Continuing with the test...")
        return True
    temperature = int(command_result.stdout.strip())
    tqdm.write(f"CPU temperature: {temperature}°C")
    return temperature < Config.get_config().limits.temperature_safe_limit


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


def generate_tasks(batch_commits: list[git.Commit]) -> list[git.Commit]:
    """Generate a list of commits repeated by (num_runs * num_repeats).

    If randomize is True, shuffle the resulting list.
    """
    tasks: list[git.Commit] = []
    num_runs: int = Config.get_config().execution_plan.num_runs
    num_repeats: int = Config.get_config().execution_plan.num_repeats
    randomize: bool = Config.get_config().execution_plan.randomize_tasks

    for commit in batch_commits:
        # If granularity is 'commits' and the commit does not contain tracked patterns, skip it
        if Config.get_config().execution_plan.granularity == "commits" and not commit_contains_patterns(
            commit,
            Config.get_config().tracked_file_extensions,
        ):
            tqdm.write(f"Skipping commit {commit.hexsha} as it does not match tracked files.")
            continue

        tqdm.write(f"Adding commit {commit.hexsha} to the task list.")
        # Each commit is scheduled for (num_runs * num_repeats) tests
        tasks.extend([commit] * (num_runs * num_repeats))

    if randomize:
        random.shuffle(tasks)

    return tasks


def verify_perf_access() -> bool:
    """Check if user is allowed to use perf without sudo."""
    # read perf_event_paranoid
    command_result = run_command("cat /proc/sys/kernel/perf_event_paranoid")
    if command_result is None:
        return False
    perf_event_paranoid = int(command_result.stdout.strip())
    return perf_event_paranoid == -1


def retrieve_common_tests(commits: list[git.Commit], repo: git.Repo) -> list[str]:
    """Retrieve common test methods between first and last commit."""
    # Checkout first commit and collect tests
    repo.git.checkout(commits[0].hexsha)
    run_command("mvn exec:java")
    f = open("discovered_tests.txt")
    tests_first = set(clean_test_output(f.read()))

    # Checkout last commit and collect tests
    repo.git.checkout(commits[-1].hexsha)
    run_command("mvn exec:java")
    f = open("discovered_tests.txt")
    tests_last = set(clean_test_output(f.read()))

    return sorted(tests_first & tests_last)  # Intersection


def clean_test_output(raw_output: str) -> list[str]:
    lines = raw_output.splitlines()
    return [line.strip().replace("()", "") for line in lines if line.strip() and "UnknownClass" not in line]


def main(config_path: str) -> None:
    """Main function for the energy consumption testing tool.

    Loads the configuration from a JSON file, clones the repository, and processes
    commits in batches. For each batch, it checks out the commit, builds the project,
    runs the energy measurement test, and saves the results to a CSV file.

    After processing each batch, it deletes the repository cache to free up space.
    Finally, it checks out back to the latest commit on the branch.
    """
    if not verify_perf_access():
        tqdm.write("Error: perf_event_paranoid must be set to -1 for perf to work.")
        return
    load_config(config_path)
    config = Config.get_config()

    project_name: str = os.path.basename(config.repo.url).replace(".git", "").strip().lower()
    project_dir: str = os.path.join("projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    repo_path: str = os.path.join(project_dir, ".cache" + project_name)
    output_file: str = os.path.join(project_dir, config.results.file)

    num_commits: int | None = config.execution_plan.num_commits
    batch_size: int = config.execution_plan.batch_size

    # Clear previous output file if exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Clone or open the repository
    repo = setup_repo(repo_path, config.repo.url, config)

    # Gather commits based on granularity and user config
    if config.execution_plan.granularity == "branches":
        branches = list(repo.remotes.origin.refs)
        commits = [branch.commit for branch in branches]  # One commit per branch
        tqdm.write(f"Branches: {branches}")
        tqdm.write(f"Commits: {commits}")
    elif config.execution_plan.granularity == "tags":
        tags = list(repo.tags)
        commits = [c for tag in tags for c in repo.iter_commits(tag, max_count=num_commits)]
    else:
        commits = list(repo.iter_commits(config.repo.branch))

        # Restrict commits if from_commit or to_commit is set
        if config.execution_plan.from_commit:
            from_commit_index = next((i for i, c in enumerate(commits) if c.hexsha == config.execution_plan.from_commit), None)
            if from_commit_index is not None:
                commits = commits[from_commit_index:]

        if config.execution_plan.to_commit:
            to_commit_index = next((i for i, c in enumerate(commits) if c.hexsha == config.execution_plan.to_commit), None)
            if to_commit_index is not None:
                # +1 because we include the to_commit itself
                commits = commits[: to_commit_index + 1]

        if num_commits:
            commits = commits[:num_commits]

    if config.execution_plan.execute_common_tests:
        common_tests = retrieve_common_tests(commits, repo)
        test_arg = ",".join(common_tests)
        config.execution_plan.test_command = f"mvn test -Dtest={test_arg}"
        tqdm.write(f"Common tests: {test_arg}")

    total_batches = (len(commits) + batch_size - 1) // batch_size
    tqdm.write(f"Total commits: {len(commits)} in {total_batches} batches (batch size: {batch_size})")

    # Run project-level setup commands
    if config.setup_commands:
        tqdm.write("\nRunning setup commands...")
        for command in config.setup_commands:
            run_command(command, repo_path)

    current_commit: str = ""
    global_task_counter = 0
    start_time = time.time()

    # A set of commit hashes to skip (because they failed once and ignore_failures==false)
    skip_commits = set()

    # Process commits batch by batch
    for batch_index in range(total_batches):
        batch_commits = commits[batch_index * batch_size : (batch_index + 1) * batch_size]
        # Build list of tasks for this batch
        tasks: list[git.Commit] = generate_tasks(batch_commits)

        tqdm.write(f"\nProcessing batch {batch_index + 1}/{total_batches} with {len(tasks)} tasks...")
        with tqdm(total=len(tasks), desc=f"Batch {batch_index + 1}/{total_batches}", leave=False) as pbar:
            i = 0
            while i < len(tasks):
                commit = tasks[i]
                # If this commit is in skip_commits, we skip directly
                if commit.hexsha in skip_commits:
                    pbar.update(1)
                    i += 1
                    continue

                build_failed = False

                # Checkout and build if we're on a new commit
                if current_commit != commit.hexsha:
                    tqdm.write(f"\n🔄 Checking out commit {commit.hexsha}...")
                    repo.git.checkout(commit.hexsha)
                    if config.execution_plan.mode == "benchmarks":
                        compile_commands = config.execution_plan.compile_commands or []
                        for cmd in compile_commands:
                            if run_command(cmd, repo_path) is None:
                                tqdm.write("Failed to build the project. Skipping this commit...")
                                build_failed = True
                                break
                        if build_failed:
                            # If building fails and ignoring is off => skip entire commit
                            if not config.execution_plan.ignore_failures:
                                skip_commits.add(commit.hexsha)
                            current_commit = commit.hexsha
                            pbar.update(1)
                            i += 1
                            continue
                    current_commit = commit.hexsha

                # Run a single energy measurement test
                success = run_single_energy_test(
                    repo_path,
                    output_file,
                    commit=commit,
                    global_task_counter=global_task_counter,
                )
                global_task_counter += 1

                # If it failed and ignoring is off => skip the rest of this commit
                if (not success) and (not config.execution_plan.ignore_failures):
                    skip_commits.add(commit.hexsha)

                # Progress bar housekeeping
                elapsed_time = int(time.time() - start_time)
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                pbar.set_postfix(global_tasks=global_task_counter, elapsed=formatted_time)
                pbar.update(1)

                i += 1  # Next task in this batch

    # Final step: checkout back to the latest commit on the configured branch
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
