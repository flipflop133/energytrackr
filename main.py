import os
import subprocess
import git
import argparse
import pandas as pd
from time import sleep
import statistics
import json
import time
from pyparsing import Any


def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as file:
        return json.load(file)


def run_energy_test(
    repo_path: str,
    output_file: str,
    config: dict[Any, Any],
) -> None:
    """
    Runs the Bash script for energy measurement multiple times.

    Args:
        repo_path (str): Path to the repository being tested.
        output_file (str): Path to the CSV file to store the energy measurement results
        config (dict): Configuration dictionary.
    """
    # Path to the Bash script to run for energy measurement
    script_path = os.path.join(os.getcwd(), "measure_energy.sh")

    # Run the Bash script for energy measurement multiple times
    for _ in range(config["test"]["num_runs"]):
        # Check if temperature is within safe limits (cpu not throttling)
        while not is_temperature_safe(config):
            print("⚠️ CPU temperature is too high. Waiting for it to cool down...")
            sleep(1)
        # Run pre-command
        if config["test"]["pre_command"]:
            subprocess.run(
                config["test"]["pre_command"],
                shell=True,
                check=True,
            )
        # Run the actual energy measurement script
        subprocess.run(
            # Use sudo to run the script, and pass the necessary arguments
            [
                "sudo",
                "bash",
                script_path,
                repo_path,
                config["test"]["command"],
                output_file,
            ],
            check=True,
        )
        # Run post-command
        if config["test"]["post_command"]:
            subprocess.run(
                config["test"]["post_command"],
                shell=True,
                check=True,
            )
        # Display current progress
        print(f"Current commit progress: {_ + 1}/{config['test']['num_runs']}")


def is_temperature_safe(config: dict[Any, Any]) -> bool:
    """Check if temperature is within safe limits.
    Safe means the CPU is not throttling.

    Returns:
        bool: _description_
    """
    # Read the current temperature from the thermal zone file
    temperature = int(
        subprocess.run(
            ["cat", f"/sys/class/thermal/{config['cpu_themal_zone']}/temp"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    # Check if the temperature is within safe limits
    if temperature < config["thresholds"]["temperature_safe_limit"]:
        return True
    else:
        return False


def analyze_results(output_file: str) -> None:
    """
    Analyzes energy usage trends and detects significant regressions.

    Args:
        output_file (str): Path to the CSV file containing the results of the energy measurements.
    """
    # Read the CSV file into a DataFrame, specifying column names
    df = pd.read_csv(
        output_file,
        header=None,
        names=["commit", "energy_used", "exit_code"],
    )

    # Convert the 'energy_used' column to integer type
    df["energy_used"] = df["energy_used"].astype(int)

    # Print the entire DataFrame to show energy consumption trends
    print("\nEnergy Consumption Trend:")
    print(df)

    # Calculate percentage increase in energy usage compared to the previous commit
    df["prev_energy"] = df["energy_used"].shift(1)
    df["increase"] = (df["energy_used"] - df["prev_energy"]) / df["prev_energy"] * 100

    # Identify significant regressions where energy usage increased by more than 20%
    regressions = df[df["increase"] > 20]

    # Print regression information or confirm no significant regressions
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
        """Reads the current energy consumption value in microjoules."""
        return int(
            subprocess.run(
                ["sudo", "cat", "/sys/class/powercap/intel-rapl:0/energy_uj"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )

    # Warm-up phase to collect baseline power samples
    power_samples = []
    prev_energy = get_energy_uj()

    print(f"Gathering baseline power data for {warmup_time} seconds...")
    for _ in range(warmup_time):
        sleep(1)
        current_energy = get_energy_uj()
        power = current_energy - prev_energy  # Power per second (μJ/s)
        power_samples.append(power)
        prev_energy = current_energy

    # Compute median and MAD
    median_power = statistics.median(power_samples)
    mad_power = (
        statistics.median([abs(x - median_power) for x in power_samples]) or 1
    )  # Avoid division by zero

    print(f"Baseline median power: {median_power / 1_000_000} W")
    print(f"Baseline MAD: {mad_power / 1_000_000} W")

    # Monitoring phase
    for _ in range(duration - warmup_time):
        sleep(1)
        current_energy = get_energy_uj()
        power = current_energy - prev_energy
        prev_energy = current_energy

        # Compute Modified Z-Score
        mz_score = 0.6745 * (power - median_power) / mad_power

        print(f"Power consumption: {power / 1_000_000} W")
        print(f"Modified Z-Score: {mz_score}")

        if abs(mz_score) > k:
            print("⚠️ System is NOT stable!")
            return False

        print("✅ System is stable.\n")

    return True


def commit_contains_c_code(commit: git.Commit, config: dict[Any, Any]) -> bool:
    files = commit.stats.files
    for file in files.keys():
        if file.endswith(tuple(config["file_extensions"])):
            return True
    return False


def main(
    config_path: str,
) -> None:
    """
    Runs the energy measurement test on a given Git repository.

    Args:
        repo_url (str): URL of the Git repository to clone.
        branch (str): Branch to use for testing.
        test_command (str): Command to run for energy measurement.
        num_commits (int): Number of commits to test.
        num_runs (int): Number of times to run each test.

    Returns:
        None
    """
    # Save start time
    start_time = time.time()
    # Setup computer for energy measurement
    subprocess.run(
        [
            "sudo",
            "bash",
            "system_setup.sh",
        ],
        check=False,
    )
    # Check if the system is stable before running the test
    # if not is_system_stable():
    #    print("❌ System is not stable. Exiting...")
    #    return
    config: dict[Any, Any] = load_config(config_path)
    project_name = os.path.basename(config["repository"]["url"]).replace(".git", "")
    project_dir = os.path.join("projects", project_name)
    os.makedirs(project_dir, exist_ok=True)

    repo_path = os.path.join(project_dir, ".cache" + project_name)
    output_file = os.path.join(project_dir, config["output"]["file"])

    # Clone repo if not already present
    if not os.path.exists(repo_path):
        print(f"Cloning {config['repository']['url']} into {repo_path}...")
        repo = git.Repo.clone_from(config["repository"]["url"], repo_path)
    else:
        print(f"Using existing repo at {repo_path}...")
        repo = git.Repo(repo_path)

    commits = list(
        repo.iter_commits(
            config["repository"]["branch"], max_count=config["test"]["num_commits"]
        )
    )

    # Clear previous results
    if os.path.exists(output_file):
        os.remove(output_file)

    for commit in commits:
        print(f"\n🔄 Checking out {commit.hexsha}...")
        repo.git.checkout(commit.hexsha)
        # Get the paths of the files changed in the commit
        if not commit_contains_c_code(commit, config):
            print("Skipping commit as it does not contain C code.")
            continue
        # build the project
        print("Building the project...")
        for command in config["compile_commands"]:
            subprocess.run(
                command,
                shell=True,
                cwd=repo_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # Run the energy test
        run_energy_test(
            repo_path,
            output_file,
            config,
        )
        # Display current progress
        print(f"Global progress: {commits.index(commit) + 1}/{len(commits)}")
        # Display elapsed time
        elapsed_time = int(time.time() - start_time)  # Convert to integer seconds

        # Format as HH:MM:SS
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        print(f"Elapsed time: {formatted_time}")
    # Checkout back to latest
    repo.git.checkout(config["repository"]["branch"])
    print("\n✅ Restored to latest commit.")

    # Analyze energy trends
    analyze_results(output_file)


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
        #    print("❌ System is not stable. Exiting...")
