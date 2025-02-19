import os
import subprocess
import git
import argparse
import pandas as pd


def run_energy_test(
    repo_path: str, test_command: str, output_file: str, num_runs: int
) -> None:
    """
    Runs the Bash script for energy measurement multiple times.

    Args:
        repo_path (str): Path to the repository being tested.
        test_command (str): Command to run the test under.
        output_file (str): Path to the CSV file to write the results to.
        num_runs (int): Number of times to run the energy measurement.
    """
    # Path to the Bash script to run for energy measurement
    script_path = os.path.join(os.getcwd(), "measure_energy.sh")

    # Run the Bash script for energy measurement multiple times
    for _ in range(num_runs):
        subprocess.run(
            # Use sudo to run the script, and pass the necessary arguments
            ["sudo", "bash", script_path, repo_path, test_command, output_file],
            check=True,
        )


def analyze_results(output_file: str) -> None:
    """
    Analyzes energy usage trends and detects significant regressions.

    Args:
        output_file (str): Path to the CSV file containing the results of the energy measurements.
    """
    # Read the CSV file into a DataFrame, specifying column names
    df = pd.read_csv(output_file, names=["commit", "energy_used", "exit_code"])

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


def main(
    repo_url: str, branch: str, test_command: str, num_commits: int, num_runs: int
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
    project_name = os.path.basename(repo_url).replace(".git", "")
    project_dir = os.path.join("projects", project_name)
    os.makedirs(project_dir, exist_ok=True)

    repo_path = os.path.join(project_dir, ".cache" + project_name)
    output_file = os.path.join(project_dir, "energy_results.csv")

    # Clone repo if not already present
    if not os.path.exists(repo_path):
        print(f"Cloning {repo_url} into {repo_path}...")
        repo = git.Repo.clone_from(repo_url, repo_path)
    else:
        print(f"Using existing repo at {repo_path}...")
        repo = git.Repo(repo_path)

    commits = list(repo.iter_commits(branch, max_count=num_commits))

    # Clear previous results
    if os.path.exists(output_file):
        os.remove(output_file)

    for commit in commits:
        print(f"\n🔄 Checking out {commit.hexsha}...")
        repo.git.checkout(commit.hexsha)
        run_energy_test(repo_path, test_command, output_file, num_runs)

    # Checkout back to latest
    repo.git.checkout(branch)
    print("\n✅ Restored to latest commit.")

    # Analyze energy trends
    analyze_results(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test energy consumption across Git commits."
    )
    parser.add_argument("repo_url", help="Git repository URL")
    parser.add_argument("branch", help="Branch to test")
    parser.add_argument("test_command", help="Command to run tests")
    parser.add_argument("num_commits", type=int, help="Number of commits to test")
    parser.add_argument(
        "num_runs", type=int, help="Number of times to measure per commit"
    )

    args = parser.parse_args()

    main(args.repo_url, args.branch, args.test_command, args.num_commits, args.num_runs)
