"""Sorts a CSV file containing commit hashes and energy values based on the commit history of a Git repository."""

import csv
import logging
import subprocess
import sys

EXPECTED_ROW_LENGTH = 2  # Number of columns expected in the CSV file


def get_commit_history(repo_path: str) -> list[str]:
    """Retrieve commit history from the Git repository in chronological order."""
    try:
        result = subprocess.run(
            ["git", "rev-list", "--reverse", "--all"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        logging.exception("Error retrieving commit history.")
        sys.exit(1)


def read_csv(file_path: str) -> list[tuple[str, str]]:
    """Read commit hashes and energy values from the CSV file, keeping duplicates."""
    data = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == EXPECTED_ROW_LENGTH:
                commit_hash, energy = row
                data.append((commit_hash, energy))  # Store as tuple to preserve duplicates
    return data


def write_csv(file_path: str, sorted_data: list[tuple[str, str]]) -> None:
    """Write the sorted commit data to a new CSV file, keeping duplicates."""
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for commit, energy in sorted_data:
            writer.writerow([commit, energy])


def reorder_commits(csv_file: str, repo_path: str, output_file: str) -> None:
    """Reorder commits based on Git history and write to a new CSV file while preserving duplicates."""
    commit_history = get_commit_history(repo_path)
    csv_data = read_csv(csv_file)

    # Sort the list while preserving duplicates
    commit_order = {commit: i for i, commit in enumerate(commit_history)}
    sorted_data = sorted(csv_data, key=lambda x: commit_order.get(x[0], float("inf")))

    write_csv(output_file, sorted_data)
    logging.info(f"Reordered CSV written to {output_file}")
