"""Sorts a CSV file containing commit hashes and energy values based on the commit history of a Git repository."""

import csv
import logging
import subprocess
import sys

EXPECTED_ROW_LENGTH = 2  # Number of columns expected in the CSV file


def get_commit_history(repo_path: str) -> list[str]:
    """Retrieve the commit history from a Git repository in chronological order.

    Args:
        repo_path (str): The file system path to the Git repository.

    Returns:
        list[str]: A list of commit hashes in chronological order (oldest to newest).

    """
    try:
        result = subprocess.run(
            ["git", "rev-list", "--reverse", "--all"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        logging.exception("Error retrieving commit history.")
        sys.exit(1)
    return result.stdout.strip().split("\n")


def read_csv(file_path: str) -> list[tuple[str, str]]:
    """Reads a CSV file containing commit hashes and energy values, preserving duplicate entries.

    Args:
        file_path (str): The path to the CSV file to read.

    Returns:
        list[tuple[str, str]]: A list of tuples, each containing a commit hash and its corresponding energy value.
    """
    data = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == EXPECTED_ROW_LENGTH:
                commit_hash, energy = row
                data.append((commit_hash, energy))  # Store as tuple to preserve duplicates
    return data


def write_csv(file_path: str, sorted_data: list[tuple[str, str]]) -> None:
    """Write the sorted commit data to a CSV file, preserving duplicates.

    Args:
        file_path (str): The path to the output CSV file.
        sorted_data (list[tuple[str, str]]): A list of tuples, each containing a commit
                                             identifier and its associated energy value.
    """
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for commit, energy in sorted_data:
            writer.writerow([commit, energy])


def reorder_commits(csv_file: str, repo_path: str, output_file: str) -> None:
    """Reorders the rows of a CSV file containing commit hashes.

    Reorders the rows of a CSV file containing commit hashes to match the chronological order of
    commits in a Git repository, preserving duplicate entries, and writes the result to a new CSV file.

    Args:
        csv_file (str): Path to the input CSV file where the first column contains commit hashes.
        repo_path (str): Path to the local Git repository to retrieve commit history.
        output_file (str): Path to the output CSV file where the reordered data will be written.

    Notes:
        - Rows with commit hashes not found in the repository history are placed at the end of the output file.
        - Assumes the existence of helper functions: get_commit_history, read_csv, and write_csv.
    """
    commit_history = get_commit_history(repo_path)
    csv_data = read_csv(csv_file)

    # Sort the list while preserving duplicates
    commit_order = {commit: i for i, commit in enumerate(commit_history)}
    sorted_data = sorted(csv_data, key=lambda x: commit_order.get(x[0], float("inf")))

    write_csv(output_file, sorted_data)
    logging.info("Reordered CSV written to %s", output_file)
