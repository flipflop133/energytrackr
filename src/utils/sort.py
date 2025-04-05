import csv
import subprocess
import sys


def get_commit_history(repo_path):
    """Retrieve commit history from the Git repository in chronological order."""
    try:
        result = subprocess.run(
            ["git", "rev-list", "--reverse", "--all"], cwd=repo_path, capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving commit history: {e}", file=sys.stderr)
        sys.exit(1)


def read_csv(file_path):
    """Read commit hashes and energy values from the CSV file, keeping duplicates."""
    data = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                commit_hash, energy = row
                data.append((commit_hash, energy))  # Store as tuple to preserve duplicates
    return data


def write_csv(file_path, sorted_data):
    """Write the sorted commit data to a new CSV file, keeping duplicates."""
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for commit, energy in sorted_data:
            writer.writerow([commit, energy])


def reorder_commits(csv_file, repo_path, output_file):
    """Reorder commits based on Git history and write to a new CSV file while preserving duplicates."""
    commit_history = get_commit_history(repo_path)
    csv_data = read_csv(csv_file)

    # Sort the list while preserving duplicates
    commit_order = {commit: i for i, commit in enumerate(commit_history)}
    sorted_data = sorted(csv_data, key=lambda x: commit_order.get(x[0], float("inf")))

    write_csv(output_file, sorted_data)
    print(f"Reordered CSV written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python reorder_commits.py <csv_file> <repo_path> <output_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    repo_path = sys.argv[2]
    output_file = sys.argv[3]

    reorder_commits(csv_file, repo_path, output_file)
