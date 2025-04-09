"""Unit tests for utils.sort module."""

import csv
import subprocess
from pathlib import Path
from typing import Any, Never

import pytest

from utils.sort import (
    get_commit_history,
    read_csv,
    reorder_commits,
    write_csv,
)


@pytest.fixture
def dummy_repo_with_commits(tmp_path: Path) -> tuple[Path, list[str]]:
    """Fixture to create a dummy git repository with 3 commits."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)

    commits = []
    for i in range(3):
        file = repo_dir / f"file{i}.txt"
        file.write_text(f"Hello {i}")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", f"commit {i}"], cwd=repo_dir, check=True)
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
        commits.append(commit_hash)

    return repo_dir, commits


def test_get_commit_history(dummy_repo_with_commits: tuple[Path, list[str]]) -> None:
    """Test get_commit_history function."""
    repo_dir, expected_commits = dummy_repo_with_commits
    history = get_commit_history(str(repo_dir))
    assert history == expected_commits


def test_get_commit_history_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_commit_history exits cleanly when subprocess fails."""

    def mock_run(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Never:  # noqa: ARG001
        """Mock subprocess.run to simulate a failure."""
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0], output="", stderr="Simulated error")

    monkeypatch.setattr("subprocess.run", mock_run)

    with pytest.raises(SystemExit):
        get_commit_history(str(tmp_path))


def test_read_csv(tmp_path: Path) -> None:
    """Test read_csv function."""
    file_path = tmp_path / "test.csv"
    rows = [
        ("abc123", "12.5"),
        ("abc123", "13.1"),
        ("def456", "10.2"),
        ("badrow",),  # should be skipped
    ]
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    data = read_csv(str(file_path))
    assert data == [("abc123", "12.5"), ("abc123", "13.1"), ("def456", "10.2")]


def test_write_csv(tmp_path: Path) -> None:
    """Test write_csv function."""
    file_path = tmp_path / "out.csv"
    sample_data = [("abc123", "12.5"), ("def456", "10.2")]
    write_csv(str(file_path), sample_data)

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows == [["abc123", "12.5"], ["def456", "10.2"]]


def test_reorder_commits(tmp_path: Path, dummy_repo_with_commits: tuple[Path, list[str]]) -> None:
    """Test reorder_commits function."""
    repo_dir, commits = dummy_repo_with_commits

    # Input CSV has commits in reverse order, with a duplicate
    input_csv = tmp_path / "input.csv"
    rows = [(commits[2], "5.0"), (commits[1], "4.5"), (commits[0], "6.1"), (commits[1], "4.6")]
    with open(input_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    output_csv = tmp_path / "sorted.csv"
    reorder_commits(str(input_csv), str(repo_dir), str(output_csv))

    with open(output_csv, newline="") as f:
        reader = csv.reader(f)
        sorted_rows = list(reader)

    # Expected order matches git history, including duplicates
    expected = [
        [commits[0], "6.1"],
        [commits[1], "4.5"],
        [commits[1], "4.6"],
        [commits[2], "5.0"],
    ]
    assert sorted_rows == expected
