"""Unit tests for SetDirectoryStage class."""

import os
from pathlib import Path

import pytest

from pipeline.core_stages.set_directory_stage import SetDirectoryStage
from utils.exceptions import MissingContextKeyError, TargetDirectoryNotFoundError


def test_set_directory_success(tmp_path: Path) -> None:
    """Test successful directory change based on commit context."""
    commit_id = "abc123"
    target_dir = tmp_path / f"repo_{commit_id}"
    target_dir.mkdir(parents=True)

    context = {"commit": commit_id, "repo_path": str(tmp_path / "repo")}

    stage = SetDirectoryStage()
    current_dir = os.getcwd()
    try:
        stage.run(context)
        assert Path(os.getcwd()) == target_dir
    finally:
        os.chdir(current_dir)  # Restore original directory to avoid side effects


def test_set_directory_missing_commit() -> None:
    """Test error when 'commit' key is missing in context."""
    context = {"repo_path": "/fake/path"}

    stage = SetDirectoryStage()
    with pytest.raises(MissingContextKeyError) as exc_info:
        stage.run(context)

    assert "commit" in str(exc_info.value)


def test_set_directory_not_found(tmp_path: Path) -> None:
    """Test error when target directory does not exist."""
    commit_id = "deadbeef"
    repo_path = tmp_path / "repo"
    context = {"commit": commit_id, "repo_path": str(repo_path)}

    expected_path = (repo_path.parent / f"{repo_path.name}_{commit_id}").resolve()

    stage = SetDirectoryStage()
    with pytest.raises(TargetDirectoryNotFoundError) as exc_info:
        stage.run(context)

    assert str(expected_path) in str(exc_info.value)
