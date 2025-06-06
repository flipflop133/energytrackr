"""Unit tests for the CheckoutStage class in the pipeline.core_stages module."""

import os
from pathlib import Path
from typing import Any

import git
import pytest

from energytrackr.pipeline.core_stages.checkout_stage import CheckoutStage


@pytest.fixture
def dummy_git_repo(tmp_path: Path) -> tuple[git.Repo, list[str]]:
    """Create a temporary Git repo with 2 commits and return it.

    Args:
        tmp_path (Path): The temporary path to create the repository.

    Returns:
        tuple[git.Repo, list[str]]: A tuple containing the Git repository and a list of commit hashes.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo = git.Repo.init(str(repo_path))
    commits = []

    for i in range(2):
        file = repo_path / f"file{i}.txt"
        file.write_text(f"Content {i}")
        repo.index.add([str(file)])
        commit = repo.index.commit(f"Commit {i}")
        commits.append(commit.hexsha)

    return repo, commits


def test_checkout_stage_success(dummy_git_repo: tuple[git.Repo, list[str]]) -> None:
    """Test the CheckoutStage class with a valid commit."""
    repo, commits = dummy_git_repo
    os.chdir(repo.working_tree_dir)

    context: dict[str, Any] = {
        "commit": commits[0],
        "repo_path": repo.working_tree_dir,
    }

    stage = CheckoutStage()
    stage.run(context)

    # Check that the working directory is at the correct commit
    assert repo.head.commit.hexsha.startswith(commits[0])
    assert "abort_pipeline" not in context or context["abort_pipeline"] is False


def test_checkout_stage_failure(dummy_git_repo: tuple[git.Repo, list[str]]) -> None:
    """Test the CheckoutStage class with an invalid commit."""
    repo, _ = dummy_git_repo
    os.chdir(repo.working_tree_dir)

    context: dict[str, Any] = {
        "commit": "nonexistentcommit",
        "repo_path": repo.working_tree_dir,
    }

    stage = CheckoutStage()
    stage.run(context)

    assert context.get("abort_pipeline") is True
