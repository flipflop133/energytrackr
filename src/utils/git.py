"""Git utility functions for cloning repositories and gathering commits."""

import os
from typing import Any

from git.objects.commit import Commit
from git.repo import Repo

from config.config_store import Config
from utils.logger import logger


def clone_or_open_repo(repo_path: str, repo_url: str, clone_options: list[str] | None = None) -> Repo:
    """Clone a Git repository from a URL or open an existing repository from a local path.

    Args:
        repo_path (str): The local path where the repository should be cloned or opened.
        repo_url (str): The URL of the remote Git repository.
        clone_options (list, optional): Additional options to pass to the clone command.

    Returns:
        git.Repo: An instance of the Git repository at the specified path.
    """
    if not os.path.exists(repo_path):
        logger.info(f"Cloning {repo_url} into {repo_path}")
        clone_opts = clone_options or []
        return Repo.clone_from(repo_url, repo_path, multi_options=clone_opts)
    else:
        logger.info(f"Using existing repo at {repo_path}")
        return Repo(repo_path)


def gather_commits(repo: Repo) -> list[Commit]:
    """Gather the commits that should be processed according to the execution plan.

    For branches, we just take one commit per branch. For tags, we take the num_commits newest
    commits for each tag. For commits, we take the specified number of commits from the specified
    branch, starting from the oldest commit. If oldest_commit is specified, we start from there.
    If newest_commit is specified, we stop at that commit.

    Args:
        repo (git.Repo): The git repository to gather commits from.

    Returns:
        list[git.Commit]: The list of commits to process.
    """
    conf = Config.get_config()
    plan = conf.execution_plan

    if plan.granularity == "branches":
        # One commit per branch
        branches = list(repo.remotes.origin.refs)
        return [branch.commit for branch in branches]

    elif plan.granularity == "tags":
        tags = list(repo.tags)
        commits = []
        for tag in tags:
            commits.extend(list(repo.iter_commits(tag, max_count=plan.num_commits)))
        return commits

    else:  # commits granularity
        # Get all commits from the branch in descending order (newest-first)
        commits = list(repo.iter_commits(conf.repo.branch))
        # Reverse to get ascending order (oldest-first) to make filtering more intuitive
        commits = list(reversed(commits))

        # If an oldest_commit is specified, start from that commit onward.
        if plan.oldest_commit:
            start_idx = next((i for i, c in enumerate(commits) if c.hexsha == plan.oldest_commit), None)
            if start_idx is not None:
                commits = commits[start_idx:]
            else:
                logger.warning("Oldest commit %s not found in commit history.", plan.oldest_commit)

        # If a newest_commit is specified, stop at that commit (inclusive)
        if plan.newest_commit:
            end_idx = next((i for i, c in enumerate(commits) if c.hexsha == plan.newest_commit), None)
            if end_idx is not None:
                commits = commits[: end_idx + 1]
            else:
                logger.warning("Newest commit %s not found in commit history.", plan.newest_commit)

        # If a number of commits is specified, take the most recent num_commits from the ascending list.
        # Since the list is ascending (oldest-first), we slice from the tail to keep the latest commits.
        if plan.num_commits:
            commits = commits[-plan.num_commits :]

        logger.info("Gathered %d commits from branch %s", len(commits), conf.repo.branch)

        return commits


def generate_commit_link(remote_url: str, commit_hash: str) -> str:
    """Generate a commit link from the remote URL and commit hash.

    Supports GitHub-style URLs.
    """
    if remote_url.startswith("git@"):
        try:
            parts = remote_url.split(":")
            domain = parts[0].split("@")[-1]
            repo_path = parts[1].replace(".git", "")
        except Exception:
            return "N/A"
        else:
            return f"https://{domain}/{repo_path}/commit/{commit_hash}"
    elif remote_url.startswith("https://"):
        repo_url = remote_url.replace(".git", "")
        return f"{repo_url}/commit/{commit_hash}"
    return "N/A"


def get_commit_details_from_git(commit_hash: str, repo: Repo) -> dict[str, Any]:
    """Retrieve commit details (summary, date, etc.) using GitPython."""
    try:
        commit_obj = repo.commit(commit_hash)
        commit_date = commit_obj.committed_datetime.strftime("%Y-%m-%d")
        commit_summary = commit_obj.summary
        commit_files = list(commit_obj.stats.files.keys())
        commit_link = "N/A"
        if repo.remotes:
            remote_url = repo.remotes[0].url
            commit_link = generate_commit_link(remote_url, commit_hash)
    except Exception:
        logger.exception(f"Error retrieving details for commit {commit_hash}")
        return {"commit_summary": "N/A", "commit_link": "N/A", "commit_date": "N/A", "files_modified": []}
    else:
        return {
            "commit_summary": commit_summary,
            "commit_link": commit_link,
            "commit_date": commit_date,
            "files_modified": commit_files,
        }
