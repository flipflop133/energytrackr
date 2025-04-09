"""Git utility functions for cloning repositories and gathering commits."""

import os

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
    branch, starting from the newest commit. If oldest_commit is specified, we start from there.
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

        # If an oldest_commit is specified, start from that commit onward
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

        # Finally, if a number of commits is specified, reverse back to descending order (newest-first)
        # then take the first num_commits. This ensures that if newest_commit is the tip, it remains first.
        if plan.num_commits:
            commits = list(reversed(commits))
            commits = commits[: plan.num_commits]

        logger.info("Gathered %d commits from branch %s", len(commits), conf.repo.branch)
        logger.info("Gathered commits: %s", commits)

        return commits
