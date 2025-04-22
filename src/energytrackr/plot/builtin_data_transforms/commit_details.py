# src/plot/builtin_data_transforms/commit_details.py
"""Fetch commit metadata for each valid commit."""

from __future__ import annotations

from pathlib import Path

from git import Repo

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Transform
from energytrackr.utils.git_utils import get_commit_details_from_git
from energytrackr.utils.logger import logger


class CommitDetails(Transform):
    """Fetch commit metadata (date, summary, files_modified, link) for each valid commit.

    Stores a dict in ctx.artefacts['commit_details'] keyed by full hash.
    """

    def apply(self, ctx: Context) -> None:  # noqa: PLR6301
        """Fetch commit metadata for each valid commit.

        Args:
            ctx (Context): The plotting context containing artefacts and statistics.
        """
        logger.info("CommitDetails: fetching commit metadata")
        valid = ctx.stats.get("valid_commits", [])
        details: dict[str, dict[str, str]] = {}
        git_repo_path = ctx.artefacts["git_repo_path"]
        if not git_repo_path or not Path(git_repo_path).is_dir():
            logger.warning("CommitDetails: no repo path; skipping metadata fetch")
        else:
            ctx.artefacts["project_name"] = Path(git_repo_path).name
            try:
                repo = Repo(git_repo_path)
            except Exception:
                logger.exception("CommitDetails: failed to open repo at %s", git_repo_path)
            else:
                for commit in valid:
                    try:
                        details[commit] = get_commit_details_from_git(commit, repo)
                    except Exception:
                        logger.exception("CommitDetails: failed to get details for %s", commit)

        logger.debug("CommitDetails: %d commits with details", len(details))
        ctx.artefacts["commit_details"] = details
