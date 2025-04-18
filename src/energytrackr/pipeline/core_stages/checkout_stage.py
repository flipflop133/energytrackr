"""Module to checkout a specific commit in a git repository."""

import os
from typing import Any

import git

from energytrackr.pipeline.stage_interface import PipelineStage
from energytrackr.utils.logger import logger


class CheckoutStage(PipelineStage):
    """Checks out the commit specified in context["current_commit"]."""

    def run(self, context: dict[str, Any]) -> None:  # noqa: PLR6301
        """Checks out the commit specified in context["current_commit"].

        Retrieves the commit specified in context["current_commit"] and the repository path
        from the configuration. It then checks out the commit using GitPython. If the checkout
        fails for any reason, it logs an exception and aborts the pipeline by setting
        context["abort_pipeline"] = True.

        Args:
            context: A dictionary containing the current execution context,
                    including the commit to checkout.
        """
        commit: str = context["commit"]
        logger.info("Repo path: %s_%s", context.get("repo_path"), commit, context=context)
        cwd = os.getcwd()
        repo = git.Repo(cwd)
        logger.info("Checking out commit %s", commit, context=context)
        try:
            repo.git.checkout(commit)
        except Exception:
            logger.exception("Failed to checkout commit %s", commit, context=context)
            context["abort_pipeline"] = True
