"""Module to checkout a specific commit in a git repository."""

import logging
from typing import Any

import git

from config.config_store import Config
from pipeline.stage_interface import PipelineStage


class CheckoutStage(PipelineStage):
    """Checks out the commit specified in context["current_commit"]."""

    def run(self, context: dict[str, Any]) -> None:
        """Checks out the commit specified in context["current_commit"].

        Retrieves the commit specified in context["current_commit"] and the repository path
        from the configuration. It then checks out the commit using GitPython. If the checkout
        fails for any reason, it logs an exception and aborts the pipeline by setting
        context["abort_pipeline"] = True.

        Args:
            context: A dictionary containing the current execution context,
                    including the commit to checkout.
        """
        commit: git.Commit = context["commit"]
        config = Config.get_config()
        logging.info(f"Repo path: {config.repo_path}_{commit}")
        repo = git.Repo()

        try:
            logging.info("Checking out commit %s", commit.hexsha)
            repo.git.checkout(commit.hexsha)
        except Exception:
            logging.exception("Failed to checkout commit %s", commit.hexsha)
            context["abort_pipeline"] = True
