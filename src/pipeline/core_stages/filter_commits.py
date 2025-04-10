"""Module to filter commits based on the provided configuration."""

from typing import Any

import git

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils.logger import logger


class FilterCommitsStage(PipelineStage):
    """Stage to filter commits based on the provided configuration."""

    def run(self, context: dict[str, Any]) -> None:
        """Filter commits based on the provided configuration."""
        commits = context.get("batch", [])
        config = Config.get_config()
        logger.error("Number of commits before filtering: %d", len(commits), context=context)

        # Modify the list in place
        i = 0
        while i < len(commits):
            commit: git.Commit = commits[i]
            for file in commit.stats.files:
                if str(file).endswith(tuple(config.tracked_file_extensions)):
                    break
            else:
                # Remove the commit if it doesn't match the criteria
                commits.pop(i)
                continue
            i += 1

        logger.error("Number of commits after filtering: %d", len(commits), context=context)
        logger.error("unique commits after filtering: %d", len(set(commits)), context=context)

        if not commits:
            logger.warning("No commits passed the filter criteria.", context=context)
            context["abort_pipeline"] = True
        else:
            logger.info(f"{len(commits)} commits passed the filter criteria.", context=context)
