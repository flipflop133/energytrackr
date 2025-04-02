"""Pipeline orchestrates the execution of stages for each commit."""

import logging
from typing import Any

import git

from config.config_store import Config
from pipeline.stage_interface import PipelineStage


class Pipeline:
    """Orchestrates the provided stages for each commit in sequence."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        """Initialize a Pipeline with a list of stages and the pipeline configuration.

        Args:
            stages: List of PipelineStage instances to run in sequence.
        """
        self.stages = stages
        self.config = Config.get_config()

    def run(self, commits: list[git.Commit]) -> None:
        """Runs the pipeline over the provided list of commits.

        Each commit is processed in sequence, and each stage in the pipeline is run
        in order. If any stage fails or requests to abort the pipeline, the remaining
        stages will not be run for that commit, and the pipeline will continue to the
        next commit.

        Args:
            commits: List of Commit objects to process in sequence.
        """
        logging.info("Starting pipeline over %d commits...", len(commits))

        for commit in commits:
            # Setup a new context for each commit
            context: dict[str, Any] = {
                "current_commit": commit,
                "build_failed": False,
                "abort_pipeline": False,
            }

            logging.info("==== Processing commit %s ====", commit.hexsha)

            # Run each stage in order
            for stage in self.stages:
                stage.run(context)
                if context.get("abort_pipeline"):
                    logging.warning("Aborting remaining stages for commit %s", commit.hexsha)
                    break

            logging.info("==== Done with commit %s ====\n", commit.hexsha)
