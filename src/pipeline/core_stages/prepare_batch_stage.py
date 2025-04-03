"""Module to build the project if in 'benchmarks' mode or skip if in 'tests' mode."""

import os
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class PrepareBatchStage(PipelineStage):
    """Builds the project if in 'benchmarks' mode, or skip if 'tests' mode has no build commands."""

    def __init__(self, pre_test_stages: list[PipelineStage]) -> None:
        """Initialize the PrepareBatchStage with a list of pre-test stages.

        Args:
            pre_test_stages (list[PipelineStage]): A list of stages to run before the main test stage.
        """
        self.pre_test_stages = pre_test_stages

    def run(self, context: dict[str, Any]) -> None:
        """Executes the `run` method for the current stage in the pipeline.

        This method processes a batch of commits by copying the repository to a new
        directory for each commit and executing the pre-test stages for each commit.

        Args:
            context (dict[str, Any]): A dictionary containing the context for the
                pipeline execution. It must include a "commits" key with a list of
                commit identifiers.

        Raises:
            KeyError: If the "commits" key is not present in the context dictionary.
            Exception: If any of the pre-test stages fail during execution.
        """
        config = Config.get_config()
        if config.execution_plan.batch_size:
            for commit in set(context["commits"]):
                progress_bar = context["bar"]
                progress_bar.set_postfix(current_commit=commit.hexsha[:8])
                progress_bar.update(1)
                # copy the repo to a new directory
                target_path = f"{config.repo_path}_{commit}"
                if not os.path.exists(target_path):
                    run_command(f"cp -r {config.repo_path} {target_path}")
                for stage in self.pre_test_stages:
                    stage.run({"commit": commit})
