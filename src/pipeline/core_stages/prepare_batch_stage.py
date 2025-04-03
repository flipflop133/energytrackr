"""Module to build the project if in 'benchmarks' mode or skip if in 'tests' mode."""

import logging
from typing import Any

from config.config_model import ModeEnum
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
        config = Config.get_config()
        if config.execution_plan.batch_size:
            for commit in set(context["commits"]):
                # copy the repo to a new directory
                run_command(f"cp -r {config.repo_path} {config.repo_path}_{commit}")
                for stage in self.pre_test_stages:
                    stage.run({"commit": commit})
