"""Module to build the project if in 'benchmarks' mode or skip if in 'tests' mode."""

import logging
from typing import Any

from config.config_model import ModeEnum
from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class BuildStage(PipelineStage):
    """Builds the project if in 'benchmarks' mode, or skip if 'tests' mode has no build commands."""

    def run(self, context: dict[str, Any]) -> None:
        """Builds the project if in 'benchmarks' mode, or skip if 'tests' mode has no build commands.

        If the pipeline is in 'benchmarks' mode, we expect compile_commands.
        If any build command fails, we abort the commit.
        If the build fails and failures are not ignored, we abort the pipeline.
        If the pipeline is in 'tests' mode, we don't build anything.
        """
        config = Config.get_config()

        # If the pipeline is in 'benchmarks' mode, we expect compile_commands.
        if config.execution_plan.mode == ModeEnum.benchmarks:
            compile_cmds = config.execution_plan.compile_commands or []
            for cmd in compile_cmds:
                logging.info("Running build command: %s", cmd)
                result = run_command(cmd, cwd=config.repo_path)
                if result.returncode != 0:
                    logging.error("Build command failed. Aborting commit.")
                    context["build_failed"] = True
                    if not config.execution_plan.ignore_failures:
                        context["abort_pipeline"] = True
                    break
        else:
            # In 'tests' mode, maybe no formal build is required.
            # Or you can put commands in compile_commands as well, if desired.
            pass
