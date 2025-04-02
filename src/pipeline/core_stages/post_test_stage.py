"""Module to run any post-test command (cleanup or teardown)."""

import logging
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class PostTestStage(PipelineStage):
    """Runs any post-test command (cleanup or teardown)."""

    def run(self, context: dict[str, Any]) -> None:
        """Runs any post-test command (cleanup or teardown).

        Retrieves the post-test command from the configuration and executes it
        within the specified working directory. If the command fails and
        failures are not ignored, the pipeline is aborted.
        """
        config = Config.get_config()
        post_cmd = config.execution_plan.post_command
        if not post_cmd:
            return

        logging.info("Running post-test command: %s", post_cmd)
        result = run_command(post_cmd, cwd=config.repo_path)
        if result.returncode != 0:
            logging.error("Post-test command failed with exit code %d", result.returncode)
            if not config.execution_plan.ignore_failures:
                context["abort_pipeline"] = True
