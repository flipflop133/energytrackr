"""TestStage class for executing user-defined tests in a pipeline."""

import logging
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class TestStage(PipelineStage):
    """Executes user-defined tests (test_command) if not already failed."""

    def run(self, context: dict[str, Any]) -> None:
        """Executes the user-defined test command unless the build has failed.

        This method retrieves the test command from the configuration and executes it
        within the specified working directory. If the build has previously failed,
        the test stage is skipped. If no test command is specified in the configuration,
        a warning is logged and the stage is skipped.

        If the test command execution fails and failures are not ignored,
        the pipeline is aborted. Otherwise, execution continues.

        Args:
            context: A dictionary containing the current execution context,
                    including flags for build failure and pipeline abortion.
        """
        config = Config.get_config()
        if context["build_failed"]:
            logging.info("Skipping tests because build failed.")
            return

        test_cmd = config.execution_plan.test_command
        if not test_cmd:
            logging.warning("No test command specified, skipping test stage.")
            return

        test_cmd_path = config.execution_plan.test_command_path or ""
        working_dir = f"{config.repo_path}{test_cmd_path}"

        logging.info("Running test command: %s in %s", test_cmd, working_dir)
        result = run_command(test_cmd, cwd=working_dir)

        if result.returncode != 0:
            logging.error("Test command exited with code %d", result.returncode)
            if not config.execution_plan.ignore_failures:
                context["abort_pipeline"] = True
            # If ignoring failures, we just continue.
