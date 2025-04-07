"""Module to run pre-build commands (e.g., setting up environment)."""

from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils.logger import logger
from utils.utils import run_command


class PreBuildStage(PipelineStage):
    """Runs any pre-build commands (e.g., setting up environment).

    Optionally only runs if certain files changed, etc.
    """

    def run(self, context: dict[str, Any]) -> None:
        """Executes pre-build commands as defined in the configuration.

        This method retrieves a pre-build command from the configuration and executes it
        within the specified repository path. If no pre-build command is defined, the stage
        is skipped. Optionally, the execution can be conditioned on changes in specific files,
        though such logic is not implemented here.

        If the pre-build command fails and failures are not ignored, the pipeline is aborted.

        Args:
            context: A dictionary containing the current execution context, including flags
                    for build failure and pipeline abortion.
        """
        config = Config.get_config()
        pre_cmd = config.execution_plan.pre_command
        if not pre_cmd:
            return  # Nothing to do

        # If user wants certain files to trigger this only if changed:
        # Check patterns in commit.stats.files, if so desired.
        # For brevity we skip that logic or replicate from your original code.

        logger.info("Running pre-build command: %s", pre_cmd)
        result = run_command(pre_cmd, cwd=context.get("repo_path"), context=context)

        if result.returncode != 0:
            logger.error("Pre-build command failed with return code %d", result.returncode)
            if not config.execution_plan.ignore_failures:
                context["abort_pipeline"] = True
