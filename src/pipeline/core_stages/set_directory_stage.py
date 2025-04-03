"""SetDirectoryStage is a pipeline stage that changes the current working directory to a commit-specific directory."""

import logging
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class SetDirectoryStage(PipelineStage):
    """Pipeline stage to change the working directory to a commit-specific directory."""

    def run(self, context: dict[str, Any]) -> None:
        """Executes the logic to change the current working directory to the commit-specific directory.

        Args:
            context (dict[str, Any]): A dictionary containing contextual information.
                Expected to include a 'commit' key with the commit identifier.

        Raises:
            KeyError: If the 'commit' key is not found in the context dictionary.
            Exception: If the command to change the directory fails.
        """
        config = Config.get_config()
        if config.execution_plan.batch_size:
            # cd into the commit directory
            logging.info("Changing directory to %s", config.repo_path)
            run_command(f"cd {config.repo_path}_{context['commit']}", cwd=config.repo_path)
