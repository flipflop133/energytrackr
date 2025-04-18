"""SetDirectoryStage: A pipeline stage to change the working directory to a commit-specific directory."""

import os
from pathlib import Path
from typing import Any

from energytrackr.pipeline.stage_interface import PipelineStage
from energytrackr.utils.exceptions import MissingContextKeyError, TargetDirectoryNotFoundError
from energytrackr.utils.logger import logger


class SetDirectoryStage(PipelineStage):
    """Pipeline stage to change the working directory to a commit-specific directory."""

    def run(self, context: dict[str, Any]) -> None:  # noqa: PLR6301
        """Executes the logic to change the current working directory to the commit-specific directory.

        Args:
            context (dict[str, Any]): A dictionary containing contextual information.
                Expected to include a 'commit' key with the commit identifier.

        Raises:
            MissingContextKeyError: If the 'commit' key is not found in the context.
            TargetDirectoryNotFoundError: If the target directory does not exist.
        """
        if "commit" not in context:
            raise MissingContextKeyError("commit")

        commit_id = context["commit"]
        target_dir = Path(f"{context.get('repo_path')}_{commit_id}").resolve()

        if not target_dir.is_dir():
            raise TargetDirectoryNotFoundError(target_dir)

        logger.info("Changing directory to %s", target_dir, context=context)
        os.chdir(target_dir)
