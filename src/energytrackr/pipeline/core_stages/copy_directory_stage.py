"""Pipeline stage to copy a directory to a target location."""

import shutil
from pathlib import Path
from typing import Any

from energytrackr.pipeline.stage_interface import PipelineStage
from energytrackr.utils.exceptions import SourceDirectoryNotFoundError
from energytrackr.utils.logger import logger


class CopyDirectoryStage(PipelineStage):
    """Pipeline stage to copy a directory to a target location."""

    def run(self, context: dict[str, Any]) -> None:  # noqa: PLR6301
        """Executes the logic to copy a directory from source to target.

        Args:
            context (dict[str, Any]): A dictionary containing contextual information.
                Expected keys:
                    - 'source_dir': str | Path — path to the source directory
                    - 'target_dir': str | Path — path to the destination directory

        Raises:
            SourceDirectoryNotFoundError: If the source directory does not exist.
        """
        source = Path(context.get("repo_path")).resolve()
        logger.info("Source directory: %s", source, context=context)
        target = Path(f"{context.get('repo_path')}_{context['commit']}").resolve()

        if not source.is_dir():
            raise SourceDirectoryNotFoundError(source)

        if target.exists():
            logger.info("Target directory already exists: %s", target, context=context)
        else:
            logger.info("Copying directory from %s to %s", source, target, context=context)
            shutil.copytree(src=source, dst=target)
            logger.info("Copy completed.", context=context)
