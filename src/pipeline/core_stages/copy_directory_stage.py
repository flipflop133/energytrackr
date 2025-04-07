"""Pipeline stage to copy a directory to a target location."""

import shutil
from pathlib import Path
from typing import Any

from pipeline.stage_interface import PipelineStage
from utils.exceptions import SourceDirectoryNotFoundError
from utils.logger import logger


class CopyDirectoryStage(PipelineStage):
    """Pipeline stage to copy a directory to a target location."""

    def run(self, context: dict[str, Any]) -> None:
        """Executes the logic to copy a directory from source to target.

        Args:
            context (dict[str, Any]): A dictionary containing contextual information.
                Expected keys:
                    - 'source_dir': str | Path — path to the source directory
                    - 'target_dir': str | Path — path to the destination directory

        Raises:
            KeyError: If required keys are missing from context.
            FileNotFoundError: If the source directory does not exist.
            FileExistsError: If the target directory already exists.
        """
        source = Path(context.get("repo_path")).resolve()
        logger.info(f"Source directory: {source}", context=context)
        target = Path(f"{context.get('repo_path')}_{context['commit']}").resolve()

        if not source.is_dir():
            raise SourceDirectoryNotFoundError(source)

        if target.exists():
            logger.info(f"Target directory already exists: {target}", context=context)
        else:
            logger.info(f"Copying directory from {source} to {target}", context=context)
            shutil.copytree(src=source, dst=target)
            logger.info("Copy completed.", context=context)
