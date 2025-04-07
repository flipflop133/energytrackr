"""Pipeline stage to copy a directory to a target location."""

import logging
import shutil
from pathlib import Path
from typing import Any

from pipeline.stage_interface import PipelineStage
from utils.exceptions import SourceDirectoryNotFoundError


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
        logging.info(f"Source directory: {source}")
        target = Path(f"{context.get('repo_path')}_{context['commit']}").resolve()

        if not source.is_dir():
            raise SourceDirectoryNotFoundError(source)

        if target.exists():
            logging.info(f"Target directory already exists: {target}")
        else:
            logging.info(f"Copying directory from {source} to {target}")
            shutil.copytree(src=source, dst=target)
            logging.info("Copy completed.")
