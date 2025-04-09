"""Utility functions."""

import subprocess
from datetime import datetime
from typing import Any

from utils.logger import logger


def run_command(
    arg: str,
    cwd: str | None = None,
    context: dict[str, Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Executes a shell command, streaming and capturing its output in real time.

    Args:
        arg (str): The command to run.
        cwd (str | None): The working directory for the command.
        context (dict[str, Any] | None): The context dictionary for logging.

    Returns:
        subprocess.CompletedProcess[str]: The completed process object containing the command's output and return code.

    """
    logger.info(f"Running command: {arg}", context=context)
    process = subprocess.Popen(
        arg,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Ensures stdout and stderr are returned as strings
    )

    stdout, stderr = process.communicate()
    retcode = process.wait(timeout=60)

    if retcode != 0:
        logger.error(f"Command failed with return code {retcode}", context=context)
        logger.error(f"stdout: {stdout}", context=context)
        logger.error(f"stderr: {stderr}", context=context)

    return subprocess.CompletedProcess(
        args=arg,
        returncode=retcode,
        stdout=stdout,
        stderr=stderr,
    )
