"""Utility functions."""

import subprocess
from typing import Any

from energytrackr.utils.logger import logger


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
    logger.info("Running command: %s", arg, context=context)
    with subprocess.Popen(
        arg,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Ensures stdout and stderr are returned as strings
    ) as process:
        stdout, stderr = process.communicate(timeout=60)
        retcode = process.returncode

    if retcode:
        logger.error("Command failed with return code %d", retcode, context=context)
        logger.error("stdout: %s", stdout, context=context)
        logger.error("stderr: %s", stderr, context=context)

    return subprocess.CompletedProcess(
        args=arg,
        returncode=retcode,
        stdout=stdout,
        stderr=stderr,
    )
