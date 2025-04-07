"""Utility functions."""

import subprocess
from datetime import datetime
from typing import Any

from utils.logger import logger


def run_command(
    arg: str,
    cwd: str | None = None,
    log_output: bool = True,
    context: dict[str, Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Executes a shell command, streaming and capturing its output in real time.

    Args:
        arg (str): The command to run.
        cwd (str | None): The working directory for the command.
        log_output (bool): If True, logs the output to a file if the command fails.
        context (dict[str, Any] | None): The context dictionary for logging.

    Returns:
        subprocess.CompletedProcess[str]: The completed process with captured output on success.
        None: If the command fails.

    Raises:
        CalledProcessError: If the command exits with a non-zero status.

    """
    logger.info(f"Running command: {arg}", context=context)
    process = subprocess.Popen(
        arg,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []

    # Read and stream the output in real time
    if process.stdout is not None:
        for line in process.stdout:
            clean_line = line.rstrip()
            if clean_line:  # avoid empty lines
                if not log_output:
                    logger.info(clean_line, context=context)
                output_lines.append(clean_line)

    retcode = process.wait()
    output = "\n".join(output_lines)
    if retcode != 0 and log_output:
        # Save the output to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"command_output_{timestamp}.log"
        with open(log_filename, "w") as f:
            f.write(output)
        logger.info(f"Command failed with return code {retcode}. Output saved to command_output.log", context=context)

    return subprocess.CompletedProcess(args=arg, returncode=retcode, stdout=output)
