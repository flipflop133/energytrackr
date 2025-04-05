"""Utility functions."""

import subprocess

from tqdm import tqdm


def run_command(arg: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Executes a shell command, streaming and capturing its output in real time.

    Args:
        arg (str): The command to run.
        cwd (str | None): The working directory for the command.

    Returns:
        subprocess.CompletedProcess[str]: The completed process with captured output on success.
        None: If the command fails.

    Raises:
        CalledProcessError: If the command exits with a non-zero status.

    """
    tqdm.write(f"Running command: {arg}")
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
                tqdm.write(clean_line)
                output_lines.append(clean_line)

    retcode = process.wait()
    output = "\n".join(output_lines)

    return subprocess.CompletedProcess(args=arg, returncode=retcode, stdout=output)
