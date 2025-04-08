import subprocess
import os
from pathlib import Path
import pytest

from utils.utils import run_command


def test_run_command_success(tmp_path: Path):
    """Test successful execution of a simple shell command."""
    result = run_command("echo Hello, world!", cwd=str(tmp_path))
    assert result.returncode == 0
    assert "Hello, world!" in result.stdout


def test_run_command_with_context(tmp_path: Path):
    """Test command with context logging (does not affect output)."""
    context = {"test": "run"}
    result = run_command("echo Context test", cwd=str(tmp_path), context=context)
    assert result.returncode == 0
    assert "Context test" in result.stdout


def test_run_command_failure_logs_file(tmp_path: Path, monkeypatch):
    """Test that a failed command creates a log file when log_output=True."""

    # Change cwd so we don't clutter project root with logs
    monkeypatch.chdir(tmp_path)

    result = run_command("exit 1", log_output=True, cwd=str(tmp_path))

    # A non-zero return code
    assert result.returncode != 0

    # Look for a generated log file
    logs = list(tmp_path.glob("command_output_*.log"))
    assert len(logs) == 1
    content = logs[0].read_text()
    assert content == ""  # "exit 1" outputs nothing, so the log will be empty


import os


def test_run_command_failure_no_log(tmp_path: Path):
    """Test that failed command does not write a log when log_output=False."""
    result = run_command("exit 2", log_output=False, cwd=str(tmp_path))
    assert result.returncode == 2

    logs = list(tmp_path.glob("command_output_*.log"))
    assert not logs  # Should be empty


from unittest.mock import patch


@patch("utils.utils.logger.info")
def test_run_command_log_streams_output_when_disabled(mock_log, tmp_path: Path):
    """Ensure logger.info is called for each line if log_output is False."""
    command = "printf 'line1\\nline2\\nline3\\n'"
    run_command(command, log_output=False, cwd=str(tmp_path))

    # The first logger.info is the "Running command" log
    # The next three are the output lines (if not empty)
    messages = [call.args[0] for call in mock_log.call_args_list]

    assert any("line1" in msg for msg in messages)
    assert any("line2" in msg for msg in messages)
    assert any("line3" in msg for msg in messages)
