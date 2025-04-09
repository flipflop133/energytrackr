"""Test suite for utility functions in the utils module."""

from pathlib import Path

from utils.utils import run_command


def test_run_command_success(tmp_path: Path) -> None:
    """Test successful execution of a simple shell command."""
    result = run_command("echo Hello, world!", cwd=str(tmp_path))
    assert result.returncode == 0
    assert "Hello, world!" in result.stdout


def test_run_command_with_context(tmp_path: Path) -> None:
    """Test command with context logging (does not affect output)."""
    context = {"test": "run"}
    result = run_command("echo Context test", cwd=str(tmp_path), context=context)
    assert result.returncode == 0
    assert "Context test" in result.stdout
