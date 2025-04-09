"""Unit tests for the CopyDirectoryStage class."""

from pathlib import Path
from typing import Any

import pytest

from pipeline.core_stages.copy_directory_stage import CopyDirectoryStage
from utils.exceptions import SourceDirectoryNotFoundError


def test_copy_directory_success(tmp_path: Path) -> None:
    """Test CopyDirectoryStage with a valid source directory."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "file.txt").write_text("test")

    context: dict[str, Any] = {"repo_path": str(source_dir), "commit": "abc123"}

    stage = CopyDirectoryStage()
    stage.run(context)

    target_dir = Path(f"{context['repo_path']}_{context['commit']}")
    assert target_dir.exists()
    assert (target_dir / "file.txt").read_text() == "test"


def test_copy_directory_target_exists(tmp_path: Path) -> None:
    """Test CopyDirectoryStage with existing target directory."""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "source_abc123"
    source_dir.mkdir()
    target_dir.mkdir()
    (source_dir / "file.txt").write_text("test")
    (target_dir / "existing.txt").write_text("exists")

    context: dict[str, Any] = {"repo_path": str(source_dir), "commit": "abc123"}

    stage = CopyDirectoryStage()
    stage.run(context)

    # Should not overwrite target
    assert (target_dir / "existing.txt").exists()
    assert not (target_dir / "file.txt").exists()  # was never copied again


def test_copy_directory_source_not_found(tmp_path: Path) -> None:
    """Test CopyDirectoryStage with a non-existent source directory."""
    context: dict[str, Any] = {"repo_path": str(tmp_path / "nonexistent"), "commit": "abc123"}

    stage = CopyDirectoryStage()
    with pytest.raises(SourceDirectoryNotFoundError):
        stage.run(context)
