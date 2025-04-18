"""Unit tests for the PostTestStage class in the pipeline module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from energytrackr.pipeline.core_stages.post_test_stage import PostTestStage


@pytest.fixture
def dummy_context(tmp_path: str) -> dict[str, str | bool]:
    """Fixture to create a dummy context for testing.

    Args:
        tmp_path (str): The temporary path to create the repository.

    Returns:
        dict[str, str | bool]: A dictionary representing the context.
    """
    return {
        "repo_path": str(tmp_path),
        "abort_pipeline": False,
    }


@pytest.fixture
def dummy_config() -> SimpleNamespace:
    """Fixture to create a dummy configuration for testing.

    Returns:
        SimpleNamespace: A dummy configuration object.
    """

    class DummyConfig(SimpleNamespace):
        """Dummy configuration class for testing."""

        class ExecutionPlan:
            """Dummy execution plan class for testing."""

            post_command = "echo cleanup"
            ignore_failures = False

        execution_plan = ExecutionPlan()

    return DummyConfig()


@patch("energytrackr.pipeline.core_stages.post_test_stage.run_command")
def test_post_test_command_success(
    mock_run: MagicMock,
    dummy_context: dict[str, str | bool],
    dummy_config: SimpleNamespace,
) -> None:
    """Test the successful execution of the post-test command."""
    mock_run.return_value = SimpleNamespace(returncode=0)

    with patch("energytrackr.pipeline.core_stages.post_test_stage.Config.get_config", return_value=dummy_config):
        stage = PostTestStage()
        stage.run(dummy_context)

    mock_run.assert_called_once()
    assert dummy_context["abort_pipeline"] is False


@patch("energytrackr.pipeline.core_stages.post_test_stage.run_command")
def test_post_test_command_failure_abort(
    mock_run: MagicMock,
    dummy_context: dict[str, str | bool],
    dummy_config: SimpleNamespace,
) -> None:
    """Test the post-test command failure when not ignored."""
    mock_run.return_value = SimpleNamespace(returncode=1)

    with patch("energytrackr.pipeline.core_stages.post_test_stage.Config.get_config", return_value=dummy_config):
        stage = PostTestStage()
        stage.run(dummy_context)

    assert dummy_context["abort_pipeline"] is True


@patch("energytrackr.pipeline.core_stages.post_test_stage.run_command")
def test_post_test_command_failure_ignored(
    mock_run: MagicMock,
    dummy_context: dict[str, str | bool],
    dummy_config: SimpleNamespace,
) -> None:
    """Test the post-test command failure when ignored."""
    dummy_config.execution_plan.ignore_failures = True
    mock_run.return_value = SimpleNamespace(returncode=1)

    with patch("energytrackr.pipeline.core_stages.post_test_stage.Config.get_config", return_value=dummy_config):
        stage = PostTestStage()
        stage.run(dummy_context)

    assert dummy_context["abort_pipeline"] is False


@patch("energytrackr.pipeline.core_stages.post_test_stage.run_command")
def test_post_test_no_command(
    mock_run: MagicMock,
    dummy_context: dict[str, str | bool],
    dummy_config: SimpleNamespace,
) -> None:
    """Test the behavior when no post-test command is provided."""
    dummy_config.execution_plan.post_command = None

    with patch("energytrackr.pipeline.core_stages.post_test_stage.Config.get_config", return_value=dummy_config):
        stage = PostTestStage()
        stage.run(dummy_context)

    mock_run.assert_not_called()
