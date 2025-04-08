import pytest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.core_stages.post_test_stage import PostTestStage


@pytest.fixture
def dummy_context(tmp_path):
    return {
        "repo_path": str(tmp_path),
        "abort_pipeline": False,
    }


@pytest.fixture
def dummy_config():
    class DummyConfig:
        class ExecutionPlan:
            post_command = "echo cleanup"
            ignore_failures = False

        execution_plan = ExecutionPlan()

    return DummyConfig()


@patch("pipeline.core_stages.post_test_stage.run_command")
def test_post_test_command_success(mock_run, dummy_context, dummy_config):
    mock_run.return_value = SimpleNamespace(returncode=0)

    with patch("pipeline.core_stages.post_test_stage.Config.get_config", return_value=dummy_config):
        stage = PostTestStage()
        stage.run(dummy_context)

    mock_run.assert_called_once()
    assert dummy_context["abort_pipeline"] is False


@patch("pipeline.core_stages.post_test_stage.run_command")
def test_post_test_command_failure_abort(mock_run, dummy_context, dummy_config):
    mock_run.return_value = SimpleNamespace(returncode=1)

    with patch("pipeline.core_stages.post_test_stage.Config.get_config", return_value=dummy_config):
        stage = PostTestStage()
        stage.run(dummy_context)

    assert dummy_context["abort_pipeline"] is True


@patch("pipeline.core_stages.post_test_stage.run_command")
def test_post_test_command_failure_ignored(mock_run, dummy_context, dummy_config):
    dummy_config.execution_plan.ignore_failures = True
    mock_run.return_value = SimpleNamespace(returncode=1)

    with patch("pipeline.core_stages.post_test_stage.Config.get_config", return_value=dummy_config):
        stage = PostTestStage()
        stage.run(dummy_context)

    assert dummy_context["abort_pipeline"] is False


@patch("pipeline.core_stages.post_test_stage.run_command")
def test_post_test_no_command(mock_run, dummy_context, dummy_config):
    dummy_config.execution_plan.post_command = None

    with patch("pipeline.core_stages.post_test_stage.Config.get_config", return_value=dummy_config):
        stage = PostTestStage()
        stage.run(dummy_context)

    mock_run.assert_not_called()
