import pytest
from unittest.mock import MagicMock

from pipeline.core_stages.build_stage import BuildStage


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.execution_plan.compile_commands = ["make build", "make test"]
    config.execution_plan.ignore_failures = False
    return config


from unittest.mock import patch, MagicMock
from pipeline.core_stages.build_stage import BuildStage


def test_build_stage_success():
    mock_result = MagicMock()
    mock_result.returncode = 0  # ✅ Simulate success

    context = {"build_failed": False, "abort_pipeline": False}

    with patch("pipeline.core_stages.build_stage.run_command", return_value=mock_result):
        with patch("config.config_store.Config.get_config") as mock_config:
            mock_config.return_value.execution_plan.compile_commands = ["make"]
            mock_config.return_value.execution_plan.ignore_failures = False
            stage = BuildStage()
            stage.run(context)

    assert context.get("build_failed") is not True
    assert context.get("abort_pipeline") is not True


def test_build_stage_fail_and_abort(monkeypatch, mock_config):
    monkeypatch.setattr("config.config_store.Config.get_config", lambda: mock_config)
    monkeypatch.setattr("utils.utils.run_command", lambda cmd, **kwargs: MagicMock(returncode=1))

    context = {}
    stage = BuildStage()
    stage.run(context)

    assert context["build_failed"] is True
    assert context["abort_pipeline"] is True


def test_build_stage_fail_ignore(monkeypatch, mock_config):
    mock_config.execution_plan.ignore_failures = True
    monkeypatch.setattr("config.config_store.Config.get_config", lambda: mock_config)
    monkeypatch.setattr("utils.utils.run_command", lambda cmd, **kwargs: MagicMock(returncode=1))

    context = {}
    stage = BuildStage()
    stage.run(context)

    assert context["build_failed"] is True
    assert context.get("abort_pipeline") is not True
