"""Unit tests for the MeasureEnergyStage class."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pipeline.core_stages.measure_stage import MeasureEnergyStage


class DummyCommit:
    def __init__(self, hexsha: str = "abc123") -> None:
        self.hexsha = hexsha


@pytest.fixture
def dummy_context(tmp_path: str) -> dict[str, str | bool]:
    return {
        "commit": DummyCommit(),
        "repo_path": str(tmp_path / "repo"),
        "build_failed": False,
        "abort_pipeline": False,
    }


@pytest.fixture
def mock_config() -> SimpleNamespace:
    class DummyConfig:
        class ExecutionPlan:
            test_command = "run-tests.sh"
            ignore_failures = False

        execution_plan = ExecutionPlan()
        repo = SimpleNamespace(url="some/repo", branch="main")

    with patch("pipeline.core_stages.measure_stage.Config.get_config", return_value=DummyConfig()):
        yield DummyConfig()


@patch("pipeline.core_stages.measure_stage.run_command")
def test_measure_energy_success(mock_run, dummy_context, mock_config) -> None:
    mock_run.return_value = SimpleNamespace(returncode=0, stdout="42 power/energy-pkg/")
    dummy_context["repo_path"] = str(Path(dummy_context["repo_path"]))
    Path(dummy_context["repo_path"]).mkdir(parents=True, exist_ok=True)

    stage = MeasureEnergyStage()
    stage.run(dummy_context)

    output_dir = Path(dummy_context["repo_path"]).parent / "energy_measurements"
    output_files = list(output_dir.glob("energy_results_*.csv"))

    assert output_files
    assert output_files[0].read_text().strip() == "abc123,42"


@patch("pipeline.core_stages.measure_stage.run_command")
def test_measure_energy_perf_failure_abort(mock_run, dummy_context, mock_config):
    mock_run.return_value = SimpleNamespace(returncode=1, stdout="")

    stage = MeasureEnergyStage()
    stage.run(dummy_context)

    assert dummy_context["abort_pipeline"] is True


@patch("pipeline.core_stages.measure_stage.run_command")
def test_measure_energy_ignore_failure_warning(mock_run, dummy_context, mock_config):
    mock_run.return_value = SimpleNamespace(returncode=1, stdout="")

    # change config to allow ignoring failures
    mock_config.execution_plan.ignore_failures = True

    stage = MeasureEnergyStage()
    stage.run(dummy_context)

    assert dummy_context["abort_pipeline"] is False


@patch("pipeline.core_stages.measure_stage.run_command")
def test_measure_energy_no_test_command(mock_run, dummy_context):
    class DummyConfig:
        class ExecutionPlan:
            test_command = ""
            ignore_failures = False

        execution_plan = ExecutionPlan()

    with patch("pipeline.core_stages.measure_stage.Config.get_config", return_value=DummyConfig()):
        stage = MeasureEnergyStage()
        stage.run(dummy_context)

    mock_run.assert_not_called()


@patch("pipeline.core_stages.measure_stage.run_command")
def test_measure_energy_build_failed_skips(mock_run, dummy_context):
    dummy_context["build_failed"] = True

    class DummyConfig:
        class ExecutionPlan:
            test_command = "run"
            ignore_failures = False

        execution_plan = ExecutionPlan()

    with patch("pipeline.core_stages.measure_stage.Config.get_config", return_value=DummyConfig()):
        stage = MeasureEnergyStage()
        stage.run(dummy_context)

    mock_run.assert_not_called()


@patch("pipeline.core_stages.measure_stage.run_command")
def test_measure_energy_no_data_found(mock_run, dummy_context, mock_config):
    mock_run.return_value = SimpleNamespace(returncode=0, stdout="no energy info here")

    stage = MeasureEnergyStage()
    stage.run(dummy_context)

    assert dummy_context["abort_pipeline"] is True


@patch("pipeline.core_stages.measure_stage.run_command")
def test_measure_energy_no_data_ignore_failure(mock_run, dummy_context, mock_config):
    mock_config.execution_plan.ignore_failures = True
    mock_run.return_value = SimpleNamespace(returncode=0, stdout="...")

    stage = MeasureEnergyStage()
    stage.run(dummy_context)

    assert dummy_context["abort_pipeline"] is False
