import builtins
import time
from unittest.mock import patch, mock_open

import pytest
from pipeline.core_stages.temperature_check_stage import TemperatureCheckStage
from config.config_store import Config
from config.config_model import PipelineConfig, RepositoryDefinition, ExecutionPlanDefinition, LimitsDefinition


@pytest.fixture
def config_with_temp_file(tmp_path):
    """Fixture to provide a Config with a mock temperature file path."""
    temp_file = tmp_path / "fake_temp"
    config = PipelineConfig(
        repo=RepositoryDefinition(url="https://example.com", branch="main"),
        execution_plan=ExecutionPlanDefinition(test_command="echo test"),
        limits=LimitsDefinition(temperature_safe_limit=60000, energy_regression_percent=20),
        tracked_file_extensions={"py"},
        cpu_thermal_file=str(temp_file),
    )
    Config.reset()
    Config.set_config(config)
    return temp_file


def test_temp_check_below_limit(config_with_temp_file):
    """Test that temperature check exits immediately when under limit."""
    config_with_temp_file.write_text("50000")
    stage = TemperatureCheckStage()

    with patch("time.sleep") as sleep_mock:
        stage.run({})
        sleep_mock.assert_not_called()


def test_temp_check_above_limit_then_ok(config_with_temp_file):
    """Test retry loop if temperature is initially above the limit."""
    stage = TemperatureCheckStage()

    temps = ["70000", "59000"]  # Above limit then below

    def side_effect_open(*args, **kwargs):
        value = temps.pop(0)
        return mock_open(read_data=value).return_value

    with patch("builtins.open", side_effect=side_effect_open), patch("time.sleep") as sleep_mock:
        stage.run({})
        sleep_mock.assert_called_once()


def test_temp_check_cannot_read(config_with_temp_file):
    """Test that pipeline proceeds if temperature file cannot be read."""
    if config_with_temp_file.exists():
        config_with_temp_file.unlink()
    stage = TemperatureCheckStage()

    with patch("time.sleep") as sleep_mock:
        stage.run({})
        sleep_mock.assert_not_called()
