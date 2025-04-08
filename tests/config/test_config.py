import json
from pathlib import Path

import pytest

from config.config_model import ExecutionPlanDefinition, ModeEnum, PipelineConfig
from config.config_store import Config
from main import (
    load_pipeline_config,
)
from utils.exceptions import ConfigurationSingletonError


def test_compile_commands_required_for_benchmarks() -> None:
    """Test that benchmarks mode requires compile_commands."""
    with pytest.raises(ValueError, match="compile_commands must be provided"):
        ExecutionPlanDefinition(
            mode=ModeEnum.benchmarks,
            granularity="commits",
            test_command="run-tests.sh",
            test_command_path=".",
            ignore_failures=True,
            num_commits=5,
            num_runs=1,
            num_repeats=1,
            batch_size=10,
            randomize_tasks=False,
            # compile_commands is omitted on purpose!
        )


def test_load_pipeline_config(tmp_path: Path) -> None:
    # Copy sample_conf.json to a temp path
    sample_path = Path("tests/config/sample_conf.json")
    config_file = tmp_path / "config.json"
    config_file.write_text(sample_path.read_text())

    # Load the config using your app code
    load_pipeline_config(str(config_file))

    # Validate it loaded correctly
    config_obj: PipelineConfig = Config.get_config()
    assert config_obj.repo.url == "https://github.com/example/project.git"
    assert config_obj.execution_plan.num_commits == 5


def test_config_singleton_identity() -> None:
    """Ensure that Config always returns the same instance."""
    instance_1 = Config()
    instance_2 = Config()
    assert instance_1 is instance_2


def test_get_config_without_set_raises() -> None:
    Config.reset()
    with pytest.raises(ConfigurationSingletonError):
        Config.get_config()


def test_set_config_twice_raises(tmp_path: Path) -> None:
    from config.config_store import Config
    from utils.exceptions import ConfigurationSingletonError

    # Load a valid sample config
    config_path = Path(__file__).parent / "sample_conf.json"
    config_data = json.loads(config_path.read_text())
    config_obj = PipelineConfig(**config_data)

    Config.reset()
    Config.set_config(config_obj)

    with pytest.raises(ConfigurationSingletonError):
        Config.set_config(config_obj)
