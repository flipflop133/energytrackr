"""Configuration model for the pipeline."""

from enum import Enum
from typing import Self

from pydantic import BaseModel, model_validator


class ModeEnum(str, Enum):
    tests = "tests"
    benchmarks = "benchmarks"


class GranularityEnum(str, Enum):
    commits = "commits"
    branches = "branches"
    tags = "tags"


class RepositoryDefinition(BaseModel):
    url: str
    branch: str
    clone_options: list[str] | None = None


class ExecutionPlanDefinition(BaseModel):
    mode: ModeEnum = ModeEnum.tests
    compile_commands: list[str] | None = None

    @model_validator(mode="after")
    def check_compile_commands_for_benchmarks(self: Self) -> Self:
        if self.mode == ModeEnum.benchmarks and not self.compile_commands:
            raise ValueError("compile_commands must be provided for 'benchmarks' mode.")
        return self

    granularity: GranularityEnum = GranularityEnum.commits
    pre_command: str | None = None
    pre_command_condition_files: set[str] = set()
    test_command: str
    test_command_path: str = ""
    ignore_failures: bool = False
    post_command: str | None = None
    num_commits: int | None = None
    num_runs: int = 1
    num_repeats: int = 1
    batch_size: int = 100
    randomize_tasks: bool = False
    oldest_commit: str | None = None
    newest_commit: str | None = None

    execute_common_tests: bool = False


class ResultsDefinition(BaseModel):
    file: str


class LimitsDefinition(BaseModel):
    temperature_safe_limit: int
    energy_regression_percent: int


class PipelineConfig(BaseModel):
    config_version: str = "1.0.0"
    repo: RepositoryDefinition
    repo_path: str | None = None
    execution_plan: ExecutionPlanDefinition
    results: ResultsDefinition
    limits: LimitsDefinition
    tracked_file_extensions: set[str]
    cpu_thermal_file: str
    setup_commands: list[str] | None = None
