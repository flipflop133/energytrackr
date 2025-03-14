"""Configuration models for strictly typed configuration."""

from enum import Enum

from pydantic import BaseModel, Field


class RepositoryConfig(BaseModel):
    """Repository configuration model."""

    url: str = Field(..., description="URL of the Git repository")
    branch: str = Field(..., description="Branch name to track", examples=["master"])
    clone_options: list[str] | None = Field(
        default=None,
        description="Additional options for Git clone command",
        examples=["--depth 1"],
    )


class ModeEnum(str, Enum):
    """ModeEnum is an enumeration that represents different modes of operation.

    Attributes:
        tests (str): Represents the 'tests' mode.
        benchmarks (str): Represents the 'benchmarks' mode.
    """

    tests = "tests"
    benchmarks = "benchmarks"


class GranularityEnum(str, Enum):
    """GranularityEnum is an enumeration that represents different levels of granularity.

    Attributes:
        commits (str): Represents the 'commits' granularity.
        branches (str): Represents the 'branches' granularity.
        tags (str): Represents the 'tags' granularity.
    """

    commits = "commits"
    branches = "branches"
    tags = "tags"


class TestConfig(BaseModel):
    """Test execution settings."""

    mode: ModeEnum = Field(ModeEnum.tests, description="Mode of testing")
    granularity: GranularityEnum = Field(GranularityEnum.commits, description="Level of granularity")
    pre_command_condition_files: list[str] | None = Field(
        None,
        description="List of files to check for in the pre-command",
        examples=["autogen.sh", "configure.ac", "Makefile.am", "CMakeLists.txt"],
    )
    pre_command: str = Field(..., description="Command to execute before tests", examples=["./autogen.sh && ./configure"])
    command: str = Field(..., description="Command to execute the tests", examples=["make -j$(nproc) check"])
    command_path: str = Field("", description="Path to execute the command")
    post_command: str | None = Field(None, description="Command to execute after tests", examples=["make clean"])
    num_commits: int = Field(..., ge=1, description="Number of commits to analyze")
    num_runs: int = Field(..., ge=1, description="Number of times each test should run")
    num_repeats: int = Field(..., ge=1, description="Number of times the entire test suite should be repeated")
    batch_size: int = Field(100, ge=1, description="Number of commits to process in a batch")
    randomize_tasks: bool = Field(False, description="Whether to randomize task execution order")


class OutputConfig(BaseModel):
    """Output configuration settings."""

    file: str = Field(..., description="File path to store the test results", examples=["energy_usage.csv"])


class ThresholdConfig(BaseModel):
    """Threshold settings for temperature and energy regression."""

    temperature_safe_limit: int = Field(..., ge=0, description="Maximum safe temperature in millidegrees Celsius")
    energy_regression_percent: int = Field(..., ge=0, le=100, description="Threshold for energy regression percentage")


class Config(BaseModel):
    """Main configuration model."""

    repository: RepositoryConfig
    test: TestConfig
    output: OutputConfig
    thresholds: ThresholdConfig
    file_extensions: list[str] = Field(..., description="List of file extensions to track for changes")
    cpu_themal_file_path: str = Field(
        ...,
        description="File path for CPU temperature monitoring",
        examples=["/sys/class/hwmon/hwmon2/temp1_input"],
    )
    compile_commands: list[str] = Field(..., description="List of commands to compile the project")
    setup_commands: list[str] | None = Field(None, description="List of setup commands to run before tests")
