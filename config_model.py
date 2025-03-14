"""Configuration models for strictly typed configuration using Pydantic."""

from enum import Enum

from pydantic import BaseModel, Field


class ModeEnum(str, Enum):
    """High-level category or mode of measurement (e.g., 'tests' or 'benchmarks')."""

    tests = "tests"
    benchmarks = "benchmarks"


class GranularityEnum(str, Enum):
    """Determines whether to test each commit, branch, or tag."""

    commits = "commits"
    branches = "branches"
    tags = "tags"


class RepositoryDefinition(BaseModel):
    """Information for cloning and tracking a Git repository.

    Corresponds to the 'repo' field in the schema.
    """

    url: str = Field(
        ...,
        min_length=1,
        description="URL of the remote Git repository.",
        examples=["https://github.com/example/project.git"],
    )
    branch: str = Field(
        ...,
        min_length=1,
        description="Name of the branch to analyze for commits.",
        examples=["main", "master"],
    )
    clone_options: list[str] | None = Field(
        default=None,
        description="Additional options for 'git clone' (e.g., shallow clone).",
        examples=[["--depth", "1"]],
    )


class ExecutionPlanDefinition(BaseModel):
    """Controls how commits are selected and tested, including optional pre/post commands.

    Corresponds to the 'executionPlan' field in the schema.
    """

    mode: ModeEnum = Field(
        ModeEnum.tests,
        description="High-level category of the tasks to run.",
        examples=["tests", "benchmarks"],
    )
    granularity: GranularityEnum = Field(
        GranularityEnum.commits,
        description="Determines whether to test each commit, each branch, or each tag.",
        examples=["commits", "branches", "tags"],
    )
    pre_command: str = Field(
        ...,
        min_length=1,
        description="Command to run before each test or build step.",
        examples=["./autogen.sh && ./configure", "setup_env.sh"],
    )
    pre_command_condition_files: set[str] | None = Field(
        default=None,
        description=("Optional list of file patterns that trigger `preCommand` only if the commit modifies one of them."),
        examples=[["configure.ac", "Makefile.am"]],
    )
    test_command: str = Field(
        ...,
        min_length=1,
        description="Main command for running tests or benchmarks.",
        examples=["make -j$(nproc) check", "pytest --maxfail=1 --disable-warnings"],
    )
    test_command_path: str = Field(
        "",
        description="Working directory in which to run `testCommand`.",
        examples=["/build-ninja", "./tests"],
    )
    post_command: str | None = Field(
        None,
        description="Command to execute after tests for cleanup, etc.",
        examples=["make clean", "teardown_env.sh"],
    )
    num_commits: int = Field(
        ...,
        ge=1,
        description="Number of recent commits (from HEAD) to evaluate, if granularity == 'commits'.",
        examples=[50, 100],
    )
    num_runs: int = Field(
        ...,
        ge=1,
        description="How many times to run the test suite per commit (to reduce variance).",
        examples=[5, 10],
    )
    num_repeats: int = Field(
        ...,
        ge=1,
        description="How many times to repeat the entire test sequence on a given commit.",
        examples=[3, 5],
    )
    batch_size: int = Field(
        100,
        ge=1,
        description="Maximum number of commits to process in each batch.",
        examples=[10, 50, 100],
    )
    randomize_tasks: bool = Field(
        False,
        description="If true, randomize the execution order of tasks across commits/runs.",
        examples=[True, False],
    )


class ResultsDefinition(BaseModel):
    """Specifies how and where results are stored.

    Corresponds to the 'results' field in the schema.
    """

    file: str = Field(
        ...,
        min_length=1,
        description="Path to the file where CSV or other structured output is written.",
        examples=["results/energy_usage.csv", "build/metrics_output.tsv"],
    )


class LimitsDefinition(BaseModel):
    """Constraints for temperature and energy regression.

    Corresponds to the 'limits' field in the schema.
    """

    temperature_safe_limit: int = Field(
        ...,
        ge=0,
        le=150000,
        description="Max CPU temperature in millidegrees Celsius to allow testing (e.g., 90000 = 90°C).",
        examples=[90000, 95000],
    )
    energy_regression_percent: int = Field(
        ...,
        ge=0,
        le=100,
        description="Allowed % increase in energy usage before flagging a regression.",
        examples=[10, 15, 20],
    )


class PipelineConfig(BaseModel):
    """Main configuration model corresponding to the entire JSON schema.

    Maps all fields from the 'Industry-Grade Energy Regression Pipeline Schema'.
    """

    config_version: str = Field(
        "1.0.0",
        description="Version of the configuration format to track schema evolution.",
        examples=["1.0.0"],
    )
    repo: RepositoryDefinition
    execution_plan: ExecutionPlanDefinition
    results: ResultsDefinition
    limits: LimitsDefinition
    tracked_file_extensions: set[str] = Field(
        ...,
        description="List of file extensions/patterns that indicate a commit is worth testing if changed.",
        examples=[["c", "CMakeLists.txt", "h"]],
        min_length=1,
    )
    cpu_thermal_file: str = Field(
        ...,
        min_length=1,
        description="Path to the file that reports CPU temperature in millidegrees Celsius.",
        examples=["/sys/class/hwmon/hwmon2/temp1_input"],
    )
    compile_commands: list[str] = Field(
        ...,
        description="Shell commands to build or prepare the project for testing.",
        min_length=1,
        examples=[["make clean", "make -j$(nproc)"]],
    )
    setup_commands: list[str] | None = Field(
        None,
        description="Optional system-wide setup commands before running tests (e.g., install dependencies).",
        examples=[["sudo apt-get install -y ccache", "pip install -r requirements.txt"]],
    )
