"""Unit tests for the Pipeline class and its methods."""

import concurrent.futures
import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import Commit

from energytrackr.config.config_store import Config
from energytrackr.config.loader import load_pipeline_config
from energytrackr.pipeline.pipeline import Pipeline, log_context_buffer, run_pre_test_stages_for_commit
from energytrackr.pipeline.stage_interface import PipelineStage


@pytest.fixture
def dummy_commit() -> Commit:
    """Fixture to create a dummy commit object.

    Returns:
        Commit: A mock commit object with a hexsha attribute.
    """
    commit = MagicMock(spec=Commit)
    commit.hexsha = "deadbeef"
    return commit


@pytest.fixture
def dummy_stages() -> dict[str, list[PipelineStage]]:
    """Fixture to create dummy stages for the pipeline.

    Returns:
        dict[str, list[PipelineStage]]: A dictionary containing lists of dummy stages.
    """
    stage_mock = MagicMock()
    stage_mock.run = MagicMock()
    return {"pre_stages": [stage_mock], "pre_test_stages": [stage_mock], "batch_stages": [stage_mock]}


@pytest.fixture
def dummy_repo_path(tmp_path: str) -> str:
    """Fixture to create a temporary repository path.

    Args:
        tmp_path (str): The temporary path to create the repository.

    Returns:
        str: The path to the temporary repository.
    """
    return f"{tmp_path} / repo"


@pytest.fixture
def dummy_context() -> dict[str, str | bool]:
    """Fixture to create a dummy context for the pipeline.

    Returns:
        dict[str, str | bool]: A dictionary representing the context.
    """
    return {
        "commit": "deadbeef",
        "build_failed": False,
        "abort_pipeline": False,
        "repo_path": "/tmp/fake_repo",
        "worker_process": True,
        "log_buffer": [(20, "Stage ran"), (30, "Warning here")],
    }


def test_run_pre_test_stages_for_commit_success(monkeypatch: pytest.MonkeyPatch, tmp_path: str, dummy_commit: Commit) -> None:
    """Test successful execution of pre-test stages for a commit."""
    Config.reset()
    sample_path = Path(__file__).parent.parent / "config" / "sample_conf.json"

    # Load the config using your app code
    load_pipeline_config(str(sample_path))

    repo = MagicMock()
    repo.commit.return_value = dummy_commit
    monkeypatch.setattr("git.Repo", lambda _path: repo)

    result = run_pre_test_stages_for_commit(dummy_commit.hexsha, str(tmp_path))

    assert result["commit"] == dummy_commit.hexsha
    assert not result["abort_pipeline"]


def test_run_pre_test_stages_for_commit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test failure in pre-test stages for a commit."""
    monkeypatch.setattr("git.Repo", lambda _path: (_ for _ in ()).throw(Exception("Repo error")))
    dummy_stage = MagicMock()

    result = run_pre_test_stages_for_commit("badsha", "/fake")
    assert result["abort_pipeline"]
    dummy_stage.run.assert_not_called()


def test_log_context_buffer_logs_messages(caplog: pytest.LogCaptureFixture) -> None:
    """Test that log_context_buffer logs messages correctly."""
    context = {
        "commit": "abc1234",
        "log_buffer": [
            (logging.INFO, "This is info"),
            (logging.WARNING, "This is a warning"),
        ],
    }

    # Ensure our logger is used and hooked up properly
    test_logger = logging.getLogger("energy-pipeline")
    test_logger.propagate = True  # Let caplog see the logs

    with caplog.at_level(logging.INFO, logger="energy-pipeline"):
        log_context_buffer(context)

    assert any("This is info" in msg for msg in caplog.messages)
    assert any("This is a warning" in msg for msg in caplog.messages)
    assert any("End of logs for abc1234" in msg for msg in caplog.messages)


def test_pipeline_run_all_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that all stages are run in the pipeline."""

    class DummyStage(PipelineStage):
        def run(self, context: dict[str, any]) -> None:
            context.setdefault("ran", []).append(self.__class__.__name__)

    # Dummy commit with fake hexsha
    @dataclass
    class DummyCommit:
        hexsha: str

    dummy_commit = DummyCommit("abc1234")

    # Set up stages
    stages = {
        "pre_stages": [DummyStage()],
        "pre_test_stages": [DummyStage()],
        "batch_stages": [DummyStage()],
    }

    # Patch Config.get_config to return a dummy config
    dummy_config = MagicMock()
    dummy_config.execution_plan.num_commits = 1
    monkeypatch.setattr("energytrackr.config.config_store.Config.get_config", lambda: dummy_config)

    # Patch ProcessPoolExecutor to return a real Future
    class DummyExecutor:
        def __enter__(self) -> "DummyExecutor":
            self.future = concurrent.futures.Future()
            self.future.set_result(
                {
                    "commit": dummy_commit.hexsha,
                    "repo_path": "fake/path",
                    "abort_pipeline": False,
                    "worker_process": True,
                    "log_buffer": [(20, "Stage ran")],
                },
            )
            return self

        def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object | None) -> None:
            pass

        def submit(self, _fn: callable, *_args: tuple, **_kwargs: dict) -> concurrent.futures.Future:
            return self.future

    pipeline = Pipeline(stages, repo_path="fake/path")

    with patch("concurrent.futures.ProcessPoolExecutor", DummyExecutor):
        pipeline.run([[dummy_commit]])
