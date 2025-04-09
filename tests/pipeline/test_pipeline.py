import pytest
from unittest.mock import MagicMock, patch
from git import Commit

from pipeline.pipeline import Pipeline, run_pre_test_stages_for_commit, log_context_buffer


@pytest.fixture
def dummy_commit():
    commit = MagicMock(spec=Commit)
    commit.hexsha = "deadbeef"
    return commit


@pytest.fixture
def dummy_stages():
    stage_mock = MagicMock()
    stage_mock.run = MagicMock()
    return {"pre_stages": [stage_mock], "pre_test_stages": [stage_mock], "batch_stages": [stage_mock]}


@pytest.fixture
def dummy_repo_path(tmp_path):
    return str(tmp_path / "repo")


@pytest.fixture
def dummy_context():
    return {
        "commit": "deadbeef",
        "build_failed": False,
        "abort_pipeline": False,
        "repo_path": "/tmp/fake_repo",
        "worker_process": True,
        "log_buffer": [(20, "Stage ran"), (30, "Warning here")],
    }


def test_run_pre_test_stages_for_commit_success(monkeypatch, tmp_path, dummy_commit):
    repo = MagicMock()
    repo.commit.return_value = dummy_commit
    monkeypatch.setattr("git.Repo", lambda path: repo)

    dummy_stage = MagicMock()
    dummy_stage.run = MagicMock()
    result = run_pre_test_stages_for_commit(dummy_commit.hexsha, [dummy_stage], str(tmp_path))

    assert result["commit"] == dummy_commit.hexsha
    assert not result["abort_pipeline"]
    dummy_stage.run.assert_called_once()


def test_run_pre_test_stages_for_commit_failure(monkeypatch):
    monkeypatch.setattr("git.Repo", lambda path: (_ for _ in ()).throw(Exception("Repo error")))
    dummy_stage = MagicMock()

    result = run_pre_test_stages_for_commit("badsha", [dummy_stage], "/fake")
    assert result["abort_pipeline"]
    dummy_stage.run.assert_not_called()


import logging
from pipeline.pipeline import log_context_buffer


def test_log_context_buffer_logs_messages(caplog):
    context = {
        "commit": "abc1234",
        "log_buffer": [
            (logging.INFO, "This is info"),
            (logging.WARNING, "This is a warning"),
        ],
    }

    # Ensure messages go to your custom logger name
    with caplog.at_level(logging.INFO, logger="energy-pipeline"):
        log_context_buffer(context)

    assert any("This is info" in msg for msg in caplog.messages)
    assert any("This is a warning" in msg for msg in caplog.messages)
    assert any("Logs for commit abc1234" in msg for msg in caplog.messages)


import concurrent.futures
from unittest.mock import patch, MagicMock
from pipeline.pipeline import Pipeline, run_pre_test_stages_for_commit
from pipeline.stage_interface import PipelineStage


from unittest.mock import patch, MagicMock
from pipeline.pipeline import Pipeline
from pipeline.stage_interface import PipelineStage
import concurrent.futures


def test_pipeline_run_all_stages(monkeypatch):
    class DummyStage(PipelineStage):
        def run(self, context):
            context.setdefault("ran", []).append(self.__class__.__name__)

    # Dummy commit with fake hexsha
    class DummyCommit:
        def __init__(self, hexsha):
            self.hexsha = hexsha

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
    monkeypatch.setattr("config.config_store.Config.get_config", lambda: dummy_config)

    # Patch ProcessPoolExecutor to return a real Future
    class DummyExecutor:
        def __enter__(self):
            self.future = concurrent.futures.Future()
            self.future.set_result(
                {
                    "commit": dummy_commit.hexsha,
                    "repo_path": "fake/path",
                    "abort_pipeline": False,
                    "worker_process": True,
                    "log_buffer": [(20, "Stage ran")],
                }
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def submit(self, fn, *args, **kwargs):
            return self.future

    pipeline = Pipeline(stages, repo_path="fake/path")

    with patch("concurrent.futures.ProcessPoolExecutor", DummyExecutor):
        pipeline.run([[dummy_commit]])
