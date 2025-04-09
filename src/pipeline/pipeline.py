"""Pipeline orchestrator for running stages on commits in a repository."""

import concurrent.futures
import os
import random
from typing import Any

import git
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config.config_store import Config
from config.loader import load_pipeline_config
from pipeline.core_stages.build_stage import BuildStage
from pipeline.core_stages.checkout_stage import CheckoutStage
from pipeline.core_stages.copy_directory_stage import CopyDirectoryStage
from pipeline.core_stages.measure_stage import MeasureEnergyStage
from pipeline.core_stages.post_test_stage import PostTestStage
from pipeline.core_stages.set_directory_stage import SetDirectoryStage
from pipeline.core_stages.temperature_check_stage import TemperatureCheckStage
from pipeline.core_stages.verify_perf_stage import VerifyPerfStage
from pipeline.custom_stages.java_setup_stage import JavaSetupStage
from pipeline.stage_interface import PipelineStage
from utils.git import clone_or_open_repo, gather_commits
from utils.logger import logger

pre_stages: list[PipelineStage] = [
    VerifyPerfStage(),
]

pre_test_stages: list[PipelineStage] = [
    CopyDirectoryStage(),
    SetDirectoryStage(),
    CheckoutStage(),
    JavaSetupStage(),
    BuildStage(),
]

batch_stages: list[PipelineStage] = [
    TemperatureCheckStage(),
    SetDirectoryStage(),
    JavaSetupStage(),
    MeasureEnergyStage(),
    PostTestStage(),
]


def compile_stages() -> dict[str, list[PipelineStage]]:
    """Compile the pipeline stages based on the execution plan.

    Returns:
        list[PipelineStage]: The compiled list of pipeline stages.
    """
    return {"pre_stages": pre_stages, "pre_test_stages": pre_test_stages, "batch_stages": batch_stages}


def measure(config_path: str) -> None:
    """Executes the measurement process for a given repository based on the provided configuration.

    This function performs the following steps:
    1. Loads the pipeline configuration from the specified path.
    2. Sets up the repository directory and clones or opens the repository.
    3. Optionally runs system-level setup commands defined in the configuration.
    4. Collects commits from the repository and divides them into batches for processing.
    5. Executes the pipeline stages on the batched tasks.
    6. Restores the repository's HEAD to the latest commit on the specified branch.

    Args:
        config_path (str): The file path to the configuration file.

    Raises:
        Any exceptions raised during the execution of the pipeline or repository operations.
    """
    load_pipeline_config(config_path)
    config = Config.get_config()
    # Set up repo path
    project_name = os.path.basename(config.repo.url).replace(".git", "").lower()
    project_dir = os.path.join("projects", project_name)
    cache_dir = os.path.join(project_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)

    repo_path: str = os.path.abspath(os.path.join(project_dir, f".cache/.cache_{project_name}"))
    repo = clone_or_open_repo(repo_path, config.repo.url, config.repo.clone_options)

    # (Optional) run system-level setup commands
    if config.setup_commands:
        for cmd in config.setup_commands:
            logger.info("Running setup command: %s", cmd)
            os.system(cmd)

    commits = gather_commits(repo)
    logger.info("Collected %d commits to process.", len(commits))

    # Divide the list of commits into batches of 'batch_size' commits each
    commit_batches = [
        commits[i : i + config.execution_plan.batch_size] for i in range(0, len(commits), config.execution_plan.batch_size)
    ]

    batches = []
    for commit_batch in commit_batches:
        batch_tasks = []
        for commit in commit_batch:
            # Add all the runs and repeats for this commit to the batch
            runs_per_commit = config.execution_plan.num_runs * config.execution_plan.num_repeats
            batch_tasks.extend([commit] * runs_per_commit)

        if config.execution_plan.randomize_tasks:
            random.shuffle(batch_tasks)

        batches.append(batch_tasks)

    pipeline = Pipeline(compile_stages(), repo_path)
    pipeline.run(batches)

    # Finally, restore HEAD
    repo.git.checkout(config.repo.branch)
    logger.info("Restored HEAD to latest commit on branch %s.", config.repo.branch)


def run_pre_test_stages_for_commit(commit_hexsha: str, stages: list[PipelineStage], repo_path: str) -> dict[str, Any]:
    """Process the pre-test stages for a single commit in a separate process.

    Instead of receiving a git.Commit object (which might not be picklable), we pass the commit's hexsha.
    Each process reopens the repository using repo_path and retrieves the commit object.

    Args:
        commit_hexsha (str): The hexsha of the commit to process.
        stages (list[PipelineStage]): The list of pre-test stages to run.
        repo_path (str): Path to the repository.

    Returns:
        dict: The context after processing the stages.
    """
    commit_context = {
        "commit": str,
        "build_failed": False,
        "abort_pipeline": False,
        "repo_path": repo_path,
        "worker_process": True,
    }
    try:
        # Re-open the repository in the worker process.
        repo = git.Repo(repo_path)
        commit = repo.commit(commit_hexsha)
        commit_context["commit"] = commit.hexsha
    except Exception:
        logger.exception(f"Failed to open repo or retrieve commit {commit_hexsha}")
        commit_context["abort_pipeline"] = True
        return commit_context

    # Execute each pre-test stage.
    for stage in stages:
        stage.run(commit_context)
        if commit_context.get("abort_pipeline"):
            break
    return commit_context


def log_context_buffer(context: dict[str, Any]) -> None:
    """Logs the buffered log messages from the provided context dictionary.

    This function retrieves a list of log messages from the "log_buffer" key
    in the context dictionary and logs them using the appropriate log level.
    It also includes the commit ID (retrieved from the "commit" key) in the
    log headers. If no log messages are present in the buffer, the function
    exits without logging anything.

    Args:
        context (dict[str, Any]): A dictionary containing the log buffer and
            commit information. Expected keys:
            - "log_buffer" (list[tuple[int, str]]): A list of tuples where each
              tuple contains a log level (int) and a log message (str).
            - "commit" (str): A string representing the commit ID. Defaults to
              "UNKNOWN" if not provided.

    Returns:
        None
    """
    logs = context.get("log_buffer", [])
    commit_id = context.get("commit", "UNKNOWN")

    if not logs:
        return

    logger.info("----- Logs for commit %s -----", commit_id[:8])
    for level, msg in logs:
        logger.log(level, msg)
    logger.info("----- End of logs for %s -----\n", commit_id[:8])


class Pipeline:
    """Orchestrates the provided stages for each commit in sequence."""

    def __init__(self, stages: dict[str, list[PipelineStage]], repo_path: str) -> None:
        """Initializes the Pipeline with the given stages and configuration.

        Args:
            stages (dict[str, list[PipelineStage]]): A dictionary where the keys are stage names (as strings)
                and the values are lists of PipelineStage objects representing the stages of the pipeline.
            repo_path (str): The path to the Git repository to be processed.

        """
        self.stages = stages
        self.config = Config.get_config()
        self.repo_path = repo_path

    def _run_stage_group(self, stages: list[PipelineStage], context: dict[str, Any]) -> bool:
        """Run a group of stages with the given context.

        Returns True if all stages ran successfully, or False if a stage requested an abort.
        """
        for stage in stages:
            stage.run(context)
            if context.get("abort_pipeline"):
                logger.warning("Aborting remaining stages for stage %s", stage.__class__.__name__)
                return False
        return True

    def run(self, batches: list[list[git.Commit]]) -> None:
        """Executes the pipeline over a list of batches, where each batch contains a list of commits.

        Args:
            batches (list[list[git.Commit]]): A list of batches, where each batch is a list of
                `git.Commit` objects to process.

        Returns:
            None

        The method processes each batch sequentially, performing the following steps:
        1. Executes pre-stages sequentially for the entire batch.
        2. Runs pre-test stages concurrently for unique commits in the batch using a
           `ProcessPoolExecutor`.
        3. Processes each commit in the batch sequentially through the pipeline stages.

        Progress is displayed using `rich.progress` with multiple progress bars for:
        - Overall pipeline progress across batches.
        - Pre-test stages progress for unique commits in a batch.
        - Batch stages progress for individual commits in a batch.

        The progress descriptions now include the cumulative number of failed commits
        (both from the pre-test and batch phases).

        Logging is used to provide detailed information about the pipeline's execution.
        """
        failed_commits: set[str] = set()

        with Progress(
            SpinnerColumn(style="green"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
            TextColumn("{task.completed:>4}/{task.total:<4}", justify="right"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            pipeline_task = progress.add_task("🔋Energy Pipeline", total=len(batches))

            for batch in batches:
                logger.info("Processing batch of %d tasks", len(batch))

                # Run pre-stages sequentially.
                pre_context = {
                    "build_failed": False,
                    "abort_pipeline": False,
                    "repo_path": self.repo_path,
                }
                if not self._run_stage_group(self.stages.get("pre_stages", []), pre_context):
                    return

                # Prepare unique commit identifiers (hexsha strings) for the pre-test stages.
                unique_commit_hexshas = list({commit.hexsha for commit in batch})
                pre_test_stages = self.stages.get("pre_test_stages", [])

                pre_test_task = progress.add_task("Pre batch stages", total=len(unique_commit_hexshas))

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = {
                        executor.submit(
                            run_pre_test_stages_for_commit,
                            commit_hexsha,
                            pre_test_stages,
                            self.repo_path,
                        ): commit_hexsha
                        for commit_hexsha in unique_commit_hexshas
                    }
                    for future in concurrent.futures.as_completed(futures):
                        commit_hexsha = futures[future]
                        try:
                            result = future.result(timeout=60)
                        except Exception:
                            logger.exception(f"Commit {commit_hexsha} generated an exception.")
                            return

                        # Display logs for the commit.
                        log_context_buffer(result)

                        # Optionally check the result for an abort signal.
                        if result.get("abort_pipeline"):
                            logger.warning("Aborting pre-test stages for commit %s", commit_hexsha)
                            failed_commits.add(commit_hexsha)

                        description = "Pre batch stages"
                        if failed_commits:
                            description += f" (failed: {len(failed_commits)})"
                        progress.update(pre_test_task, advance=1, description=description)

                progress.remove_task(pre_test_task)

                batch_to_process = [commit for commit in batch if commit.hexsha not in failed_commits]

                batch_stage_task = progress.add_task(
                    "[green]🧪Batch stages",
                    total=len(batch_to_process),
                )

                logger.info("Starting pipeline over %d commits...", len(batch_to_process))
                for commit in batch_to_process:
                    if commit.hexsha in failed_commits:
                        logger.warning("Skipping failed commit %s", commit.hexsha)
                        continue

                    # Update the progress description to include the current commit and failed count.
                    progress.update(
                        batch_stage_task,
                        description=f"🧪Batch stages ({commit.hexsha[:8]}) (failed: {len(failed_commits)})",
                    )
                    progress.advance(batch_stage_task)

                    commit_context = {
                        "commit": commit,
                        "build_failed": False,
                        "abort_pipeline": False,
                        "repo_path": self.repo_path,
                    }
                    logger.info("==== Processing commit %s ====", commit.hexsha)

                    if not self._run_stage_group(self.stages.get("batch_stages", []), commit_context):
                        logger.warning(f"Commit {commit.hexsha} failed to process.")
                        failed_commits.add(commit.hexsha)
                        continue

                    logger.info("==== Done with commit %s ====\n", commit.hexsha)

                # Log the summary of batch processing.
                logger.info("Batch stages completed with %d failed commits.", len(failed_commits))
                progress.remove_task(batch_stage_task)
                progress.advance(pipeline_task)
