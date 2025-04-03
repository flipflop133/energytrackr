import logging
import os
from typing import Any

import git
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.config_store import Config
from pipeline.stage_interface import PipelineStage


class Pipeline:
    """Orchestrates the provided stages for each commit in sequence."""

    def __init__(self, stages: dict[str, list[PipelineStage]]) -> None:
        self.stages = stages
        self.config = Config.get_config()

    def _run_stage_group(self, stages: list[PipelineStage], context: dict[str, Any]) -> bool:
        """
        Run a group of stages with the given context sequentially.
        Returns True if all stages ran successfully, or False if a stage requested an abort.
        """
        for stage in stages:
            stage.run(context)
            if context.get("abort_pipeline"):
                logging.warning("Aborting remaining stages for stage %s", stage.__class__.__name__)
                return False
        return True

    def run(self, batches: list[list[git.Commit]]) -> None:
        with tqdm(total=len(batches), desc="Energy Pipeline", unit="batch") as progress_bar:
            for batch in batches:
                logging.info("Processing batch of %d tasks", len(batch))

                # Run pre-stages sequentially
                pre_context = {"build_failed": False, "abort_pipeline": False}
                if not self._run_stage_group(self.stages.get("pre_stages", []), pre_context):
                    return

                # Run pre-batch stages concurrently
                with tqdm(total=len(batch), desc="Pre batch stages", unit="commit", leave=False) as inner_progress_bar:
                    pre_batch_context = {
                        "commits": batch,
                        "bar": inner_progress_bar,
                        "build_failed": False,
                        "abort_pipeline": False,
                    }
                    pre_batch_stages = self.stages.get("pre_batch_stages", [])
                    # Create a thread pool with as many workers as there are CPU cores
                    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                        futures = {executor.submit(stage.run, pre_batch_context): stage for stage in pre_batch_stages}
                        for future in as_completed(futures):
                            try:
                                future.result()
                            except Exception as e:
                                logging.exception("Exception in pre-batch stage: %s", e)
                                pre_batch_context["abort_pipeline"] = True
                                return
                            if pre_batch_context.get("abort_pipeline"):
                                stage = futures[future]
                                logging.warning("Aborting remaining pre-batch stages for stage %s", stage.__class__.__name__)
                                executor.shutdown(cancel_futures=True)
                                return

                # Run pipeline stages for each commit in the batch
                with tqdm(total=len(batch), desc="Batch stages", unit="commit", leave=False) as inner_progress_bar:
                    logging.info("Starting pipeline over %d commits...", len(batch))
                    for commit in batch:
                        inner_progress_bar.set_postfix(current_commit=commit.hexsha[:8])
                        inner_progress_bar.update(1)
                        commit_context = {
                            "commit": commit,
                            "build_failed": False,
                            "abort_pipeline": False,
                        }
                        logging.info("==== Processing commit %s ====", commit.hexsha)

                        if not self._run_stage_group(self.stages.get("batch_stages", []), commit_context):
                            logging.warning("Aborting remaining stages for commit %s", commit.hexsha)
                            return

                        logging.info("==== Done with commit %s ====\n", commit.hexsha)

                progress_bar.update(1)
