"""Module to filter commits and add regression context based on configuration.

This merged stage first applies filtering criteria to the commit list,
then uses the original commit order along with the configuration parameters
(min_commits_before and min_commits_after) to ensure that each candidate commit
has a previous and/or a next commit for comparison. If a neighboring commit was removed
during filtering, it is added back from the original list.
"""

from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils.logger import logger


class FilterAndRegressionStage(PipelineStage):
    """Stage to filter commits and augment them with regression context.

    Combined stage to filter commits and augment them with regression context
    based on the commit order and configuration parameters.
    """

    def run(self, context: dict[str, Any]) -> None:
        """Executes the filtering and regression augmentation stage of the pipeline.

        This method processes a batch of commits provided in the context, filters them
        based on specified criteria, and augments the filtered commits with their
        neighboring commits in the original order to provide additional regression context.

        Args:
            context (dict[str, Any]): A dictionary containing the pipeline context.
                Expected keys:
                - "batch": A list of commits to process. Each commit is expected to have
                  attributes such as `hexsha` and `stats.files`.
                - "abort_pipeline" (optional): A flag to indicate whether the pipeline should
                  be aborted.

        Behavior:
            - If no commits are provided in the "batch", the pipeline is aborted.
            - Commits are filtered based on the following criteria:
                - Files modified in the commit should not belong to ignored directories.
                - At least one file modified in the commit should have a tracked file extension.
            - After filtering, the method augments the filtered commits with their neighbors
              from the original list, based on the `min_commits_before` and `min_commits_after`
              configuration values.
            - The final list of commits is sorted in the original order and stored back in
              the "batch" key of the context.

        Logging:
            - Logs the number of commits before and after filtering.
            - Logs the number of unique commits after filtering.
            - Logs the final number of commits after regression augmentation.
            - Logs warnings if no commits pass the filter criteria or if the final commit
              count exceeds the original count.

        Raises:
            None
        """
        # Save the original commit list in order (assumed to be meaningful)
        original_commits = context.get("batch", [])
        original_count = len(original_commits)
        if not original_commits:
            logger.warning("No commits to process.", context=context)
            context["abort_pipeline"] = True
            return

        # Create a mapping for quick lookup of a commit's position in the original list.
        commit_index = {commit.hexsha: i for i, commit in enumerate(original_commits)}

        config = Config.get_config()
        regression_config = config.regression_detection
        min_before = regression_config.min_commits_before
        min_after = regression_config.min_commits_after

        logger.error("Number of commits before filtering: %d", original_count, context=context)

        # --- Filtering Step ---
        filtered_commits = []
        for commit in original_commits:
            remove_commit = False
            has_tracked_file = False

            # Check each file modified in the commit.
            for file in commit.stats.files:
                # Mark for removal if file is in an ignored directory.
                if any(file.startswith(directory) for directory in config.ignored_directories):
                    remove_commit = True
                    break
                # Mark as acceptable if file extension is among tracked ones.
                if file.endswith(tuple(config.tracked_file_extensions)):
                    has_tracked_file = True

            if not remove_commit and has_tracked_file:
                filtered_commits.append(commit)

        logger.error("Number of commits after filtering: %d", len(filtered_commits), context=context)
        logger.error("Unique commits after filtering: %d", len({c.hexsha for c in filtered_commits}), context=context)

        if not filtered_commits:
            logger.warning("No commits passed the filter criteria.", context=context)
            context["abort_pipeline"] = True
            return
        else:
            logger.info("%d commits passed the filter criteria.", len(filtered_commits), context=context)

        # --- Regression Context Augmentation using Original Order ---
        # Build a dictionary (or set) keyed by commit hexsha for fast membership checking.
        augmented = {commit.hexsha: commit for commit in filtered_commits}

        # For each commit in the filtered list, check its original neighbors.
        for commit in filtered_commits:
            pos = commit_index.get(commit.hexsha)
            # Add preceding commits if needed.
            for i in range(1, min_before + 1):
                neighbor_pos = pos - i
                if neighbor_pos < 0:
                    break  # No more commits before.
                neighbor = original_commits[neighbor_pos]
                if neighbor.hexsha not in augmented:
                    augmented[neighbor.hexsha] = neighbor
            # Add succeeding commits if needed.
            for i in range(1, min_after + 1):
                neighbor_pos = pos + i
                if neighbor_pos >= original_count:
                    break  # No more commits after.
                neighbor = original_commits[neighbor_pos]
                if neighbor.hexsha not in augmented:
                    augmented[neighbor.hexsha] = neighbor

        final_commits = [augmented[h] for h in augmented]
        # Sort the final list in the original order using commit_index.
        final_commits.sort(key=lambda c: commit_index[c.hexsha])

        logger.info("Final number of commits after regression augmentation: %d", len(final_commits), context=context)
        # It should never exceed the original count because we only add neighbors from the original list.
        if len(final_commits) > original_count:
            logger.warning(
                "Final commit count (%d) exceeds original commit count (%d)!",
                len(final_commits),
                original_count,
                context=context,
            )
        context["batch"] = final_commits
