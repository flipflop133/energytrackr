"""Module to check if user is allowed to use perf without sudo."""

import logging
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class VerifyPerfStage(PipelineStage):
    """Runs any pre-build commands (e.g., setting up environment).

    Optionally only runs if certain files changed, etc.
    """

    def run(self, context: dict[str, Any]) -> None:
        """Check if user is allowed to use perf without sudo."""
        # read perf_event_paranoid
        command_result = run_command("cat /proc/sys/kernel/perf_event_paranoid")
        perf_event_paranoid = int(command_result.stdout.strip())
        if perf_event_paranoid != -1:
            logging.error("Perf_event_paranoid is not set to -1")
            if not Config.get_config().execution_plan.ignore_failures:
                context["abort_pipeline"] = True
