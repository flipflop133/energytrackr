"""Module for measuring energy consumption using perf."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class MeasureEnergyStage(PipelineStage):
    """Uses `perf` to measure energy consumption. Appends the data to a results file."""

    def __init__(self) -> None:
        """Initialize the MeasureEnergyStage with a timestamp.

        The timestamp is used to uniquely identify the results file
        generated during the energy measurement process.
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self, context: dict[str, Any]) -> None:
        """Runs the energy measurement and appends the data to a results file.

        If the build failed, or if there is no test command, or if the perf command fails,
        it will abort the pipeline unless ignore_failures is set.

        Args:
            context (dict): The context dictionary containing the current commit and other configuration.

        """
        config = Config.get_config()
        if context["build_failed"]:
            logging.info("Skipping energy measurement because build failed.")
            return

        test_cmd = config.execution_plan.test_command
        if not test_cmd:
            logging.warning("No test command => no energy measurement performed.")
            return

        repo_path = config.repo_path
        perf_command = f"perf stat -e power/energy-pkg/ {test_cmd}"

        logging.info("Measuring energy with: %s", perf_command)
        result = run_command(perf_command, cwd=repo_path)

        # If `perf` fails:
        if result.returncode != 0:
            logging.error("Perf command failed (code %d).", result.returncode)
            if not config.execution_plan.ignore_failures:
                context["abort_pipeline"] = True
                return
            else:
                logging.warning("Ignoring failures; continuing anyway.")

        # Extract the reading from perf output
        energy_pkg = self.extract_energy_value(result.stdout, "power/energy-pkg/")
        if energy_pkg is None:
            logging.warning("No energy data found in perf output.")
            if not config.execution_plan.ignore_failures:
                context["abort_pipeline"] = True
                return

        # Log to CSV
        commit_hash = context["commit"].hexsha
        assert repo_path is not None, "Repository path is not set in the configuration."
        output_file = Path(repo_path).parent / "energy_measurements" / f"energy_results_{self.timestamp}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("a") as fh:
            fh.write(f"{commit_hash},{energy_pkg}\n")

        logging.info("Appended energy data to %s", output_file)

    @staticmethod
    def extract_energy_value(perf_output: str, event_name: str) -> str | None:
        """Extracts the value of the specified event from perf output.

        Args:
            perf_output (str): The output from the perf command.
            event_name (str): The name of the event to extract, e.g. "power/energy-pkg/".

        Returns:
            str | None: The value of the event as a string, or None if not found.
        """
        for line in perf_output.split("\n"):
            if event_name in line:
                parts = line.split()
                if parts and "<not" not in parts[0]:
                    return parts[0].replace(",", "")
        return None
