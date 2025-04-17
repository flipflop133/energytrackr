"""Plotting module for energy consumption data analysis."""

import os
import sys
from datetime import datetime

from utils.logger import logger  # Assuming you have a logger module; otherwise, use logging.getLogger(__name__)

from .data import EnergyData
from .page import ReportPage


def plot(input_path: str, git_repo_path: str | None = None) -> None:
    """Plotting function for energy consumption data.

    Coordinator function for loading energy consumption data, processing it,
    and generating interactive plots and summary reports in both text and HTML formats.

    Args:
        input_path: Path to the CSV file containing the data.
        git_repo_path: Optional path to the Git repository to enrich commit details.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        Exception: For any other errors encountered during processing.
    """
    # Determine the project folder and project name from the input path.
    folder = os.path.dirname(input_path)
    project_name = os.path.basename(folder)
    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define CSV column names and energy columns to process.
    column_names = ["commit", "energy-pkg", "energy-core", "energy-gpu"]
    energy_columns = ["energy-pkg", "energy-core", "energy-gpu"]

    # Instantiate the data object and load the CSV data.
    energy_data = EnergyData(csv_path=input_path, energy_columns=energy_columns)
    if not os.path.isfile(input_path):
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    try:
        energy_data.load_data(column_names=column_names)
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    # Process each energy column independently.
    for col in energy_columns:
        # Check whether the column has non-empty data.
        if energy_data.df[col].dropna().empty:
            logger.info(f"Skipping column '{col}' (empty data).")
            continue

        # Process the column: outlier filtering, commit-wise stats, distribution analysis, and change detection.
        energy_data.process_column(col)

        # Check if the processing yielded valid commits for plotting.
        stats = energy_data.stats.get(col)
        if not stats or not stats.valid_commits:
            logger.info(f"No valid commits for column '{col}'. Skipping plot for this energy metric.")
            continue

        # Build and generate the report page: the ReportPage object coordinates plotting and summary export.
        report = ReportPage(
            energy_data=energy_data,
            energy_column=col,
            git_repo_path=git_repo_path,
            template_dir=os.path.join(os.path.dirname(__file__), "templates"),
        )
        report.project_name = project_name  # Optionally update the project name.
        report.timestamp = timestamp_now

        # Export text and HTML reports to the same folder as the CSV file.
        report.generate_text_summary(output_folder=folder)
        report.generate_html_summary(output_folder=folder)
