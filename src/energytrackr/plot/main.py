"""Plotting module for energy consumption data analysis."""

import os
import sys
from datetime import datetime

from energytrackr.plot.data import EnergyData
from energytrackr.plot.page import ReportPage
from energytrackr.utils.logger import logger


def plot(input_path: str, git_repo_path: str | None = None) -> None:
    """Plotting function for energy consumption data.

    Coordinator function for loading energy consumption data, processing it,
    and generating interactive plots and summary reports in both text and HTML formats.

    Args:
        input_path: Path to the CSV file containing the data.
        git_repo_path: Optional path to the Git repository to enrich commit details.
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
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    try:
        energy_data.load_data(column_names=column_names)
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    # Process each energy column independently.
    for col in energy_columns:
        # Check whether the column has non-empty data.
        if energy_data.df is None or energy_data.df[col].dropna(inplace=False).empty:
            logger.info("Skipping column '%s' (empty data).", col)
            continue

        # Process the column: outlier filtering, commit-wise stats, distribution analysis, and change detection.
        energy_data.process_column(col)

        # Check if the processing yielded valid commits for plotting.
        stats = energy_data.stats.get(col)
        if not stats or not stats.valid_commits:
            logger.info("No valid commits for column '%s'. Skipping plot for this energy metric.", col)
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

        # Export HTML report to the same folder as the CSV file.
        report.generate_html_summary(output_folder=folder)
