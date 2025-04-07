"""Create a plot of median energy consumption from CSV file."""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import shapiro

# Parameters
MIN_MEASUREMENTS = 2
NORMALITY_P_THRESHOLD = 0.05
PELT_PENALTY = 10  # Adjust this penalty value to tune sensitivity
MIN_VALUES_FOR_NORMALITY_TEST = 3


@dataclass
class EnergyPlotData:
    """EnergyPlotData is a data structure that encapsulates information required for plotting energy-related data.

    Attributes:
        x_indices (np.ndarray): An array of indices representing the x-axis values.
        short_hashes (list[str]): A list of short hash strings corresponding to data points.
        y_medians (list[float]): A list of median values for the y-axis.
        y_errors (list[float]): A list of error values associated with the y-axis medians.
        distribution_data (list[Any]): A list of distribution data arrays for each data point.
        normality_flags (list[bool]): A list of boolean flags indicating whether the
        data at each point is normally distributed.
        change_points (list[int]): A list of indices where significant changes occur in the data.
        energy_column (str): The name of the energy-related column being analyzed or plotted.
    """

    x_indices: np.ndarray[Any, np.dtype[np.int64]]
    short_hashes: list[str]
    y_medians: list[float]
    y_errors: list[float]
    distribution_data: list[Any]
    normality_flags: list[bool]
    change_points: list[int]
    energy_column: str


def prepare_commit_statistics(
    df: pd.DataFrame,
    energy_column: str,
) -> tuple[list[str], list[str], np.ndarray[Any, np.dtype[np.int64]], list[float], list[float]]:
    """Prepare commit statistics for plotting.

    Processes a DataFrame to compute commit statistics, including median and standard deviation
    of energy values, while filtering out commits with insufficient measurements.

    Args:
        df (pd.DataFrame): The input DataFrame containing commit data. It must include the columns
            "commit" and the specified energy_column.
        energy_column (str): The name of the column in the DataFrame containing energy values.

    Returns:
        tuple: A tuple containing:
            - valid_commits (list[str]): A list of full commit hashes that passed the filtering criteria.
            - short_hashes (list[str]): A list of shortened commit hashes (first 7 characters).
            - x_indices (np.ndarray): An array of indices corresponding to the valid commits.
            - y_medians (list[float]): A list of median energy values for the valid commits.
            - y_errors (list[float]): A list of standard deviation values for the valid commits.
    """
    commit_counts = df.groupby("commit").size().reset_index(name="count")
    df_median = df.groupby("commit", sort=False)[energy_column].median().reset_index()
    df_std = df.groupby("commit", sort=False)[energy_column].std().reset_index()
    df_median = df_median.merge(df_std, on="commit", suffixes=("", "_std"))
    df_median = df_median.merge(commit_counts, on="commit")
    df_median = df_median[df_median["count"] >= MIN_MEASUREMENTS]
    df_median["commit_short"] = df_median["commit"].str[:7]

    valid_commits = df_median["commit"].tolist()
    short_hashes = df_median["commit_short"].tolist()
    x_indices = np.arange(len(valid_commits))
    y_medians = df_median[energy_column].tolist()
    y_errors = df_median[f"{energy_column}_std"].tolist()

    return valid_commits, short_hashes, x_indices, y_medians, y_errors


def compute_distribution_and_normality(
    df: pd.DataFrame,
    valid_commits: list[str],
    energy_column: str,
) -> tuple[list[np.ndarray[Any, Any]], list[bool]]:
    """Computes the distribution of values and tests for normality for each commit.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        valid_commits (list[str]): A list of commit identifiers to process.
        energy_column (str): The name of the column containing the energy values.

    Returns:
        tuple[list[ArrayLike], list[bool]]:
            - A list of arrays, where each array contains the energy values for a specific commit.
            - A list of boolean flags indicating whether the values for each commit pass the Shapiro-Wilk normality test.
              If the number of values is below the minimum required for the test, the flag defaults to True.

    Notes:
        - The function uses the Shapiro-Wilk test to assess normality.
        - Constants `MIN_VALUES_FOR_NORMALITY_TEST` and `NORMALITY_P_THRESHOLD` are expected to be defined in the scope
          where this function is used.
    """
    distribution_data: list[np.ndarray[Any, Any]] = []
    normality_flags = []
    for commit in valid_commits:
        values = df[df["commit"] == commit][energy_column].values
        distribution_data.append(np.asarray(values))
        if len(values) >= MIN_VALUES_FOR_NORMALITY_TEST:
            _, p_shapiro = shapiro(values)
            normality_flags.append(p_shapiro >= NORMALITY_P_THRESHOLD)
        else:
            normality_flags.append(True)
    return distribution_data, normality_flags


def detect_change_points(y_medians: list[float]) -> list[int]:
    """Detects change points in a given time series using the PELT (Pruned Exact Linear Time) algorithm.

    Args:
        y_medians (list[float]): A list of median values representing the time series data.

    Returns:
        list[int]: A list of indices indicating the positions of detected change points in the time series.

    Notes:
        - The function uses the "l2" cost model for the PELT algorithm.
        - The penalty value for the PELT algorithm is determined by the constant `PELT_PENALTY`,
          which must be defined elsewhere in the code.
        - The input `y_medians` is converted to a NumPy array for processing.
    """
    energy_series = np.array(y_medians)
    algo = rpt.Pelt(model="l2").fit(energy_series)
    change_points = algo.predict(pen=PELT_PENALTY)
    return list(map(int, change_points))


def plot_energy_data(ax: Axes, plot_data: EnergyPlotData) -> None:
    """Plot the energy data using the bundled plot_data.

    This function draws violin plots, error bars, change point indicators,
    and breaking commit markers using the information provided in plot_data.
    """
    # Unpack data for readability
    x_indices = plot_data.x_indices
    short_hashes = plot_data.short_hashes
    y_medians = plot_data.y_medians
    y_errors = plot_data.y_errors
    distribution_data = plot_data.distribution_data
    normality_flags = plot_data.normality_flags
    change_points = plot_data.change_points
    energy_column = plot_data.energy_column

    # Violin plots for the distribution per commit
    violin_parts = ax.violinplot(
        distribution_data,
        positions=x_indices,
        widths=0.5,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    bodies = cast(list[PathCollection], violin_parts["bodies"])
    for pc, is_normal in zip(bodies, normality_flags, strict=False):
        pc.set_facecolor("lightgrey" if is_normal else "lightcoral")
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)
        pc.set_zorder(1)

    # Error bars (median ± std)
    ax.errorbar(
        x_indices,
        y_medians,
        yerr=y_errors,
        marker="o",
        linestyle="-",
        color="b",
        label=f"Median {energy_column}",
        zorder=2,
    )

    # Mark detected change points and highlight breaking commits
    regression_label_added = False
    break_commit_label_added = False
    for cp in change_points[:-1]:
        if not regression_label_added:
            ax.axvline(x=cp - 0.5, color="green", linestyle="--", label="Regression Start")
            regression_label_added = True
        else:
            ax.axvline(x=cp - 0.5, color="green", linestyle="--")

        if cp < len(x_indices):
            if not break_commit_label_added:
                ax.scatter(
                    x_indices[cp],
                    y_medians[cp],
                    marker="*",
                    color="purple",
                    s=150,
                    zorder=6,
                    label="Breaking Commit",
                )
                break_commit_label_added = True
            else:
                ax.scatter(
                    x_indices[cp],
                    y_medians[cp],
                    marker="*",
                    color="purple",
                    s=150,
                    zorder=6,
                )
            ax.text(x_indices[cp], y_medians[cp], short_hashes[cp], fontsize=8, ha="left", color="purple")

    # Axis labels, ticks, and title
    ax.set_xticks(x_indices)
    ax.set_xticklabels(short_hashes, rotation=45, ha="right")
    ax.set_xlabel("Commit Hash (sorted by date, oldest to newest)")
    ax.set_ylabel(f"Median Energy ({energy_column})")
    ax.set_title(f"Energy Consumption Trend (Median per Commit) - {energy_column}")
    ax.grid(True)

    # Custom legend
    custom_handles = [
        Line2D([0], [0], marker="o", color="blue", label=f"Median {energy_column}", linestyle="-"),
        Line2D([0], [0], linestyle="--", color="green", label="Regression Start"),
        Line2D([0], [0], marker="*", color="purple", linestyle="None", markersize=10, label="Breaking Commit"),
        Patch(facecolor="lightgrey", edgecolor="black", label="Normal Distribution"),
        Patch(facecolor="lightcoral", edgecolor="black", label="Non-Normal (Shapiro-Wilk p < 0.05)"),
    ]
    ax.legend(handles=custom_handles)


def create_energy_plot(df: pd.DataFrame, energy_column: str, output_filename: str) -> None:
    """Create a plot of median energy consumption for each commit from a DataFrame."""
    # Prepare commit statistics
    valid_commits, short_hashes, x_indices, y_medians, y_errors = prepare_commit_statistics(df, energy_column)
    # Compute distributions and normality flags
    distribution_data, normality_flags = compute_distribution_and_normality(df, valid_commits, energy_column)
    # Detect change points
    change_points = detect_change_points(y_medians)

    # Bundle all plot data into a single dataclass instance
    plot_data = EnergyPlotData(
        x_indices=x_indices,
        short_hashes=short_hashes,
        y_medians=y_medians,
        y_errors=y_errors,
        distribution_data=distribution_data,
        normality_flags=normality_flags,
        change_points=change_points,
        energy_column=energy_column,
    )

    # Plot the energy data
    plt.figure(figsize=(40, 10))
    ax = plt.gca()
    plot_energy_data(ax, plot_data)
    plt.tight_layout()
    plt.savefig(output_filename)


def create_energy_plots(input_path: str) -> None:
    """Generates energy consumption plots for a given input CSV file.

    The function reads energy consumption data from a CSV file, processes it,
    and creates plots for each energy metric (e.g., "energy-pkg", "energy-core",
    "energy-gpu"). The generated plots are saved as PNG files in the same
    directory as the input file, with filenames that include the project name,
    energy metric, and a timestamp.

    Args:
        input_path (str): The file path to the input CSV file containing energy
                          consumption data. The file is expected to have no
                          header and four columns: "commit", "energy-pkg",
                          "energy-core", and "energy-gpu".

    Raises:
        SystemExit: If the input file does not exist, the function logs an error
                    message and exits the program.

    Notes:
        - The project name is derived from the name of the folder containing
          the input file.
        - The timestamp is generated in the format "YYYYMMDD_HHMMSS".
    """
    if not os.path.isfile(input_path):
        logging.info(f"Error: File not found: {input_path}")
        sys.exit(1)

    folder = os.path.dirname(input_path)
    project_name = os.path.basename(folder)
    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(
        input_path,
        header=None,
        names=["commit", "energy-pkg", "energy-core", "energy-gpu"],
    )

    for column in ["energy-pkg", "energy-core", "energy-gpu"]:
        output_filename = os.path.join(folder, f"{project_name}_{column}_{timestamp_now}.png")
        create_energy_plot(df, column, output_filename)
