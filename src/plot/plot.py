import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from scipy.stats import shapiro, ttest_ind

# ---------------------------
# Parameters and Constants
# ---------------------------
MIN_MEASUREMENTS = 2
NORMALITY_P_THRESHOLD = 0.05
MIN_VALUES_FOR_NORMALITY_TEST = 3
WELCH_P_THRESHOLD = 0.05  # Significance threshold for Welch's t-test
MIN_PCT_INCREASE = 0.05  # Practical threshold for change (10%)
WINDOW_SIZE = 15  # Sliding window size


# ---------------------------
# Dataclasses
# ---------------------------
@dataclass
class ChangeEvent:
    index: int
    severity: float  # Always positive (e.g., 0.25 for a 25% change)
    direction: str  # "increase" for regressions (worse energy) or "decrease" for improvements


@dataclass
class EnergyPlotData:
    x_indices: np.ndarray[Any, np.dtype[np.int64]]
    short_hashes: list[str]
    y_medians: list[float]
    y_errors: list[float]
    distribution_data: list[Any]
    normality_flags: list[bool]
    change_events: list[ChangeEvent]
    energy_column: str


# ---------------------------
# Data Preparation Functions
# ---------------------------
def prepare_commit_statistics(
    df: pd.DataFrame, energy_column: str
) -> tuple[list[str], list[str], np.ndarray[Any, np.dtype[np.int64]], list[float], list[float]]:
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
    df: pd.DataFrame, valid_commits: list[str], energy_column: str
) -> tuple[list[np.ndarray[Any, Any]], list[bool]]:
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


# ---------------------------
# Energy Change Detection Function
# ---------------------------
def detect_energy_changes(
    distribution_data: list[np.ndarray[Any, Any]],
    y_medians: list[float],
    short_hashes: list[str],
    window_size: int = WINDOW_SIZE,
    min_pct_change: float = MIN_PCT_INCREASE,
    p_threshold: float = WELCH_P_THRESHOLD,
) -> list[ChangeEvent]:
    """
    Detect energy changes using a sliding window approach.
    This function returns a list of ChangeEvent indicating the commit index,
    the severity (as a fraction), and the direction ("increase" or "decrease").
    """
    changes = []
    i = window_size

    while i < len(distribution_data) - window_size:
        # Create the baseline and test windows
        baseline = np.concatenate(distribution_data[i - window_size : i])
        test = np.concatenate(distribution_data[i : i + window_size])
        if len(baseline) < MIN_VALUES_FOR_NORMALITY_TEST or len(test) < MIN_VALUES_FOR_NORMALITY_TEST:
            i += 1
            continue

        baseline_median = np.median(baseline)
        test_median = np.median(test)
        # Perform statistical test between the two windows
        _, p_value = ttest_ind(baseline, test, equal_var=False)

        if p_value < p_threshold:
            # Check for regression (energy increase)
            if test_median >= baseline_median * (1 + min_pct_change):
                for j in range(i, i + window_size - 1):
                    if y_medians[j + 1] >= y_medians[j] * (1 + min_pct_change):
                        severity = (y_medians[j + 1] - baseline_median) / baseline_median
                        changes.append(ChangeEvent(index=j + 1, severity=severity, direction="increase"))
                        break
                i += window_size
                continue
            # Check for improvement (energy decrease)
            elif test_median <= baseline_median * (1 - min_pct_change):
                for j in range(i, i + window_size - 1):
                    if y_medians[j + 1] <= y_medians[j] * (1 - min_pct_change):
                        severity = (baseline_median - y_medians[j + 1]) / baseline_median
                        changes.append(ChangeEvent(index=j + 1, severity=severity, direction="decrease"))
                        break
                i += window_size
                continue
        i += 1

    return changes


# ---------------------------
# Plotting Functions
# ---------------------------
def plot_energy_data(ax: Axes, plot_data: EnergyPlotData) -> None:
    """
    Plot energy data with violin plots for the distributions, errorbars for the medians,
    and vertical shaded areas indicating energy changes with colors and opacity representing the severity.
    """
    x_indices = plot_data.x_indices
    short_hashes = plot_data.short_hashes
    y_medians = plot_data.y_medians
    y_errors = plot_data.y_errors
    distribution_data = plot_data.distribution_data
    normality_flags = plot_data.normality_flags
    change_events = plot_data.change_events
    energy_column = plot_data.energy_column

    # Plot violin plots for distribution data
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

    # Plot median energy values with error bars
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

    # Draw vertical shaded areas for each detected energy change
    for event in change_events:
        cp = event.index
        # Cap severity (e.g., maximum of a 50% change) for color mapping
        capped_severity = min(event.severity, 0.5)
        # Map severity to opacity: 0% → 0.2 opacity and 50% → 0.8 opacity
        opacity = (capped_severity / 0.5) * (0.8 - 0.2) + 0.2

        if event.direction == "increase":
            color = to_rgba((1.0, 0.0, 0.0, opacity))  # red for regressions
            text_color = "darkred"
            sign = "+"
        else:
            color = to_rgba((0.0, 0.8, 0.0, opacity))  # green for improvements
            text_color = "darkgreen"
            sign = "-"

        ax.axvspan(cp - 0.5, cp + 0.5, color=color, zorder=0)
        ax.text(
            cp,
            max(y_medians) * 1.05,
            f"{sign}{int(event.severity * 100)}%",
            fontsize=8,
            ha="center",
            color=text_color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            zorder=7,
        )

    # Format the X-axis with commit short hashes
    ax.set_xticks(x_indices)
    ax.set_xticklabels(short_hashes, rotation=45, ha="right")
    ax.set_xlabel("Commit Hash (sorted by date, oldest to newest)")
    ax.set_ylabel(f"Median Energy ({energy_column})")
    ax.set_title(f"Energy Consumption Trend (Median per Commit) - {energy_column}")
    ax.grid(True)

    # Custom legend handles
    custom_handles = [
        Line2D([0], [0], marker="o", color="blue", label=f"Median {energy_column}", linestyle="-"),
        Patch(facecolor="lightgrey", edgecolor="black", label="Normal Distribution"),
        Patch(facecolor="lightcoral", edgecolor="black", label="Non-Normal (Shapiro-Wilk p < 0.05)"),
        Patch(facecolor=to_rgba((1.0, 0.0, 0.0, 0.5)), edgecolor="none", label="Regression (↑ energy)"),
        Patch(facecolor=to_rgba((0.0, 0.8, 0.0, 0.5)), edgecolor="none", label="Improvement (↓ energy)"),
    ]
    ax.legend(handles=custom_handles)


def create_energy_plot(df: pd.DataFrame, energy_column: str, output_filename: str) -> None:
    """
    Create and save a plot for a given energy column from the CSV data.
    """
    valid_commits, short_hashes, x_indices, y_medians, y_errors = prepare_commit_statistics(df, energy_column)
    distribution_data, normality_flags = compute_distribution_and_normality(df, valid_commits, energy_column)
    change_events = detect_energy_changes(distribution_data, y_medians, short_hashes)

    plot_data = EnergyPlotData(
        x_indices=x_indices,
        short_hashes=short_hashes,
        y_medians=y_medians,
        y_errors=y_errors,
        distribution_data=distribution_data,
        normality_flags=normality_flags,
        change_events=change_events,
        energy_column=energy_column,
    )

    plt.figure(figsize=(40, 10))
    ax = plt.gca()
    plot_energy_data(ax, plot_data)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def create_energy_plots(input_path: str) -> None:
    """
    Load the CSV data, generate plots for each energy metric, and save the images.
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
