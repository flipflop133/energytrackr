"""Plotting script for energy consumption data."""

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
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import shapiro, ttest_ind

# ---------------------------
# Parameters and Constants
# ---------------------------
MIN_MEASUREMENTS = 2
NORMALITY_P_THRESHOLD = 0.05
MIN_VALUES_FOR_NORMALITY_TEST = 3
WELCH_P_THRESHOLD = 0.05  # Significance threshold for Welch's t-test
MIN_PCT_INCREASE = 0.02  # Practical threshold for change (2%)
WINDOW_SIZE = 15  # Sliding window size


# ---------------------------
# Dataclasses
# ---------------------------
@dataclass
class ChangeEvent:
    """Data structure to hold energy change events.

    This structure is used to represent significant changes in energy consumption.

    Attributes:
        index (int): The index of the commit where the change occurred.
        severity (float): The severity of the change (e.g., 0.25 for a 25% change).
        direction (str): The direction of the change ("increase" for regressions or "decrease" for improvements).
    """

    index: int
    severity: float  # Always positive (e.g., 0.25 for a 25% change)
    direction: str  # "increase" for regressions (worse energy) or "decrease" for improvements


@dataclass
class EnergyPlotData:
    """Data structure to hold energy plot data.

    This structure is used to pass all the necessary data for plotting energy consumption trends.

    Attributes:
        x_indices (np.ndarray): X-axis indices for plotting.
        short_hashes (list[str]): Short commit hashes for labeling.
        y_medians (list[float]): Median energy values for each commit.
        y_errors (list[float]): Standard deviation of energy values for each commit.
        distribution_data (list[np.ndarray]): Distribution data for each commit.
        normality_flags (list[bool]): Flags indicating normality of distributions.
        change_events (list[ChangeEvent]): Detected energy change events.
        energy_column (str): The name of the energy column being plotted.
    """

    x_indices: np.ndarray[Any, np.dtype[np.int64]]
    short_hashes: list[str]
    y_medians: list[float]
    y_errors: list[float]
    distribution_data: list[Any]
    normality_flags: list[bool]
    change_events: list[ChangeEvent]
    energy_column: str


# ---------------------------
# Helper Functions for Git Information
# ---------------------------
def generate_commit_link(remote_url: str, commit_hash: str) -> str:
    """
    Generate a commit link from the remote URL and commit hash.

    Supports basic parsing for GitHub URLs.
    """
    if remote_url.startswith("git@"):
        # Convert git@github.com:user/repo.git to https://github.com/user/repo/commit/<hash>
        try:
            parts = remote_url.split(":")
            domain = parts[0].split("@")[-1]  # e.g., github.com
            repo_path = parts[1].replace(".git", "")
            return f"https://{domain}/{repo_path}/commit/{commit_hash}"
        except Exception:
            return "N/A"
    elif remote_url.startswith("https://"):
        # Remove trailing .git if present.
        repo_url = remote_url.replace(".git", "")
        return f"{repo_url}/commit/{commit_hash}"
    return "N/A"


def get_commit_details_from_git(commit_hash: str, repo) -> dict:
    """
    Retrieve commit details using GitPython.

    Returns a dictionary with keys:
      - commit_summary: The commit message summary.
      - commit_link: A link to the commit (if a remote URL can be parsed).
    """
    try:
        commit_obj = repo.commit(commit_hash)
        commit_summary = commit_obj.summary  # or commit_obj.message.strip() for full message
        commit_link = "N/A"
        if repo.remotes:
            # Use the first remote's URL
            remote_url = repo.remotes[0].url
            commit_link = generate_commit_link(remote_url, commit_hash)
        return {"commit_summary": commit_summary, "commit_link": commit_link}
    except Exception as e:
        logging.error(f"Error retrieving details for commit {commit_hash}: {e}")
        return {"commit_summary": "N/A", "commit_link": "N/A"}


# ---------------------------
# Data Preparation Functions
# ---------------------------
def prepare_commit_statistics(
    df: pd.DataFrame,
    energy_column: str,
) -> tuple[list[str], list[str], np.ndarray[Any, np.dtype[np.int64]], list[float], list[float]]:
    """Prepares commit statistics for energy data.

    This function computes the median and standard deviation of energy values
    associated with each commit, filtering out commits with insufficient measurements.
    It also generates short commit hashes for plotting.

    Args:
        df (pd.DataFrame): The input DataFrame containing energy data.
                           It must include columns 'commit' and the specified energy column.
        energy_column (str): The name of the column in the DataFrame containing energy values.

    Returns:
        tuple[list[str], list[str], np.ndarray[Any, np.dtype[np.int64]], list[float], list[float]]:
            - A list of valid commit identifiers.
            - A list of short commit hashes.
            - An array of x indices for plotting.
            - A list of median energy values for each commit.
            - A list of standard deviation values for each commit.
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
    """Computes the distribution data and normality flags for energy values associated with a list of valid commits.

    Args:
        df (pd.DataFrame): The input DataFrame containing energy data.
                           It must include columns 'commit' and the specified energy column.
        valid_commits (list[str]): A list of commit identifiers to filter the data.
        energy_column (str): The name of the column in the DataFrame containing energy values.

    Returns:
        tuple[list[np.ndarray[Any, Any]], list[bool]]:
            - A list of numpy arrays, where each array contains the energy values
              for a specific commit.
            - A list of boolean flags indicating whether the energy values for each
              commit passed the Shapiro-Wilk normality test (True if normal, False otherwise).
              If the number of values is below the threshold for the test, the flag defaults to True.

    Notes:
        - The function uses a global constant `MIN_VALUES_FOR_NORMALITY_TEST` to determine
          the minimum number of values required to perform the Shapiro-Wilk test.
        - The global constant `NORMALITY_P_THRESHOLD` is used as the p-value threshold
          for determining normality.
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


# ---------------------------
# Energy Change Detection Functions
# ---------------------------
def get_change_direction(baseline_median: float, test_median: float, min_pct_change: float) -> str | None:
    """Determine the direction of change given baseline and test medians.

    Args:
        baseline_median (float): The median energy of the baseline window.
        test_median (float): The median energy of the test window.
        min_pct_change (float): The minimum percentage change threshold.

    Returns:
      "increase" if test_median is at least (1 + min_pct_change)*baseline_median,
      "decrease" if test_median is at most (1 - min_pct_change)*baseline_median,
      Otherwise, None.
    """
    if test_median >= baseline_median * (1 + min_pct_change):
        return "increase"
    if test_median <= baseline_median * (1 - min_pct_change):
        return "decrease"
    return None


def find_first_change(
    y_medians: list[float],
    window_range: tuple[int, int],
    baseline_median: float,
    min_pct_change: float,
    direction: str,
) -> tuple[int, float] | None:
    """Search within indices [start, end) for the first commit meeting the change threshold.

    Parameters:
      y_medians      : List of median energy values.
      window_range   : Tuple (start, end) defining the range to search.
      baseline_median: The median energy of the baseline window.
      min_pct_change : The minimum percentage change threshold.
      direction      : 'increase' for regressions, 'decrease' for improvements.

    Returns:
      A tuple (commit_index, severity) if a change is found; otherwise, None.
    """
    for j in range(window_range[0], window_range[1] - 1):
        if direction == "increase" and y_medians[j + 1] >= y_medians[j] * (1 + min_pct_change):
            severity = (y_medians[j + 1] - baseline_median) / baseline_median
            return j + 1, severity
        if direction == "decrease" and y_medians[j + 1] <= y_medians[j] * (1 - min_pct_change):
            severity = (baseline_median - y_medians[j + 1]) / baseline_median
            return j + 1, severity
    return None


def detect_energy_changes(
    distribution_data: list[np.ndarray[Any, Any]],
    y_medians: list[float],
    min_pct_change: float = MIN_PCT_INCREASE,
    p_threshold: float = WELCH_P_THRESHOLD,
) -> list[ChangeEvent]:
    """Detect energy changes by comparing each commit to its immediate predecessor.

    Returns a list of ChangeEvent (with commit index, severity, and change direction),
    where changes could be energy increases ("increase") or improvements ("decrease").

    Args:
        distribution_data (list[np.ndarray[Any, Any]]): Distribution data for each commit.
        y_medians (list[float]): Median energy values for each commit.
        min_pct_change (float): Minimum percentage change to consider significant.
        p_threshold (float): P-value threshold for Welch's t-test.

    Returns:
        list[ChangeEvent]: List of detected energy change events.
    """
    changes = []
    # Start from the second commit, comparing to the previous one.
    for i in range(1, len(distribution_data)):
        baseline = distribution_data[i - 1]
        test = distribution_data[i]

        # Ensure there is a minimum number of values in both commits to run the test.
        if len(baseline) < MIN_VALUES_FOR_NORMALITY_TEST or len(test) < MIN_VALUES_FOR_NORMALITY_TEST:
            continue

        baseline_median = np.median(baseline)
        test_median = np.median(test)
        _, p_value = ttest_ind(baseline, test, equal_var=False)
        # Only proceed if the difference is statistically significant.
        if p_value >= p_threshold:
            continue

        direction = get_change_direction(baseline_median, test_median, min_pct_change)
        if direction is None:
            continue

        severity = (
            (test_median - baseline_median) / baseline_median
            if direction == "increase"
            else (baseline_median - test_median) / baseline_median
        )
        changes.append(ChangeEvent(index=i, severity=severity, direction=direction))

    return changes


# ---------------------------
# Plotting Functions
# ---------------------------
def plot_energy_data(ax: Axes, plot_data: EnergyPlotData) -> None:
    """Plot energy data with violin plots and error bars.

    Plot energy data with violin plots for the distributions, errorbars for the medians,
    and vertical shaded areas indicating energy changes with colors and opacity representing the severity.

    Args:
        ax (Axes): The matplotlib Axes object to plot on.
        plot_data (EnergyPlotData): The data to be plotted, including commit indices, short hashes,
            median values, error values, distribution data, normality flags, and change events.
    """
    x_indices = plot_data.x_indices
    short_hashes = plot_data.short_hashes
    y_medians = plot_data.y_medians
    y_errors = plot_data.y_errors
    distribution_data = plot_data.distribution_data
    normality_flags = plot_data.normality_flags
    change_events = plot_data.change_events
    energy_column = plot_data.energy_column

    # Plot violin plots for each commit's distribution.
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

    # Plot the medians with error bars.
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

    # Mark change events with vertical shaded areas.
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
    """Create and save a plot for a given energy column from the CSV data.

    Args:
        df (pd.DataFrame): DataFrame containing the energy data.
        energy_column (str): The name of the energy column to plot.
        output_filename (str): The filename to save the plot.
    """
    valid_commits, short_hashes, x_indices, y_medians, y_errors = prepare_commit_statistics(df, energy_column)
    distribution_data, normality_flags = compute_distribution_and_normality(df, valid_commits, energy_column)
    change_events = detect_energy_changes(distribution_data, y_medians)

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


# ---------------------------
# Export Summary Function
# ---------------------------
def export_change_events_summary(
    valid_commits: list[str],
    short_hashes: list[str],
    change_events: list[ChangeEvent],
    df: pd.DataFrame,
    energy_column: str,
    folder: str,
    project_name: str,
    timestamp_now: str,
    git_repo_path: str | None = None,
) -> None:
    """
    Exports a textual summary of detected energy change events.

    For each flagged commit, the summary lists:
      - Full commit hash and its short version.
      - Change direction and severity.
      - Commit summary message.
      - Commit link.

    If the CSV lacks commit details, and a git_repo_path is provided, GitPython is used to retrieve them.
    """
    commit_details = {}
    # If CSV already contains extra columns, try to pick them up
    if "commit_summary" in df.columns and "commit_link" in df.columns:
        for commit in valid_commits:
            row = df[df["commit"] == commit].iloc[0]
            commit_details[commit] = {
                "commit_summary": row.get("commit_summary", "N/A"),
                "commit_link": row.get("commit_link", "N/A"),
            }
    else:
        for commit in valid_commits:
            commit_details[commit] = {"commit_summary": "N/A", "commit_link": "N/A"}

    # If a git repository is provided, update missing details using GitPython.
    if git_repo_path:
        try:
            from git import Repo

            repo = Repo(git_repo_path)
            for commit in valid_commits:
                if commit_details[commit]["commit_summary"] == "N/A" or commit_details[commit]["commit_link"] == "N/A":
                    details = get_commit_details_from_git(commit, repo)
                    commit_details[commit] = details
        except Exception as e:
            logging.error(f"Error loading Git repository from {git_repo_path}: {e}")

    summary_lines = []
    summary_lines.append(f"Energy Consumption Change Summary for '{energy_column}'")
    summary_lines.append(f"Project: {project_name}")
    summary_lines.append(f"Date: {timestamp_now}")
    summary_lines.append("=" * 80)

    if not change_events:
        summary_lines.append("No significant energy changes detected.")
    else:
        for event in change_events:
            commit_hash = valid_commits[event.index]
            short_hash = short_hashes[event.index]
            details = commit_details.get(commit_hash, {"commit_summary": "N/A", "commit_link": "N/A"})
            direction_str = "Regression (Increase)" if event.direction == "increase" else "Improvement (Decrease)"
            summary_lines.append(f"Commit: {commit_hash} (Short: {short_hash})")
            summary_lines.append(f"Direction: {direction_str}")
            summary_lines.append(f"Severity: {int(event.severity * 100)}%")
            summary_lines.append(f"Commit Message: {details['commit_summary']}")
            summary_lines.append(f"Commit Link: {details['commit_link']}")
            summary_lines.append("-" * 80)

    summary_filename = os.path.join(folder, f"{project_name}_{energy_column}_{timestamp_now}_summary.txt")
    with open(summary_filename, "w") as f:
        f.write("\n".join(summary_lines))
    logging.info(f"Exported energy change summary to {summary_filename}")


def filter_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Filter out outliers from the given DataFrame column using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name on which to apply the filtering.
        multiplier (float): The multiplier for the IQR to set the bounds (default: 1.5).

    Returns:
        pd.DataFrame: A DataFrame with outliers removed for the specified column.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Optional: Log or print the bounds for debugging
    print(f"Filtering '{column}': Q1={q1}, Q3={q3}, IQR={iqr}, lower_bound={lower_bound}, upper_bound={upper_bound}")
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df


# ---------------------------
# Main Function for Plot Creation
# ---------------------------
def create_energy_plots(input_path: str, git_repo_path: str | None = None) -> None:
    """
    Loads the CSV data, filters out outliers, generates plots for each energy metric,
    and exports a textual summary of energy change events.

    Parameters:
      - input_path: Path to the CSV file containing energy data.
      - git_repo_path (optional): Path to the local Git repository. If provided, commit messages and links are retrieved.
    """
    if not os.path.isfile(input_path):
        logging.info(f"Error: File not found: {input_path}")
        sys.exit(1)

    folder = os.path.dirname(input_path)
    project_name = os.path.basename(folder)
    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # In this version, the CSV is expected to have only commit hash and energy values.
    df = pd.read_csv(input_path, header=None, names=["commit", "energy-pkg", "energy-core", "energy-gpu"])

    # Process only non-empty columns
    for column in ["energy-pkg", "energy-core", "energy-gpu"]:
        if df[column].dropna().empty:
            logging.info(f"Skipping processing for column '{column}' because it is empty.")
            continue

        # Filter outliers for the current energy column
        df_filtered = filter_outliers_iqr(df, column, multiplier=1.5)

        # Check if there's sufficient data after filtering
        if df_filtered.empty:
            logging.info(f"All data filtered out for column '{column}'. Skipping plot generation.")
            continue

        output_filename = os.path.join(folder, f"{project_name}_{column}_{timestamp_now}.png")
        create_energy_plot(df_filtered, column, output_filename)

        # Prepare summary with change events and use git information if available.
        valid_commits, short_hashes, _, _, _ = prepare_commit_statistics(df_filtered, column)
        distribution_data, _ = compute_distribution_and_normality(df_filtered, valid_commits, column)
        # Compute medians from distributions (needed for change detection)
        change_events = detect_energy_changes(distribution_data, [np.median(d) for d in distribution_data])
        export_change_events_summary(
            valid_commits, short_hashes, change_events, df_filtered, column, folder, project_name, timestamp_now, git_repo_path
        )
