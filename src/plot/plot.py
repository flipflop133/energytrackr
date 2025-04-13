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


# ---------------------------
# Dataclasses
# ---------------------------
@dataclass
class ChangeEvent:
    """
    Data structure to hold energy change events.
    """

    index: int  # The index of the commit (in the list of valid commits)
    severity: float  # Severity of the change (e.g., 0.25 for +25%)
    direction: str  # "increase" or "decrease"
    cohen_d: float  # Effect size


@dataclass
class EnergyPlotData:
    """
    Data structure to hold energy plot data for a single energy column.
    """

    x_indices: np.ndarray[Any, np.dtype[np.int64]]
    short_hashes: list[str]
    y_medians: list[float]
    y_errors: list[float]
    distribution_data: list[np.ndarray[Any, Any]]
    normality_flags: list[bool]
    change_events: list[ChangeEvent]
    energy_column: str


# ---------------------------
# Helper Functions for Git Information
# ---------------------------
def generate_commit_link(remote_url: str, commit_hash: str) -> str:
    """
    Generate a commit link from the remote URL and commit hash.
    Supports GitHub-type URLs. Returns "N/A" if parsing fails.
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
    Returns a dictionary with commit_summary, commit_link, commit_date, and files_modified.
    """
    try:
        commit_obj = repo.commit(commit_hash)
        commit_date = commit_obj.committed_datetime.strftime("%Y-%m-%d")
        commit_summary = commit_obj.summary
        commit_files = list(commit_obj.stats.files.keys())

        commit_link = "N/A"
        if repo.remotes:
            remote_url = repo.remotes[0].url
            commit_link = generate_commit_link(remote_url, commit_hash)

        return {
            "commit_summary": commit_summary,
            "commit_link": commit_link,
            "commit_date": commit_date,
            "files_modified": commit_files,
        }
    except Exception as e:
        logging.error(f"Error retrieving details for commit {commit_hash}: {e}")
        return {"commit_summary": "N/A", "commit_link": "N/A", "commit_date": "N/A", "files_modified": []}


# ---------------------------
# Distribution and Normality
# ---------------------------
def compute_distribution_and_normality(
    df: pd.DataFrame,
    valid_commits: list[str],
    energy_column: str,
) -> tuple[list[np.ndarray[Any, Any]], list[bool]]:
    """
    Computes distribution data (list of arrays) and normality flags
    for each commit in `valid_commits`, for the specified energy column.
    """
    distribution_data = []
    normality_flags = []
    for commit in valid_commits:
        values = df[df["commit"] == commit][energy_column].values
        distribution_data.append(values)
        if len(values) >= MIN_VALUES_FOR_NORMALITY_TEST:
            _, p_shapiro = shapiro(values)
            normality_flags.append(p_shapiro >= NORMALITY_P_THRESHOLD)
        else:
            # If not enough data to test, treat as "normal" by default
            normality_flags.append(True)
    return distribution_data, normality_flags


# ---------------------------
# Change Detection
# ---------------------------
def get_change_direction(baseline_median: float, test_median: float, min_pct_change: float) -> str | None:
    """
    Determine if test_median is an 'increase' or 'decrease' from baseline_median
    given `min_pct_change` threshold. Returns None if there's no significant direction.
    """
    if test_median >= baseline_median * (1 + min_pct_change):
        return "increase"
    if test_median <= baseline_median * (1 - min_pct_change):
        return "decrease"
    return None


def detect_energy_changes(
    distribution_data: list[np.ndarray[Any, Any]],
    y_medians: list[float],
    min_pct_change: float = MIN_PCT_INCREASE,
    p_threshold: float = WELCH_P_THRESHOLD,
) -> list[ChangeEvent]:
    """
    Detect significant changes (Welch's t-test + practical threshold) between each commit and its predecessor.
    """
    changes = []
    for i in range(1, len(distribution_data)):
        baseline = distribution_data[i - 1]
        test = distribution_data[i]

        if len(baseline) < MIN_VALUES_FOR_NORMALITY_TEST or len(test) < MIN_VALUES_FOR_NORMALITY_TEST:
            # Not enough data on either side
            continue

        baseline_median = np.median(baseline)
        test_median = np.median(test)

        # Statistical test
        _, p_value = ttest_ind(baseline, test, equal_var=False)
        if p_value < p_threshold:
            # Practical threshold test
            direction = get_change_direction(baseline_median, test_median, min_pct_change)
            if direction:
                # Compute Cohen's d
                mean_baseline = np.mean(baseline)
                mean_test = np.mean(test)
                var_baseline = np.var(baseline, ddof=1)
                var_test = np.var(test, ddof=1)
                pooled_std = np.sqrt((var_baseline + var_test) / 2.0)
                cohen_d = (mean_test - mean_baseline) / pooled_std if pooled_std != 0 else 0.0

                # Severity is the percentage
                if direction == "increase":
                    severity = (test_median - baseline_median) / baseline_median
                else:
                    severity = (baseline_median - test_median) / baseline_median

                changes.append(ChangeEvent(index=i, severity=severity, direction=direction, cohen_d=cohen_d))
    return changes


# ---------------------------
# Plotting
# ---------------------------
def plot_energy_data(ax: Axes, plot_data: EnergyPlotData) -> None:
    """
    Plot distributions (violin), median + std dev lines, mark areas for changes,
    and annotate changes with severity and Cohen's d.
    """
    x_indices = plot_data.x_indices
    short_hashes = plot_data.short_hashes
    y_medians = plot_data.y_medians
    y_errors = plot_data.y_errors
    distribution_data = plot_data.distribution_data
    normality_flags = plot_data.normality_flags
    change_events = plot_data.change_events
    energy_column = plot_data.energy_column

    # Violin plots
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

    # Median + std error bars
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

    # Annotate significant changes
    placed_bboxes = []
    max_y = max(y_medians) if y_medians else 0.0
    for event in change_events:
        cp = event.index
        # Limit severity to 50% for shading intensity
        capped_severity = min(event.severity, 0.5)
        opacity = (capped_severity / 0.5) * (0.8 - 0.2) + 0.2
        if event.direction == "increase":
            color = to_rgba((1.0, 0.0, 0.0, opacity))
            text_color = "darkred"
            sign = "+"
        else:
            color = to_rgba((0.0, 0.8, 0.0, opacity))
            text_color = "darkgreen"
            sign = "-"

        ax.axvspan(cp - 0.5, cp + 0.5, color=color, zorder=0)

        base_y = max_y * 1.05
        annotation_str = f"{sign}{int(event.severity * 100)}% (d={event.cohen_d:.2f})"

        txt = ax.text(
            cp,
            base_y,
            annotation_str,
            fontsize=8,
            ha="center",
            color=text_color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            zorder=7,
        )

        # Try to resolve label collisions
        ax.figure.canvas.draw()
        text_bbox = txt.get_window_extent(renderer=ax.figure.canvas.get_renderer())
        shift_increment = 0.05 * max_y

        while any(text_bbox.overlaps(b) for b in placed_bboxes):
            cx, cy = txt.get_position()
            txt.set_position((cx, cy + shift_increment))
            ax.figure.canvas.draw()
            text_bbox = txt.get_window_extent(renderer=ax.figure.canvas.get_renderer())

        placed_bboxes.append(text_bbox)

    # X axis
    ax.set_xticks(x_indices)
    ax.set_xticklabels(short_hashes, rotation=45, ha="right")
    ax.set_xlabel("Commit (oldest → newest)")
    ax.set_ylabel(f"Median Energy (J)")
    ax.set_title(f"Energy Consumption Trend - {energy_column}")
    ax.grid(True)

    # Legend
    custom_handles = [
        Line2D([0], [0], marker="o", color="blue", label=f"Median {energy_column}", linestyle="-"),
        Patch(facecolor="lightgrey", edgecolor="black", label="Normal Dist."),
        Patch(facecolor="lightcoral", edgecolor="black", label="Non-Normal Dist. (Shapiro-Wilk p < 0.05)"),
        Patch(facecolor=to_rgba((1.0, 0.0, 0.0, 0.5)), edgecolor="none", label="Regression (↑ energy)"),
        Patch(facecolor=to_rgba((0.0, 0.8, 0.0, 0.5)), edgecolor="none", label="Improvement (↓ energy)"),
        Line2D([0], [0], marker="", color="black", label="Annotation: ±X% (Cohen's d)"),
    ]
    ax.legend(handles=custom_handles)


# ---------------------------
# Commit Statistics
# ---------------------------
def prepare_commit_statistics(
    df: pd.DataFrame,
    energy_column: str,
) -> tuple[pd.DataFrame, list[str], list[str], np.ndarray[Any, np.dtype[np.int64]], list[float], list[float]]:
    """
    Computes median, std, count per commit. Returns:
      - df_median: DataFrame with columns [commit, {energy_column}, {energy_column}_std, count, commit_short]
      - valid_commits
      - short_hashes
      - x_indices
      - y_medians
      - y_errors
    """
    commit_counts = df.groupby("commit").size().reset_index(name="count")
    df_median = df.groupby("commit", sort=False)[energy_column].median().reset_index()
    df_std = df.groupby("commit", sort=False)[energy_column].std().reset_index()
    df_median = df_median.merge(df_std, on="commit", suffixes=("", "_std"))
    df_median = df_median.merge(commit_counts, on="commit")
    df_median = df_median[df_median["count"] >= MIN_MEASUREMENTS].copy()
    df_median["commit_short"] = df_median["commit"].str[:7]

    valid_commits = df_median["commit"].tolist()
    short_hashes = df_median["commit_short"].tolist()
    x_indices = np.arange(len(valid_commits))
    y_medians = df_median[energy_column].tolist()
    y_errors = df_median[f"{energy_column}_std"].tolist()

    return df_median, valid_commits, short_hashes, x_indices, y_medians, y_errors


# ---------------------------
# Outlier Filtering
# ---------------------------
def filter_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Removes outliers using IQR * multiplier from the specified column.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# ---------------------------
# Statistical Summary Computation
# ---------------------------
def compute_statistical_summary(
    df: pd.DataFrame,
    distribution_data: list[np.ndarray[Any, Any]],
    normality_flags: list[bool],
    change_events: list[ChangeEvent],
    outliers_removed_count: int,
    energy_column: str,
) -> dict:
    """
    Compute overall statistical metrics to display in the summary box:
      - total_commits, significant_changes, regressions_count, improvements_count, ...
      - mean, median, std, max inc/dec, average cohen's d, normal/non-normal counts, etc.
    """
    stats_summary = {}
    stats_summary["total_commits"] = len(distribution_data)  # # of valid commits
    stats_summary["significant_changes"] = len(change_events)
    stats_summary["regressions"] = sum(e.direction == "increase" for e in change_events)
    stats_summary["improvements"] = sum(e.direction == "decrease" for e in change_events)

    # Basic stats from the (already filtered) entire dataset
    stats_summary["mean_energy"] = df[energy_column].mean()
    stats_summary["median_energy"] = df[energy_column].median()
    stats_summary["std_energy"] = df[energy_column].std()

    # Max increase & decrease
    inc_events = [e for e in change_events if e.direction == "increase"]
    dec_events = [e for e in change_events if e.direction == "decrease"]
    stats_summary["max_increase"] = max(e.severity for e in inc_events) if inc_events else 0.0
    stats_summary["max_decrease"] = max(e.severity for e in dec_events) if dec_events else 0.0

    # Average Cohen's d
    if change_events:
        stats_summary["avg_cohens_d"] = np.mean([abs(e.cohen_d) for e in change_events])
    else:
        stats_summary["avg_cohens_d"] = 0.0

    # Normal vs non-normal
    stats_summary["normal_count"] = sum(normality_flags)
    stats_summary["non_normal_count"] = len(normality_flags) - stats_summary["normal_count"]

    stats_summary["outliers_removed"] = outliers_removed_count

    return stats_summary


# ---------------------------
# Export Summaries
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
    Text-file summary of the change events. (Mostly the same as before,
    but you can keep or trim as needed for a textual log.)
    """
    commit_details = {}
    # If CSV has these columns, try to use them
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

    # If git repo path is given, fill missing details
    if git_repo_path:
        try:
            from git import Repo

            repo = Repo(git_repo_path)
            for commit in valid_commits:
                if commit_details[commit]["commit_summary"] == "N/A" or commit_details[commit]["commit_link"] == "N/A":
                    details = get_commit_details_from_git(commit, repo)
                    commit_details[commit].update(details)
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
            details = commit_details.get(commit_hash, {})
            direction_str = "Regression (Increase)" if event.direction == "increase" else "Improvement (Decrease)"
            summary_lines.append(f"Commit: {commit_hash} (Short: {short_hash})")
            summary_lines.append(f"Direction: {direction_str}")
            summary_lines.append(f"Severity: {int(event.severity * 100)}%")
            summary_lines.append(f"Commit Message: {details.get('commit_summary', 'N/A')}")
            summary_lines.append(f"Commit Link: {details.get('commit_link', 'N/A')}")
            summary_lines.append(f"Cohen's d: {event.cohen_d:.2f}")
            summary_lines.append(
                f"Median {energy_column} (J): {np.median(df[df['commit'] == commit_hash][energy_column].values):.2f}"
            )
            summary_lines.append(f"Files modified: {details.get('files_modified', 'N/A')}")
            summary_lines.append("-" * 80)

    summary_filename = os.path.join(folder, f"{project_name}_{energy_column}_{timestamp_now}_summary.txt")
    with open(summary_filename, "w") as f:
        f.write("\n".join(summary_lines))
    logging.info(f"Exported energy change summary to {summary_filename}")


def export_change_events_html_summary(
    df_median: pd.DataFrame,
    valid_commits: list[str],
    short_hashes: list[str],
    distribution_data: list[np.ndarray[Any, Any]],
    normality_flags: list[bool],
    change_events: list[ChangeEvent],
    df: pd.DataFrame,
    energy_column: str,
    folder: str,
    project_name: str,
    timestamp_now: str,
    plot_img_path: str,
    stats_summary: dict,
    git_repo_path: str | None = None,
) -> None:
    """
    Exports an HTML report with:
      - Plot image
      - Statistical summary
      - Table of ALL commits (not just changed ones) with columns:
         Commit, Change, Severity, Median (J), Std Dev (J), Normality, n, Cohen's d, Files, Message
    """
    # We need details from git or from the df columns if provided
    commit_details = {}
    if "commit_summary" in df.columns and "commit_link" in df.columns:
        for commit in valid_commits:
            row = df[df["commit"] == commit].iloc[0]
            commit_details[commit] = {
                "commit_summary": row.get("commit_summary", "N/A"),
                "commit_link": row.get("commit_link", "N/A"),
                "commit_date": row.get("commit_date", "N/A"),
                "files_modified": row.get("files_modified", []),
            }
    else:
        for commit in valid_commits:
            commit_details[commit] = {
                "commit_summary": "N/A",
                "commit_link": "N/A",
                "commit_date": "N/A",
                "files_modified": [],
            }

    repo = None
    if git_repo_path:
        try:
            from git import Repo

            repo = Repo(git_repo_path)
        except Exception as e:
            logging.error(f"Error loading Git repository from {git_repo_path}: {e}")
            repo = None

    # Fill missing details from git
    if repo:
        for commit in valid_commits:
            # Only if missing
            if commit_details[commit]["commit_summary"] == "N/A":
                details = get_commit_details_from_git(commit, repo)
                commit_details[commit].update(details)

    # Build a quick lookup for changes: commit_index -> ChangeEvent
    change_lookup = {ch.index: ch for ch in change_events}

    # Create table rows for every valid commit
    table_rows = ""
    for i, commit in enumerate(valid_commits):
        short_hash = short_hashes[i]
        commit_info = commit_details[commit]
        # from df_median row, get median, std, count
        row_stats = df_median[df_median["commit"] == commit].iloc[0]
        median_val = row_stats[energy_column]
        std_val = row_stats[f"{energy_column}_std"]
        n_val = int(row_stats["count"])
        is_normal = normality_flags[i]

        # See if there's a significant change event
        if i in change_lookup:
            evt = change_lookup[i]
            direction_str = "increase" if evt.direction == "increase" else "decrease"
            if direction_str == "increase":
                row_class = "increase"
                change_str = "Regression (Increase)"
            else:
                row_class = "decrease"
                change_str = "Improvement (Decrease)"
            severity_str = f"{int(evt.severity * 100)}%"
            cohen_str = f"{evt.cohen_d:.2f}"
        else:
            row_class = ""
            change_str = "None"
            severity_str = "0%"
            cohen_str = "0.00"

        # Normality string
        normality_str = "Normal" if is_normal else "Non-normal"
        # Build file list
        files_mod = commit_info.get("files_modified", [])
        if files_mod:
            files_modified_html = "<ul>" + "".join(f"<li>{file}</li>" for file in files_mod) + "</ul>"
        else:
            files_modified_html = "N/A"

        commit_link = commit_info.get("commit_link", "#")
        commit_message = commit_info.get("commit_summary", "N/A")

        table_rows += f"""
        <tr class="{row_class}">
            <td><a href="{commit_link}" target="_blank">{short_hash}</a></td>
            <td>{change_str}</td>
            <td>{severity_str}</td>
            <td>{median_val:.2f}</td>
            <td>{std_val:.2f}</td>
            <td>{normality_str}</td>
            <td>{n_val}</td>
            <td>{cohen_str}</td>
            <td>{files_modified_html}</td>
            <td>{commit_message}</td>
        </tr>
        """

    # Attempt to get the oldest and newest commit dates
    oldest_commit_date = "N/A"
    newest_commit_date = "N/A"
    if len(valid_commits) >= 2 and repo is not None:
        oldest_commit_date = get_commit_details_from_git(valid_commits[0], repo)["commit_date"]
        newest_commit_date = get_commit_details_from_git(valid_commits[-1], repo)["commit_date"]

    # Insert real stats from stats_summary
    total_commits = stats_summary["total_commits"]
    significant_changes = stats_summary["significant_changes"]
    regressions = stats_summary["regressions"]
    improvements = stats_summary["improvements"]
    mean_energy = stats_summary["mean_energy"]
    median_energy = stats_summary["median_energy"]
    std_energy = stats_summary["std_energy"]
    max_increase = stats_summary["max_increase"]  # fraction
    max_decrease = stats_summary["max_decrease"]  # fraction
    avg_cohens_d = stats_summary["avg_cohens_d"]
    normal_count = stats_summary["normal_count"]
    non_normal_count = stats_summary["non_normal_count"]
    outliers_removed = stats_summary["outliers_removed"]

    # Convert fraction → percentage
    max_inc_pct = f"{max_increase * 100:.1f}%" if max_increase != 0 else "0.0%"
    max_dec_pct = f"{max_decrease * 100:.1f}%" if max_decrease != 0 else "0.0%"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Energy Consumption Change Summary - {energy_column}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        header img {{
            max-width: 100%;
            height: auto;
        }}
        h1 {{
            color: #333;
        }}
        h2, h3 {{
            color: #555;
        }}
        .summary-box {{
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #fff;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #ddd;
        }}
        tr.increase {{
            background-color: rgba(255, 0, 0, 0.1);
        }}
        tr.decrease {{
            background-color: rgba(0, 255, 0, 0.1);
        }}
        .note {{
            font-size: 0.9em;
            color: #777;
            text-align: center;
            margin-top: 20px;
        }}
        td ul {{
            margin: 0;
            padding-left: 20px;
            text-align: left;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Energy Consumption Report for {project_name}</h1>
        <img src="{os.path.basename(plot_img_path)}" alt="Energy Consumption Plot">
        <p>Note: Right click and open image in new tab for full resolution and to be able to zoom more easily.</p>
    </header>

    <div class="summary-box">
      <h2>General Summary</h2>
      <ul>
        <li><strong>Project:</strong> {project_name}</li>
        <li><strong>Energy Metric:</strong> {energy_column}</li>
        <li><strong>Commit Range:</strong>{short_hashes[0]} ({oldest_commit_date}) → {short_hashes[-1]} ({newest_commit_date})</li>
        <li><strong>Number of commits (after filtering):</strong> {total_commits}</li>
      </ul>
      <h2>Statistical Summary</h2>
      <ul>
        <li><strong>Total commits analyzed:</strong> {total_commits}</li>
        <li><strong>Significant changes detected:</strong> {significant_changes}
          <ul>
            <li>Regressions (↑): {regressions}</li>
            <li>Improvements (↓): {improvements}</li>
          </ul>
        </li>
        <li><strong>Mean energy:</strong> {mean_energy:.2f} J</li>
        <li><strong>Median energy:</strong> {median_energy:.2f} J</li>
        <li><strong>Std. deviation:</strong> {std_energy:.2f} J</li>
        <li><strong>Max increase severity:</strong> {max_inc_pct}</li>
        <li><strong>Max decrease severity:</strong> {max_dec_pct}</li>
        <li><strong>Average Cohen’s d:</strong> {avg_cohens_d:.2f}</li>
        <li><strong>Normal distributions:</strong> {normal_count}</li>
        <li><strong>Non-normal distributions:</strong> {non_normal_count}</li>
        <li><strong>Outliers removed:</strong> {outliers_removed}</li>
      </ul>
    </div>

    <main>
        <table>
            <thead>
                <tr>
                    <th>Commit</th>
                    <th>Change</th>
                    <th>Severity</th>
                    <th>Median (J)</th>
                    <th>Std Dev (J)</th>
                    <th>Normality</th>
                    <th>n</th>
                    <th>Cohen's d</th>
                    <th>Files</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        <p class="note">
            Rows are highlighted red or green when a significant change (Welch's t-test + 2% threshold) is detected.
        </p>
    </main>
</body>
</html>
"""

    summary_filename = os.path.join(folder, f"{project_name}_{energy_column}_{timestamp_now}_summary.html")
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    logging.info(f"Exported energy change summary HTML to {summary_filename}")


# ---------------------------
# Single Plot Entry
# ---------------------------
def create_energy_plot(
    df: pd.DataFrame, energy_column: str, output_filename: str
) -> tuple[list[str], list[str], list[np.ndarray[Any, Any]], list[bool], list[ChangeEvent]]:
    """
    Create and save the plot for a single energy column.
    Returns (valid_commits, short_hashes, distribution_data, normality_flags, change_events)
    so they can be reused in the summaries.
    """
    df_median, valid_commits, short_hashes, x_indices, y_medians, y_errors = prepare_commit_statistics(df, energy_column)
    if not valid_commits:
        # Not enough data after outlier filtering or no commits meet MIN_MEASUREMENTS
        logging.info(f"No valid commits for column '{energy_column}'. Skipping plot.")
        return [], [], [], [], []

    distribution_data, normality_flags = compute_distribution_and_normality(df, valid_commits, energy_column)
    change_events = detect_energy_changes(distribution_data, y_medians)

    # Build plot
    plt.figure(figsize=(40, 10))
    ax = plt.gca()
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
    plot_energy_data(ax, plot_data)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

    return valid_commits, short_hashes, distribution_data, normality_flags, change_events


# ---------------------------
# Main Orchestrator
# ---------------------------
def create_energy_plots(input_path: str, git_repo_path: str | None = None) -> None:
    """
    Loads the CSV data, filters outliers, generates plots and text/html summaries for each energy column.
    """
    if not os.path.isfile(input_path):
        logging.error(f"Error: File not found: {input_path}")
        sys.exit(1)

    folder = os.path.dirname(input_path)
    project_name = os.path.basename(folder)
    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Suppose CSV has 4 columns: commit, energy-pkg, energy-core, energy-gpu
    df = pd.read_csv(input_path, header=None, names=["commit", "energy-pkg", "energy-core", "energy-gpu"])

    # For each energy column
    for column in ["energy-pkg", "energy-core", "energy-gpu"]:
        if df[column].dropna().empty:
            logging.info(f"Skipping column '{column}' (empty data).")
            continue

        # Filter outliers
        original_len = len(df)
        df_filtered = filter_outliers_iqr(df, column, multiplier=1.5)
        outliers_removed_count = original_len - len(df_filtered)

        if df_filtered.empty:
            logging.info(f"After outlier removal, no data remains for '{column}'. Skipping.")
            continue

        # Create and save plot
        output_filename = os.path.join(folder, f"{project_name}_{column}_{timestamp_now}.png")
        valid_commits, short_hashes, distribution_data, normality_flags, change_events = create_energy_plot(
            df_filtered, column, output_filename
        )

        if not valid_commits:
            continue  # no valid commits for this column

        # Export plain-text summary
        export_change_events_summary(
            valid_commits,
            short_hashes,
            change_events,
            df_filtered,
            column,
            folder,
            project_name,
            timestamp_now,
            git_repo_path,
        )

        # Compute stats for the HTML summary
        # We'll re-use the same median/stdev info from prepare_commit_statistics
        df_median, _, _, _, _, _ = prepare_commit_statistics(df_filtered, column)

        stats_summary = compute_statistical_summary(
            df_filtered,
            distribution_data,
            normality_flags,
            change_events,
            outliers_removed_count,
            column,
        )

        # Finally export the HTML summary
        export_change_events_html_summary(
            df_median=df_median,
            valid_commits=valid_commits,
            short_hashes=short_hashes,
            distribution_data=distribution_data,
            normality_flags=normality_flags,
            change_events=change_events,
            df=df_filtered,
            energy_column=column,
            folder=folder,
            project_name=project_name,
            timestamp_now=timestamp_now,
            plot_img_path=output_filename,
            stats_summary=stats_summary,
            git_repo_path=git_repo_path,
        )
