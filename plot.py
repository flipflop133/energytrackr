"""Create a plot of median energy consumption from CSV file."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import shapiro

# Parameters
MIN_MEASUREMENTS = 2
NORMALITY_P_THRESHOLD = 0.05
PELT_PENALTY = 10  # Adjust this penalty value to tune sensitivity


def create_energy_plot(df: pd.DataFrame, energy_column: str, output_filename: str) -> None:
    # Preprocessing: group by commit and calculate median and std
    """Create a plot of median energy consumption for each commit from a DataFrame.

    This function processes a DataFrame to calculate the median and standard deviation
    of energy consumption values grouped by commit. It performs change point detection
    to identify regressions in energy consumption trends and visualizes the results in a plot.

    Args:
        df (pd.DataFrame): The DataFrame containing energy data with commit hashes.
        energy_column (str): The column name in `df` representing energy values.
        output_filename (str): The filename to save the generated plot.

    Returns:
        None: This function does not return a value; it saves the plot as an image file.

    The plot includes:
    - Violin plots showing the distribution of energy values per commit.
    - Error bars representing median ± standard deviation.
    - Change point detection lines indicating regression start points.
    - Markers for breaking commits where significant energy regressions occur.
    - The x-axis represents commit hashes, and the y-axis represents median energy values.
    - The plot is saved to the specified `output_filename`.
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

    # Compute median and error values directly from the aggregated dataframe
    y_medians = df_median[energy_column].tolist()
    y_errors = df_median[f"{energy_column}_std"].tolist()

    # Compute distribution and normality flags per commit for violin plot coloring
    distribution_data = []
    normality_flags = []
    for commit in valid_commits:
        values = df[df["commit"] == commit][energy_column].values
        distribution_data.append(values)
        if len(values) >= 3:
            _, p_shapiro = shapiro(values)
            normality_flags.append(p_shapiro >= NORMALITY_P_THRESHOLD)
        else:
            normality_flags.append(True)

    # Use PELT change point detection on the median energy time series
    energy_series = np.array(y_medians)
    algo = rpt.Pelt(model="l2").fit(energy_series)
    # The change_points list gives indices marking the end of each segment.
    # The last index is always the end of the series so we ignore it for regression start.
    change_points = algo.predict(pen=PELT_PENALTY)

    # Plotting
    plt.figure(figsize=(40, 10))

    # Violin plots for the distribution per commit
    violin_parts = plt.violinplot(
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
    plt.errorbar(
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
        # Draw a vertical dashed line to indicate the boundary between segments.
        if not regression_label_added:
            plt.axvline(x=cp - 0.5, color="green", linestyle="--", label="Regression Start")
            regression_label_added = True
        else:
            plt.axvline(x=cp - 0.5, color="green", linestyle="--")

        # Highlight the breaking commit (first commit of the new segment)
        # Ensure cp is within range (it should be, as cp is the start index of the new segment)
        if cp < len(valid_commits):
            if not break_commit_label_added:
                plt.scatter(x_indices[cp], y_medians[cp], marker="*", color="purple", s=150, zorder=6, label="Breaking Commit")
                break_commit_label_added = True
            else:
                plt.scatter(
                    x_indices[cp],
                    y_medians[cp],
                    marker="*",
                    color="purple",
                    s=150,
                    zorder=6,
                )
            # Annotate with the short commit hash
            plt.text(x_indices[cp], y_medians[cp], short_hashes[cp], fontsize=8, ha="left", color="purple")

    # Axis labels, ticks, and title
    plt.xticks(ticks=x_indices, labels=short_hashes, rotation=45, ha="right")
    plt.xlabel("Commit Hash (sorted by date, oldest to newest)")
    plt.ylabel(f"Median Energy ({energy_column})")
    plt.title(f"Energy Consumption Trend (Median per Commit) - {energy_column}")
    plt.grid(True)
    plt.tight_layout()

    # Final legend (explicit and complete)
    custom_handles = [
        Line2D([0], [0], marker="o", color="blue", label=f"Median {energy_column}", linestyle="-"),
        Line2D([0], [0], linestyle="--", color="green", label="Regression Start"),
        Line2D([0], [0], marker="*", color="purple", linestyle="None", markersize=10, label="Breaking Commit"),
        Patch(facecolor="lightgrey", edgecolor="black", label="Normal Distribution"),
        Patch(facecolor="lightcoral", edgecolor="black", label="Non-Normal (Shapiro-Wilk p < 0.05)"),
    ]
    plt.legend(handles=custom_handles)

    # Save the plot
    plt.savefig(output_filename)


if __name__ == "__main__":
    df = pd.read_csv(
        "sorted_energy_data_new_3.csv",
        header=None,
        names=["commit", "energy-pkg", "energy-core", "energy-gpu"],
    )

    create_energy_plot(df, "energy-pkg", "plot_pkg.png")
    create_energy_plot(df, "energy-core", "plot_core.png")
    create_energy_plot(df, "energy-gpu", "plot_gpu.png")
