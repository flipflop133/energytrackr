import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_energy_plot(df, energy_column, output_filename):
    """
    Create a plot of median energy consumption with outlier detection using error bars,
    and display the distribution (via violin plots) for each commit's energy measurements.

    Outliers are defined as points outside the overall median ± one standard deviation.
    """
    # Compute median and standard deviation for the specified energy_column per commit
    df_median = df.groupby("commit", sort=False)[energy_column].median().reset_index()
    df_std = df.groupby("commit", sort=False)[energy_column].std().reset_index()

    # Merge median and standard deviation data
    df_median = df_median.merge(df_std, on="commit", suffixes=("", "_std"))

    # Shorten commit hashes for display
    df_median["commit_short"] = df_median["commit"].str[:7]

    # Extract X (commit hash), Y (median energy) and Y error (std dev)
    x = df_median["commit_short"]
    y = df_median[energy_column]
    y_std = df_median[f"{energy_column}_std"]

    # Convert x-axis labels to numerical indices
    x_indices = np.arange(len(x))

    # Extract the full distribution data for each commit
    distribution_data = [group[energy_column].values for _, group in df.groupby("commit", sort=False)]

    # Create the figure
    plt.figure(figsize=(20, 10))

    # Draw the violin plot for each commit's distribution (drawn first so it stays in the background)
    violin_parts = plt.violinplot(
        distribution_data,
        positions=x_indices,
        widths=0.5,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for pc in violin_parts["bodies"]:
        pc.set_facecolor("lightgrey")
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)
        pc.set_zorder(1)

    # Compute overall median and standard deviation for outlier detection
    overall_median = np.median(y)
    overall_std = np.std(y)
    outliers = (y > overall_median + overall_std) | (y < overall_median - overall_std)

    # Plot main line with error bars
    plt.errorbar(
        x_indices,
        y,
        yerr=y_std,
        marker="o",
        linestyle="-",
        color="b",
        label=f"Median {energy_column}",
        zorder=2,
    )

    # Overlay outlier points in red
    plt.scatter(
        x_indices[outliers],
        y[outliers],
        color="r",
        label="Outliers",
        zorder=3,
    )

    # Label outlier commits
    for i in np.where(outliers)[0]:
        plt.text(
            x_indices[i],
            y.iloc[i],
            x.iloc[i],
            fontsize=8,
            ha="right",
            color="red",
        )

    # Plot horizontal lines for overall median ± standard deviation
    plt.axhline(
        overall_median + overall_std,
        color="r",
        linestyle="--",
        label="Overall Median + Std",
    )
    plt.axhline(
        overall_median - overall_std,
        color="r",
        linestyle="--",
        label="Overall Median - Std",
    )

    # Adjust x-axis to show commit hashes
    plt.xticks(ticks=x_indices, labels=x, rotation=45, ha="right")

    # Add labels, title, legend, and grid
    plt.xlabel("Commit Hash")
    plt.ylabel(f"Median Energy ({energy_column})")
    plt.title(f"Energy Consumption Trend (Median per Commit) - {energy_column}")
    plt.legend()
    plt.grid(True)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save and display the plot
    plt.savefig(output_filename)
    plt.show()


# ----------------------------
# MAIN SCRIPT
# ----------------------------
if __name__ == "__main__":
    # Load CSV file (Assumes no headers)
    df = pd.read_csv(
        "projects/portable/energy_usage.csv",
        header=None,
        names=["commit", "energy-pkg", "energy-core", "energy-gpu"],
    )

    # Generate three separate plots
    create_energy_plot(df, "energy-pkg", "plot_pkg.png")
    create_energy_plot(df, "energy-core", "plot_core.png")
    create_energy_plot(df, "energy-gpu", "plot_gpu.png")
