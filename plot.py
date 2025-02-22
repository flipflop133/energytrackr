import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def create_energy_plot(df, energy_column, output_filename, sigma=2):
    """
    Create a plot of median energy consumption with outlier detection and smoothing.
    """
    # Compute median and std for the specified energy_column
    df_median = df.groupby("commit")[energy_column].median().reset_index()
    df_std = df.groupby("commit")[energy_column].std().reset_index()

    # Merge median and standard deviation data
    df_median = df_median.merge(df_std, on="commit", suffixes=("", "_std"))

    # Shorten commit hashes
    df_median["commit_short"] = df_median["commit"].str[:7]

    # Extract X, Y, and standard deviation
    x = df_median["commit_short"]
    y = df_median[energy_column]
    y_std = df_median[f"{energy_column}_std"]

    # Compute upper and lower bounds
    y_upper = y + y_std
    y_lower = y - y_std

    # Apply Gaussian smoothing
    y_upper_smooth = gaussian_filter1d(y_upper, sigma=sigma)
    y_lower_smooth = gaussian_filter1d(y_lower, sigma=sigma)

    # Convert x-axis labels to numerical indices
    x_indices = range(len(x))

    # Identify outliers
    outliers = (y > y_upper_smooth) | (y < y_lower_smooth)

    # Create the figure
    plt.figure(figsize=(20, 10))

    # Plot main line
    plt.plot(
        x_indices,
        y,
        marker="o",
        linestyle="-",
        color="b",
        label=f"Median {energy_column}",
    )

    # Overlay outlier points in red
    plt.scatter(
        np.array(x_indices)[outliers],
        np.array(y)[outliers],
        color="r",
        label="Outliers",
        zorder=3,
    )
    # Label outlier commits
    for i in np.where(outliers)[0]:
        plt.text(
            x_indices[i],
            y[i],
            x[i],
            fontsize=8,
            ha="right",
            color="red",
        )

    # Plot smoothed upper/lower bounds
    plt.plot(
        x_indices, y_upper_smooth, linestyle="--", color="r", label="Upper Bound (+1σ)"
    )
    plt.plot(
        x_indices, y_lower_smooth, linestyle="--", color="r", label="Lower Bound (-1σ)"
    )

    # Plot standard deviation lines based on the entire dataset
    median_val = np.median(y)
    sd_val = np.std(y)
    plt.axhline(median_val + sd_val, color="r", linestyle="--")
    plt.axhline(median_val - sd_val, color="r", linestyle="--")

    # Fill the area between the smoothed bounds
    plt.fill_between(x_indices, y_lower_smooth, y_upper_smooth, color="r", alpha=0.2)

    # Adjust X-axis to show commit hashes
    plt.xticks(ticks=x_indices, labels=x, rotation=45, ha="right")

    # Labeling
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
        "projects/xz/energy_usage.csv",
        header=None,
        names=["commit", "energy-pkg", "energy-core", "energy-gpu"],
    )

    # Generate three separate plots
    create_energy_plot(df, "energy-pkg", "plot_pkg.png", sigma=2)
    create_energy_plot(df, "energy-core", "plot_core.png", sigma=2)
    create_energy_plot(df, "energy-gpu", "plot_gpu.png", sigma=2)
