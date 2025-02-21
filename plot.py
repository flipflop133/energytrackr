import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Load CSV file (Assumes no headers)
df = pd.read_csv(
    "projects/xz/energy_usage.csv", header=None, names=["commit", "energy", "other"]
)

# Compute median energy per commit
df_median = df.groupby("commit")["energy"].median().reset_index()

# Compute the standard deviation per commit
df_std = df.groupby("commit")["energy"].std().reset_index()

# Merge median and standard deviation data
df_median = df_median.merge(df_std, on="commit", suffixes=("", "_std"))

# Shorten commit hashes to first 7 characters
df_median["commit_short"] = df_median["commit"].str[:7]

# Extract commit hashes (X) and median energy values (Y)
x = df_median["commit_short"]
y = df_median["energy"]
y_std = df_median["energy_std"]

# Compute upper and lower bounds
y_upper = y + y_std
y_lower = y - y_std

# Apply Gaussian smoothing
sigma = 2  # Adjust for smoother/less smooth curves
y_upper_smooth = gaussian_filter1d(y_upper, sigma=sigma)
y_lower_smooth = gaussian_filter1d(y_lower, sigma=sigma)

# Convert x-axis labels to numerical indices
x_indices = range(len(x))

# Identify outliers
outliers = (y > y_upper_smooth) | (y < y_lower_smooth)

# Create the figure
plt.figure(figsize=(20, 10))

# 1. Plot full line with normal color
plt.plot(
    x_indices,
    y,
    marker="o",
    linestyle="-",
    color="b",
    label="Median Energy Consumption",
)

# 2. Overlay outlier points in red
plt.scatter(
    np.array(x_indices)[outliers],  # X values for outliers
    np.array(y)[outliers],  # Y values for outliers
    color="r",
    label="Outliers",
    zorder=3,  # Ensure outliers are on top
)
# Display outliers commits
for i in np.where(outliers)[0]:
    plt.text(
        x_indices[i],
        y[i],
        df_median["commit_short"][i],
        fontsize=8,
        ha="right",
        color="red",
    )


# plt.errorbar(x_indices, y, yerr=y_std)
# Plot smoothed upper and lower bounds
plt.plot(
    x_indices, y_upper_smooth, linestyle="--", color="r", label="Upper Bound (+1σ)"
)
plt.plot(
    x_indices, y_lower_smooth, linestyle="--", color="r", label="Lower Bound (-1σ)"
)

# Plot standard deviation
median = np.median(y)
sd = np.std(y)
plt.axhline(median + sd, color="r", linestyle="--")
plt.axhline(median - sd, color="r", linestyle="--")


# Fill the area between the smoothed bounds
plt.fill_between(x_indices, y_lower_smooth, y_upper_smooth, color="r", alpha=0.2)

# Adjust X-axis to show commit hashes
plt.xticks(ticks=x_indices, labels=x, rotation=45, ha="right")

# Label the axes
plt.xlabel("Commit Hash")
plt.ylabel("Median Energy (uJ)")
plt.title("Energy Consumption Trend (Median per Commit)")
plt.legend()
plt.grid(True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save and display the plot
plt.savefig("plot.png")
plt.show()
