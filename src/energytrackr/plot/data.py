"""Data processing and statistical analysis for energy measurements."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind

# Configuration constants can either be hard-coded or
# passed to the constructor for better flexibility.
MIN_MEASUREMENTS = 2
NORMALITY_P_THRESHOLD = 0.05
MIN_VALUES_FOR_NORMALITY_TEST = 3
WELCH_P_THRESHOLD = 0.05
MIN_PCT_INCREASE = 0.02


def nice_number(x: float) -> float:
    """Rounds a given number to a "nice" number, which is a value that is easy to interpret.

    The function determines the order of magnitude of the input number and
    selects a "nice" fraction based on predefined thresholds.

    Args:
        x (float): The input number to be rounded.

    Returns:
        float: A "nice" number that is close to the input value.
    """
    if not x:
        return 0
    thresholds = [1.5, 3, 7]
    exponent = math.floor(math.log10(x))
    fraction = x / (10**exponent)
    if fraction < thresholds[0]:
        nice_fraction = 1
    elif fraction < thresholds[1]:
        nice_fraction = 2
    elif fraction < thresholds[2]:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


@dataclass
class ChangeEvent:
    """Represents a change event with associated metadata.

    Attributes:
        index (int): The index of the change event.
        severity (float): The severity level of the change event.
        direction (str): The direction of the change event (e.g., "up", "down").
        cohen_d (float): The Cohen's d effect size associated with the change event.
    """

    index: int
    severity: float
    direction: str
    cohen_d: float


@dataclass
class EnergyStats:
    """EnergyStats is a class that encapsulates statistical data related to energy metrics.

    Attributes:
        valid_commits (list[str]): A list of valid commit identifiers.
        short_hashes (list[str]): A list of short hash representations of commits.
        x_indices (np.ndarray): An array of x-axis indices for plotting.
        y_medians (list[float]): A list of median values for the y-axis.
        y_errors (list[float]): A list of error values corresponding to the y-axis medians.
        df_median (pd.DataFrame): A DataFrame containing median-related data for further analysis.
    """

    valid_commits: list[str]
    short_hashes: list[str]
    x_indices: np.ndarray
    y_medians: list[float]
    y_errors: list[float]
    df_median: pd.DataFrame


class EnergyData:
    """A class to process and analyze energy data from a CSV file."""

    def __init__(self, csv_path: str, energy_columns: list[str]) -> None:
        """Initialize the EnergyData class."""
        self.csv_path = csv_path
        self.energy_columns = energy_columns
        self.df = None
        self.stats = {}  # Dictionary keyed by energy column
        self.distributions = {}  # Raw distributions for each commit per column
        self.normality_flags = {}
        self.change_events = {}

    def load_data(self, column_names: list[str]) -> None:
        """Load data from a CSV file into a DataFrame.

        Args:
            column_names (list[str]): A list of column names for the DataFrame.
        """
        self.df = pd.read_csv(self.csv_path, header=None, names=column_names)

    @staticmethod
    def _filter_outliers(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """Filters outliers from a DataFrame column based on the Interquartile Range (IQR) method.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            column (str): The name of the column to filter for outliers.
            multiplier (float, optional): The multiplier for the IQR to define the outlier bounds.
                Defaults to 1.5.

        Returns:
            pd.DataFrame: A DataFrame with outliers removed from the specified column.
        """
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    @staticmethod
    def _compute_commit_statistics(df: pd.DataFrame, energy_column: str) -> EnergyStats:
        """Compute statistics for energy consumption grouped by commit.

        This method calculates the median, standard deviation, and count of energy
        consumption values for each commit in the provided DataFrame. It filters out
        commits with fewer measurements than the defined minimum threshold and generates
        additional metadata for visualization.

        Args:
            df (pd.DataFrame): The input DataFrame containing energy consumption data.
                It must include a "commit" column and the specified energy column.
            energy_column (str): The name of the column in the DataFrame that contains
                energy consumption values.

        Returns:
            EnergyStats: Data class containing the computed statistics.
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
        y_errors = df_median[f"{energy_column}_std"].fillna(0.0).tolist()
        return EnergyStats(valid_commits, short_hashes, x_indices, y_medians, y_errors, df_median)

    @staticmethod
    def _compute_distribution_and_normality(
        df: pd.DataFrame,
        valid_commits: list[str],
        energy_column: str,
    ) -> tuple[list[Any], list[Any]]:
        """Compute distributions and normality flags for energy consumption data.

        This method generates distributions of energy consumption values for each commit
        and performs a normality test (Shapiro-Wilk) on each distribution.

        Args:
            df (pd.DataFrame): The input DataFrame containing energy consumption data.
                It must include a "commit" column and the specified energy column.
            valid_commits (list[str]): A list of valid commit identifiers.
            energy_column (str): The name of the column in the DataFrame that contains
                energy consumption values.

        Returns:
            tuple: A tuple containing two lists:
                - distributions (list[Any]): A list of distributions for each commit.
                - normality_flags (list[Any]): A list of boolean flags indicating
                    whether each distribution is normal.
        """
        distributions = []
        normality_flags = []
        for commit in valid_commits:
            values = df[df["commit"] == commit][energy_column].values
            distributions.append(values)
            if len(values) >= MIN_VALUES_FOR_NORMALITY_TEST:
                _, p_shapiro = shapiro(values)
                normality_flags.append(p_shapiro >= NORMALITY_P_THRESHOLD)
            else:
                normality_flags.append(True)  # Assume normality with too few data points
        return distributions, normality_flags

    @staticmethod
    def _get_change_direction(baseline_median: float, test_median: float) -> str | None:
        """Determine the direction of change between a baseline median and a test median.

        Args:
            baseline_median (float): The median value of the baseline data.
            test_median (float): The median value of the test data.

        Returns:
            str | None:
                - "increase" if the test median is greater than or equal to the baseline median
                  by at least the percentage defined by MIN_PCT_INCREASE.
                - "decrease" if the test median is less than or equal to the baseline median
                  by at least the percentage defined by MIN_PCT_INCREASE.
                - None if the change is within the threshold defined by MIN_PCT_INCREASE.
        """
        if test_median >= baseline_median * (1 + MIN_PCT_INCREASE):
            return "increase"
        if test_median <= baseline_median * (1 - MIN_PCT_INCREASE):
            return "decrease"
        return None

    def detect_energy_changes(self, distributions: list[np.ndarray]) -> list[ChangeEvent]:
        """Detects significant energy changes between consecutive distributions.

        This method analyzes a list of numerical distributions and identifies
        significant changes in energy levels between consecutive distributions
        using statistical tests and effect size calculations.

        Args:
            distributions (list[np.ndarray]): A list of numerical distributions
                (as numpy arrays) to analyze for energy changes.

        Returns:
            list[ChangeEvent]: A list of `ChangeEvent` objects representing
                detected changes. Each `ChangeEvent` includes the index of the
                change, severity of the change, direction ("increase" or "decrease"),
                and Cohen's d effect size.

        Notes:
            - Distributions with fewer than `MIN_VALUES_FOR_NORMALITY_TEST` values
              are skipped.
            - The Welch's t-test is used to determine if the change is statistically
              significant, with a threshold defined by `WELCH_P_THRESHOLD`.
            - The severity of the change is calculated as the relative difference
              in medians between the two distributions.
            - Cohen's d is used to quantify the effect size of the change.
        """
        changes = []
        for i in range(1, len(distributions)):
            baseline = distributions[i - 1]
            test = distributions[i]
            if len(baseline) < MIN_VALUES_FOR_NORMALITY_TEST or len(test) < MIN_VALUES_FOR_NORMALITY_TEST:
                continue
            baseline_median = np.median(baseline)
            test_median = np.median(test)
            _, p_value = ttest_ind(baseline, test, equal_var=False)
            if (p_value < WELCH_P_THRESHOLD) and (direction := self._get_change_direction(baseline_median, test_median)):
                var_baseline = np.var(baseline, ddof=1)
                var_test = np.var(test, ddof=1)
                pooled_std: int = np.sqrt((var_baseline + var_test) / 2.0)
                cohen_d = (np.mean(test) - np.mean(baseline)) / pooled_std if pooled_std else 0.0
                severity = (
                    (test_median - baseline_median) / baseline_median
                    if direction == "increase"
                    else (baseline_median - test_median) / baseline_median
                )
                changes.append(ChangeEvent(index=i, severity=severity, direction=direction, cohen_d=cohen_d))
        return changes

    def process_column(self, energy_column: str) -> None:
        """Processes a specified energy column.

        Processes a specified energy column by filtering outliers, computing statistics,
        analyzing distribution and normality, and detecting energy changes.

        Args:
            energy_column (str): The name of the energy column to process.

        Workflow:
            1. Filters outliers in the specified energy column using `filter_outliers`.
            2. Computes commit statistics for the filtered data using `compute_commit_statistics`.
            3. Analyzes the distribution and normality of the filtered data using
               `compute_distribution_and_normality`.
            4. Detects energy changes in the distribution using `detect_energy_changes`.
            5. Stores the results in instance attributes keyed by the energy column:
               - `self.stats`: Stores computed statistics.
               - `self.distributions`: Stores distribution data.
               - `self.normality_flags`: Stores normality test results.
               - `self.change_events`: Stores detected energy change events.
        """
        # Assume self.df is already loaded
        df_filtered = self._filter_outliers(self.df, energy_column)
        stats = self._compute_commit_statistics(df_filtered, energy_column)
        distribution, normality = self._compute_distribution_and_normality(df_filtered, stats.valid_commits, energy_column)
        changes = self.detect_energy_changes(distribution)
        # Store the results in a dict keyed by energy column
        self.stats[energy_column] = stats
        self.distributions[energy_column] = distribution
        self.normality_flags[energy_column] = normality
        self.change_events[energy_column] = changes

    def compute_overall_summary(self, energy_column: str) -> dict[str, Any]:
        """Compute an overall summary of energy-related statistics for a given column.

        Args:
            energy_column (str): The name of the column containing energy data.

        Returns:
            dict: A dictionary containing the following summary statistics:
                - total_commits (int): Total number of data points in the distribution.
                - significant_changes (int): Number of significant change events.
                - regressions (int): Count of changes with an "increase" direction.
                - improvements (int): Count of changes with a "decrease" direction.
                - mean_energy (float): Mean value of the energy column.
                - median_energy (float): Median value of the energy column.
                - std_energy (float): Standard deviation of the energy column.
                - max_increase (float): Maximum severity of "increase" changes.
                - max_decrease (float): Maximum severity of "decrease" changes.
                - avg_cohens_d (float): Average absolute Cohen's d value for changes.
                - normal_count (int): Number of normality flags set to True.
                - non_normal_count (int): Number of normality flags set to False.
                - outliers_removed (int): Number of outliers removed from the dataset.
        """
        distribution = self.distributions[energy_column]
        normality = self.normality_flags[energy_column]
        changes = self.change_events[energy_column]
        outliers_removed_count = len(self.df) - len(self._filter_outliers(self.df, energy_column))
        summary = {
            "total_commits": len(distribution),
            "significant_changes": len(changes),
            "regressions": sum(1 for e in changes if e.direction == "increase"),
            "improvements": sum(1 for e in changes if e.direction == "decrease"),
            "mean_energy": self.df[energy_column].mean(),
            "median_energy": self.df[energy_column].median(),
            "std_energy": self.df[energy_column].std(),
            "max_increase": max((e.severity for e in changes if e.direction == "increase"), default=0.0),
            "max_decrease": max((e.severity for e in changes if e.direction == "decrease"), default=0.0),
            "avg_cohens_d": np.mean([abs(e.cohen_d) for e in changes]) if changes else 0.0,
            "normal_count": sum(normality),
            "non_normal_count": len(normality) - sum(normality),
            "outliers_removed": outliers_removed_count,
        }
        return summary
