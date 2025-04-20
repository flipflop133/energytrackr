"""Data processing and statistical analysis for energy measurements."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import shapiro, ttest_ind

MIN_MEASUREMENTS = 2
NORMALITY_P_THRESHOLD = 0.05
MIN_VALUES_FOR_NORMALITY_TEST = 3
WELCH_P_THRESHOLD = 0.05
MIN_PCT_INCREASE = 0.02
COHEN_D_THRESHOLDS: dict[str, float] = {
    "negligible": 0.2,
    "small": 0.5,
    "medium": 0.8,
    "large": 1.2,
}
PCT_CHANGE_THRESHOLDS: dict[str, float] = {
    "minor": 0.05,
    "moderate": 0.10,
    "major": float("inf"),
}

PRACTICAL_THRESHOLDS = {
    "info": 0.05,
    "warning": 0.1,
    "critical": 0.2,
}
PRACTICAL_LEVEL_4_TRIGGERS = [
    level for level, threshold in PRACTICAL_THRESHOLDS.items() if threshold > PRACTICAL_THRESHOLDS["info"]
]


@dataclass
class EffectSize:
    """Represents the effect size of a statistical comparison.

    Attributes:
        cohen_d (float): The calculated Cohen's d value indicating the standardized difference between two means.
        category (str): A qualitative description of the effect size (e.g., 'small', 'medium', 'large').
    """

    cohen_d: float
    category: str


@dataclass
class ChangeMagnitude:
    """Represents the magnitude of change between two values, including both percentage and absolute differences.

    Attributes:
        pct_change (float): The percentage change between two values.
        pct_change_level (str): A qualitative description of the percentage change (e.g., 'low', 'moderate', 'high').
        abs_diff (float): The absolute difference between two values.
        practical_level (str): A qualitative assessment of the practical significance of the change.
    """

    pct_change: float
    pct_change_level: str
    abs_diff: float
    practical_level: str


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
    direction: str
    p_value: float
    effect_size: EffectSize
    change_magnitude: ChangeMagnitude
    context_tags: list[str] | None  # e.g. [“cpu”, “module:io”]
    level: int


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
    x_indices: np.ndarray[Any, Any]
    y_medians: list[float]
    y_errors: list[float]
    df_median: pd.DataFrame


class EnergyData:
    """A class to process and analyze energy data from a CSV file."""

    def __init__(self, csv_path: str, energy_columns: list[str]) -> None:
        """Initialize the EnergyData class."""
        self.csv_path: str = csv_path
        self.energy_columns: list[str] = energy_columns
        self.df: pd.DataFrame | None = None
        self.stats: dict[str, Any] = {}  # Dictionary keyed by energy column
        self.distributions: dict[str, list[np.ndarray[Any, Any]]] = {}  # Raw distributions for each commit per column
        self.normality_flags: dict[str, list[bool]] = {}
        self.change_events: dict[str, list[ChangeEvent]] = {}

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
        q1: float = df[column].quantile(0.25)
        q3: float = df[column].quantile(0.75)
        iqr: float = q3 - q1
        lower_bound: float = q1 - multiplier * iqr
        upper_bound: float = q3 + multiplier * iqr
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
        commit_counts: DataFrame = df.groupby("commit").size().reset_index(name="count")
        df_median: DataFrame = df.groupby("commit", sort=False)[energy_column].median().reset_index()
        df_std: DataFrame = df.groupby("commit", sort=False)[energy_column].std().reset_index()
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
                arr = np.asarray(values, dtype=float)
                _, p_shapiro = shapiro(arr)
                normality_flags.append(p_shapiro >= NORMALITY_P_THRESHOLD)
            else:
                normality_flags.append(True)  # Assume normality with too few data points
        return distributions, normality_flags

    @staticmethod
    def _get_change_direction(baseline_median: np.floating[Any], test_median: np.floating[Any]) -> str | None:
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

    @staticmethod
    def _get_raw_direction(baseline_median: float, test_median: float) -> str:
        """Return 'increase' if test > baseline, else 'decrease'."""
        return "increase" if test_median > baseline_median else "decrease"

    def _detect_energy_changes(self, distributions: list[np.ndarray]) -> list[ChangeEvent]:  # pylint: disable=R0914  # noqa: PLR0914
        """Detects significant changes in energy distributions across sequential samples.

        This method compares each pair of consecutive distributions using statistical tests
        (Welch's t-test), effect size (Cohen's d), percentage change, and practical thresholds
        to identify and categorize meaningful changes in energy usage.

        Args:
            distributions (list[np.ndarray]): A list of numpy arrays, each representing a sample
                of energy measurements for a given interval.

        Returns:
            list[ChangeEvent]: A list of ChangeEvent objects, each describing a detected change
                between two consecutive distributions, including statistical significance,
                effect size, percentage and absolute change, practical relevance, context tags,
                and a severity level.
        """
        changes: list[ChangeEvent] = []

        for i in range(1, len(distributions)):
            baseline = distributions[i - 1]
            test = distributions[i]
            # skip tiny samples
            if len(baseline) < MIN_VALUES_FOR_NORMALITY_TEST or len(test) < MIN_VALUES_FOR_NORMALITY_TEST:
                continue

            # medians & p-value
            baseline_med = np.median(baseline)
            test_med = np.median(test)
            _, p_value = ttest_ind(baseline, test, equal_var=False)

            # Level-1: we now only filter by p_value
            if p_value < WELCH_P_THRESHOLD:
                # pooled std for effect size
                var_b = np.var(baseline, ddof=1)
                var_t = np.var(test, ddof=1)
                pooled = np.sqrt((var_b + var_t) / 2.0) or 1.0
                cohen_d = float((np.mean(test) - np.mean(baseline)) / pooled)

                # Level-2: effect-size bucket
                effect_cat = self.classify_effect_size(float(cohen_d))
                effect_size = EffectSize(cohen_d=cohen_d, category=effect_cat)

                # Level-3: % change & bucket
                pct = abs((test_med - baseline_med) / baseline_med)
                pct_cat = self.classify_pct_change(float(pct))

                # Level-4: absolute joules & bucket (example thresholds)
                abs_diff = float(test_med - baseline_med)
                practical = self.classify_practical(abs_diff, float(baseline_med))
                change_magnitude = ChangeMagnitude(
                    pct_change=float(pct),
                    pct_change_level=pct_cat,
                    abs_diff=abs_diff,
                    practical_level=practical,
                )

                # Level-5: context tags
                ctx = None  # Placeholder for context tags
                # Direction: may be None if pct < MIN_PCT_INCREASE
                direction = self._get_raw_direction(float(baseline_med), float(test_med))

                # after you compute p_value, effect_cat, pct_cat, practical, ctx...
                lvl = 1

                # Level 2: statistical + small effect size
                if abs(cohen_d) >= COHEN_D_THRESHOLDS["negligible"]:  # i.e., ≥ 0.2
                    lvl = max(lvl, 2)

                # Level 3: relative change in % is meaningful (e.g. ≥ 2%)
                if pct >= MIN_PCT_INCREASE:
                    lvl = max(lvl, 3)

                # Level 4: practical absolute change (warning or critical)
                if practical in PRACTICAL_LEVEL_4_TRIGGERS:
                    lvl = max(lvl, 4)

                # Level 5: context tags exist (e.g., touching important modules)
                if ctx:
                    lvl = max(lvl, 5)

                changes.append(
                    ChangeEvent(
                        index=i,
                        direction=direction,
                        p_value=float(p_value),
                        effect_size=effect_size,
                        change_magnitude=change_magnitude,
                        context_tags=ctx,
                        level=lvl,
                    ),
                )
        return changes

    def load_data(self, column_names: list[str]) -> None:
        """Load data from a CSV file into a DataFrame.

        Args:
            column_names (list[str]): A list of column names for the DataFrame.
        """
        self.df = pd.read_csv(self.csv_path, header=None, names=column_names)

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

        Raises:
            ValueError: If the DataFrame is not loaded before processing.
        """
        # Assume self.df is already loaded
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Please call load_data() before processing columns.")  # noqa: TRY003
        df_filtered = self._filter_outliers(self.df, energy_column)
        stats = self._compute_commit_statistics(df_filtered, energy_column)
        distribution, normality = self._compute_distribution_and_normality(df_filtered, stats.valid_commits, energy_column)
        changes = self._detect_energy_changes(distribution)
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
        if self.df is not None:
            outliers_removed_count = len(self.df) - len(self._filter_outliers(self.df, energy_column))
            mean_energy = self.df[energy_column].mean()
            median_energy = self.df[energy_column].median()
            std_energy = self.df[energy_column].std()
        else:
            outliers_removed_count = 0
            mean_energy = None
            median_energy = None
            std_energy = None

        summary = {
            "total_commits": len(distribution),
            "significant_changes": len(changes),
            "regressions": sum(1 for e in changes if e.direction == "increase"),
            "improvements": sum(1 for e in changes if e.direction == "decrease"),
            "mean_energy": mean_energy,
            "median_energy": median_energy,
            "std_energy": std_energy,
            "avg_cohens_d": np.mean([abs(e.effect_size.cohen_d) for e in changes]) if changes else 0.0,
            "normal_count": sum(normality),
            "non_normal_count": len(normality) - sum(normality),
            "outliers_removed": outliers_removed_count,
        }
        return summary

    @staticmethod
    def classify_effect_size(d: float) -> str:
        """Classifies the magnitude of an effect size (Cohen's d) into categorical labels.

        Args:
            d (float): The effect size value (Cohen's d) to classify.

        Returns:
            str: The category label corresponding to the magnitude of the effect size,
                based on predefined thresholds in COHEN_D_THRESHOLDS.

        Notes:
            - The function uses the absolute value of d for classification.
            - If d exceeds all defined thresholds, the last category in COHEN_D_THRESHOLDS is returned.
        """
        abs_d = abs(d)
        for category, thresh in COHEN_D_THRESHOLDS.items():
            if abs_d <= thresh:
                return category
        # fallback if > max threshold
        return list(COHEN_D_THRESHOLDS.keys())[-1]

    @staticmethod
    def classify_pct_change(pct: float) -> str:
        """Classifies a percentage change value into a category label based on predefined thresholds.

        Args:
            pct (float): The percentage change value to classify.

        Returns:
            str: The label corresponding to the range in which the percentage change falls.
                 If the value exceeds all thresholds, returns the last label as a fallback.

        Note:
            The classification thresholds are defined in the global dictionary `PCT_CHANGE_THRESHOLDS`,
            where keys are labels and values are threshold values.
        """
        for label, threshold in PCT_CHANGE_THRESHOLDS.items():
            if pct < threshold:
                return label
        return list(PCT_CHANGE_THRESHOLDS.keys())[-1]  # fallback

    @staticmethod
    def classify_practical(abs_diff: float, baseline_median: float) -> str:
        """Classifies the practical significance of an absolute difference relative to a baseline median.

        Parameters:
            abs_diff (float): The absolute difference to classify.
            baseline_median (float): The baseline median value used to scale thresholds.

        Returns:
            str: The classification level as a string, determined by comparing abs_diff to scaled thresholds.
                 Returns the corresponding level if abs_diff is below a threshold, or "critical" if above all.
        """
        # Build concrete thresholds from relative ones
        thresholds = {level: factor * baseline_median for level, factor in PRACTICAL_THRESHOLDS.items()}
        # Sort levels by ascending threshold value
        for level, limit in sorted(thresholds.items(), key=lambda x: x[1]):
            if abs_diff < limit:
                return level
        return "critical"  # fallback if above all thresholds
