"""Data transform to filter out outlier commits."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas import Series

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, Transform


@dataclass(frozen=True)
class OutlierFilterConfig:
    """Configuration for filtering energy outliers."""

    window: int = 20
    multiplier: float = 1.5
    max_run_length: int = 2
    agg: str = "median"
    commit_col: str = "commit_hash"
    energy_col: str = "energy_median"


class FilterOutliers(Transform, Configurable[OutlierFilterConfig]):
    """Filter out transient outlier commits from the DataFrame."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the outlier filter with configuration parameters."""
        super().__init__(OutlierFilterConfig, **params)

    @property
    def window(self) -> int:
        """Get the rolling window size for outlier detection."""
        return self.config.window

    @property
    def multiplier(self) -> float:
        """Get the multiplier for the interquartile range (IQR) in outlier detection."""
        return self.config.multiplier

    @property
    def max_run_length(self) -> int:
        """Get the maximum run length for transient outlier detection."""
        return self.config.max_run_length

    @property
    def agg(self) -> str:
        """Get the aggregation function used for energy measurements."""
        return self.config.agg

    @property
    def commit_col(self) -> str:
        """Get the name of the commit column in the DataFrame."""
        return self.config.commit_col

    @property
    def energy_col(self) -> str:
        """Get the name of the energy column in the DataFrame."""
        return self.config.energy_col

    def apply(self, ctx: Context) -> None:
        """Filters out transient outlier commits from the DataFrame stored in the context artefacts.

        Args:
            ctx (Context): The context containing the DataFrame and artefacts.
        """
        df: pd.DataFrame = ctx.artefacts["df"]

        commit_scores = self._aggregate_commit_scores(df)
        q1, q3, iqr = self._compute_rolling_quartiles(commit_scores)
        is_outlier = self._detect_outliers(commit_scores, q1, q3, iqr)
        transient_commits = self._find_transient_outliers(commit_scores, is_outlier)

        df_filtered = self._remove_commits(df, transient_commits)
        ctx.artefacts["df"] = df_filtered
        ctx.stats["commits_removed"] = len(transient_commits)

    def _aggregate_commit_scores(self, df: pd.DataFrame) -> pd.Series:
        agg_series = getattr(df.groupby(self.commit_col)[self.energy_col], self.agg)()
        commits = df[self.commit_col].drop_duplicates().tolist()
        return pd.Series(agg_series.loc[commits].values, index=commits)

    def _compute_rolling_quartiles(self, commit_scores: pd.Series) -> tuple[Series[Any], Series[Any], Series[Any]]:
        q1 = commit_scores.rolling(self.window, center=True, min_periods=1).quantile(0.25)
        q3 = commit_scores.rolling(self.window, center=True, min_periods=1).quantile(0.75)
        iqr = q3 - q1
        return q1, q3, iqr

    def _detect_outliers(self, commit_scores: pd.Series, q1: pd.Series, q3: pd.Series, iqr: pd.Series) -> pd.Series:
        low = q1 - self.multiplier * iqr
        high = q3 + self.multiplier * iqr
        return (commit_scores < low) | (commit_scores > high)

    def _find_transient_outliers(self, commit_scores: pd.Series, is_outlier: pd.Series) -> pd.Index:
        group_id = (is_outlier != is_outlier.shift(fill_value=False)).cumsum()
        run_size = is_outlier.groupby(group_id).transform("size")
        return commit_scores.index[is_outlier & (run_size <= self.max_run_length)]

    def _remove_commits(self, df: pd.DataFrame, commits_to_remove: pd.Index) -> pd.DataFrame:
        mask = ~df[self.commit_col].isin(commits_to_remove)
        return df[mask].copy()
