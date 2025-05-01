"""SMA line plot object."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from energytrackr.plot.builtin_plot_objects.series_line import SeriesLine


@dataclass(frozen=True)
class SMALineConfig:
    """Configuration for the Simple Moving Average (SMA) line plot object."""

    window: int = 50
    color: str = "orange"
    line_width: int = 2
    radius: float = 0.3
    legend: str = "SMA"
    default_visible: bool = True


class SMALine(SeriesLine[SMALineConfig]):
    """Draws a rolling-window simple moving average over the median values."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the SMA line with optional parameters overriding defaults."""
        super().__init__(SMALineConfig, **params)

    def compute(self, series: pd.Series) -> pd.Series:
        """Compute the simple moving average of the series.

        Args:
            series: Input median series.

        Returns:
            SMA series with the same index.
        """
        return series.rolling(window=self.config.window, min_periods=1).mean()
