"""Plotting object for Exponential Moving Average (EMA) line."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from energytrackr.plot.builtin_plot_objects.series_line import SeriesLine


@dataclass(frozen=True)
class EMALineConfig:
    """Configuration for the Exponential Moving Average (EMA) line plot object."""

    span: int = 20
    color: str = "green"
    line_width: int = 2
    radius: float = 0.3
    legend: str = "EMA"
    default_visible: bool = True


class EMALine(SeriesLine[EMALineConfig]):
    """Draws an exponential moving average over the median values."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the EMA line with optional parameters overriding defaults."""
        super().__init__(EMALineConfig, **params)

    def compute(self, series: pd.Series) -> pd.Series:
        """Compute the exponential moving average of the series.

        Args:
            series: Input median series.

        Returns:
            EMA series with the same index.
        """
        return series.ewm(span=self.config.span, adjust=False).mean()
