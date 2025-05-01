"""Bollinger Bands (SMA ± n_std x rolling std) around a moving average."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, PlotObj


@dataclass(frozen=True)
class OutlierFilterConfig:
    """Configuration for bollinger bands."""

    window: int = 20
    n_std: float = 2.0
    band_color: str = "orange"
    ma_color: str = "navy"
    line_width: int = 2
    legend: str = "Bollinger Bands"
    default_visible: bool = True


class BollingerBands(PlotObj, Configurable[OutlierFilterConfig]):
    """Draws Bollinger Bands (SMA ± n_std x rolling std) around a moving average."""

    def __init__(self, **kwargs: dict) -> None:
        """Initialize the BollingerBands object with config-driven arguments."""
        super().__init__(OutlierFilterConfig, **kwargs)

    def add(self, ctx: Context, fig: figure) -> None:
        """Adds Bollinger Bands and a Simple Moving Average (SMA) line to the given Bokeh figure.

        This method computes the SMA and standard deviation over a rolling window from the median values
        in the provided context statistics. It then calculates the upper and lower Bollinger Bands,
        creates a data source, and adds a shaded area (band) between the bands and a line for the SMA.
        A hover tool is also added to display commit and SMA values.

        Args:
            ctx (Context): The plotting context containing statistics and labels.
            fig (figure): The Bokeh figure to which the Bollinger Bands and SMA will be added.
        """
        med = pd.Series(ctx.stats["medians"], name="median")
        sma = med.rolling(self.config.window, min_periods=1).mean()
        std = med.rolling(self.config.window, min_periods=1).std().fillna(0)
        upper = sma + self.config.n_std * std
        lower = sma - self.config.n_std * std

        x: list[int] = list(range(len(med)))
        labels: Sequence[str] = ctx.stats["short_hashes"]
        src = ColumnDataSource(
            data={
                "x": x,
                "sma": sma.tolist(),
                "upper": upper.tolist(),
                "lower": lower.tolist(),
                "commit": labels,
            },
        )

        # shaded band
        fig.varea(
            x="x",
            y1="lower",
            y2="upper",
            source=src,
            fill_color=self.config.band_color,
            fill_alpha=0.2,
            legend_label=self.config.legend,
            visible=self.config.default_visible,
        )
        # SMA line
        renderer = fig.line(
            x="x",
            y="sma",
            source=src,
            line_color=self.config.ma_color,
            line_width=self.config.line_width,
            legend_label=self.config.legend,
            visible=self.config.default_visible,
        )

        # hover on the central SMA
        hover = HoverTool(renderers=[renderer], tooltips=[("Commit", "@commit"), ("SMA", "@sma{0.00}")])
        fig.add_tools(hover)
