"""Plotting object for MACD (Moving Average Convergence Divergence) line."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, PlotObj


@dataclass(frozen=True)
class MACDConfig:
    """Configuration for MACD calculation and styling."""

    fast_span: int = 12
    slow_span: int = 26
    signal_span: int = 9
    macd_color: str = "purple"
    signal_color: str = "black"
    line_width: int = 2
    default_visible: bool = True


class MACDLine(PlotObj, Configurable[MACDConfig]):
    """Draws MACD (fast EMA - slow EMA) plus its signal line (EMA of MACD)."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the macd with configuration parameters."""
        super().__init__(MACDConfig, **params)

    @property
    def legend_macd(self) -> str:
        """Generate the legend label for the MACD line using the configured fast and slow spans.

        Returns:
            str: A string representing the MACD legend in the format 'MACD(fast_span,slow_span)'.
        """
        return f"MACD({self.config.fast_span},{self.config.slow_span})"

    @property
    def legend_signal(self) -> str:
        """Returns the legend label for the signal line in the MACD plot.

        The label includes the signal span value from the configuration.

        Returns:
            str: The formatted legend label for the signal line.
        """
        return f"Signal({self.config.signal_span})"

    def add(self, ctx: Context, fig: figure) -> None:
        """Adds MACD (Moving Average Convergence Divergence) and signal lines to the given Bokeh figure.

        This method computes the MACD and signal lines from the median values in the provided context,
        creates a data source for plotting, and adds the corresponding lines to the figure. It also
        attaches a hover tool to display commit hashes and line values.

        Args:
            ctx (Context): The context containing statistical data, including medians and commit hashes.
            fig (figure): The Bokeh figure to which the MACD and signal lines will be added.
        """
        medians = pd.Series(ctx.stats["medians"], name="median")
        ema_fast = medians.ewm(span=self.config.fast_span, adjust=False).mean()
        ema_slow = medians.ewm(span=self.config.slow_span, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.config.signal_span, adjust=False).mean()

        x_vals = list(range(len(macd)))
        commits: Sequence[str] = ctx.stats["short_hashes"]
        src = ColumnDataSource({
            "x": x_vals,
            "macd": macd.tolist(),
            "signal": signal.tolist(),
            "commit": commits,
        })

        r_macd = fig.line(
            x="x",
            y="macd",
            source=src,
            line_color=self.config.macd_color,
            line_width=self.config.line_width,
            legend_label=self.legend_macd,
            visible=self.config.default_visible,
        )
        r_signal = fig.line(
            x="x",
            y="signal",
            source=src,
            line_color=self.config.signal_color,
            line_width=self.config.line_width,
            line_dash="dashed",
            legend_label=self.legend_signal,
            visible=self.config.default_visible,
        )

        hover = HoverTool(
            renderers=[r_macd, r_signal],
            tooltips=[
                ("Commit", "@commit"),
                (self.legend_macd, "@macd{0.00}"),
                (self.legend_signal, "@signal{0.00}"),
            ],
        )
        fig.add_tools(hover)
