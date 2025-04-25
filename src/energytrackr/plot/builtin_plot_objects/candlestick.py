"""Candlestick - optional object that shows min/max & open/close per commit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PlotObj
from energytrackr.utils.logger import logger


class Candlestick(PlotObj):
    """Draws a candlestick chart: open, close, low, high per commit."""

    def __init__(self, default_visible: bool = False, body_width: float = 0.6) -> None:
        """Initialize the Candlestick object.

        Args:
            default_visible (bool): Whether the candlestick is visible by default.
            body_width (float): Width of the candlestick body.
        """
        self.visible = default_visible
        self.body_width = body_width

    def add(self, ctx: Context, fig: figure) -> None:
        """Adds a candlestick plot to the given context's figure using statistical data.

        This method constructs a candlestick chart by computing the open, close, low, and high values
        from the provided medians and distributions. It then creates a Bokeh ColumnDataSource with these
        values and draws the candlestick bodies using the `quad` glyph and the wicks using the `segment` glyph.

        Side Effects:
            Modifies the figure in the context by adding candlestick and wick glyphs.
            Logs a debug message indicating whether the candlestick is visible.

        Args:
            ctx (Context): The plotting context containing statistical data and the figure to plot on.
                - ctx.stats["x_indices"]: Sequence of x-axis indices for each candlestick.
                - ctx.stats["medians"]: Sequence of median values for each candlestick.
                - ctx.artefacts["distributions"]: Sequence of distributions (arrays) for each candlestick.
            fig (figure): The Bokeh figure to which the candlestick plot will be added.
        """
        x: Sequence[int] = ctx.stats["x_indices"]
        medians: Sequence[float] = ctx.stats["medians"]

        # Compute open (previous median) and close (current), low and high
        opens = [medians[0], *medians[:-1]]
        closes = list(medians)
        distributions = ctx.artefacts["distributions"]
        lows = [float(np.min(d)) for d in distributions]
        highs = [float(np.max(d)) for d in distributions]
        colors = ["green" if close < open_ else "red" for open_, close in zip(opens, closes, strict=True)]

        # Build a ColumnDataSource with named fields
        data = {
            "x": x,
            "open": opens,
            "close": closes,
            "low": lows,
            "high": highs,
            "color": colors,
            # Precompute top/bottom/left/right for quad
            "top": [max(o, c) for o, c in zip(opens, closes, strict=True)],
            "bottom": [min(o, c) for o, c in zip(opens, closes, strict=True)],
            "left": [xi - self.body_width / 2 for xi in x],
            "right": [xi + self.body_width / 2 for xi in x],
        }
        src = ColumnDataSource(data)

        # Use data column names for quad glyph
        fig.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            fill_color="color",
            line_color="black",
            source=src,
            legend_label="Candlestick",
            visible=self.visible,
        )
        # Draw wicks using segment glyph referencing source fields
        fig.segment(
            x0="x",
            y0="low",
            x1="x",
            y1="high",
            source=src,
            line_color="black",
            legend_label="Candle Wicks",
            visible=self.visible,
        )
        logger.debug("Candlestick added (%s)", "visible" if self.visible else "hidden by default")
