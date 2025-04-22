"""ErrorBars - vertical segments showing ±sigma around the median."""

from __future__ import annotations

from collections.abc import Sequence

from bokeh.models import ColumnDataSource

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PlotObj
from energytrackr.utils.exceptions import PlotObjectDidNotInitializeFigureError
from energytrackr.utils.logger import logger


class ErrorBars(PlotObj):
    """A plot object for rendering error bars on a plot.

    Attributes:
        line_width (int): Width of the error bar lines.
        color (str): Color of the error bars.
        legend (str): Label for the error bars in the plot legend.

    Methods:
        add(ctx: Context, **params: Any) -> None:
            Adds error bars to the given plot context using median values and their associated errors.
            The error bars are drawn as vertical segments from (x, median - error) to (x, median + error).
            Uses data from ctx.stats: "x_indices", "medians", and "y_errors".
    """

    def __init__(self, line_width: int = 2, color: str = "black", legend: str = "Error Bars") -> None:
        """Initialize the ErrorBars object with line width, color, and legend label.

        Args:
            line_width (int): Width of the error bar lines.
            color (str): Color of the error bars.
            legend (str): Label for the error bars in the plot legend.
        """
        self.line_width = line_width
        self.color = color
        self.legend = legend

    def add(self, ctx: Context) -> None:
        """Adds vertical error bars to the plot using the provided context.

        Retrieves x-coordinates, median y-values, and y-error values from the context's statistics.
        Computes the lower and upper bounds for the error bars, constructs a data source,
        and adds the error bars as vertical segments to the figure.

        Args:
            ctx (Context): The plotting context containing figure and statistical data.

        Raises:
            PlotObjectDidNotInitializeFigureError: If the figure is not initialized in the context.
        """
        x: Sequence[int] = ctx.stats["x_indices"]
        y: Sequence[float] = ctx.stats["medians"]
        err: Sequence[float] = ctx.stats["y_errors"]
        lower = [m - e for m, e in zip(y, err, strict=True)]
        upper = [m + e for m, e in zip(y, err, strict=True)]
        src = ColumnDataSource({"x": x, "y_lower": lower, "y_upper": upper})
        if not (fig := ctx.fig):
            raise PlotObjectDidNotInitializeFigureError(self.__class__.__name__)
        fig.segment(
            "x",
            "y_lower",
            "x",
            "y_upper",
            source=src,
            line_width=self.line_width,
            color=self.color,
            legend_label=self.legend,
        )
        logger.debug("ErrorBars added with %d segments", len(x))
