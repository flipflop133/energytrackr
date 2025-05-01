"""Moving-average plotting objects: a generic base plus EMA and SMA implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, PlotObj


# Protocol defining common config attributes
@runtime_checkable
class SeriesLineConfigProtocol(Protocol):
    """Protocol that all series line configs must follow."""

    @property
    def color(self) -> str:
        """Color of the line and markers."""
        raise NotImplementedError

    @property
    def line_width(self) -> int:
        """Width of the line."""
        raise NotImplementedError

    @property
    def radius(self) -> float:
        """Radius of the markers."""
        raise NotImplementedError

    @property
    def legend(self) -> str:
        """Label for the legend."""
        raise NotImplementedError

    @property
    def default_visible(self) -> bool:
        """Default visibility of the line and markers."""
        raise NotImplementedError


ConfigT = TypeVar("ConfigT", bound=SeriesLineConfigProtocol)


class SeriesLine[ConfigT: SeriesLineConfigProtocol](PlotObj, Configurable[ConfigT]):
    """Base class for line plot objects over a single Pandas Series."""

    def compute(self, series: pd.Series) -> pd.Series:  # noqa: PLR6301
        """Transform the input series before plotting; default is identity.

        Args:
            series (pd.Series): The input series to transform.

        Returns:
            pd.Series: The transformed series.
        """
        return series

    def add(self, ctx: Context, fig: figure) -> None:
        """Add the line, markers, and hover tool to the Bokeh figure.

        Args:
            ctx (Context): The plotting context containing figure and statistical data.
            fig (figure): The Bokeh figure to which the line and markers will be added.
        """
        med = pd.Series(ctx.stats.get("medians", []), name="median")
        out = self.compute(med)
        x = list(range(len(out)))
        labels: Sequence[str] = ctx.stats.get("short_hashes", [])
        src = ColumnDataSource({"x": x, "y": out.tolist(), "commit": labels})

        fig.line(
            "x",
            "y",
            source=src,
            color=self.config.color,
            line_width=self.config.line_width,
            legend_label=self.config.legend,
            visible=self.config.default_visible,
        )
        renderer: GlyphRenderer = fig.circle(
            "x",
            "y",
            source=src,
            radius=self.config.radius,
            color=self.config.color,
            legend_label=self.config.legend,
            visible=self.config.default_visible,
        )
        hover = HoverTool(
            renderers=[renderer],
            tooltips=[("Commit", "@commit"), (self.config.legend, "@y{0.00}")],
        )
        fig.add_tools(hover)
