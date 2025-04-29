"""Base template class for all energytrackr plots.

Defines the common build pipeline and hooks for composition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models.layouts import Column
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context


class BasePlot(ABC):
    """Canonical build pipeline for all plots.

      1. Create figure
      2. Prepare data sources
      3. Draw glyphs
      4. Configure axes, ranges, fonts
      5. Wrap into layout
    Concrete subclasses implement only the minimal hooks.
    """

    def build(self, ctx: Context) -> None:
        """Assemble and store the plot in the given context.

        Args:
            ctx (Context): The context object containing the data and configuration.
        """
        fig = self._make_figure(ctx)
        sources = self._make_sources(ctx)
        self._draw_glyphs(fig, sources, ctx)
        self._configure(fig, ctx)
        layout = self._wrap_layout(fig, ctx)
        ctx.plots[self._key(ctx)] = layout

    @abstractmethod
    def _make_sources(self, ctx: Context) -> dict[str, Any]:
        """Return the data sources (e.g. ColumnDataSource objects) needed for rendering.

        Args:
            ctx (Context): The context object containing the data and configuration.
        """

    @abstractmethod
    def _draw_glyphs(self, fig: figure, sources: dict[str, ColumnDataSource], ctx: Context) -> None:
        """Draw plot-specific glyphs on the provided Bokeh figure.

        Args:
            fig (figure): The Bokeh figure to draw on.
            sources (dict[str, Any]): The data sources for the plot.
            ctx (Context): The context object containing the data and configuration.
        """

    def _make_figure(self, ctx: Context) -> figure:
        """Initialize a Bokeh figure with default tools, sizing, and title.

        Args:
            ctx (Context): The context object containing the data and configuration.

        Returns:
            figure: A Bokeh figure object.
        """
        return figure(
            title=self._title(ctx),
            sizing_mode="stretch_width",
            tools="pan,box_zoom,reset,save,wheel_zoom,hover",
            toolbar_location="above",
        )

    @abstractmethod
    def _configure(self, fig: figure, ctx: Context) -> None:
        """Hook for configuring axes labels, fonts, ranges, etc.

        Override in mixins or subclasses as needed.

        Args:
            fig (figure): The Bokeh figure to configure.
            ctx (Context): The context object containing the data and configuration.
        """

    def _wrap_layout(self, fig: figure, ctx: Context) -> Column:  # noqa: ARG002, PLR6301 # pylint: disable=unused-argument
        """Wrap the figure into a layout (default: single-column).

        Override to add widgets or custom arrangements.

        Args:
            fig (figure): The Bokeh figure to wrap.
            ctx (Context): The context object containing the data and configuration.

        Returns:
            layout: The wrapped layout object.
        """
        return column(fig, sizing_mode="stretch_width")

    def _key(self, ctx: Context) -> str:  # noqa: ARG002 # pylint: disable=unused-argument
        """Key under which the layout is stored in ctx.plots (default: class name).

        Args:
            ctx (Context): The context object containing the data and configuration.

        Returns:
            str: The key for storing the layout.
        """
        return type(self).__name__

    def _title(self, ctx: Context) -> str:
        """Title for the figure (default: same as key).

        Args:
            ctx (Context): The context object containing the data and configuration.

        Returns:
            str: The title for the figure.
        """
        return self._key(ctx)
