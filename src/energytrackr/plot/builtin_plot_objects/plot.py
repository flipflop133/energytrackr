"""Candlestick - optional object that shows min/max & open/close per commit."""

from __future__ import annotations

from bokeh.models.annotations.labels import Title
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PlotObj


class Plot(PlotObj):
    """Draws a candlestick chart: open, close, low, high per commit."""

    def add(self, ctx: Context) -> None:  # noqa: PLR6301
        """Return an empty Bokeh figure with basic labels.

        Args:
            ctx (Context): The plotting context containing artefacts, statistics, and the figure.
        """
        ctx.fig = figure(
            title=f"Energy Consumption - {ctx.energy_fields[0]}",
            sizing_mode="stretch_width",
            height=400,
            tools="pan,box_zoom,reset,save,wheel_zoom",
        )
        ctx.fig.xaxis[0].axis_label = "Commit (oldest → newest)"
        ctx.fig.xaxis[0].major_label_text_font = "Roboto"
        ctx.fig.xaxis[0].axis_label_text_font = "Roboto"

        ctx.fig.yaxis[0].axis_label = f"Median {ctx.energy_fields[0]} (J)"
        ctx.fig.yaxis[0].axis_label_text_font = "Roboto"
        ctx.fig.yaxis[0].major_label_text_font = "Roboto"

        title: Title | str | None = ctx.fig.title
        assert isinstance(title, Title)
        title.text_font = "Roboto"
        title.text_font_size = "16pt"
        title.align = "center"
        title.text_color = "black"
