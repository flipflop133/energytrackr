"""Candlestick - optional object that shows min/max & open/close per commit."""

from __future__ import annotations

from bokeh.layouts import column
from bokeh.models.annotations.labels import Title
from bokeh.models.layouts import Column
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.plot_interface import Plot
from energytrackr.plot.config import Settings, get_settings
from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PlotObj


class EvolutionPlot(Plot):
    """Base plot object."""

    def __init__(self, objects: list[str], template: str | None = None) -> None:
        """Initialize the Plot object.

        Args:
            objects (list[PlotObj]): List of plot objects to be added to the figure.
            template (str | None): Path to the template file. If None, use the default template.
        """
        self.template_path = template if template else "templates/base_plot.html"
        self.objects: list[str] = objects

    def build(self, ctx: Context) -> None:
        """Return an empty Bokeh figure with basic labels.

        Args:
            ctx (Context): The plotting context containing artefacts, statistics, and the figure.

        Raises:
            TypeError: If the plot object is not of type PlotObj.
        """
        fig = figure(
            title=f"Energy Consumption - {ctx.energy_fields[0]}",
            sizing_mode="stretch_width",
            tools="pan,box_zoom,reset,save,wheel_zoom",
            toolbar_location="above",
        )
        settings: Settings = get_settings()
        font: str = settings.energytrackr.report.font
        fig.xaxis[0].axis_label = "Commit (oldest → newest)"
        fig.xaxis[0].major_label_text_font = font
        fig.xaxis[0].axis_label_text_font = font

        fig.yaxis[0].axis_label = f"Median {ctx.energy_fields[0]} (J)"
        fig.yaxis[0].axis_label_text_font = font
        fig.yaxis[0].major_label_text_font = font

        title: Title | str | None = fig.title
        assert isinstance(title, Title)
        title.text_font = font
        title.text_font_size = "16pt"
        title.align = "center"
        title.text_color = "black"
        layout: Column = column(
            fig,
            sizing_mode="stretch_width",
        )
        for obj_name in self.objects:
            obj = ctx.plot_objects[obj_name]
            if not isinstance(obj, PlotObj):
                raise TypeError()
            obj.add(ctx, fig)
        ctx.plots["Evolution"] = layout
