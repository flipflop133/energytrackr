"""EvolutionPlot using the BasePlot template and FontMixin."""

from __future__ import annotations

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.base import BasePlot
from energytrackr.plot.builtin_plots.mixins import FontMixin, draw_additional_objects
from energytrackr.plot.builtin_plots.registry import register_plot
from energytrackr.plot.core.context import Context


@register_plot
class EvolutionPlot(FontMixin, BasePlot):
    """Candlestick-like evolution of energy per commit."""

    def __init__(self, objects: list[str], template: str | None = None) -> None:
        """Initialize the EvolutionPlot with a list of plot objects."""
        self.objects = objects
        self.template_path = template or "templates/base_plot.html"

    def _make_sources(self, ctx: Context) -> dict[str, ColumnDataSource]:  # noqa: ARG002, PLR6301
        return {}

    def _draw_glyphs(self, fig: figure, sources: dict[str, ColumnDataSource], ctx: Context) -> None:  # noqa: ARG002
        draw_additional_objects(self.objects, fig, ctx)

    def _configure(self, fig: figure, ctx: Context) -> None:
        # Apply font settings
        super()._configure(fig, ctx)
        # Axis labels
        fig.xaxis[0].axis_label = "Commit (oldest → newest)"
        fig.yaxis[0].axis_label = f"Median {ctx.energy_fields[0]} (J)"

    def _title(self, ctx: Context) -> str:  # noqa: PLR6301
        # Custom title including field
        return f"Energy Consumption - {ctx.energy_fields[0]}"

    def _key(self, ctx: Context) -> str:  # noqa: ARG002, PLR6301
        # Store under 'Evolution'
        return "Evolution"
