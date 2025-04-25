"""MedianLine - default line/glyph object for the modular plot system.

YAML usage
~~~~~~~~~~
```yaml
plot:
  objects:
    - {module: plot.builtin_plot_objects.median_line:MedianLine,
       params: {color: blue, line_width: 2}}
```
"""

from __future__ import annotations

from collections.abc import Sequence

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.glyphs import Circle
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PlotObj
from energytrackr.utils.logger import logger


class MedianLine(PlotObj):
    """Draws median points connected by a line."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        color: str = "blue",
        line_width: int = 1,
        radius: float = 0.3,
        legend: str | None = None,
        default_visible: bool = True,
    ) -> None:
        """Initialize the MedianLine object.

        Args:
            color (str): Color of the line and points.
            line_width (int): Width of the line.
            radius (float): Radius of the points.
            legend (str | None): Legend label for the line. If None, defaults to "Median".
            default_visible (bool): Whether the line is visible by default.
        """
        self.color = color
        self.line_width = line_width
        self.radius = radius
        self.legend = legend or "Median"
        self.default_visible = default_visible

    def add(self, ctx: Context, fig: figure) -> None:
        """Adds a median line and interactive hover tool to the provided Bokeh figure context.

        This method extracts x-indices, median values, and commit short hashes from the context's statistics,
        creates a data source, and plots a line and circle glyphs representing the median values.
        It also attaches a hover tool to display commit and median information on mouseover.

        Args:
            ctx (Context): The plotting context containing statistics and the Bokeh figure.
            fig (figure): The Bokeh figure to which the median line will be added.
        """
        x: Sequence[int] = ctx.stats["x_indices"]
        y: Sequence[float] = ctx.stats["medians"]
        src = ColumnDataSource(data={"x": x, "y": y, "commit": ctx.stats["short_hashes"]})

        fig.line("x", "y", source=src, color=self.color, line_width=self.line_width, legend_label=f"{self.legend} line")

        logger.debug("self.radius = %s", self.radius)
        median_renderer: GlyphRenderer[Circle] = fig.circle(
            "x",
            "y",
            source=src,
            radius=self.radius,
            color=self.color,
            legend_label=self.legend,
            visible=self.default_visible,
        )
        hover: HoverTool = HoverTool(
            renderers=[median_renderer],
            tooltips=[("Commit", "@commit"), ("Median", "@y")],
        )
        fig.add_tools(hover)

        logger.debug("MedianLine added to figure with %d points", len(x))
