"""Distribution scatter plot object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PlotObj


@dataclass(frozen=True)
class DistributionScatterStyle:
    """Style for the distribution scatter plot."""

    normal_color: str = "blue"
    nonnormal_color: str = "orange"
    radius: float = 0.3
    alpha: float = 0.5
    normal_visible: bool = False
    normal_label: str = "Normal"
    nonnormal_label: str = "Non-Normal"


class DistributionScatter(PlotObj):
    """Scatter-plots each raw measurement point per commit."""

    def __init__(self, style: Mapping[str, object] | DistributionScatterStyle | None = None) -> None:
        """Initialize the DistributionScatter plot object.

        Args:
            style (Mapping[str, object] | DistributionScatterStyle): Style configuration for the scatter plot.
                If a dictionary is provided, it should contain keys matching the attributes of DistributionScatterStyle.
                If a DistributionScatterStyle object is provided, it will be used directly.
        """
        # Accept either a raw dict (from YAML) or our Style object
        if isinstance(style, DistributionScatterStyle):
            self.style = style
        else:
            data = dict(style or {})
            self.style = DistributionScatterStyle(
                normal_color=str(data.get("normal_color", DistributionScatterStyle.normal_color)),
                nonnormal_color=str(data.get("nonnormal_color", DistributionScatterStyle.nonnormal_color)),
                radius=float(str(data.get("radius", DistributionScatterStyle.radius))),
                alpha=float(str(data.get("alpha", DistributionScatterStyle.alpha))),
                normal_visible=bool(data.get("normal_visible", DistributionScatterStyle.normal_visible)),
                normal_label=str(data.get("normal_label", DistributionScatterStyle.normal_label)),
                nonnormal_label=str(data.get("nonnormal_label", DistributionScatterStyle.nonnormal_label)),
            )

    def add(self, ctx: Context, fig: figure) -> None:
        """Add the distribution scatter plot to the figure.

        Args:
            ctx (Context): The context object containing artefacts and figure.
                It should contain the following artefacts:
                - "distributions": List of distributions for each commit.
                - "normality_flags": List of booleans indicating if the distribution is normal or not.
            fig (figure): The Bokeh figure to which the scatter plot will be added.
        """
        dists = ctx.artefacts.get("distributions", [])
        flags = ctx.artefacts.get("normality_flags", [])

        normal_x, normal_y, nonnorm_x, nonnorm_y = [], [], [], []
        for i, vals in enumerate(dists):
            is_norm = flags[i] if i < len(flags) else True
            for v in vals:
                if is_norm:
                    normal_x.append(i)
                    normal_y.append(v)
                else:
                    nonnorm_x.append(i)
                    nonnorm_y.append(v)

        # normal
        normal_src = ColumnDataSource(data={"x": normal_x, "y": normal_y})
        fig.circle(
            x="x",
            y="y",
            source=normal_src,
            radius=self.style.radius,
            alpha=self.style.alpha,
            color=self.style.normal_color,
            legend_label=self.style.normal_label,
            visible=self.style.normal_visible,
        )

        # non-normal
        nonnorm_src = ColumnDataSource(data={"x": nonnorm_x, "y": nonnorm_y})
        fig.circle(
            x="x",
            y="y",
            source=nonnorm_src,
            radius=self.style.radius,
            alpha=self.style.alpha,
            color=self.style.nonnormal_color,
            legend_label=self.style.nonnormal_label,
            visible=True,
        )
