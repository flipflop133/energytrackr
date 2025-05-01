"""Distribution scatter plot object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, PlotObj


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


@dataclass(frozen=True)
class DistributionScatterConfig:
    """Configuration for the distribution scatter plot."""

    normal_color: str = DistributionScatterStyle.normal_color
    nonnormal_color: str = DistributionScatterStyle.nonnormal_color
    radius: float = DistributionScatterStyle.radius
    alpha: float = DistributionScatterStyle.alpha
    normal_visible: bool = DistributionScatterStyle.normal_visible
    normal_label: str = DistributionScatterStyle.normal_label
    nonnormal_label: str = DistributionScatterStyle.nonnormal_label


class DistributionScatter(PlotObj, Configurable[DistributionScatterConfig]):
    """Scatter-plots each raw measurement point per commit."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the DistributionScatter plot object.

        Args:
            **params: Configuration parameters for the distribution scatter plot.
        """
        # Accept either a raw dict (from YAML) or our Style object
        super().__init__(DistributionScatterConfig, **params)

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
            radius=self.config.radius,
            alpha=self.config.alpha,
            color=self.config.normal_color,
            legend_label=self.config.normal_label,
            visible=self.config.normal_visible,
        )

        # non-normal
        nonnorm_src = ColumnDataSource(data={"x": nonnorm_x, "y": nonnorm_y})
        fig.circle(
            x="x",
            y="y",
            source=nonnorm_src,
            radius=self.config.radius,
            alpha=self.config.alpha,
            color=self.config.nonnormal_color,
            legend_label=self.config.nonnormal_label,
            visible=True,
        )
