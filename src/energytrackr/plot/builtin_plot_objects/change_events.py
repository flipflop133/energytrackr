"""ChangeEventMarkers - Draws regression/improvement markers and optional level circles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bokeh.models import BoxAnnotation
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, PlotObj
from energytrackr.utils.logger import logger


@dataclass(frozen=True)
class ChangeEventMarkersConfig:
    """Configuration for ChangeEventMarkers."""

    show_levels: bool = True
    radius_base: float = 6


class ChangeEventMarkers(PlotObj, Configurable[ChangeEventMarkersConfig]):
    """Draws regression/improvement markers and optional level circles."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the ChangeEventMarkers object.

        Args:
            **params: Configuration parameters for ChangeEventMarkers.
        """
        super().__init__(ChangeEventMarkersConfig, **params)

    def add(self, ctx: Context, fig: figure) -> None:
        """Adds visual markers for change events to the plot in the given context.

        This method retrieves change events from the context's artefacts and adds
        box annotations to the plot to indicate regressions (in red) and improvements (in green)
        at the corresponding indices. If `self.show_levels` is True, it also adds circles
        at the median values for each event, colored and sized according to their level.

        Side Effects:
            Modifies the plot in `ctx.fig` by adding box annotations and, optionally, level circles.
            Logs information and debug messages about the process.

        Args:
            ctx (Context): The plotting context containing artefacts, statistics, and the figure.
            fig (figure): The Bokeh figure to which the change event markers will be added.
        """
        if not (events := ctx.artefacts.get("change_events", [])):
            logger.info("ChangeEventMarkers: no events; skipping")
            return
        # Regression / improvement boxes
        for e in events:
            color = "red" if e.direction == "increase" else "green"
            fig.add_layout(
                BoxAnnotation(
                    left=e.index - 0.4,
                    right=e.index + 0.4,
                    fill_color=color,
                    fill_alpha=0.3,
                    line_alpha=0,
                    level="annotation",
                ),
            )

        # Level circles
        if self.config.show_levels:
            color_map = {1: "gray", 2: "blue", 3: "orange", 4: "purple", 5: "red"}
            for lvl, color in color_map.items():
                if not (xs := [e.index for e in events if e.level == lvl]):
                    continue
                ys = [ctx.stats["medians"][i] for i in xs]
                fig.circle(
                    x=xs,
                    y=ys,
                    radius=lvl * 0.3,
                    color=color,
                    alpha=0.4,
                    legend_label=f"Level {lvl}",
                )

        logger.debug("ChangeEventMarkers added: %d events", len(events))
