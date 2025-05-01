# src/energytrackr/plot/builtin_plot_objects/dynamic_ticks.py

"""DynamicTicks - Adaptive x-axis ticks + label overrides using commit hashes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bokeh.core.properties import TextLike
from bokeh.models import CustomJS, FixedTicker
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, PlotObj


@dataclass(frozen=True)
class DynamicTicksConfig:
    """Configuration for DynamicTicks."""

    max_ticks: int = 30


class DynamicTicks(PlotObj, Configurable[DynamicTicksConfig]):
    """Adaptive x-axis ticks + label overrides using commit hashes."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the DynamicTicks object with configuration parameters.

        Args:
            **params: Configuration parameters for the DynamicTicks object.
        """
        super().__init__(DynamicTicksConfig, **params)

    def add(self, ctx: Context, fig: figure) -> None:
        """Adds dynamic tick labeling to the x-axis of the plot based on the current view range.

        This method initializes the x-axis with a fixed set of ticks and label overrides using short hashes.
        It also attaches a JavaScript callback to the x-axis range, so that when the user pans or zooms,
        the tick positions are recalculated to maintain a readable number of ticks, and the labels are updated accordingly.

        Args:
            ctx (Context): The plotting context containing the figure and statistics.
            fig (figure): The Bokeh figure to which the dynamic ticks will be added.
        """
        # 1) Build mapping from index → short_hash
        hashes = ctx.stats["short_hashes"]

        # 2) Set up initial ticker
        x_min = 0
        x_max = len(hashes) - 1
        raw_step = (x_max - x_min) / self.config.max_ticks
        step = max(1, int(raw_step))
        ticker = FixedTicker(ticks=list(range(x_min, x_max + 1, step)))
        overrides: Mapping[float | str, TextLike] = {float(i): h for i, h in enumerate(hashes)}
        for ax in fig.xaxis:
            ax.ticker = ticker
            ax.major_label_overrides = overrides
            ax.major_label_orientation = math.pi / 4

        # 3) Dynamic callback to recalc ticks on pan/zoom
        callback = CustomJS(
            args={"ticker": ticker, "full_length": x_max, "max_ticks": self.config.max_ticks},
            code="""
                const start = cb_obj.start;
                const end   = cb_obj.end;
                const visible = end - start;
                function niceNumber(x) {
                  const exp = Math.floor(Math.log(x)/Math.LN10);
                  const frac = x/Math.pow(10,exp);
                  let nf;
                  if(frac<1.5) nf=1;
                  else if(frac<3) nf=2;
                  else if(frac<7) nf=5;
                  else nf=10;
                  return nf*Math.pow(10,exp);
                }
                const raw = visible/max_ticks;
                const interval = Math.max(1, niceNumber(raw));
                const new_ticks = [];
                for(let t=0; t<=full_length; t+=interval){
                  if(t>=start && t<=end) new_ticks.push(t);
                }
                ticker.ticks = new_ticks;
            """,
        )
        fig.x_range.js_on_change("start", callback)
        fig.x_range.js_on_change("end", callback)
