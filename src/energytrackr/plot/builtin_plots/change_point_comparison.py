"""Change-Point Comparison Plot."""

from __future__ import annotations

import numpy as np
import ruptures as rpt
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.layouts import Column
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.plot_interface import Plot
from energytrackr.plot.core.context import Context


class ChangePointComparison(Plot):
    """Renders a median-trend plot with change-point detection overlays."""

    def build(self, ctx: Context) -> None:  # noqa: PLR6301
        """Builds the change-point comparison plot.

        Builds a Bokeh plot visualizing change-point detection on median energy values across commits.

        This function performs the following steps:
        1. Extracts commit labels and computes the median of energy distributions for each commit.
        2. Detects change-points in the sequence of medians using the PELT algorithm with an RBF cost model.
        3. Prepares a Bokeh data source for the median line.
        4. Determines the y-axis span for plotting vertical change-point segments.
        5. Constructs a Bokeh figure showing the median trend line across commits.
        6. Overlays vertical dashed segments at each detected change-point.
        7. Adds a hover tool to display precise commit and median values.
        8. Stores the resulting plot layout in the context under the key "change_point_comparison".

        Args:
            ctx (Context): The plotting context containing statistics, artefacts, and configuration.
        """
        # 1) Extract commit labels & compute medians
        labels = ctx.stats["short_hashes"]
        dists = ctx.artefacts["distributions"]
        medians = np.array([float(np.median(arr)) for arr in dists])

        # 2) Detect change-points with PELT + RBF cost
        algo = rpt.Pelt(model="rbf").fit(medians)
        breakpoints = algo.predict(pen=3)
        # drop the final index (always == len(medians))
        cps = [bp for bp in breakpoints if bp < len(medians)]

        # 3) Prepare data source for the median line
        source = ColumnDataSource(
            data={
                "commit": labels,
                "med": medians.tolist(),
            },
        )

        # 5) Build the figure
        p = figure(
            x_range=labels,
            sizing_mode="stretch_width",
            title=f"Change-Point Detection: {ctx.energy_fields[0]} Medians",
            x_axis_label="Commit",
            y_axis_label=f"Median {ctx.energy_fields[0]} (J)",
            tools="pan,box_zoom,reset,save,wheel_zoom,hover",
            toolbar_location="above",
        )
        # median-trend line
        median_line = p.line(
            x="commit",
            y="med",
            source=source,
            line_width=2,
            line_color="navy",
            name="median_line",
        )

        # 6) Overlay vertical segments at each detected change-point
        if cps:
            segment_source = ColumnDataSource({
                "commit": [labels[cp] for cp in cps],
                "y0": [medians.min()] * len(cps),
                "y1": [medians.max()] * len(cps),
            })
            p.segment(
                x0="commit",
                y0="y0",
                x1="commit",
                y1="y1",
                source=segment_source,
                line_dash="dashed",
                line_color="firebrick",
                line_width=2,
            )

        # 7) Hover tool for precise values (only for the median line)
        hover = HoverTool(
            tooltips=[
                ("Commit", "@commit"),
                ("Median", "@med{0.00} J"),
            ],
            mode="vline",
            renderers=[median_line],
        )
        p.add_tools(hover)
        for axis in p.xaxis:
            axis.major_label_orientation = 0.8

        # 8) Add the plot to the context
        layout: Column = column(p, sizing_mode="stretch_width")
        ctx.plots["Change Point Detection"] = layout
