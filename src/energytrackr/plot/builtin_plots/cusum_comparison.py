"""CUSUM comparison plot for energy consumption regression analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, Label, Range1d, Span
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.plot_interface import Plot
from energytrackr.plot.core.context import Context


@dataclass
class CUSStats:
    """Computed statistics for CUSUM plot."""

    medians: np.ndarray
    baseline: float
    deviations: np.ndarray
    cusum: np.ndarray
    sigma: float
    upper_limit: float
    lower_limit: float


@dataclass
class CUSSources:
    """ColumnDataSource bundle for CUSUM plot."""

    main: ColumnDataSource


@dataclass
class CUSFigure:
    """Encapsulates the Bokeh figure and its annotation glyphs."""

    p: figure
    upper_span: Span
    lower_span: Span
    upper_label: Label
    lower_label: Label


class CUSUMComparison(Plot):
    """Renders an enhanced CUSUM control chart highlighting regressions."""

    def build(self, ctx: Context) -> None:  # noqa: PLR6301
        """Builds the CUSUM comparison chart and stores it in the context.

        Fetches commit labels and distributions, computes CUSUM statistics,
        creates data sources and figure, then assembles into a layout.

        Args:
            ctx: Plotting context with stats, artefacts, energy_fields, and plots dict.
        """
        labels = ctx.stats["short_hashes"]
        dists = ctx.artefacts["distributions"]

        stats = CUSUMComparison._compute_stats(dists)
        sources = CUSUMComparison._make_sources(labels, stats)
        fig = CUSUMComparison._create_figure(ctx.energy_fields[0], labels, stats, sources)

        layout = column(fig.p, sizing_mode="stretch_width")
        ctx.plots["CUSUM"] = layout

    @staticmethod
    def _compute_stats(dists: list[list[float]]) -> CUSStats:
        """Compute CUSUM statistics from distributions.

        Compute per-commit medians, baseline deviations, and CUSUM values,
        plus control limits at ±2sigma of the deviations.

        Args:
            dists: List of numeric arrays representing distributions.

        Returns:
            CUSStats: Object containing computed statistics for CUSUM.
        """
        med = np.array([np.median(arr) for arr in dists], dtype=float)
        base = med[0]
        dev = med - base
        cus = np.cumsum(dev)
        sigma = float(dev.std())
        return CUSStats(
            medians=med,
            baseline=base,
            deviations=dev,
            cusum=cus,
            sigma=sigma,
            upper_limit=2 * sigma,
            lower_limit=-2 * sigma,
        )

    @staticmethod
    def _make_sources(labels: list[str], stats: CUSStats) -> CUSSources:
        """Create data sources for CUSUM plot.

        Prepare a ColumnDataSource with CUSUM and masks for positive (regression)
        and negative (improvement) deviations.

        Args:
            labels: List of commit labels.
            stats: CUSStats object containing computed statistics.

        Returns:
            CUSSources: Data sources for the CUSUM plot.
        """
        data = {
            "commit": labels,
            "cusum": stats.cusum.tolist(),
            "pos": np.maximum(stats.cusum, 0).tolist(),
            "neg": np.minimum(stats.cusum, 0).tolist(),
        }
        return CUSSources(main=ColumnDataSource(data=data))

    @staticmethod
    def _create_figure(
        energy_field: str,
        labels: list[str],
        stats: CUSStats,
        src: CUSSources,
    ) -> CUSFigure:
        """Create the CUSUM plot with shaded areas and control limits.

        Build the Bokeh figure:
          - shaded vareas for pos/neg CUSUM
          - CUSUM line and points
          - control limit spans and labels
          - hover tool and axis styling

        Args:
            energy_field: Name of the energy field being analyzed.
            labels: List of commit labels.
            stats: CUSUM statistics object.
            src: Data sources for the plot.

        Returns:
            CUSFigure: The figure with annotations and data sources.
        """
        p = figure(
            x_range=FactorRange(*labels),
            sizing_mode="stretch_width",
            title=f"CUSUM Chart: {energy_field}",
            x_axis_label="Commit",
            y_axis_label="Cumulative deviation (J)",
            tools="pan,box_zoom,reset,save,wheel_zoom,hover",
            toolbar_location="above",
        )

        # shaded areas
        p.varea(x="commit", y1=0, y2="pos", source=src.main, fill_alpha=0.3, legend_label="Regression")
        p.varea(x="commit", y1=0, y2="neg", source=src.main, fill_alpha=0.3, legend_label="Improvement")

        # CUSUM line + points
        p.line(x="commit", y="cusum", source=src.main, line_width=2, legend_label="CUSUM")
        p.circle(x="commit", y="cusum", source=src.main, radius=0.02, fill_color="white", line_width=2)

        # control limit spans
        up = Span(location=stats.upper_limit, dimension="width", line_dash="dashed", line_color="red", line_width=1)
        lo = Span(location=stats.lower_limit, dimension="width", line_dash="dashed", line_color="green", line_width=1)
        p.add_layout(up)
        p.add_layout(lo)

        # limit labels
        y_max = max(stats.cusum.max(), stats.upper_limit) * 1.1
        up_lbl = Label(x=0, y=stats.upper_limit, text=f"+2sigma = {stats.upper_limit:.2f} J", text_color="red", y_offset=5)
        lo_lbl = Label(x=0, y=stats.lower_limit, text=f"-2sigma = {stats.lower_limit:.2f} J", text_color="green", y_offset=-10)
        p.add_layout(up_lbl)
        p.add_layout(lo_lbl)

        # hover & styling
        hover = HoverTool(tooltips=[("Commit", "@commit"), ("CUSUM", "@cusum{0.00} J")], mode="vline")
        p.add_tools(hover)
        for ax in p.xaxis:
            ax.major_label_orientation = 0.8
        y_start = min(stats.lower_limit, stats.cusum.min()) * 1.1
        p.y_range = Range1d(start=y_start, end=y_max)
        if p.legend:
            p.legend[0].location = "top_left"
            p.legend[0].click_policy = "hide"

        return CUSFigure(p=p, upper_span=up, lower_span=lo, upper_label=up_lbl, lower_label=lo_lbl)
