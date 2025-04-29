"""CUSUMComparison using BasePlot and mixins for cleaner composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from bokeh.models import ColumnDataSource, FactorRange, Label, Range1d, Span
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.base import BasePlot
from energytrackr.plot.builtin_plots.mixins import FontMixin, HoverMixin
from energytrackr.plot.builtin_plots.registry import register_plot
from energytrackr.plot.core.context import Context


@dataclass
class CUSStats:
    """Statistics for CUSUM control chart."""

    medians: npt.NDArray[np.float64]
    baseline: float
    deviations: npt.NDArray[np.float64]
    cusum: npt.NDArray[np.float64]
    sigma: float
    upper_limit: float
    lower_limit: float


@register_plot
class CUSUMComparison(FontMixin, HoverMixin, BasePlot):
    """Renders an enhanced CUSUM control chart highlighting regressions and improvements."""

    def __init__(self) -> None:
        """Initialize the CUSUMComparison plot."""
        super().__init__()
        self._upper_span: Span | None = None
        self._lower_span: Span | None = None
        self._upper_label: Label | None = None
        self._lower_label: Label | None = None
        self._stats: Any | None = None
        self._src: ColumnDataSource | None = None

    def _make_figure(self, ctx: Context) -> figure:
        fig = super()._make_figure(ctx)

        labels = ctx.stats["short_hashes"]
        fig.x_range = FactorRange(*labels)

        return fig

    def _make_sources(self, ctx: Context) -> dict[str, Any]:
        labels: list[str] = ctx.stats["short_hashes"]
        dists: list[list[float]] = ctx.artefacts["distributions"]
        stats: CUSStats = self._compute_stats(dists)

        data = {
            "commit": labels,
            "cusum": stats.cusum.tolist(),
            "pos": np.maximum(stats.cusum, 0).tolist(),
            "neg": np.minimum(stats.cusum, 0).tolist(),
        }
        src = ColumnDataSource(data=data)

        # store for drawing and configuration
        self._stats = stats
        self._src = src
        return {"src": src}

    def _draw_glyphs(self, fig: figure, sources: dict[str, ColumnDataSource], ctx: Context) -> None:  # noqa: ARG002
        src = sources["src"]
        # shaded areas for regression (pos) and improvement (neg)
        fig.varea(
            x="commit",
            y1=0,
            y2="pos",
            source=src,
            fill_alpha=0.3,
            legend_label="Regression",
        )
        fig.varea(
            x="commit",
            y1=0,
            y2="neg",
            source=src,
            fill_alpha=0.3,
            legend_label="Improvement",
        )
        # CUSUM line and points
        fig.line(
            x="commit",
            y="cusum",
            source=src,
            line_width=2,
            legend_label="CUSUM",
        )
        fig.circle(
            x="commit",
            y="cusum",
            source=src,
            radius=0.02,
            fill_color="white",
            line_width=2,
        )
        assert self._stats is not None
        # control limit spans
        stats = self._stats
        self._upper_span = Span(
            location=stats.upper_limit,
            dimension="width",
            line_dash="dashed",
            line_color="red",
            line_width=1,
        )
        self._lower_span = Span(
            location=stats.lower_limit,
            dimension="width",
            line_dash="dashed",
            line_color="green",
            line_width=1,
        )
        fig.add_layout(self._upper_span)
        fig.add_layout(self._lower_span)

        # labels for limits
        self._upper_label = Label(
            x=0,
            y=stats.upper_limit,
            text=f"+2sigma = {stats.upper_limit:.2f} J",
            text_color="red",
            y_offset=5,
        )
        self._lower_label = Label(
            x=0,
            y=stats.lower_limit,
            text=f"-2sigma = {stats.lower_limit:.2f} J",
            text_color="green",
            y_offset=-10,
        )
        fig.add_layout(self._upper_label)
        fig.add_layout(self._lower_label)

    def _configure(self, fig: figure, ctx: Context) -> None:
        super()._configure(fig, ctx)
        # axis labels
        fig.x_range = FactorRange(*ctx.stats["short_hashes"])
        fig.xaxis[0].axis_label = "Commit"
        fig.yaxis[0].axis_label = "Cumulative deviation (J)"
        # rotate x-axis labels
        for ax in fig.xaxis:
            ax.major_label_orientation = 0.8
        # set y-range based on data
        stats = self._stats
        assert stats is not None
        y_min = min(stats.lower_limit, float(np.min(stats.cusum))) * 1.1
        y_max = max(stats.upper_limit, float(np.max(stats.cusum))) * 1.1
        fig.y_range = Range1d(start=y_min, end=y_max)

    @staticmethod
    def _compute_stats(dists: list[list[float]]) -> CUSStats:
        med = np.array([float(np.median(arr)) for arr in dists], dtype=float)
        base = med[0]
        dev = med - base
        cus = np.cumsum(dev)
        sigma = float(np.std(dev))
        return CUSStats(
            medians=med,
            baseline=base,
            deviations=dev,
            cusum=cus,
            sigma=sigma,
            upper_limit=2 * sigma,
            lower_limit=-2 * sigma,
        )

    def _hover_tooltips(self, ctx: Context) -> list[tuple[str, str]]:  # noqa: ARG002, PLR6301
        return [
            ("Commit", "@commit"),
            ("CUSUM", "@cusum{0.00} J"),
        ]
