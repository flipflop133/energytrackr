"""Bootstrap CI histogram for Δ-median between two commits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bokeh.models import ColumnDataSource, Label, Span
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.plot_interface import Plot
from energytrackr.plot.builtin_plots.utils import get_labels_and_dists, make_selectors, wrap_layout
from energytrackr.plot.core.context import Context


@dataclass
class BootStats:
    """Holds bootstrap histogram counts, bin edges, and confidence interval bounds."""

    counts: list[int]
    lefts: list[float]
    rights: list[float]
    low_ci: float
    high_ci: float


@dataclass
class BootSources:
    """Bundles the Bokeh ColumnDataSources used in the bootstrap comparison plot."""

    raw: ColumnDataSource
    hist: ColumnDataSource
    ci: ColumnDataSource


@dataclass
class BootFigure:
    """Encapsulates the Bokeh figure and its CI span/label annotations."""

    p: figure
    low_span: Span
    high_span: Span
    low_label: Label
    high_label: Label


class BootstrapComparison(Plot):
    """Renders an interactive bootstrap CI histogram for Δ-median between two commits."""

    def build(self, ctx: Context) -> None:
        """Builds the bootstrap comparison layout and stores it in the plotting context.

        Steps:
        1. Compute bootstrap statistics for initial (oldest vs newest) commits.
        2. Create data sources for raw distributions, histogram, and CI.
        3. Construct the Bokeh figure with histogram and CI annotations.
        4. Set up AutocompleteInput widgets and wire the JS callback.
        5. Assemble into a column layout and assign to ctx.plots.

        Args:
            ctx (Context): Plotting context with `stats`, `artefacts`, and `plots` dict.
        """
        labels, dists = get_labels_and_dists(ctx)
        stats = self._compute_boot_stats(dists)
        sources = self._make_sources(labels, dists, stats)
        fig = self._create_figure(ctx.energy_fields[0], sources, stats)

        sel1, sel2 = make_selectors(
            labels,
            js_code_path=Path(__file__).parent / "static" / "bootstrap_comparison.js",
            cb_args={
                "raw": sources.raw,
                "hist_source": sources.hist,
                "ci_source": sources.ci,
                "low_span": fig.low_span,
                "high_span": fig.high_span,
                "low_label": fig.low_label,
                "high_label": fig.high_label,
                "sel1": None,
                "sel2": None,
                "plot": fig.p,
            },
        )

        wrap_layout(sel1, sel2, fig.p, "Bootstrap", ctx)

    @staticmethod
    def _compute_boot_stats(dists: list[list[float]], num_bootstraps: int = 1000) -> BootStats:
        """Computes bootstrap statistics for the Δ-median CI.

        Performs a bootstrap sampling of the median difference (%) between
        the oldest and newest distributions and computes a histogram + 95% CI.

        Args:
            dists: List of energy measurement lists for each commit.
            num_bootstraps: Number of bootstrap resamples to draw.

        Returns:
            BootStats: counts, bin edges, and CI bounds for the bootstrap distribution.
        """
        arr1, arr2 = np.asarray(dists[0], float), np.asarray(dists[-1], float)
        diffs = []
        for _ in range(num_bootstraps):
            m1 = np.median(np.random.choice(arr1, size=len(arr1), replace=True))
            m2 = np.median(np.random.choice(arr2, size=len(arr2), replace=True))
            diffs.append((m2 / m1 - 1) * 100)

        counts, edges = np.histogram(diffs, bins="sturges")
        low_ci, high_ci = np.percentile(diffs, [2.5, 97.5])

        return BootStats(
            counts=[int(x) for x in counts],
            lefts=[float(x) for x in edges[:-1]],
            rights=[float(x) for x in edges[1:]],
            low_ci=low_ci,
            high_ci=high_ci,
        )

    @staticmethod
    def _make_sources(labels: list[str], dists: list[list[float]], stats: BootStats) -> BootSources:
        """Makes the Bokeh ColumnDataSource objects for the plot.

        Creates ColumnDataSource objects for:
          - raw distributions (for JS callbacks)
          - histogram bins and counts
          - confidence interval bounds

        Args:
            labels: List of commit hashes.
            dists: List of measurement arrays.
            stats: Precomputed bootstrap statistics.

        Returns:
            BootSources: Bundle of three ColumnDataSource instances.
        """
        raw = ColumnDataSource({"commit": labels, "raw": dists})
        hist = ColumnDataSource({
            "top": stats.counts,
            "left": stats.lefts,
            "right": stats.rights,
        })
        ci = ColumnDataSource({"low": [stats.low_ci], "high": [stats.high_ci]})
        return BootSources(raw, hist, ci)

    @staticmethod
    def _create_figure(energy_field: str, src: BootSources, stats: BootStats) -> BootFigure:
        """Creates the Bokeh figure for the bootstrap comparison.

        Constructs the Bokeh figure with:
          - Histogram of bootstrap Δ-median
          - Vertical dashed spans at 2.5% and 97.5% CI
          - Labels above each CI bound

        Args:
            energy_field: Name of the energy metric for the title.
            src: Sources for histogram and CI.
            stats: Bootstrap statistics for binning and CI.

        Returns:
            BootFigure: The figure plus Span/Label annotations.
        """
        p = figure(
            sizing_mode="stretch_width",
            title=f"Bootstrap Δ-Median CI: {energy_field}",
            x_axis_label="Δ-median (%)",
            y_axis_label="Count",
            tools="pan,box_zoom,reset,save,wheel_zoom,hover",
            toolbar_location="above",
        )
        p.quad(
            top="top",
            bottom=0,
            left="left",
            right="right",
            source=src.hist,
            fill_alpha=0.5,
            line_color="black",
        )

        low_span = Span(
            location=stats.low_ci,
            dimension="height",
            line_color="firebrick",
            line_dash="dashed",
            line_width=2,
        )
        high_span = Span(
            location=stats.high_ci,
            dimension="height",
            line_color="firebrick",
            line_dash="dashed",
            line_width=2,
        )
        p.add_layout(low_span)
        p.add_layout(high_span)

        y_max = max(stats.counts) * 0.9
        low_label = Label(
            x=stats.low_ci,
            y=y_max,
            x_units="data",
            y_units="data",
            text=f"2.5%: {stats.low_ci:.2f}%",
            text_align="center",
            background_fill_alpha=0.7,
        )
        high_label = Label(
            x=stats.high_ci,
            y=y_max,
            x_units="data",
            y_units="data",
            text=f"97.5%: {stats.high_ci:.2f}%",
            text_align="center",
            background_fill_alpha=0.7,
        )
        p.add_layout(low_label)
        p.add_layout(high_label)

        return BootFigure(p, low_span, high_span, low_label, high_label)
