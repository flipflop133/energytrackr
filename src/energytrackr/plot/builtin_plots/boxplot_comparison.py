"""Boxplot comparison plot for energy distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bokeh.models import ColumnDataSource, DataRenderer, FactorRange, HoverTool, Range1d, Whisker
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.transform import jitter

from energytrackr.plot.builtin_plots.plot_interface import Plot
from energytrackr.plot.builtin_plots.utils import get_labels_and_dists, initial_commits, make_selectors, wrap_layout
from energytrackr.plot.core.context import Context


@dataclass
class Quartiles:
    """Quartile values for a distribution (Q1, median, Q3)."""

    q1: float
    median: float
    q3: float


@dataclass
class Whiskers:
    """Lower and upper whisker values for a boxplot."""

    lower: float
    upper: float


@dataclass
class Notch:
    """Lower and upper bounds of the notch around the median."""

    low: float
    high: float


@dataclass
class SingleStats:
    """Statistical summary for a single commit's distribution."""

    commit: str
    quartiles: Quartiles
    whiskers: Whiskers
    notch: Notch


@dataclass
class PlotSources:
    """Container for all Bokeh ColumnDataSource objects used."""

    full: ColumnDataSource
    raw: ColumnDataSource
    box: ColumnDataSource
    scatter1: ColumnDataSource
    scatter2: ColumnDataSource
    med_ticks: ColumnDataSource
    line: ColumnDataSource


class BoxplotComparison(Plot):
    """Interactive boxplot comparison for two selected commits with hover tooltips."""

    def __init__(self, template: str | None = None) -> None:
        """Initialize the BoxplotComparison plot.

        Args:
            template (str | None): Path to the HTML template file. If None, uses the default template.
        """
        tpl_dir = Path(__file__).parent / "templates"
        self.template_path = Path(template) if template else tpl_dir / "boxplot_comparison.html"

    def build(self, ctx: Context) -> None:
        """Builds the boxplot comparison plot and adds it to the context.

        This function performs the following steps:
        1. Retrieves labels and distributions from the context.
        2. Computes quartile statistics for the boxplot.
        3. Identifies the initial and final commits for comparison.
        4. Creates data sources for the plot.
        5. Generates the boxplot figure.
        6. Creates interactive widgets for selection.
        7. Arranges the widgets and plot into a layout and stores it in the context.

        Args:
            ctx (Context): The plotting context containing statistics, artefacts, and configuration.
        """
        labels, dists = get_labels_and_dists(ctx)
        stats = self._compute_stats(labels, dists)
        init1, init2 = initial_commits(labels)
        src = self._make_sources(stats, dists, init1, init2)
        fig = self._create_figure(stats, src, ctx.energy_fields[0], init1, init2)

        common_args = {
            "full": src.full,
            "raw": src.raw,
            "box": src.box,
            "scatter1": src.scatter1,
            "scatter2": src.scatter2,
            "tick": src.med_ticks,
            "line": src.line,
            "plot": fig,
            "color_list": src.box.data["color"],
        }

        sel1, sel2 = make_selectors(
            labels=labels,
            js_code_path=Path(__file__).parent / "static" / "boxplot_comparison.js",
            cb_args=common_args,
        )

        wrap_layout(sel1, sel2, fig, "Boxplot", ctx)

    @staticmethod
    def _stats_for_commit(commit: str, arr: list[float]) -> SingleStats:
        data = np.sort(np.asarray(arr, float))
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        lower = max(data.min(), q1 - 1.5 * iqr)
        upper = min(data.max(), q3 + 1.5 * iqr)
        notch = 1.57 * iqr / np.sqrt(len(data))
        return SingleStats(
            commit=commit,
            quartiles=Quartiles(q1, median, q3),
            whiskers=Whiskers(lower, upper),
            notch=Notch(median - notch, median + notch),
        )

    @staticmethod
    def _compute_stats(labels: list[str], dists: list[list[float]]) -> list[SingleStats]:
        return [BoxplotComparison._stats_for_commit(lbl, arr) for lbl, arr in zip(labels, dists, strict=False)]

    @staticmethod
    def _make_full_and_raw(stats: list[SingleStats], dists: list[list[float]]) -> tuple[ColumnDataSource, ColumnDataSource]:
        full = ColumnDataSource({
            "commit": [s.commit for s in stats],
            "q1": [s.quartiles.q1 for s in stats],
            "q2": [s.quartiles.median for s in stats],
            "q3": [s.quartiles.q3 for s in stats],
            "lower": [s.whiskers.lower for s in stats],
            "upper": [s.whiskers.upper for s in stats],
            "n_low": [s.notch.low for s in stats],
            "n_high": [s.notch.high for s in stats],
        })
        raw = ColumnDataSource({"commit": [s.commit for s in stats], "raw": dists})
        return full, raw

    @staticmethod
    def _make_box_and_scatters(
        stats: list[SingleStats],
        dists: list[list[float]],
        init1: str,
        init2: str,
    ) -> tuple[ColumnDataSource, ColumnDataSource, ColumnDataSource]:
        # find stats
        s1 = next(s for s in stats if s.commit == init1)
        s2 = next(s for s in stats if s.commit == init2)
        palette = Category10[3]
        colors = palette[:2]
        box = ColumnDataSource({
            "commit": [s1.commit, s2.commit],
            "q1": [s1.quartiles.q1, s2.quartiles.q1],
            "q2": [s1.quartiles.median, s2.quartiles.median],
            "q3": [s1.quartiles.q3, s2.quartiles.q3],
            "lower": [s1.whiskers.lower, s2.whiskers.lower],
            "upper": [s1.whiskers.upper, s2.whiskers.upper],
            "n_low": [s1.notch.low, s2.notch.low],
            "n_high": [s1.notch.high, s2.notch.high],
            "color": colors,
        })
        i1, i2 = [stats.index(s) for s in (s1, s2)]
        sc1 = ColumnDataSource({"x": [init1] * len(dists[i1]), "y": dists[i1]})
        sc2 = ColumnDataSource({"x": [init2] * len(dists[i2]), "y": dists[i2]})
        return box, sc1, sc2

    @staticmethod
    def _make_med_ticks_and_line(
        stats: list[SingleStats],
        init1: str,
        init2: str,
    ) -> tuple[ColumnDataSource, ColumnDataSource]:
        s1 = next(s for s in stats if s.commit == init1)
        s2 = next(s for s in stats if s.commit == init2)
        lowers = [s.whiskers.lower for s in stats]
        uppers = [s.whiskers.upper for s in stats]
        h = (max(uppers) - min(lowers)) * 0.05 * 0.1
        ticks = ColumnDataSource({
            "commit": [s1.commit, s2.commit],
            "med": [s1.quartiles.median, s2.quartiles.median],
            "w": [0.6, 0.6],
            "h": [h, h],
        })
        line = ColumnDataSource({
            "x": [s1.commit, s2.commit],
            "y": [s1.quartiles.median, s2.quartiles.median],
        })
        return ticks, line

    @staticmethod
    def _make_sources(stats: list[SingleStats], dists: list[list[float]], init1: str, init2: str) -> PlotSources:
        full, raw = BoxplotComparison._make_full_and_raw(stats, dists)
        box, sc1, sc2 = BoxplotComparison._make_box_and_scatters(stats, dists, init1, init2)
        ticks, line = BoxplotComparison._make_med_ticks_and_line(stats, init1, init2)
        return PlotSources(full, raw, box, sc1, sc2, ticks, line)

    @staticmethod
    def _create_figure(stats: list[SingleStats], src: PlotSources, energy_field: str, init1: str, init2: str) -> figure:
        y_start, y_end = BoxplotComparison.__compute_y_range(stats)
        palette = Category10[3]
        colors = palette[:2]

        p = figure(
            x_range=FactorRange([init1, init2]),
            y_range=Range1d(start=y_start, end=y_end),
            tools="pan,box_zoom,reset,save,wheel_zoom",
            toolbar_location="above",
            sizing_mode="stretch_width",
            title=f"Distribution Boxplot: {energy_field}",
        )

        BoxplotComparison.__add_box_layers(p, src)
        BoxplotComparison.__add_whiskers(p, src)
        BoxplotComparison.__add_median_markers(p, src)
        BoxplotComparison.__add_raw_points(p, src, colors)
        BoxplotComparison.__add_hover(p)

        for xaxis in p.xaxis:
            xaxis.major_label_orientation = 0.8
            xaxis.axis_label = f"{energy_field} (J)"

        return p

    @staticmethod
    def __compute_y_range(stats: list[SingleStats]) -> tuple[float, float]:
        lowers = [s.whiskers.lower for s in stats]
        uppers = [s.whiskers.upper for s in stats]
        lo, hi = min(lowers), max(uppers)
        margin = (hi - lo) * 0.05
        return lo - margin, hi + margin

    @staticmethod
    def __add_box_layers(p: figure, src: PlotSources) -> None:
        layers = [("q1", "q3", 0.6), ("n_low", "n_high", 0.3)]
        for bottom, top, width in layers:
            p.vbar(
                x="commit",
                bottom=bottom,
                top=top,
                width=width,
                source=src.box,
                fill_color="color",
                fill_alpha=0.3,
                line_color="black",
            )

    @staticmethod
    def __add_whiskers(p: figure, src: PlotSources) -> None:
        whisk = Whisker(source=src.box, base="commit", lower="lower", upper="upper")
        for head in (whisk.upper_head, whisk.lower_head):
            if head:
                head.size = 8
        p.add_layout(whisk)

    @staticmethod
    def __add_median_markers(p: figure, src: PlotSources) -> None:
        p.line(x="x", y="y", source=src.line, line_dash="dashed", line_color="firebrick", line_width=2)
        p.rect(x="commit", y="med", width="w", height="h", source=src.med_ticks, fill_color="firebrick", line_color=None)

    @staticmethod
    def __add_raw_points(p: figure, src: PlotSources, colors: list[str]) -> None:
        for scatter, color in ((src.scatter1, colors[0]), (src.scatter2, colors[1])):
            p.circle(x=jitter("x", width=0.3, range=p.x_range), y="y", source=scatter, radius=0.02, alpha=0.4, color=color)

    @staticmethod
    def __add_hover(p: figure) -> None:
        # Find the first DataRenderer (e.g., GlyphRenderer) for the hover tool
        data_renderers = [r for r in p.renderers if isinstance(r, DataRenderer)]
        tool = HoverTool(
            tooltips=[
                ("Commit", "@commit"),
                ("Q1", "@q1{0.00} J"),
                ("Median", "@q2{0.00} J"),
                ("Q3", "@q3{0.00} J"),
                ("Lower", "@lower{0.00} J"),
                ("Upper", "@upper{0.00} J"),
            ],
            renderers=[data_renderers[0]] if data_renderers else "auto",
        )
        p.add_tools(tool)
