"""Violin plot comparing two commits with working hover tooltips."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from bokeh.models import ColumnDataSource, Label, Range1d, Span
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import figure
from scipy.stats import gaussian_kde

from energytrackr.plot.builtin_plots.plot_interface import Plot
from energytrackr.plot.builtin_plots.utils import get_labels_and_dists, make_selectors, wrap_layout
from energytrackr.plot.core.context import Context

if TYPE_CHECKING:
    from bokeh.models.sources import DataDict


class ViolinComparison(Plot):
    """Interactive violin plot comparing two selected commits with hover tooltips."""

    # pylint: disable=too-many-locals
    def build(self, ctx: Context) -> None:  # noqa: PLR0914
        """Build an interactive violin comparison plot in the given context.

        Args:
            ctx: Plotting context containing stats, artefacts, energy_fields, and a plots dict.
        """
        labels, dists = get_labels_and_dists(ctx)
        full_kde, medians = self._compute_kdes(dists)

        # after computing full_kde & medians
        p: figure = self._create_figure(ctx.energy_fields[0])

        # 1) initial indices
        i1, i2 = 0, -1
        init1, init2 = labels[i1], labels[i2]

        # 2) initial patch data
        v1_ds = ColumnDataSource(self._make_patch_ds(full_kde, i1, 0, init1, 0.4))
        v2_ds = ColumnDataSource(self._make_patch_ds(full_kde, i2, 1, init2, 0.4))

        # 3) render
        self._render_violin(p, v1_ds, fill_color="lightsteelblue")
        self._render_violin(p, v2_ds, fill_color="lightcoral" if medians[i2] > medians[i1] else "lightgreen")

        # 4) spans & label
        span1, span2 = self._add_median_spans(p, medians, i1, i2)
        label_diff = self._add_diff_label(p, medians, i1, i2, full_kde)

        sel1, sel2 = make_selectors(
            labels,
            js_code_path=Path(__file__).parent / "static" / "violin_comparison.js",
            cb_args={
                "full_kde": full_kde,
                "violin1_ds": v1_ds,
                "violin2_ds": v2_ds,
                "span1": span1,
                "span2": span2,
                "label_diff": label_diff,
                "sel1": None,
                "sel2": None,
                "width": 0.4,
                "labels": labels,
                "ticker": p.xaxis[0].ticker,
                "v2": p.renderers[-1],
            },
        )
        ViolinComparison._configure_xaxis(p, sel1.value, sel2.value)
        wrap_layout(sel1, sel2, p, "Violin", ctx)

    @staticmethod
    def _compute_kdes(dists: Sequence[Sequence[float] | np.ndarray]) -> tuple[ColumnDataSource, list[float]]:
        """Compute normalized KDEs and medians for given distributions.

        Args:
            dists: A list of numeric arrays.

        Returns:
            Tuple containing a ColumnDataSource with 'kde_x', 'kde_y', 'median' keys and the medians list.
        """
        kde_x_list: list[list[float]] = []
        kde_y_list: list[list[float]] = []
        medians: list[float] = []
        for arr in dists:
            a = np.asarray(arr, float)
            grid = np.linspace(a.min(), a.max(), 200)
            dens = gaussian_kde(a)(grid)
            kde_x_list.append((dens / dens.max()).tolist())
            kde_y_list.append(grid.tolist())
            medians.append(float(np.median(a)))
        src = ColumnDataSource({"kde_x": kde_x_list, "kde_y": kde_y_list, "median": medians})
        return src, medians

    @staticmethod
    def _create_figure(field: str) -> figure:
        """Initialize base Bokeh figure for violin plot.

        Args:
            field: Energy field name for y-axis and title.

        Returns:
            Configured Bokeh Figure.
        """
        p = figure(
            title=f"Violin Plot: {field}",
            sizing_mode="stretch_width",
            x_range=Range1d(-0.6, 1.6),
            y_axis_label=f"{field} (J)",
            tools="pan,box_zoom,reset,save,wheel_zoom",
            toolbar_location="above",
        )
        return p

    @staticmethod
    def _make_patch_ds(
        source: ColumnDataSource,
        idx: int,
        pos: int,
        label: str,
        width: float,
    ) -> DataDict:
        norm = source.data["kde_x"][idx]
        grid = source.data["kde_y"][idx]

        xs = [pos - n * width for n in norm] + [pos + n * width for n in reversed(norm)]
        ys = list(grid) + list(reversed(grid))
        commits = [label] * len(xs)

        # dict[str, Sequence[Any] | NDArray[Any]] is accepted as DataDict
        return {
            "x": xs,  # list[float] is a Sequence[Any]
            "y": ys,
            "commit": commits,
        }

    @staticmethod
    def _render_violin(p: figure, ds: ColumnDataSource, fill_color: str) -> GlyphRenderer:
        """Render a violin patch onto the figure.

        Args:
            p: Bokeh Figure.
            ds: ColumnDataSource for patch.
            fill_color: Fill color string.

        Returns:
            The Bokeh GlyphRenderer for the patch.
        """
        return p.patch("x", "y", source=ds, fill_color=fill_color, fill_alpha=0.6, line_color="black")

    @staticmethod
    def _add_median_spans(p: figure, medians: list[float], idx1: int, idx2: int) -> tuple[Span, Span]:
        """Add dashed spans for medians.

        Args:
            p: Bokeh Figure.
            medians: List of medians.
            idx1: First index.
            idx2: Second index.

        Returns:
            Tuple of two Span objects.
        """
        span1 = Span(location=medians[idx1], dimension="width", line_color="blue", line_dash="dashed", line_width=2)
        span2 = Span(
            location=medians[idx2],
            dimension="width",
            line_color="red" if medians[idx2] > medians[idx1] else "green",
            line_dash="dashed",
            line_width=2,
        )
        p.add_layout(span1)
        p.add_layout(span2)
        return span1, span2

    @staticmethod
    def _add_diff_label(p: figure, medians: list[float], idx1: int, idx2: int, source: ColumnDataSource) -> Label:
        """Annotate median difference.

        Args:
            p: Bokeh Figure.
            medians: List of medians.
            idx1: First index.
            idx2: Second index.
            source: ColumnDataSource with 'kde_y'.

        Returns:
            A Bokeh Label renderer.
        """
        delta = medians[idx2] - medians[idx1]
        y_max = max(source.data["kde_y"][-1])
        label = Label(
            x=0.5,
            y=y_max * 1.05,
            text=f"Δ median = {delta:.2f} J",
            text_align="center",
            text_font_size="12pt",
            text_color="red" if delta > 0 else "green",
        )
        p.add_layout(label)
        return label

    @staticmethod
    def _configure_xaxis(p: figure, label1: str, label2: str) -> None:
        """Configure fixed x-axis ticks and labels.

        Args:
            p: Bokeh Figure.
            label1: First commit label.
            label2: Second commit label.
        """
        ticker = FixedTicker(ticks=[0, 1])
        p.xaxis[0].ticker = ticker
        p.xaxis[0].major_label_overrides = {0: label1, 1: label2}
        p.xaxis[0].axis_label = "Commit"
