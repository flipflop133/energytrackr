"""QQ-plot comparison of two commits."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    HoverTool,
    Label,
    LinearColorMapper,
    Range1d,
)
from bokeh.models.renderers import GlyphRenderer
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.plot_interface import Plot
from energytrackr.plot.builtin_plots.utils import (
    get_labels_and_dists,
    initial_commits,
    make_selectors,
    wrap_layout,
)
from energytrackr.plot.core.context import Context


class QQComparison(Plot):
    """Interactive QQ-plot comparing two commits with percentile coloring and tooltips."""

    def build(self, ctx: Context) -> None:
        """Build the QQ-plot comparison layout in the provided context.

        1. Fetch labels+distributions
        2. Compute full-percentile table
        3. Pick initial commits & build sources
        4. Render glyphs and configure axes
        5. Create selectors + JS callback
        6. Wrap everything in the shared layout helper

        Args:
            ctx: Plotting context containing stats, artefacts, energy_fields, and a plots dict.
        """
        # 1) labels and data
        labels, dists = get_labels_and_dists(ctx)

        # 2) full percentile table
        full_quant: dict[str, list | np.ndarray] = self._compute_quantiles(dists)

        # 3) initial commits
        init1, init2 = initial_commits(labels)
        i1, i2 = labels.index(init1), labels.index(init2)

        qq_src = ColumnDataSource(pd.DataFrame(self._make_qq_ds(full_quant, i1, i2)))
        idl_src = ColumnDataSource(pd.DataFrame(self._make_identity_ds(qq_src)))

        # 4) figure + glyphs
        p = self._create_figure(ctx.energy_fields[0])
        self._render_scatter(p, qq_src)
        self._render_identity(p, idl_src)
        self._add_colorbar(p)
        self._add_annotation(p)
        self._configure_ranges(p, qq_src)

        # 5) selectors + callback
        sel1, sel2 = make_selectors(
            labels,
            js_code_path=Path(__file__).parent / "static" / "qq_comparison.js",
            cb_args={
                "full_quant": full_quant,
                "commits": labels,
                "qq_src": qq_src,
                "idl_src": idl_src,
                "labels": labels,
                "plot": p,
            },
        )

        # 6) final layout
        wrap_layout(sel1, sel2, p, "Quantile Quantile", ctx)

    @staticmethod
    def _compute_quantiles(dists: Sequence[Sequence[float]]) -> dict[str, list | np.ndarray]:
        """Compute 0-100 percentiles for each distribution.

        Args:
            dists: List of numeric arrays.

        Returns:
            ColumnDataSource with key 'quant': list of percentile lists.
        """
        perc: np.ndarray = np.linspace(0, 100, 101)
        table = [np.percentile(np.asarray(arr), perc).tolist() for arr in dists]
        return {"quant": table, "percent": perc}

    @staticmethod
    def _make_qq_ds(full_quant: dict[str, list | np.ndarray], i: int, j: int) -> dict[str, Sequence[Any]]:
        """Prepare QQ data mapping percentiles to x and y.

        Args:
            full_quant: Source with 'quant' and 'percent'.
            i: Index for A's distribution.
            j: Index for B's distribution.

        Returns:
            Dict with 'x', 'y', and 'percent'.
        """
        quants = full_quant["quant"]
        perc = np.asarray(full_quant["percent"])
        # Ensure all values are numpy arrays (sequences)
        return cast(
            dict[str, Sequence[Any]],
            {
                "x": np.asarray(quants[i], dtype=np.float64),
                "y": np.asarray(quants[j], dtype=np.float64),
                "percent": perc,
            },
        )

    @staticmethod
    def _make_identity_ds(qq_src: ColumnDataSource) -> dict[str, Sequence[Any]]:
        xs = np.asarray(qq_src.data["x"]).flatten()
        ys = np.asarray(qq_src.data["y"]).flatten()
        vmin = float(np.min(np.concatenate([xs, ys])))
        vmax = float(np.max(np.concatenate([xs, ys])))
        return cast(
            dict[str, Sequence[Any]],
            {
                "x": np.array([vmin, vmax], dtype=np.float64),
                "y": np.array([vmin, vmax], dtype=np.float64),
            },
        )

    @staticmethod
    def _create_figure(field: str) -> figure:
        """Initialize the QQ-plot figure with hover tool.

        Args:
            field: Energy field name for y-axis and title.

        Returns:
            Configured Bokeh Figure.
        """
        p = figure(
            title=f"QQ-Plot: {field}",
            x_axis_label=f"Commit A {field} (J)",
            y_axis_label=f"Commit B {field} (J)",
            sizing_mode="stretch_width",
            tools="pan,box_zoom,reset,save,wheel_zoom,hover",
            toolbar_location="above",
        )
        hover = HoverTool(
            tooltips=[("Pct", "@percent%"), ("A", "@x{0.00} J"), ("B", "@y{0.00} J")],
            mode="mouse",
        )
        p.add_tools(hover)
        return p

    @staticmethod
    def _render_scatter(p: figure, ds: ColumnDataSource) -> GlyphRenderer:
        """Render colored QQ scatter.

        Args:
            p: Bokeh Figure.
            ds: QQ ColumnDataSource.

        Returns:
            GlyphRenderer for circles.
        """
        mapper = LinearColorMapper(palette=Viridis256, low=0, high=100)
        return p.circle(
            "x",
            "y",
            source=ds,
            radius=0.04,
            fill_color={"field": "percent", "transform": mapper},
            line_color=None,
        )

    @staticmethod
    def _render_identity(p: figure, ds: ColumnDataSource) -> GlyphRenderer:
        """Render the identity line.

        Args:
            p: Bokeh Figure.
            ds: Identity line ColumnDataSource.

        Returns:
            GlyphRenderer for line.
        """
        return p.line(
            "x",
            "y",
            source=ds,
            line_dash="dashed",
            line_color="black",
            line_width=2,
        )

    @staticmethod
    def _add_colorbar(p: figure) -> None:
        """Add a percentile colorbar on the right."""
        mapper = LinearColorMapper(palette=Viridis256, low=0, high=100)
        cb = ColorBar(color_mapper=mapper, label_standoff=12, border_line_color=None, location=(0, 0), title="Percentile")
        p.add_layout(cb, "right")

    @staticmethod
    def _configure_ranges(p: figure, ds: ColumnDataSource) -> None:
        """Auto-scale both axes with a small margin."""
        xs, ys = ds.data["x"], ds.data["y"]
        # Ensure xs and ys are flat lists of numbers
        xs_flat = np.asarray(xs).flatten()
        ys_flat = np.asarray(ys).flatten()
        vmin = float(np.min([*xs_flat, *ys_flat]))
        vmax = float(np.max([*xs_flat, *ys_flat]))
        m = (vmax - vmin) * 0.02
        p.x_range = Range1d(vmin - m, vmax + m)
        p.y_range = Range1d(vmin - m, vmax + m)

    @staticmethod
    def _add_annotation(p: figure) -> Label:
        """Annotate the region where B > A.

        Args:
            p: Bokeh Figure.

        Returns:
            The Label renderer.
        """
        lbl = Label(
            x=15,
            y=15,
            x_units="screen",
            y_units="screen",
            text="Above line → B > A",
            text_font_size="10pt",
            text_color="firebrick",
        )
        p.add_layout(lbl)
        return lbl
