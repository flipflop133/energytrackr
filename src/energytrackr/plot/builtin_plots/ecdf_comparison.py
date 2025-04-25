"""ECDF comparison plot for two commits."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Range1d, Select
from bokeh.models.renderers import GlyphRenderer
from bokeh.palettes import Category10
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.plot_interface import Plot
from energytrackr.plot.core.context import Context


class ECDFComparison(Plot):
    """Interactive ECDF comparison between two commits."""

    def build(self, ctx: Context) -> None:  # noqa: PLR6301
        """Build the ECDF comparison layout in the given context.

        Args:
            ctx: Plotting context with stats, artefacts, energy_fields, and plots dict.
        """
        labels = ctx.stats["short_hashes"]
        full_ecdf, (x_min, x_max) = ECDFComparison._compute_ecdf(ctx.artefacts["distributions"])
        sel1, sel2 = ECDFComparison._create_widgets(labels)
        p = ECDFComparison._create_figure(ctx.energy_fields[0], x_min, x_max)

        raw_a = ECDFComparison._make_ecdf_ds(full_ecdf, 0)
        raw_b = ECDFComparison._make_ecdf_ds(full_ecdf, -1)

        ds_a = ColumnDataSource(pd.DataFrame({"x": raw_a["x"], "y": raw_a["y"]}))
        ds_b = ColumnDataSource(pd.DataFrame({"x": raw_b["x"], "y": raw_b["y"]}))
        palette = Category10[3]
        ECDFComparison._render_steps(p, ds_a, palette[0], "Commit A")
        ECDFComparison._render_steps(p, ds_b, palette[1], "Commit B")

        ECDFComparison._configure_plot(p)

        args = {
            "full_ecdf": full_ecdf,
            "src_a": ds_a,
            "src_b": ds_b,
            "sel1": sel1,
            "sel2": sel2,
            "plot": p,
            "labels": labels,
        }
        ECDFComparison._attach_callback(args)

        ctx.plots["ecdf_comparison"] = column(
            row(sel1, sel2, sizing_mode="stretch_width"),
            p,
            sizing_mode="stretch_width",
        )

    @staticmethod
    def _compute_ecdf(dists: Sequence[Sequence[float]]) -> tuple[ColumnDataSource, tuple[float, float]]:
        """Compute ECDF coordinates and global x-range.

        Args:
            dists: List of numeric arrays.

        Returns:
            Tuple of ColumnDataSource with 'ecdf_x' and 'ecdf_y', and (min, max) of all values.
        """
        xs_list: list[list[float]] = []
        ys_list: list[list[float]] = []
        for arr in dists:
            a = np.sort(np.asarray(arr, float))
            n = len(a)

            # cast to tell Pylance “this really is list[float]”
            xs = cast(list[float], a.tolist())
            ys = cast(list[float], (np.arange(1, n + 1) / n).tolist())

            xs_list.append(xs)
            ys_list.append(ys)

        all_vals = np.concatenate([np.asarray(arr, float) for arr in dists])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        src = ColumnDataSource({"ecdf_x": xs_list, "ecdf_y": ys_list})
        return src, (vmin, vmax)

    @staticmethod
    def _create_widgets(labels: list[str | None]) -> tuple[Select, Select]:
        """Create dropdown selectors for commits.

        Args:
            labels: List of commit identifiers.

        Returns:
            Two Select widgets for A and B.
        """
        sel1 = Select(title="Commit A", value=labels[0], options=labels)
        sel2 = Select(title="Commit B", value=labels[-1], options=labels)
        return sel1, sel2

    @staticmethod
    def _create_figure(field: str, x_min: float, x_max: float) -> figure:
        """Initialize the Bokeh figure for ECDF.

        Args:
            field: Energy field name for axis label.
            x_min: Minimum x-range.
            x_max: Maximum x-range.

        Returns:
            Configured Bokeh Figure with ECDF axes.
        """
        p = figure(
            x_range=Range1d(start=x_min, end=x_max),
            sizing_mode="stretch_width",
            title=f"Empirical CDF: {field}",
            x_axis_label=f"{field} (J)",
            y_axis_label="ECDF",
        )
        return p

    @staticmethod
    def _make_ecdf_ds(full_ecdf: ColumnDataSource, idx: int) -> dict[str, list[float]]:
        """Extract ECDF x and y for a given index.

        Args:
            full_ecdf: Source with 'ecdf_x' and 'ecdf_y'.
            idx: Distribution index.

        Returns:
            Dict with 'x' and 'y' lists.
        """
        return {
            "x": full_ecdf.data["ecdf_x"][idx],
            "y": full_ecdf.data["ecdf_y"][idx],
        }

    @staticmethod
    def _render_steps(p: figure, ds: ColumnDataSource, color: str, legend: str) -> GlyphRenderer:
        """Render an ECDF step line.

        Args:
            p: Bokeh Figure.
            ds: ECDF ColumnDataSource.
            color: Line color.
            legend: Legend label.

        Returns:
            GlyphRenderer for the step.
        """
        return p.step(x="x", y="y", source=ds, mode="after", line_width=2, line_color=color, legend_label=legend)

    @staticmethod
    def _configure_plot(p: figure) -> None:
        """Configure plot aesthetics like legend location.

        Args:
            p: Bokeh Figure.
        """
        p.legend[0].location = "bottom_right"

    @staticmethod
    def _attach_callback(args: dict[str, Any]) -> None:
        """Attach JS callback to update ECDFs when selection changes.

        Args:
            args: Mapping of callback arguments.
        """
        code = Path(__file__).parent.joinpath("static", "bootstrap_comparison.js").read_text(encoding="utf-8")
        cb = CustomJS(args=args, code=code)
        args["sel1"].js_on_change("value", cb)
        args["sel2"].js_on_change("value", cb)
