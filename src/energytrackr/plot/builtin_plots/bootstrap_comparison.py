"""BootstrapComparison using BasePlot and mixins for cleaner composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from bokeh.models import ColumnDataSource, Label, Span
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.mixins import ComparisonBase, get_labels_and_dists
from energytrackr.plot.builtin_plots.registry import register_plot
from energytrackr.plot.core.context import Context


class BootStats(NamedTuple):
    """Statistical summary for bootstrap CI histogram."""

    counts: list[int]
    lefts: list[float]
    rights: list[float]
    low_ci: float
    high_ci: float


@register_plot
class BootstrapComparison(ComparisonBase):
    """Interactive bootstrap CI histogram for Δ-median between two commits."""

    def __init__(self) -> None:
        """Initialize the BootstrapComparison plot."""
        super().__init__()
        self._low_span: Span | None = None
        self._high_span: Span | None = None
        self._low_label: Label | None = None
        self._high_label: Label | None = None
        self._sources: dict[str, ColumnDataSource] | None = None

    def _make_sources(self, ctx: Context) -> dict[str, Any]:
        labels, dists = get_labels_and_dists(ctx)
        stats = self._compute_boot_stats(dists)

        raw = ColumnDataSource({"commit": labels, "raw": dists})
        hist = ColumnDataSource({
            "top": stats.counts,
            "left": stats.lefts,
            "right": stats.rights,
        })
        ci = ColumnDataSource({"low": [stats.low_ci], "high": [stats.high_ci]})

        # store for JS callbacks
        self._sources = {
            "raw": raw,
            "hist": hist,
            "ci": ci,
        }
        return self._sources

    def _draw_glyphs(self, fig: figure, sources: dict[str, ColumnDataSource], ctx: Context) -> None:  # noqa: ARG002
        # histogram
        fig.quad(
            top="top",
            bottom=0,
            left="left",
            right="right",
            source=sources["hist"],
            fill_alpha=0.5,
            line_color="black",
        )
        # CI spans
        low = sources["ci"].data["low"][0]
        high = sources["ci"].data["high"][0]
        self._low_span = Span(location=low, dimension="height", line_color="firebrick", line_dash="dashed", line_width=2)
        self._high_span = Span(location=high, dimension="height", line_color="firebrick", line_dash="dashed", line_width=2)
        fig.add_layout(self._low_span)
        fig.add_layout(self._high_span)
        # CI labels
        y_max = max(sources["hist"].data["top"]) * 0.9
        self._low_label = Label(
            x=low,
            y=y_max,
            x_units="data",
            y_units="data",
            text=f"2.5%: {low:.2f}%",
            text_align="center",
            background_fill_alpha=0.7,
        )
        self._high_label = Label(
            x=high,
            y=y_max,
            x_units="data",
            y_units="data",
            text=f"97.5%: {high:.2f}%",
            text_align="center",
            background_fill_alpha=0.7,
        )
        fig.add_layout(self._low_label)
        fig.add_layout(self._high_label)

    def _title(self, ctx: Context) -> str:  # noqa: PLR6301
        return f"Bootstrap Δ-Median CI: {ctx.energy_fields[0]}"

    def _key(self, ctx: Context) -> str:  # noqa: ARG002, PLR6301
        return "Bootstrap"

    def _callback_js_path(self) -> Path:  # noqa: PLR6301
        return Path(__file__).parent / "static" / "bootstrap_comparison.js"

    def _callback_args(self, fig: figure, ctx: Context) -> dict[str, Any]:  # noqa: ARG002
        assert self._sources is not None
        self._sources.update()
        return {
            **self._sources,
            "low_span": self._low_span,
            "high_span": self._high_span,
            "low_label": self._low_label,
            "high_label": self._high_label,
            "plot": fig,
        }

    def _hover_tooltips(self, ctx: Context) -> list[tuple[str, str]]:  # noqa: ARG002, PLR6301
        return [
            ("Bin start", "@left{0.00}%"),
            ("Count", "@top"),
        ]

    @staticmethod
    def _compute_boot_stats(dists: list[np.ndarray], num_bootstraps: int = 1000) -> BootStats:
        arr1, arr2 = np.asarray(dists[0], float), np.asarray(dists[-1], float)
        diffs = []
        for _ in range(num_bootstraps):
            m1 = np.median(np.random.choice(arr1, len(arr1), replace=True))
            m2 = np.median(np.random.choice(arr2, len(arr2), replace=True))
            diffs.append((m2 / m1 - 1) * 100)
        counts, edges = np.histogram(diffs, bins="sturges")
        low_ci, high_ci = np.percentile(diffs, [2.5, 97.5])

        return BootStats(
            counts=[int(c) for c in counts],
            lefts=edges[:-1].tolist(),
            rights=edges[1:].tolist(),
            low_ci=low_ci,
            high_ci=high_ci,
        )
