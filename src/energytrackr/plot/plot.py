"""Plotting module for energy consumption data analysis (refactored, strongly typed)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, Column, ColumnDataSource, CustomJS, FixedTicker, HoverTool, Range1d, Toggle
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting import figure

from energytrackr.plot.data import ChangeEvent, EnergyData, EnergyStats, nice_number

DataDictLike = Mapping[str, Sequence[int | float | str]]
DEFAULT_MAX_TICKS: int = 30


@dataclass
class ChartRenderers:
    """Container for Bokeh renderers used in the EnergyPlot class."""

    median_renderer: GlyphRenderer
    error_renderer: GlyphRenderer
    norm_renderer: GlyphRenderer
    nonnorm_renderer: GlyphRenderer
    candle_body: GlyphRenderer
    candle_wick: GlyphRenderer


class EnergyPlot:
    """EnergyPlot is a class for visualizing energy consumption trends over commits (refactored, strongly typed)."""

    energy_column: str
    stats: EnergyStats
    distribution: list[np.ndarray]
    normality_flags: list[bool]
    change_events: list[ChangeEvent]
    fig: figure | None

    def __init__(self, energy_column: str, stats: EnergyStats, energy_data: EnergyData) -> None:
        """Initializes the EnergyPlot with energy column, statistics, distribution data, and change events."""
        self.energy_column = energy_column
        self.stats = stats
        self.fig = None
        self.change_events = energy_data.change_events[self.energy_column]
        self.distribution = energy_data.distributions[self.energy_column]
        self.normality_flags = energy_data.normality_flags[self.energy_column]

    def _make_change_event_sources(self) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        regression: DataDictLike = {"x": [], "y": [], "commit": [], "severity": [], "cohen_d": []}
        improvement: DataDictLike = {"x": [], "y": [], "commit": [], "severity": [], "cohen_d": []}
        for event in self.change_events:
            idx: int = event.index
            target: dict[str, list[Any]] = regression if event.direction == "increase" else improvement
            target["x"].append(idx)
            target["y"].append(self.stats.y_medians[idx])
            target["commit"].append(self.stats.short_hashes[idx])
            target["severity"].append(f"{int(event.severity * 100)}%")
            target["cohen_d"].append(f"{event.cohen_d:.2f}")
        return regression, improvement

    def create_figure(self) -> Column:
        """Assembles the full Bokeh layout by delegating to helper methods.

        Returns:
            Column: The Bokeh layout containing the plot and toggle.
        """
        x_min: int = 0
        x_max: int = len(self.stats.short_hashes) - 1
        fig: figure = self._init_figure(x_min, x_max)

        median_renderer, error_renderer = self._plot_median_line_and_errors(fig)
        norm_renderer, nonnorm_renderer = self._plot_distributions(fig)
        self._plot_change_event_markers(fig)
        self._add_change_event_annotations(fig)
        candle_body, candle_wick = self._plot_candlestick(fig)

        for legend in fig.legend:
            legend.click_policy = "hide"

        self._setup_ticks(fig, x_min, x_max)
        self._add_hover_tool(fig, median_renderer)

        renderers = ChartRenderers(
            median_renderer=median_renderer,
            error_renderer=error_renderer,
            norm_renderer=norm_renderer,
            nonnorm_renderer=nonnorm_renderer,
            candle_body=candle_body,
            candle_wick=candle_wick,
        )
        toggle: Toggle = self._create_toggle(renderers)
        layout: Column = column(toggle, fig, sizing_mode="stretch_width")
        return layout

    def _init_figure(self, x_min: int, x_max: int) -> figure:
        """Initializes the base figure with axes, labels, and ranges.

        Args:
            x_min (int): Minimum x-axis value.
            x_max (int): Maximum x-axis value.

        Returns:
            Figure: The initialized Bokeh figure.
        """
        fig: figure = figure(
            title=f"Energy Consumption Trend - {self.energy_column}",
            x_range=Range1d(x_min, x_max),
            tools="pan,box_zoom,reset,save,wheel_zoom",
            toolbar_location="above",
            sizing_mode="stretch_width",
            height=400,
        )
        for ax in fig.xaxis:
            ax.axis_label = "Commit (oldest → newest)"
        for ax in fig.yaxis:
            ax.axis_label = f"Median {self.energy_column} (J)"
        for ax in fig.xaxis:
            ax.major_label_orientation = pi / 4
            ax.major_label_overrides = dict(enumerate(self.stats.short_hashes))
        return fig

    def _plot_median_line_and_errors(self, fig: figure) -> tuple[GlyphRenderer, GlyphRenderer]:
        """Plots median points and error bars, returns their renderers.

        Args:
            fig (Figure): The Bokeh figure to plot on.

        Returns:
            tuple[GlyphRenderer, GlyphRenderer]: The median and error bar renderers.
        """
        xs: list[int] = list(range(len(self.stats.short_hashes)))
        median_source: ColumnDataSource = ColumnDataSource(
            data={"x": xs, "y": self.stats.y_medians, "commit": self.stats.short_hashes},
        )
        fig.line(
            "x",
            "y",
            source=median_source,
            color="blue",
            line_width=1,
            legend_label=f"Median Line ({self.energy_column})",
        )
        median_renderer: GlyphRenderer = fig.circle(
            "x",
            "y",
            source=median_source,
            radius=0.1,
            color="blue",
            legend_label=f"Median ({self.energy_column})",
        )
        lower: np.ndarray = np.array(self.stats.y_medians) - np.array(self.stats.y_errors)
        upper: np.ndarray = np.array(self.stats.y_medians) + np.array(self.stats.y_errors)
        error_source: ColumnDataSource = ColumnDataSource(data={"x": xs, "y_lower": lower, "y_upper": upper})
        error_renderer: GlyphRenderer = fig.segment(
            "x",
            "y_lower",
            "x",
            "y_upper",
            source=error_source,
            line_width=2,
            legend_label="Error Bars",
        )
        return median_renderer, error_renderer

    def _plot_distributions(self, fig: figure) -> tuple[GlyphRenderer, GlyphRenderer]:
        """Plots scatter for normal and non-normal distributions.

        Args:
            fig (Figure): The Bokeh figure to plot on.

        Returns:
            tuple[GlyphRenderer, GlyphRenderer]: The normal and non-normal distribution renderers.
        """
        normal_x: list[int] = []
        normal_y: list[float] = []
        nonnorm_x: list[int] = []
        nonnorm_y: list[float] = []
        for i, vals in enumerate(self.distribution):
            for v in vals:
                if self.normality_flags[i]:
                    normal_x.append(i)
                    normal_y.append(v)
                else:
                    nonnorm_x.append(i)
                    nonnorm_y.append(v)
        norm_renderer: GlyphRenderer = fig.circle(
            x=normal_x,
            y=normal_y,
            radius=0.3,
            alpha=0.5,
            legend_label="Normal",
            visible=False,
        )
        nonnorm_renderer: GlyphRenderer = fig.circle(
            x=nonnorm_x,
            y=nonnorm_y,
            radius=0.3,
            alpha=0.5,
            color="orange",
            legend_label="Non-Normal",
        )
        return norm_renderer, nonnorm_renderer

    def _plot_change_event_markers(self, fig: figure) -> None:
        """Adds circle markers for regression and improvement events."""
        regression, improvement = self._make_change_event_sources()
        if regression["x"]:
            reg_source: ColumnDataSource = ColumnDataSource(data=regression)
            fig.circle("x", "y", source=reg_source, radius=1, alpha=0.6, color="red", legend_label="Regression (↑ energy)")
        if improvement["x"]:
            imp_source: ColumnDataSource = ColumnDataSource(data=improvement)
            fig.circle("x", "y", source=imp_source, radius=1, alpha=0.6, color="green", legend_label="Improvement (↓ energy)")

    def _plot_candlestick(self, fig: figure) -> tuple[GlyphRenderer, GlyphRenderer]:
        """Plots a candlestick chart, returns body and wick renderers.

        Args:
            fig (Figure): The Bokeh figure to plot on.

        Returns:
            tuple[GlyphRenderer, GlyphRenderer]: The body and wick renderers of the candlestick chart.
        """
        xs: list[int] = list(range(len(self.stats.short_hashes)))
        medians: list[float] = self.stats.y_medians
        opens: list[float] = [medians[0], *medians[:-1]]
        closes: list[float] = medians
        lows: list[float] = [float(np.min(dist)) for dist in self.distribution]
        highs: list[float] = [float(np.max(dist)) for dist in self.distribution]
        candle_width: float = 0.6
        colors: list[str] = ["green" if close < open_ else "red" for open_, close in zip(opens, closes, strict=True)]
        body: GlyphRenderer = fig.quad(
            top=[max(o, c) for o, c in zip(opens, closes, strict=True)],
            bottom=[min(o, c) for o, c in zip(opens, closes, strict=True)],
            left=[x - candle_width / 2 for x in xs],
            right=[x + candle_width / 2 for x in xs],
            fill_color=colors,
            line_color="black",
            legend_label="Candlestick (body)",
            visible=False,
        )
        wick: GlyphRenderer = fig.segment(
            x0=xs,
            x1=xs,
            y0=lows,
            y1=highs,
            line_width=1,
            legend_label="Candlestick (wick)",
            visible=False,
        )
        return body, wick

    def _add_change_event_annotations(self, fig: figure) -> None:
        """Shades background for significant change events."""
        for e in self.change_events:
            idx: int = e.index
            box: BoxAnnotation = BoxAnnotation(
                left=idx - 0.4,
                right=idx + 0.4,
                fill_color="red" if e.direction == "increase" else "green",
                fill_alpha=0.15 + min(float(e.severity), 0.5),
            )
            fig.add_layout(box)

    def _setup_ticks(self, fig: figure, x_min: int, x_max: int) -> None:
        """Configures dynamic x-axis ticks."""
        full_ticks: list[int] = list(range(len(self.stats.short_hashes)))
        ticker: FixedTicker = FixedTicker(ticks=full_ticks)
        for ax in fig.xaxis:
            ax.ticker = ticker
        raw_step: float = (x_max - x_min) / DEFAULT_MAX_TICKS
        step: int = max(1, int(nice_number(raw_step)))
        ticker.ticks = list(range(x_min, x_max + 1, step))
        callback: CustomJS = CustomJS(
            args={"ticker": ticker, "full_length": len(self.stats.short_hashes) - 1, "max_ticks": DEFAULT_MAX_TICKS},
            code="""
                var start = cb_obj.start;
                var end = cb_obj.end;
                var range_visible = end - start;
                function niceNumber(x) {
                    var exponent = Math.floor(Math.log(x) / Math.LN10);
                    var fraction = x / Math.pow(10, exponent);
                    var niceFraction;
                    if (fraction < 1.5) { niceFraction = 1; }
                    else if (fraction < 3) { niceFraction = 2; }
                    else if (fraction < 7) { niceFraction = 5; }
                    else { niceFraction = 10; }
                    return niceFraction * Math.pow(10, exponent);
                }
                var rawStep = range_visible / max_ticks;
                var step = Math.max(1, niceNumber(rawStep));
                var new_ticks = [];
                for (var t = 0; t <= full_length; t += step) {
                    if (t >= start && t <= end) new_ticks.push(t);
                }
                ticker.ticks = new_ticks;
            """,
        )
        fig.x_range.js_on_change("start", callback)
        fig.x_range.js_on_change("end", callback)

    @staticmethod
    def _add_hover_tool(fig: figure, median_renderer: GlyphRenderer) -> None:
        """Adds hover tooltip for median points."""
        hover: HoverTool = HoverTool(
            renderers=[median_renderer],
            tooltips=[("Commit", "@commit"), ("Median", "@y")],
        )
        fig.add_tools(hover)

    @staticmethod
    def _create_toggle(
        renderers: ChartRenderers,
    ) -> Toggle:
        """Creates and returns a toggle widget to switch chart views.

        Args:
            renderers (ChartRenderers): The renderers for median, error, and candlestick charts.

        Returns:
            Toggle: The toggle widget for switching between views.
        """
        toggle: Toggle = Toggle(label="Switch to Candlestick", button_type="success", active=False)
        callback: CustomJS = CustomJS(
            args={
                "median_renderer": renderers.median_renderer,
                "error_renderer": renderers.error_renderer,
                "norm_renderer": renderers.norm_renderer,
                "nonnorm_renderer": renderers.nonnorm_renderer,
                "candle_body": renderers.candle_body,
                "candle_wick": renderers.candle_wick,
                "toggle_widget": toggle,
            },
            code="""
                if(cb_obj.active) {
                    median_renderer.visible = false;
                    error_renderer.visible = false;
                    norm_renderer.visible = false;
                    nonnorm_renderer.visible = false;
                    candle_body.visible = true;
                    candle_wick.visible = true;
                    toggle_widget.label = "Switch to Line Chart";
                } else {
                    median_renderer.visible = true;
                    error_renderer.visible = true;
                    norm_renderer.visible = true;
                    nonnorm_renderer.visible = true;
                    candle_body.visible = false;
                    candle_wick.visible = false;
                    toggle_widget.label = "Switch to Candlestick";
                }
            """,
        )
        toggle.js_on_change("active", callback)
        toggle.width_policy = "max"
        return toggle
