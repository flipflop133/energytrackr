"""Plotting module for energy consumption data analysis."""

from math import pi
from typing import Any

import numpy as np
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, ColumnDataSource, CustomJS, FixedTicker, HoverTool, Toggle
from bokeh.models.layouts import Column
from bokeh.plotting import figure

from .data import ChangeEvent, EnergyStats, nice_number

DEFAULT_MAX_TICKS = 30


class EnergyPlot:
    """EnergyPlot is a class for visualizing energy consumption trends over a series of commits.

    It provides functionality to create interactive plots, including line charts, candlestick charts,
    and annotations for significant change events.

    Attributes:
        energy_column (str): The name of the energy column being analyzed.
        stats (EnergyStats): An object containing statistical data such as medians, errors, and commit hashes.
        distribution (list[np.ndarray]): A list of distributions for each commit.
        normality_flags (list[bool]): Flags indicating whether the distribution for each commit is normal.
        change_events (list[ChangeEvent]): A list of significant change events (e.g., regressions or improvements).
        figure (bokeh.plotting.figure.Figure): The Bokeh figure object for the plot.

    Methods:
        _make_change_event_sources():
            Creates data sources for regression and improvement events to be plotted on the chart.

        create_figure():
            Creates and returns a Bokeh layout containing the energy consumption trend plot.
            Includes line charts, candlestick charts, and interactive widgets.

        _add_change_event_annotations(fig):
            Adds shaded box annotations to the plot to highlight significant change events.

        _setup_ticks(fig, x_min, x_max):
            Configures the x-axis ticks for the plot, ensuring they adapt dynamically to the visible range.
    """

    def __init__(
        self,
        energy_column: str,
        stats: EnergyStats,
        distribution: list[np.ndarray],
        normality_flags: list[bool],
        change_events: list[ChangeEvent],
    ) -> None:
        """Initializes the EnergyPlot instance."""
        self.energy_column = energy_column
        self.stats = stats
        self.distribution = distribution
        self.normality_flags = normality_flags
        self.change_events = change_events
        self.figure = None

    def _make_change_event_sources(self) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        regression = {"x": [], "y": [], "commit": [], "severity": [], "cohen_d": []}
        improvement = {"x": [], "y": [], "commit": [], "severity": [], "cohen_d": []}
        for event in self.change_events:
            idx = event.index
            if event.direction == "increase":
                regression["x"].append(idx)
                regression["y"].append(self.stats.y_medians[idx])
                regression["commit"].append(self.stats.short_hashes[idx])
                regression["severity"].append(f"{int(event.severity * 100)}%")
                regression["cohen_d"].append(f"{event.cohen_d:.2f}")
            else:
                improvement["x"].append(idx)
                improvement["y"].append(self.stats.y_medians[idx])
                improvement["commit"].append(self.stats.short_hashes[idx])
                improvement["severity"].append(f"{int(event.severity * 100)}%")
                improvement["cohen_d"].append(f"{event.cohen_d:.2f}")
        return regression, improvement

    def create_figure(self) -> Column:
        """Creates a Bokeh figure visualizing energy consumption trends with multiple chart types.

        The figure includes:
        - A line chart showing median energy consumption with error bars.
        - Scatter plots for normal and non-normal data distributions.
        - Markers for regression and improvement events in energy consumption.
        - A candlestick chart representing open-high-low-close (OHLC) data for energy consumption.

        Additionally, a toggle widget is provided to switch between the line chart and candlestick chart views.

        Returns:
            Column: A Bokeh layout containing the toggle widget and the figure.
        """
        x_min, x_max = 0, len(self.stats.short_hashes) - 1
        fig = figure(
            title=f"Energy Consumption Trend - {self.energy_column}",
            x_range=(x_min, x_max),
            tools="pan,box_zoom,reset,save,wheel_zoom",
            toolbar_location="above",
            sizing_mode="stretch_width",
            height=400,
        )
        fig.xaxis.axis_label = "Commit (oldest → newest)"
        fig.yaxis.axis_label = f"Median {self.energy_column} (J)"
        fig.xaxis.major_label_orientation = pi / 4
        fig.xaxis.major_label_overrides = {i: short for i, short in enumerate(self.stats.short_hashes)}

        # -- LINE CHART: plot median points and error bars --
        median_source = ColumnDataSource(
            data={
                "x": list(range(len(self.stats.short_hashes))),
                "y": self.stats.y_medians,
                "commit": self.stats.short_hashes,
            },
        )
        median_renderer = fig.circle(
            "x",
            "y",
            source=median_source,
            radius=0.1,
            color="blue",
            legend_label=f"Median ({self.energy_column})",
        )
        lower = np.array(self.stats.y_medians) - np.array(self.stats.y_errors)
        upper = np.array(self.stats.y_medians) + np.array(self.stats.y_errors)
        error_source = ColumnDataSource(
            data={"x": list(range(len(self.stats.short_hashes))), "y_lower": lower, "y_upper": upper},
        )
        error_renderer = fig.segment(
            "x",
            "y_lower",
            "x",
            "y_upper",
            source=error_source,
            line_width=2,
            color="black",
            legend_label="Error Bars",
            visible=True,  # Adjust visibility as desired
        )

        # -- LINE CHART: plot distributions --
        normal_x, normal_y, nonnormal_x, nonnormal_y = [], [], [], []
        for i, values in enumerate(self.distribution):
            for val in values:
                if self.normality_flags[i]:
                    normal_x.append(i)
                    normal_y.append(val)
                else:
                    nonnormal_x.append(i)
                    nonnormal_y.append(val)
        norm_renderer = fig.circle(
            x=normal_x,
            y=normal_y,
            radius=0.3,
            color="lightgray",
            alpha=0.5,
            legend_label="Normal",
            visible=False,
        )
        nonnorm_renderer = fig.circle(
            x=nonnormal_x,
            y=nonnormal_y,
            radius=0.3,
            color="lightcoral",
            alpha=0.5,
            legend_label="Non-Normal",
        )

        # -- LINE CHART: plot change events --
        regression, improvement = self._make_change_event_sources()
        if regression["x"]:
            reg_source = ColumnDataSource(regression)
            fig.circle("x", "y", source=reg_source, color="red", radius=1, alpha=0.6, legend_label="Regression (↑ energy)")
        if improvement["x"]:
            imp_source = ColumnDataSource(improvement)
            fig.circle("x", "y", source=imp_source, color="green", radius=1, alpha=0.6, legend_label="Improvement (↓ energy)")

        # -- Add change event annotations --
        self._add_change_event_annotations(fig)
        self._setup_ticks(fig, x_min, x_max)

        # -- Add hover for medians --
        hover = HoverTool(renderers=[median_renderer], tooltips=[("Commit", "@commit"), ("Median", "@y")])
        fig.add_tools(hover)

        # -- CANDLESTICK CHART: compute OHLC from the distributions --
        # -- CANDLESTICK CHART: compute OHLC from medians and distributions --
        xs = list(range(len(self.stats.short_hashes)))
        medians = self.stats.y_medians

        # Compute open: for commit 0 use its own median; for later commits, use the previous commit's median.
        opens = [medians[0]] + medians[:-1]
        closes = medians

        # Low and high from the distribution of each commit.
        lows = [np.min(dist) for dist in self.distribution]
        highs = [np.max(dist) for dist in self.distribution]
        candle_width = 0.6

        # Determine candle colors based on whether energy consumption improved (lower energy) or regressed.
        # (Assuming lower energy is better, so close < open means improvement → green)
        candle_colors = ["green" if close < open else "red" for open, close in zip(opens, closes, strict=False)]

        # The candle body: the quad is drawn from the open to the close, colored accordingly.
        candle_body = fig.quad(
            top=[max(o, c) for o, c in zip(opens, closes, strict=False)],
            bottom=[min(o, c) for o, c in zip(opens, closes, strict=False)],
            left=[x - candle_width / 2 for x in xs],
            right=[x + candle_width / 2 for x in xs],
            fill_color=candle_colors,
            line_color="black",
            legend_label="Candlestick (body)",
            visible=False,
        )

        # The wick: drawn as a segment from the low to the high.
        candle_wick = fig.segment(
            x0=xs,
            x1=xs,
            y0=lows,
            y1=highs,
            color="black",
            line_width=1,
            legend_label="Candlestick (wick)",
            visible=False,
        )

        # -- Add a Toggle widget to switch between line and candlestick views --
        toggle = Toggle(label="Switch to Candlestick", button_type="success", active=False)
        toggle_callback = CustomJS(
            args=dict(
                median_renderer=median_renderer,
                error_renderer=error_renderer,
                norm_renderer=norm_renderer,
                nonnorm_renderer=nonnorm_renderer,
                candle_body=candle_body,
                candle_wick=candle_wick,
                toggle_widget=toggle,
            ),
            code="""
                // When active==true, we switch to candlestick view.
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
        toggle.js_on_change("active", toggle_callback)

        self.figure = fig

        # Ensure toggle doesn't constrain the figure's full width
        toggle.width_policy = "max"
        layout = column(toggle, fig, sizing_mode="stretch_width")
        return layout

    def _add_change_event_annotations(self, fig: figure) -> None:
        for event in self.change_events:
            idx = event.index
            box = BoxAnnotation(
                left=idx - 0.4,
                right=idx + 0.4,
                fill_color="red" if event.direction == "increase" else "green",
                fill_alpha=0.15 + min(event.severity, 0.5),
            )
            fig.add_layout(box)

    def _setup_ticks(self, fig: figure, x_min: int, x_max: int) -> None:
        full_ticks = list(range(len(self.stats.short_hashes)))
        ticker = FixedTicker(ticks=full_ticks)
        fig.xaxis.ticker = ticker
        raw_step = (x_max - x_min) / DEFAULT_MAX_TICKS
        step = max(1, int(nice_number(raw_step)))
        new_ticks = list(range(x_min, x_max + 1, step))
        ticker.ticks = new_ticks

        callback = CustomJS(
            args=dict(ticker=ticker, full_length=len(self.stats.short_hashes) - 1, max_ticks=DEFAULT_MAX_TICKS),
            code="""
                var start = cb_obj.start;
                var end = cb_obj.end;
                var range_visible = end - start;
                function niceNumber(x) {
                    var exponent = Math.floor(Math.log(x) / Math.LN10);
                    var fraction = x / Math.pow(10, exponent);
                    var niceFraction;
                    if (fraction < 1.5) {
                        niceFraction = 1;
                    } else if (fraction < 3) {
                        niceFraction = 2;
                    } else if (fraction < 7) {
                        niceFraction = 5;
                    } else {
                        niceFraction = 10;
                    }
                    return niceFraction * Math.pow(10, exponent);
                }
                var rawStep = range_visible / max_ticks;
                var step = Math.max(1, niceNumber(rawStep));
                var new_ticks = [];
                for (var t = 0; t <= full_length; t += step) {
                    if (t >= start && t <= end) {
                        new_ticks.push(t);
                    }
                }
                ticker.ticks = new_ticks;
            """,
        )
        fig.x_range.js_on_change("start", callback)
        fig.x_range.js_on_change("end", callback)
