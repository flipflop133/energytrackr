"""Reusable mixin classes for common plot behavior."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import AutocompleteInput, ColorBar, CustomJS, HoverTool, LinearColorMapper
from bokeh.models.annotations.labels import Title
from bokeh.models.layouts import Column
from bokeh.models.renderers import DataRenderer
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from energytrackr.plot.builtin_plots.base import BasePlot
from energytrackr.plot.config import get_settings
from energytrackr.plot.core.context import Context


def make_selectors(
    labels: list[str],
    js_code_path: str | Path,
    cb_args: Mapping[str, Any],
) -> tuple[AutocompleteInput, AutocompleteInput]:
    """Create two AutocompleteInput widgets wired to the same CustomJS callback.

    Args:
        labels (list[str]): List of labels to be used for the AutocompleteInput widgets.
        js_code_path (str | Path): Path to the JavaScript code file for the CustomJS callback.
        cb_args (Mapping[str, Any]): Additional arguments to be passed to the CustomJS callback.

    Returns:
        tuple[AutocompleteInput, AutocompleteInput]: A tuple containing two AutocompleteInput widgets.
    """
    # 1) build the two widgets
    sel1 = AutocompleteInput(
        title="Commit A",
        value=labels[0],
        completions=labels,
        width=300,
        case_sensitive=False,
        min_characters=1,
    )
    sel2 = AutocompleteInput(
        title="Commit B",
        value=labels[-1],
        completions=labels,
        width=300,
        case_sensitive=False,
        min_characters=1,
    )

    # 2) inject them into the callback args
    full_args = dict(cb_args)
    full_args["sel1"] = sel1
    full_args["sel2"] = sel2

    # 3) load JS and hook it up
    js_code = Path(js_code_path).read_text(encoding="utf-8")
    callback = CustomJS(args=full_args, code=js_code)
    sel1.js_on_change("value", callback)
    sel2.js_on_change("value", callback)

    return sel1, sel2


def get_labels_and_dists(ctx: Context) -> tuple[list[str], list[np.ndarray]]:
    """Extracts and returns the labels and corresponding distributions from the given context.

    Args:
        ctx (Context): The context object containing statistical data and artefacts.

    Returns:
        tuple[list[str], list[np.ndarray]]: A tuple containing a list of short hash labels and a list of distributions.
    """
    return ctx.stats["short_hashes"], ctx.artefacts["distributions"]


def initial_commits(labels: Sequence[str]) -> tuple[str, str]:
    """Returns the first and last elements from a sequence of label strings.

    Args:
        labels (Sequence[str]): A sequence of label strings.

    Returns:
        tuple[str, str]: A tuple containing the first and last label from the input sequence.
    """
    return labels[0], labels[-1]


class CommitSelectorsMixin:
    """Adds two Autocomplete selectors for commit A/B with JS callback wiring."""

    def _wrap_layout(self, fig: figure, ctx: Context) -> Column:
        labels = ctx.stats["short_hashes"]
        sel1, sel2 = make_selectors(
            labels=labels,
            js_code_path=self._callback_js_path(),
            cb_args=self._callback_args(fig, ctx),
        )
        return column(
            row(sel1, sel2, sizing_mode="stretch_width"),
            fig,
            sizing_mode="stretch_width",
        )

    @abstractmethod
    def _callback_js_path(self) -> str | Path:
        """Path to the JavaScript callback file."""

    def _callback_args(self, fig: figure, ctx: Context) -> Mapping[str, Any]:  # noqa: ARG002, PLR6301 # pylint: disable=unused-argument
        """Additional args passed to the JS callback (default: plot).

        Args:
            fig (figure): The Bokeh figure object.
            ctx (Context): The context object containing the data and configuration.

        Returns:
            Mapping[str, Any]: A dictionary of additional arguments for the JS callback.
        """
        return {"plot": fig}


class FontMixin:
    """Configures consistent fonts for axes and title."""

    def _configure(self, fig: figure, ctx: Context) -> None:
        super()._configure(fig, ctx)  # type: ignore[attr-defined]
        settings = get_settings()
        font = settings.energytrackr.report.font

        for ax in fig.xaxis + fig.yaxis:
            ax.axis_label_text_font = font
            ax.major_label_text_font = font

        title: Title | str | None = fig.title
        if isinstance(title, Title):
            title.text_font = font
            title.text_font_size = "16pt"
            title.align = "center"
            title.text_color = "black"


class ColorbarMixin:
    """Adds a Viridis palette colorbar on the right."""

    def _configure(self, fig: figure, ctx: Context) -> None:
        super()._configure(fig, ctx)  # type: ignore[attr-defined]
        mapper = LinearColorMapper(palette=Viridis256, low=0, high=100)
        colorbar = ColorBar(
            color_mapper=mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            title="Percentile",
        )
        fig.add_layout(colorbar, "right")


class HoverMixin:
    """Adds a hover tool to the first data renderer with custom tooltips."""

    def _configure(self, fig: figure, ctx: Context) -> None:
        super()._configure(fig, ctx)  # type: ignore[attr-defined]
        if not (renderers := [r for r in fig.renderers if isinstance(r, DataRenderer)]):
            return
        tool = HoverTool(
            tooltips=self._hover_tooltips(ctx),
            renderers=[renderers[0]],
        )
        fig.add_tools(tool)

    @abstractmethod
    def _hover_tooltips(self, ctx: Context) -> list[tuple[str, str]]:
        """Returns the tooltip list for the hover tool."""


class ComparisonBase(CommitSelectorsMixin, FontMixin, HoverMixin, BasePlot):
    """Abstract base for any “compare-two commits” plot."""


def draw_additional_objects(objects: list[str], fig: figure, ctx: Context) -> None:
    """Draw additional plot objects on the figure.

    Args:
        objects (list[str]): List of object names to be drawn.
        fig (figure): The Bokeh figure object.
        ctx (Context): The context object containing the data and configuration.
    """
    for name in objects:
        ctx.plot_objects[name].add(ctx, fig)
