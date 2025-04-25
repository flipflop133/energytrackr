"""Utility functions for builtin plots."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bokeh.layouts import column, row
from bokeh.models import AutocompleteInput, CustomJS
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context


def get_labels_and_dists(ctx: Context) -> tuple[list[str], list[list[float]]]:
    """Extracts and returns the labels and corresponding distributions from the given context.

    Args:
        ctx (Context): The context object containing statistical data and artefacts.

    Returns:
        tuple[list[str], list[list[float]]]:
            A tuple where the first element is a list of label strings (short hashes),
            and the second element is a list of distributions (each distribution is a list of floats).
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


def wrap_layout(sel1: AutocompleteInput, sel2: AutocompleteInput, fig: figure, key: str, ctx: Context) -> None:
    """Assembles column(row(sel1, sel2), fig) and stores under ctx.plots[key].

    Args:
        sel1 (AutocompleteInput): First selector.
        sel2 (AutocompleteInput): Second selector.
        fig (figure): Bokeh figure to display.
        key (str): Key to store the layout in ctx.plots.
        ctx (Any): Context object with plots attribute.
    """
    layout = column(row(sel1, sel2, sizing_mode="stretch_width"), fig, sizing_mode="stretch_width")
    ctx.plots[key] = layout
