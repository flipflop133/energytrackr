"""Stable abstract interfaces for plug-ins (analysis, plot, page).

Other parts of the application import these **only for type-checking**, never
for behaviour; every concrete implementation lives in a plug-in module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models.layouts import Column
from bokeh.plotting import figure
from jinja2 import Environment

from energytrackr.plot.core.context import Context
from energytrackr.utils.exceptions import NotADataclassTypeError, UnexpectedArgumentError


# Define a type-safe protocol for dataclass types
@runtime_checkable
class DataclassType(Protocol):
    """Protocol for a dataclass type."""

    __dataclass_fields__: dict


ConfigT = TypeVar("ConfigT", bound=DataclassType)


class Configurable[ConfigT]:
    """Base class for objects that can be configured with a dataclass."""

    def __init__(self, config_cls: type[ConfigT], **params: dict[str, Any]) -> None:
        """Initialize the object with a configuration dataclass and parameters.

        Raises:
            NotADataclassTypeError: If *config_cls* is not a dataclass type.
            UnexpectedArgumentError: If any parameter is not part of the configuration dataclass.
        """
        if not is_dataclass(config_cls):
            raise NotADataclassTypeError(config_cls)
        valid_keys = {f.name for f in fields(config_cls)}
        if invalid := set(params) - valid_keys:
            raise UnexpectedArgumentError(list(invalid), self.__class__.__name__)
        self.config: ConfigT = config_cls(**params)


class Transform(ABC):
    """A data-processing step executed in sequence."""

    @abstractmethod
    def apply(self, ctx: Context) -> None:
        """Mutate *ctx* in-place or add artefacts used by later stages.

        Args:
            ctx (Context): The context object containing the energy data and other artefacts.
        """
        raise NotImplementedError


class PlotObj(ABC):
    """Adds glyphs/renderers to a Bokeh *Figure*."""

    @abstractmethod
    def add(self, ctx: Context, fig: figure) -> None:
        """Mutate *ctx* in-place or add artefacts used by later stages.

        Args:
            ctx (Context): The context object containing the energy data and other artefacts.
            fig (figure): The Bokeh figure object to which the plot elements will be added.
        """
        raise NotImplementedError


class PageObj(ABC):
    """Produces an HTML snippet rendered into the final report."""

    @abstractmethod
    def render(self, env: Environment, ctx: Context) -> str:
        """Render the HTML snippet using the provided Jinja2 environment.

        Args:
            env (Environment): The Jinja2 environment for rendering.
            ctx (Context): The context object containing the energy data and other artefacts.

        Returns:
            str: The rendered HTML snippet.
        """
        raise NotImplementedError


class BasePlot(ABC):
    """Canonical build pipeline for all plots.

      1. Create figure
      2. Prepare data sources
      3. Draw glyphs
      4. Configure axes, ranges, fonts
      5. Wrap into layout
    Concrete subclasses implement only the minimal hooks.
    """

    def build(self, ctx: Context) -> None:
        """Assemble and store the plot in the given context.

        Args:
            ctx (Context): The context object containing the data and configuration.
        """
        fig = self._make_figure(ctx)
        sources = self._make_sources(ctx)
        self._draw_glyphs(fig, sources, ctx)
        self._configure(fig, ctx)
        layout = self._wrap_layout(fig, ctx)
        ctx.plots[self._key(ctx)] = layout

    @abstractmethod
    def _make_sources(self, ctx: Context) -> dict[str, Any]:
        """Return the data sources (e.g. ColumnDataSource objects) needed for rendering.

        Args:
            ctx (Context): The context object containing the data and configuration.
        """

    @abstractmethod
    def _draw_glyphs(self, fig: figure, sources: dict[str, ColumnDataSource], ctx: Context) -> None:
        """Draw plot-specific glyphs on the provided Bokeh figure.

        Args:
            fig (figure): The Bokeh figure to draw on.
            sources (dict[str, Any]): The data sources for the plot.
            ctx (Context): The context object containing the data and configuration.
        """

    def _make_figure(self, ctx: Context) -> figure:
        """Initialize a Bokeh figure with default tools, sizing, and title.

        Args:
            ctx (Context): The context object containing the data and configuration.

        Returns:
            figure: A Bokeh figure object.
        """
        return figure(
            title=self._title(ctx),
            sizing_mode="stretch_width",
            tools="pan,box_zoom,reset,save,wheel_zoom",
            toolbar_location="above",
        )

    @abstractmethod
    def _configure(self, fig: figure, ctx: Context) -> None:
        """Hook for configuring axes labels, fonts, ranges, etc.

        Override in mixins or subclasses as needed.

        Args:
            fig (figure): The Bokeh figure to configure.
            ctx (Context): The context object containing the data and configuration.
        """

    def _wrap_layout(self, fig: figure, ctx: Context) -> Column:  # noqa: ARG002, PLR6301 # pylint: disable=unused-argument
        """Wrap the figure into a layout (default: single-column).

        Override to add widgets or custom arrangements.

        Args:
            fig (figure): The Bokeh figure to wrap.
            ctx (Context): The context object containing the data and configuration.

        Returns:
            layout: The wrapped layout object.
        """
        return column(fig, sizing_mode="stretch_width")

    def _key(self, ctx: Context) -> str:  # noqa: ARG002 # pylint: disable=unused-argument
        """Key under which the layout is stored in ctx.plots (default: class name).

        Args:
            ctx (Context): The context object containing the data and configuration.

        Returns:
            str: The key for storing the layout.
        """
        return type(self).__name__

    def _title(self, ctx: Context) -> str:
        """Title for the figure (default: same as key).

        Args:
            ctx (Context): The context object containing the data and configuration.

        Returns:
            str: The title for the figure.
        """
        return self._key(ctx)
