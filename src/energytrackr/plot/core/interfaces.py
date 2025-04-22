"""Stable abstract interfaces for plug-ins (analysis, plot, page).

Other parts of the application import these **only for type-checking**, never
for behaviour; every concrete implementation lives in a plug-in module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from jinja2 import Environment

from energytrackr.plot.core.context import Context


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
    def add(self, ctx: Context) -> None:
        """Mutate *ctx* in-place or add artefacts used by later stages.

        Args:
            ctx (Context): The context object containing the energy data and other artefacts.
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
