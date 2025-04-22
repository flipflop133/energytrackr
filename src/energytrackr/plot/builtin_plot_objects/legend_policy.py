# src/plot/builtin_plot_objects/legend_policy.py

"""LegendPolicy - sets click_policy on all legend entries."""

from __future__ import annotations

from typing import Never, cast

from bokeh.core.enums import LegendClickPolicy, LegendClickPolicyType
from bokeh.models import Legend

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PlotObj
from energytrackr.utils.exceptions import InvalidLegendClickPolicyError, PlotObjectDidNotInitializeFigureError


class LegendPolicy(PlotObj):
    """A PlotObj that sets click_policy on all legend entries.

    YAML usage:
      - module: plot.builtin_plot_objects.legend_policy:LegendPolicy
        params:
          policy: hide             # "hide", "mute", "disable"
    """

    def __init__(self, *, policy: LegendClickPolicyType | None = None) -> None:
        """Initialize the LegendPolicy with a click policy.

        Args:
            policy (LegendClickPolicyType): The click policy for the legend. Options are "hide", "mute", or "disable".

        Raises:
            InvalidLegendClickPolicyError: If the provided policy is not valid.
        """
        if policy is None:
            policy = cast(LegendClickPolicyType, LegendClickPolicy._default)
        if (policy_str := str(policy)) not in LegendClickPolicy:
            valid = list(LegendClickPolicy)
            raise InvalidLegendClickPolicyError(policy_str, valid)
        self.policy: LegendClickPolicyType = policy

    def add(self, ctx: Context) -> None:
        """Adds legend policy and styling to the figure's legends in the given context.

        This method sets the `click_policy` and `label_text_font` attributes for each
        `Legend` object found in the figure associated with the provided context.

        Args:
            ctx (Context): The plotting context containing the figure (`fig`) to modify.

        Raises:
            PlotObjectDidNotInitializeFigureError: If the figure is not initialized in the context.
        """
        if not (fig := ctx.fig):
            raise PlotObjectDidNotInitializeFigureError(self.__class__.__name__)

        legends: list[Legend] | list[Never] = fig.legend if isinstance(fig.legend, (list, tuple)) else [fig.legend]
        for legend in legends:
            if isinstance(legend, Legend):
                legend.click_policy = self.policy
                legend.label_text_font = "Roboto"
