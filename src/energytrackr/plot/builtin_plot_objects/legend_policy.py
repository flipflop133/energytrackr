"""LegendPolicy - sets click_policy on all legend entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Never, cast

from bokeh.core.enums import LegendClickPolicy, LegendClickPolicyType
from bokeh.models import Legend
from bokeh.plotting import figure

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, PlotObj
from energytrackr.utils.exceptions import InvalidLegendClickPolicyError


@dataclass(frozen=True)
class LegendPolicyConfig:
    """Configuration for LegendPolicy.

    Attributes:
        policy (LegendClickPolicyType): The click policy for the legend. Options are "hide", "mute", or "disable".
    """

    policy: LegendClickPolicyType = field(default="hide")


class LegendPolicy(PlotObj, Configurable[LegendPolicyConfig]):
    """A PlotObj that sets click_policy on all legend entries."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the LegendPolicy with a click policy.

        Args:
            **params: Configuration parameters for the LegendPolicy.

        Raises:
            InvalidLegendClickPolicyError: If the provided policy is not valid.
        """
        super().__init__(LegendPolicyConfig, **params)
        if (policy := self.config.policy) is None:
            policy = cast(LegendClickPolicyType, LegendClickPolicy._default)
        if (policy_str := str(policy)) not in LegendClickPolicy:
            valid = list(LegendClickPolicy)
            raise InvalidLegendClickPolicyError(policy_str, valid)

    def add(self, ctx: Context, fig: figure) -> None:  # noqa: ARG002
        """Adds legend policy and styling to the figure's legends in the given context.

        This method sets the `click_policy` and `label_text_font` attributes for each
        `Legend` object found in the figure associated with the provided context.

        Args:
            ctx (Context): The plotting context containing the figure (`fig`) to modify.
            fig (figure): The Bokeh figure to which the legend policy will be applied.
        """
        legends: list[Legend] | list[Never] = fig.legend if isinstance(fig.legend, (list, tuple)) else [fig.legend]
        for legend in legends:
            if isinstance(legend, Legend):
                legend.click_policy = self.config.policy
                legend.label_text_font = "Roboto"
                legend.location = "top_left"
