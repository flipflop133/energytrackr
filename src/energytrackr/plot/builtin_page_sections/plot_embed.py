"""PlotEmbed - embeds registered Bokeh plots into HTML report using Tabs."""

from __future__ import annotations

from bokeh.embed import components
from bokeh.models import TabPanel, Tabs
from bokeh.resources import CDN
from jinja2 import Environment

from energytrackr.plot.builtin_plots.registry import get_registered_plots
from energytrackr.plot.core.context import Context


class PlotEmbed:
    """Page section that injects the interactive Bokeh charts into HTML."""

    def __init__(self, div_class: str = "bokeh-chart") -> None:
        """Initialize the PlotEmbed section."""
        self.div_class = div_class

    def render(self, env: Environment, ctx: Context) -> str:  # noqa: ARG002 # pylint: disable=unused-argument
        """Return HTML snippet containing Bokeh resources, script, and div.

        Uses registry order to maintain consistent tab order.

        Args:
            env (Environment): Jinja2 environment for rendering.
            ctx (Context): Context object containing plot data.

        Returns:
            str: HTML snippet with Bokeh resources, script, and div.
        """
        # Build TabPanels in registry order if present in ctx.plots
        tabs: list[TabPanel] = [
            TabPanel(child=plot_layout, title=name) for name in get_registered_plots() if (plot_layout := ctx.plots.get(name))
        ]

        # Fallback: include any remaining
        for title, layout in ctx.plots.items():
            if title not in get_registered_plots():
                tabs.append(TabPanel(child=layout, title=title))

        tabs_widget = Tabs(tabs=tabs, sizing_mode="stretch_width")
        script, div = components(tabs_widget)

        # Generate CDN resource tags
        cdn_js = CDN.js_files
        cdn_css = CDN.css_files
        cdn_tags = [f'<link rel="stylesheet" href="{url}">' for url in cdn_css]
        cdn_tags += [f'<script src="{url}"></script>' for url in cdn_js]
        cdn_html = "\n".join(cdn_tags)

        # Combine resources, script, and div
        return f'{cdn_html}\n{script}\n<div class="{self.div_class}">{div}</div>'
