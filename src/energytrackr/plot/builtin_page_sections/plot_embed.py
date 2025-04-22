"""PlotEmbed - embeds the Bokeh figure into the HTML report."""

from __future__ import annotations

from bokeh.embed import components
from bokeh.resources import CDN
from jinja2 import Environment

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PageObj


class PlotEmbed(PageObj):
    """Page section that injects the interactive Bokeh chart into HTML."""

    def __init__(self, div_class: str = "bokeh-chart") -> None:
        """Initialize the PlotEmbed page section.

        Args:
            div_class (str): CSS class for the div element containing the Bokeh plot.
                Default is "bokeh-chart".
        """
        self.div_class = div_class

    def render(self, env: Environment, ctx: Context) -> str:  # noqa: ARG002
        """Return HTML snippet containing Bokeh resources, script, and div.

        Args:
            env (Any): Jinja2 environment (not used here).
            ctx (Context): Context object containing the Bokeh figure.

        Returns:
            str: HTML snippet with Bokeh resources, script, and div.
        """
        if ctx.fig is None:
            return "<p><strong>Error:</strong> no figure to embed.</p>"

        # Generate the standalone components
        script, div = components(ctx.fig)
        cdn_js = CDN.js_files
        cdn_css = CDN.css_files

        # Build CDN resource tags
        cdn_tags = [f'<link rel="stylesheet" href="{url}">' for url in cdn_css] + [
            f'<script src="{url}"></script>' for url in cdn_js
        ]
        cdn_html = "\n".join(cdn_tags)

        # Combine resources, script, and div
        html = f'{cdn_html}\n{script}\n<div class="{self.div_class}">{div}</div>'
        return html
