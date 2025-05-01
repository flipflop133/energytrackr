"""PlotEmbed - embeds registered Bokeh plots into HTML report using Tabs."""

from __future__ import annotations

from bokeh.core.validation.check import ValidationIssues, check_integrity
from bokeh.embed import components
from bokeh.models import TabPanel, Tabs
from bokeh.resources import CDN
from jinja2 import Environment

from energytrackr.plot.builtin_plots.registry import get_registered_plots
from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PageObj
from energytrackr.utils.exceptions import BokehValidationIssuesError
from energytrackr.utils.logger import logger


class PlotEmbed(PageObj):
    """Page section that injects the interactive Bokeh charts into HTML."""

    def __init__(self, div_class: str = "bokeh-chart") -> None:
        """Initialize the PlotEmbed section."""
        self.div_class = div_class

    def render(self, env: Environment, ctx: Context) -> str:  # noqa: ARG002
        """Return HTML snippet containing Bokeh resources, script, and div.

        Args:
            env (Environment): Jinja2 environment for rendering.
            ctx (Context): Context object containing the plots to embed.

        Returns:
            str: HTML snippet with Bokeh resources and embedded plots.
        """
        tabs = self._build_tabs(ctx)
        self._validate_tabs(tabs)
        return self._render_html(tabs)

    @staticmethod
    def _build_tabs(ctx: Context) -> list[TabPanel]:
        """Build TabPanels in registry order, then add any extras.

        Args:
            ctx (Context): Context object containing the plots to embed.

        Returns:
            list[TabPanel]: List of TabPanels for each plot.
        """
        tabs: list[TabPanel] = [
            TabPanel(child=plot_layout, title=name) for name in get_registered_plots() if (plot_layout := ctx.plots.get(name))
        ]
        for title, layout in ctx.plots.items():
            if title not in get_registered_plots():
                tabs.append(TabPanel(child=layout, title=title))
        return tabs

    @staticmethod
    def _validate_tabs(tabs: list[TabPanel]) -> None:
        """Validate each tab's Bokeh layout, raise if issues found.

        Args:
            tabs (list[TabPanel]): List of TabPanels to validate.

        Raises:
            BokehValidationIssuesError: If any validation issues are found.
        """
        for panel in tabs:
            root = panel.child
            refs = root.references()
            issues: ValidationIssues = check_integrity(refs)
            if issues.error or issues.warning:
                logger.error("🔍 Bokeh validation issues in tab '%s':", panel.title)
                for issue in issues.error:
                    logger.error("    E-%d %s: %s", issue.code, issue.name, issue.text)
                for issue in issues.warning:
                    logger.warning("    W-%d %s: %s", issue.code, issue.name, issue.text)
                raise BokehValidationIssuesError(panel.title, issues)

    def _render_html(self, tabs: list[TabPanel]) -> str:
        """Render the final HTML snippet with CDN resources and Bokeh components.

        Args:
            tabs (list[TabPanel]): List of TabPanels to render.

        Returns:
            str: HTML snippet with CDN resources and embedded plots.
        """
        tabs_widget = Tabs(tabs=tabs, sizing_mode="stretch_width")
        script, div = components(tabs_widget)
        cdn_js = CDN.js_files
        cdn_css = CDN.css_files
        cdn_tags = [f'<link rel="stylesheet" href="{url}">' for url in cdn_css]
        cdn_tags += [f'<script src="{url}"></script>' for url in cdn_js]
        cdn_html = "\n".join(cdn_tags)
        return f'{cdn_html}\n{script}\n<div class="{self.div_class}">{div}</div>'
