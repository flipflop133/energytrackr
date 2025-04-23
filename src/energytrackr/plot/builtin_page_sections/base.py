"""Base page section for energytrackr."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2.environment import Template

from energytrackr.plot.config import get_settings
from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PageObj
from energytrackr.utils.logger import logger
from energytrackr.utils.utils import get_local_env


class Base(PageObj):
    """Base page section for energytrackr.

    This class serves as a base for creating page sections in the energytrackr application.
    It provides a template path and a render method to generate HTML content using Jinja2 templates.
    """

    def __init__(self, template: str | None = None) -> None:
        """Initialize the Base page section.

        Args:
            template (str | None): Optional path to a custom template file.
                If None, defaults to the package template located in templates/base.html.
        """
        # fallback to package template under templates/base.html
        self.template_path = template or str(Path(__file__).with_name("templates") / "base.html")

    def render(self, env: Environment, ctx: Context) -> str:  # noqa: ARG002
        """Render the page section using Jinja2 templates.

        Args:
            env (Environment): Jinja2 environment for rendering templates.
            ctx (Context): Context object containing data for rendering.

        Returns:
            str: Rendered HTML content.
        """
        if not Path(self.template_path).is_file():
            logger.error("LevelLegend: template '%s' not found.", self.template_path)
            return "<p><strong>Error:</strong> level legend template missing.</p>"

        settings = get_settings()
        tmpl: Template = get_local_env(env, self.template_path).get_template(Path(self.template_path).name)
        return tmpl.render(font=settings.energytrackr.report.font)
