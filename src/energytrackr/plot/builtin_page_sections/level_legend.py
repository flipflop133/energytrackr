"""Level Legend Page Section."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

from energytrackr.plot.config import get_settings
from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PageObj
from energytrackr.utils.logger import logger
from energytrackr.utils.utils import get_local_env


class LevelLegend(PageObj):
    """Renders the Change-Event Level Legend as an HTML fragment.

    Reads threshold values from the YAML config via `settings`.
    """

    def __init__(self, template: str | None = None) -> None:
        """Initialize the LevelLegend page section.

        Args:
            template (str | None): Optional path to a custom template file.
                If None, defaults to the package template located in templates/level_legend.html.
        """
        # fallback to package template under templates/level_legend.html
        self.template_path = template or str(Path(__file__).with_name("templates") / "level_legend.html")

    def render(self, env: Environment, ctx: Context) -> str:  # noqa: ARG002
        """Renders the level legend template using the provided Jinja2 environment and context.

        This method checks if the specified template file exists. If not, it logs an error and returns an HTML error message.
        If the template exists, it ensures the correct Jinja2 environment is used (either the provided one or a new one
        with the appropriate loader), loads the template, and renders it with threshold values from the application settings.

        Args:
            env (Environment): The Jinja2 environment to use for template rendering.
            ctx (Context): The rendering context (currently unused).

        Returns:
            str: The rendered HTML string of the level legend, or an error message if the template is missing.
        """
        if not Path(self.template_path).is_file():
            logger.error("LevelLegend: template '%s' not found.", self.template_path)
            return "<p><strong>Error:</strong> level legend template missing.</p>"

        tmpl = get_local_env(env, self.template_path).get_template(Path(self.template_path).name)
        th = get_settings().energytrackr.analysis.thresholds
        return tmpl.render(
            welch_p=th.welch_p,
            cohen_d=th.cohen_d,
            pct_change=th.pct_change,
            practical=th.practical,
        )
