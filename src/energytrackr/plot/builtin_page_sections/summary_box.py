"""SummaryBox - first HTML section in the default report.

Renders a quick statistics overview for the active energy column.  Uses a Jinja
partial `summary_box.html` that ships with *plot*; users can override by
encoding their own PageObj in YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PageObj
from energytrackr.utils.logger import logger
from energytrackr.utils.utils import get_local_env


class SummaryBox(PageObj):
    """Render general & statistical summary values."""

    def __init__(self, template: str | None = None) -> None:
        """Initialize the SummaryBox page section.

        Args:
            template (str | None): Optional path to a custom template file.
                If None, defaults to the package template located in templates/summary_box.html.
        """
        # fallback to package template under templates/summary_box.html
        self.template_path = template or str(Path(__file__).with_name("templates") / "summary_box.html")

    def render(self, env: Environment, ctx: Context) -> str:
        """Renders the summary box using a Jinja2 template.

        This method checks if the specified template file exists. If not, it logs an error and returns an HTML error message.
        If the template is outside the default environment's loader path, it creates a new Jinja2 Environment with the
        appropriate loader.
        It then loads the template, computes the overall summary for the first energy field in the context, and renders the
        template with the computed summary and the column name.

        Args:
            env (Environment): The Jinja2 environment to use for template rendering.
            ctx (Context): The context containing energy fields and other relevant data.

        Returns:
            str: The rendered HTML string for the summary box, or an error message if the template is missing.
        """
        if not Path(self.template_path).is_file():
            logger.error("SummaryBox: template '%s' not found.", self.template_path)
            return "<p><strong>Error:</strong> summary template missing.</p>"

        tmpl = get_local_env(env, self.template_path).get_template(Path(self.template_path).name)
        summary = self.compute_overall_summary(ctx.energy_fields[0], ctx)
        return tmpl.render(column=ctx.energy_fields[0], **summary)

    def compute_overall_summary(self, energy_column: str, ctx: Context) -> dict[str, Any]:
        """Compute an overall summary of energy-related statistics for a given column.

        Args:
            energy_column (str): The name of the column containing energy data.
            ctx (Context): The context containing artefacts and statistics.

        Returns:
            dict: A dictionary containing the following summary statistics:
                - total_commits (int): Total number of data points in the distribution.
                - significant_changes (int): Number of significant change events.
                - regressions (int): Count of changes with an "increase" direction.
                - improvements (int): Count of changes with a "decrease" direction.
                - mean_energy (float): Mean value of the energy column.
                - median_energy (float): Median value of the energy column.
                - std_energy (float): Standard deviation of the energy column.
                - max_increase (float): Maximum severity of "increase" changes.
                - max_decrease (float): Maximum severity of "decrease" changes.
                - avg_cohens_d (float): Average absolute Cohen's d value for changes.
                - normal_count (int): Number of normality flags set to True.
                - non_normal_count (int): Number of normality flags set to False.
                - outliers_removed (int): Number of outliers removed from the dataset.
        """
        normality = ctx.artefacts["normality_flags"]
        changes = ctx.artefacts["change_events"]
        if (df := ctx.artefacts.get("df", pd.DataFrame())) is not None:
            outliers_removed_count = len(df) - len(self._filter_outliers(df, energy_column))
            mean_energy = df[energy_column].mean()
            median_energy = df[energy_column].median()
            std_energy = df[energy_column].std()
        else:
            outliers_removed_count = 0
            mean_energy = None
            median_energy = None
            std_energy = None
        valid_commits = ctx.stats["valid_commits"]
        oldest_commit = valid_commits[-1] if valid_commits else None
        latest_commit = valid_commits[0] if valid_commits else None

        summary = {
            "project_name": ctx.artefacts["project_name"],
            "oldest_commit_hash": oldest_commit[:7] if oldest_commit else None,
            "oldest_commit_date": ctx.artefacts["commit_details"].get(oldest_commit, {}).get("commit_date")
            if oldest_commit
            else None,
            "latest_commit_hash": latest_commit[:7] if latest_commit else None,
            "latest_commit_date": ctx.artefacts["commit_details"].get(latest_commit, {}).get("commit_date")
            if latest_commit
            else None,
            "total_commits": len(ctx.stats["valid_commits"]),
            "significant_changes": len(changes),
            "regressions": sum(1 for e in changes if e.direction == "increase"),
            "improvements": sum(1 for e in changes if e.direction == "decrease"),
            "mean_energy": mean_energy,
            "median_energy": median_energy,
            "std_energy": std_energy,
            "avg_cohens_d": np.mean([abs(e.effect_size.cohen_d) for e in changes]) if changes else 0.0,
            "normal_count": sum(normality),
            "non_normal_count": len(normality) - sum(normality),
            "outliers_removed": outliers_removed_count,
        }
        return summary

    @staticmethod
    def _filter_outliers(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """Filters outliers from a DataFrame column based on the Interquartile Range (IQR) method.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            column (str): The name of the column to filter for outliers.
            multiplier (float, optional): The multiplier for the IQR to define the outlier bounds.
                Defaults to 1.5.

        Returns:
            pd.DataFrame: A DataFrame with outliers removed from the specified column.
        """
        q1: float = df[column].quantile(0.25)
        q3: float = df[column].quantile(0.75)
        iqr: float = q3 - q1
        lower_bound: float = q1 - multiplier * iqr
        upper_bound: float = q3 + multiplier * iqr
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
