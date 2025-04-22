"""Defines a ChangeTable class for rendering a table of commit changes with statistical metrics."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import PageObj
from energytrackr.utils.logger import logger
from energytrackr.utils.utils import get_local_env

# ---------------------------------------------------------------------------
# Helper - expand column groups & formatting
# ---------------------------------------------------------------------------

group_map = {
    "stats": [
        ("n_val", "n"),
        ("normality", "Normality"),
        ("median_val", "Median (J)"),
        ("std_val", "Std Dev (J)"),
    ],
    "tests": [
        ("p_value", "p-value"),
        ("cohen_str", "Cohen d"),
        ("effect_cat", "Effect"),
        ("pct_change", "Δ %"),
        ("pct_cat", "Δ cat"),
        ("abs_diff", "Δ J"),
        ("practical", "Practical"),
    ],
}


class ChangeTable(PageObj):
    """Renders a table of every commit with computed metrics."""

    def __init__(self, template: str | None = None, columns: list[dict[str, str]] | None = None) -> None:
        """Initializes the ChangeTable object.

        Args:
            template (str | None): Optional path to a custom template file.
                If None, defaults to the package template located in templates/change_table.html.
            columns (list[dict[str, str]] | None): Optional list of dictionaries defining the columns to be displayed.
        """
        # Columns and template location
        cols: list[dict[str, str]] = []
        if columns is None:
            self.columns = [
                {"key": "short_hash", "label": "Commit"},
                {"key": "message", "label": "Message"},
                {"key": "commit_date", "label": "Date"},
                {"key": "commit_files", "label": "Files"},
                {"key": "commit_link", "label": "Link"},
                {"group": "stats"},
                {"group": "tests"},
            ]
        else:
            for c in columns:
                if "group" in c:
                    keys = group_map[c["group"]]
                    if include := c.get("include"):
                        keys = [pair for pair in keys if pair[0] in include]
                    if exclude := c.get("exclude"):
                        keys = [pair for pair in keys if pair[0] not in exclude]
                    cols.extend({"key": k, "label": label} for k, label in keys)
                else:
                    cols.append({"key": c["key"], "label": c.get("label", c["key"])})
            self.columns = cols
        tpl_dir = Path(__file__).with_name("templates")
        self.template_path = Path(template) if template else tpl_dir / "change_table.html"

    def render(self, env: Environment, ctx: Context) -> str:
        """Renders the change table section as an HTML string using a Jinja2 template.

        Args:
            env (Environment): The Jinja2 environment used for template rendering.
            ctx (Context): The context object containing artefacts, statistics, and energy fields.

        Returns:
            str: The rendered HTML string for the change table. If the template is missing, returns an error message.

        Workflow:
            - Checks if the template file exists; logs an error and returns an error message if not found.
            - Determines whether to use the provided environment or create a local one based on the template's location.
            - Loads the template.
            - Prepares table rows by extracting commit and statistical data from the context.
            - Formats each row with commit details, statistical values, and change event information.
            - Renders the template with the prepared columns and rows.
        """
        # Load template
        if not self.template_path.is_file():
            logger.error("ChangeTable: template '%s' not found.", self.template_path)
            return "<p><strong>Error:</strong> change table template missing.</p>"

        tmpl = get_local_env(env, str(self.template_path)).get_template(self.template_path.name)
        # Prepare rows
        logger.info("energy fields: %s", ctx.energy_fields)
        logger.info("stats fields: %s", ctx.stats.keys())
        col = ctx.energy_fields[0]
        stats = ctx.stats
        rows = {e.index: e for e in ctx.artefacts["change_events"]}
        df_m = stats["df_median"]

        table_rows = []
        for i, commit in enumerate(stats["valid_commits"]):
            rs = df_m[df_m["commit"] == commit].iloc[0]
            ev = rows.get(i)
            commit_details = ctx.artefacts["commit_details"].get(commit, {})
            table_rows.append({
                "short_hash": stats["short_hashes"][i],
                "message": commit_details.get("commit_summary", ""),
                "commit_date": commit_details.get("commit_date", ""),
                "commit_files": commit_details.get("files_modified", ""),
                "commit_link": commit_details.get("commit_link", ""),
                "n_val": int(rs["count"]),
                "normality": "Normal" if ctx.artefacts["normality_flags"][i] else "Non-normal",
                "median_val": f"{rs[col]:.2f}",
                "std_val": f"{rs[f'{col}_std']:.2f}",
                "p_value": f"{ev.p_value:.3g}" if ev else "N/A",
                "cohen_str": f"{ev.effect_size.cohen_d:.2f}" if ev else "N/A",
                "effect_cat": ev.effect_size.category if ev else "N/A",
                "pct_change": f"{ev.change_magnitude.pct_change * 100:.1f}%" if ev else "0%",
                "pct_cat": ev.change_magnitude.pct_change_level if ev else "N/A",
                "abs_diff": f"{ev.change_magnitude.abs_diff:.2f}" if ev else "0.0",
                "practical": ev.change_magnitude.practical_level if ev else "N/A",
                "level": ev.level if ev else "-",
                "row_class": (
                    "increase"
                    if ev and ev.direction == "increase"
                    else "decrease"
                    if ev and ev.direction == "decrease"
                    else ""
                ),
            })

        # Render with Jinja2
        return tmpl.render(columns=self.columns, rows=table_rows)
