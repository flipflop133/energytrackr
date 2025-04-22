"""Main orchestration module for energy consumption data analysis and reporting."""

import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from energytrackr.plot.config import get_settings
from energytrackr.plot.core.context import Context
from energytrackr.plot.core.loader import load_callable
from energytrackr.utils.exceptions import (
    CantFindFileError,
    MissingEnergyFieldError,
    PlotObjectDidNotInitializeFigureError,
)
from energytrackr.utils.logger import logger


def _resolve_csv_path(input_path: str) -> Path:
    csv_path = Path(input_path).expanduser().resolve()
    if not csv_path.is_file():
        raise CantFindFileError(str(csv_path))
    return csv_path


def _locate_config(csv_path: Path, config_path: str | None) -> Path:
    if config_path:
        cfg = Path(config_path).expanduser().resolve()
        if not cfg.is_file():
            raise CantFindFileError(str(cfg))
        return cfg

    candidates = [
        Path.cwd() / "plot.yml",
        csv_path.with_suffix(".yml"),
        csv_path.parent / "plot.yml",
    ]
    for p in candidates:
        if p.is_file():
            return p

    tried = ", ".join(str(p) for p in candidates)
    raise CantFindFileError(tried)


def _extend_pythonpath(plugin_dirs: Sequence[str]) -> None:
    for plugin_dir in plugin_dirs:
        sys.path.insert(0, str(Path(plugin_dir).expanduser().resolve()))


def _build_context(csv_path: Path, git_repo_path: str | None, energy_fields: list[str]) -> Context:
    if not energy_fields:
        raise MissingEnergyFieldError()

    ctx = Context(input_path=str(csv_path), energy_fields=list(energy_fields))
    if git_repo_path:
        ctx.artefacts["git_repo_path"] = git_repo_path
    return ctx


def _apply_transforms(ctx: Context, transforms: Sequence[dict[str, Any]]) -> None:
    for spec in transforms:
        cls = load_callable(spec["module"])
        params = spec.get("params", {}) or {}
        logger.info("▶ Transform %s(%s)", spec["module"], params)
        cls(**params).apply(ctx)


def _build_plot_objects(ctx: Context, plot_objs: Sequence[dict[str, Any]]) -> None:
    for idx, spec in enumerate(plot_objs):
        cls = load_callable(spec["module"])
        params = spec.get("params", {}) or {}
        logger.info("▶ PlotObj %s(%s)", spec["module"], params)
        obj = cls(**params)
        obj.add(ctx)
        if not idx and ctx.fig is None:
            raise PlotObjectDidNotInitializeFigureError(spec["module"])


def _render_page_sections(ctx: Context, page_objs: Sequence[dict[str, Any]]) -> str:
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(templates_dir))
    sections: list[str] = []
    for spec in page_objs:
        cls = load_callable(spec["module"])
        params = spec.get("params", {}) or {}
        logger.info("▶ PageObj %s(%s)", spec["module"], params)
        sections.append(cls(**params).render(env, ctx))
    return "\n".join(sections)


def plot(
    input_path: str,
    git_repo_path: str | None = None,
    config_path: str | None = None,
) -> None:
    """Main entry point: orchestrate loading, plotting, and report generation.

    Args:
        input_path (str): Path to the CSV file containing energy data.
        git_repo_path (str | None): Optional path to a Git repository.
        config_path (str | None): Optional path to the configuration file.
    """
    # Resolve paths
    csv_path = _resolve_csv_path(input_path)
    cfg_path = _locate_config(csv_path, config_path)

    # Load settings
    settings = get_settings(str(cfg_path))

    # Extend PYTHONPATH for plugins
    _extend_pythonpath(settings.energytrackr.plugins.paths)

    # Prepare context
    ctx = _build_context(
        csv_path,
        git_repo_path,
        list(settings.energytrackr.data.energy_fields),
    )

    # Apply data transforms
    _apply_transforms(ctx, settings.energytrackr.plot.transforms)

    # Build plot objects
    _build_plot_objects(ctx, settings.energytrackr.plot.objects)

    # Render HTML report
    html_content = _render_page_sections(ctx, settings.energytrackr.plot.page)

    # Write out report
    output_file = csv_path.with_suffix(".plot.html")
    output_file.write_text(html_content, encoding="utf-8")
    logger.info("✔ Report exported to %s", output_file)

    # Optionally open in browser
    if settings.energytrackr.report.chart.get("open", False):
        webbrowser.open(output_file.as_uri())
