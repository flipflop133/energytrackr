"""Shared, mutable blackboard passed to every plug-in object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bokeh.plotting import figure

if TYPE_CHECKING:
    from energytrackr.plot.core.interfaces import PlotObj


@dataclass
class Context:
    """Context class holds the state and artefacts required for energy data analysis and plotting.

    Attributes:
        energy_data (EnergyData): The main energy data set to be analyzed or plotted.
        energy_column (str): The name of the column in energy_data representing energy values.
        artefacts (dict[str, Any]): Runtime artefacts produced by data transforms and plot objects.
        stats (dict[str, Any]): Statistical summaries or metrics computed during analysis.
        fig (figure | None): The figure object created by the pipeline and manipulated by plot objects.
    """

    input_path: str
    energy_fields: list[str]

    # Runtime artefacts produced by transforms & plot objects
    artefacts: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    fig: figure | None = None  # created by pipeline, manipulated by objects
    plots: dict[str, Any] = field(default_factory=dict)
    plot_objects: dict[str, PlotObj] = field(default_factory=dict)
