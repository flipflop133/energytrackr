"""MedianLine - default line/glyph object for the modular plot system.

YAML usage
~~~~~~~~~~
```yaml
plot:
  objects:
    - {module: plot.builtin_plot_objects.median_line:MedianLine,
       params: {color: blue, line_width: 2}}
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from energytrackr.plot.builtin_plot_objects.series_line import SeriesLine


@dataclass(frozen=True)
class MedianLineConfig:
    """Configuration for the Median line plot object."""

    color: str = "blue"
    line_width: int = 1
    radius: float = 0.3
    legend: str = "Median"
    default_visible: bool = True


class MedianLine(SeriesLine[MedianLineConfig]):
    """Draws median points connected by a line (identity series)."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the MedianLine object with color, line width, and legend label."""
        super().__init__(MedianLineConfig, **params)
