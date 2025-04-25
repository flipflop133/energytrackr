"""Plot interface for energy analysis visualizations."""

from abc import ABC, abstractmethod

from energytrackr.plot.core.context import Context


class Plot(ABC):
    """Abstract base for commit-to-commit energy comparisons."""

    @abstractmethod
    def build(self, ctx: Context) -> None:
        """Build the plot layout.

        Args:
            ctx (Context): The plotting context containing statistics, artefacts, and a container for the resulting plot.
        """
