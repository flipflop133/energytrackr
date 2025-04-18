"""Interface for pipeline stages."""

from abc import ABC, abstractmethod
from typing import Any


class PipelineStage(ABC):
    """Abstract base for pipeline stages.

    Each stage receives a shared 'context' dict and can read/write data.
    """

    @abstractmethod
    def run(self, context: dict[str, Any]) -> None:
        """Execute the logic for this stage, possibly modifying context.

        If the stage fails critically, set context["abort_pipeline"] = True.

        Args:
            context (dict[str, Any]): Shared context for the pipeline.
        """
