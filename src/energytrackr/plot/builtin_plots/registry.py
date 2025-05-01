"""Registry for BasePlot subclasses, enabling automatic discovery of available plots."""

from energytrackr.plot.core.interfaces import BasePlot
from energytrackr.utils.exceptions import PlotAlreadyRegisteredError

_PLOT_REGISTRY: dict[str, type[BasePlot]] = {}


def register_plot(plot_cls: type[BasePlot]) -> type[BasePlot]:
    """Class decorator to register a BasePlot subclass.

    Usage::

        @register_plot
        class MyPlot(BasePlot): ...

    Registered under its class name.

    Args:
        plot_cls (type[BasePlot]): A subclass of BasePlot

    Returns:
        type[BasePlot]: The registered plot class.

    Raises:
        PlotAlreadyRegisteredError: If the plot class is already registered.
    """
    if (name := plot_cls.__name__) in _PLOT_REGISTRY:
        raise PlotAlreadyRegisteredError(name)
    _PLOT_REGISTRY[name] = plot_cls
    return plot_cls


def get_registered_plots() -> dict[str, type[BasePlot]]:
    """Return mapping of plot names to BasePlot subclasses.

    Returns:
        dict[str, type[BasePlot]]: Mapping of plot names to BasePlot subclasses.
    """
    return dict(_PLOT_REGISTRY)
