"""Custom exceptions for the application."""

from pathlib import Path


class UnknownCommandError(Exception):
    """Exception raised for unknown commands.

    Attributes:
        command (str): The command that caused the exception.
    """

    def __init__(self, command: str) -> None:
        """Initialize the exception with the unknown command.

        Args:
            command (str): The command that caused the exception.
        """
        super().__init__(f"Unknown command: {command}")


class ConfigurationSingletonAlreadySetError(ValueError):
    """Custom ValueError for errors related to the configuration singleton."""

    def __str__(self) -> str:
        """Returns the custom error message.

        Returns:
            str: The custom error message.
        """
        return "Configuration already initialized."


class ConfigurationSingletonNotSetError(Exception):
    """Custom Exception for errors related to the configuration singleton."""

    def __str__(self) -> str:
        """Returns the custom error message.

        Returns:
            str: The custom error message.
        """
        return "Configuration not initialized."


class SourceDirectoryNotFoundError(Exception):
    """Exception raised when the source directory is not found."""

    def __init__(self, path: Path) -> None:
        """Initialize the exception with a custom message.

        Args:
            path (Path): The path to the source directory that was not found.
        """
        super().__init__(f"Source directory not found: {path.resolve()}")


class TargetDirectoryNotFoundError(Exception):
    """Exception raised when the target directory is not found."""

    def __init__(self, path: Path) -> None:
        """Initialize the exception with a custom message.

        Args:
            path (Path): The path to the target directory that was not found.
        """
        super().__init__(f"Target directory not found: {path.resolve()}")


class MissingContextKeyError(Exception):
    """Exception raised when a required key is missing in the context."""

    def __init__(self, key: str) -> None:
        """Initialize the exception with a custom message.

        Args:
            key (str): The missing key that caused the exception.
        """
        super().__init__(f"Missing required key: {key}")


class CantFindFileError(FileNotFoundError):
    """Exception raised when a file cannot be found.

    Inherits from FileNotFoundError to provide a more specific error message.
    """

    def __init__(self, path: str) -> None:
        """Initialize the exception with a custom message."""
        super().__init__(f"File not found: {path}")


class SettingsReadOnlyError(AttributeError):
    """Exception raised when attempting to modify a read-only Settings object."""

    def __init__(self) -> None:
        """Initialize the exception with a custom message."""
        super().__init__("Settings object is read-only")


class MissingEnergyTrackrKeyError(KeyError):
    """Exception raised when the top-level 'energytrackr' key is missing in the YAML configuration."""

    def __init__(self) -> None:
        """Initialize the exception with a custom message."""
        super().__init__("Top-level 'energytrackr' key missing in YAML.")


class PlotObjectDidNotInitializeFigureError(RuntimeError):
    """Raised when the first PlotObj does not initialize ctx.fig."""

    def __init__(self, module: str) -> None:
        """Initialize the exception with a custom message.

        Args:
            module (str): The module name of the PlotObj that failed to initialize.
        """
        super().__init__(f"First PlotObj '{module}' did not initialize ctx.fig.")


class MissingEnergyFieldError(ValueError):
    """Exception raised when no energy fields are specified in the configuration."""

    def __init__(self) -> None:
        """Initialize the exception with a custom message."""
        super().__init__("Configuration must specify at least one energy field.")


class InvalidDottedPathError(ValueError):
    """Raised when the dotted path is not in the expected 'mod:attr' format."""

    def __init__(self, dotted: str) -> None:
        """Initialize the exception with a custom message.

        Args:
            dotted (str): The invalid dotted path that caused the exception.
        """
        super().__init__(f"Invalid dotted path '{dotted}', expected 'mod:attr'.")


class AttributeNotFoundError(ImportError):
    """Raised when the attribute is not found in the specified module."""

    def __init__(self, mod_path: str, attr: str) -> None:
        """Initialize the exception with a custom message.

        Args:
            mod_path (str): The module path where the attribute was expected to be found.
            attr (str): The attribute that was not found in the module.
        """
        super().__init__(f"Module '{mod_path}' has no attribute '{attr}'.")


class InvalidLegendClickPolicyError(ValueError):
    """Exception raised when an invalid legend click policy is provided."""

    def __init__(self, policy: str, valid: list) -> None:
        """Initialize with the invalid policy and valid options.

        Args:
            policy (str): The invalid policy value.
            valid (list): The list of valid policy values.
        """
        super().__init__(f"Invalid legend-click policy {policy!r}; must be one of {valid}")


class CommitStatsMissingOrEmptyDataFrameError(ValueError):
    """Exception raised when the DataFrame is missing or empty in CommitStats, or the specified column is not found."""

    def __init__(self, column: str) -> None:
        """Initialize the exception with a custom message.

        Args:
            column (str): The column that was expected in the DataFrame.
        """
        super().__init__(f"CommitStats: missing or empty DataFrame for column '{column}'")


class PlotAlreadyRegisteredError(ValueError):
    """Exception raised when a plot class is already registered."""

    def __init__(self, name: str) -> None:
        """Initialize the exception with the plot class name.

        Args:
            name (str): The name of the plot class that is already registered.
        """
        super().__init__(f"Plot '{name}' already registered.")


class PlotLabelsNotSetError(ValueError):
    """Exception raised when plot labels are not set."""

    def __init__(self, plot_key: str) -> None:
        """Initialize the exception with the plot key.

        Args:
            plot_key (str): The key identifying the plot.
        """
        super().__init__(f"{plot_key} plot labels not set.")


class PlotSourcesNotSetError(ValueError):
    """Exception raised when plot sources are not set."""

    def __init__(self, plot_key: str) -> None:
        """Initialize the exception with the plot key.

        Args:
            plot_key (str): The key identifying the plot.
        """
        super().__init__(f"Sources not set for graph {plot_key}")


class NotAPlotObjTypeError(TypeError):
    """Exception raised when an object is not a PlotObj."""

    def __init__(self, name: str) -> None:
        """Initialize the exception with the object name.

        Args:
            name (str): The name of the object that is not a PlotObj.
        """
        super().__init__(f"Plot object '{name}' is not a PlotObj")
