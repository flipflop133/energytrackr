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


class ConfigurationSingletonError(ValueError):
    """Custom ValueError for errors related to the configuration singleton."""

    def __str__(self) -> str:
        """Returns the custom error message.

        Returns:
            str: The custom error message.
        """
        return "Configuration already initialized."


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
