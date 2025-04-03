"""Module to store the pipeline configuration."""

from typing import Optional

from config.config_model import PipelineConfig


class ConfigurationSingletonError(ValueError):
    """Custom ValueError for errors related to the configuration singleton."""

    def __str__(self) -> str:
        """Returns the custom error message.

        Returns:
            str: The custom error message.
        """
        return "Configuration already initialized."


class Config:
    """Class to retrieve the unique configuration of the pipeline."""

    _instance: Optional["Config"] = None
    _pipeline_config: PipelineConfig | None = None

    def __new__(cls) -> "Config":
        """Creates a new instance of the Config class.

        Returns:
            Config: The singleton instance of the Config class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_config(cls, new_config: PipelineConfig) -> None:
        """Sets the pipeline configuration.

        Args:
            new_config (PipelineConfig): The new pipeline configuration to set.

        Raises:
            ConfigurationSingletonError: If the configuration has already been set.
        """
        if cls._pipeline_config is not None:
            raise ConfigurationSingletonError()
        cls._pipeline_config = new_config

    @classmethod
    def get_config(cls) -> PipelineConfig:
        """Gets the pipeline configuration.

        Returns:
            PipelineConfig: The pipeline configuration.

        Raises:
            ConfigurationSingletonError: If the configuration has not been set.
        """
        if cls._pipeline_config is None:
            raise ConfigurationSingletonError()
        return cls._pipeline_config
