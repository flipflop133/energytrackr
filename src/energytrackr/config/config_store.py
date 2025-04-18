"""Module to store the pipeline configuration."""

from energytrackr.config.config_model import PipelineConfig
from energytrackr.utils.exceptions import ConfigurationSingletonAlreadySetError, ConfigurationSingletonNotSetError


class Config:
    """Class to retrieve the unique configuration of the pipeline."""

    _instance: "Config | None" = None
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
            ConfigurationSingletonAlreadySetError: If the configuration has already been set.
        """
        if cls._pipeline_config is not None:
            raise ConfigurationSingletonAlreadySetError()
        cls._pipeline_config = new_config

    @classmethod
    def get_config(cls) -> PipelineConfig:
        """Gets the pipeline configuration.

        Returns:
            PipelineConfig: The pipeline configuration.

        Raises:
            ConfigurationSingletonNotSetError: If the configuration has not been set.
        """
        if cls._pipeline_config is None:
            raise ConfigurationSingletonNotSetError()
        return cls._pipeline_config

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing purposes only)."""
        cls._pipeline_config = None
        cls._instance = None
