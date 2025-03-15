"""Singleton class to store the configuration object."""

from typing import Optional

from config_model import PipelineConfig


class ConfigurationSingletonError(ValueError):
    """Raised when the configuration has already been initialized."""

    def __init__(self) -> None:
        """Initialize the error message."""
        super().__init__("Configuration has already been initialized.")


class Config:
    """Singleton class to store the configuration object."""

    _instance: Optional["Config"] = None
    _pipeline_config: PipelineConfig | None = None

    def __new__(cls) -> "Config":
        """Ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_config(cls, new_pipeline_config: PipelineConfig) -> None:
        """Set the configuration object."""
        if cls._pipeline_config is not None:
            raise ConfigurationSingletonError()
        cls._pipeline_config = new_pipeline_config

    @classmethod
    def get_config(cls) -> PipelineConfig:
        """Get the configuration object."""
        if cls._pipeline_config is None:
            raise ConfigurationSingletonError()
        return cls._pipeline_config
