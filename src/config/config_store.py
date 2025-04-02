"""Module to store the pipeline configuration."""

from typing import Optional

from config.config_model import PipelineConfig


class ConfigurationSingletonError(ValueError):
    pass


class Config:
    _instance: Optional["Config"] = None
    _pipeline_config: PipelineConfig | None = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_config(cls, new_config: PipelineConfig) -> None:
        if cls._pipeline_config is not None:
            raise ConfigurationSingletonError("Configuration already initialized.")
        cls._pipeline_config = new_config

    @classmethod
    def get_config(cls) -> PipelineConfig:
        if cls._pipeline_config is None:
            raise ConfigurationSingletonError("Configuration not set.")
        return cls._pipeline_config
