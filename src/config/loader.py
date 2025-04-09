"""This module provides a function to load the pipeline configuration from a JSON file."""

from config.config_model import PipelineConfig
from config.config_store import Config


def load_pipeline_config(config_path: str) -> None:
    """Load the pipeline configuration from the specified file and set it as the configuration for the pipeline.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        None
    """
    import json

    with open(config_path) as f:
        data = json.load(f)
    pipeline_config = PipelineConfig(**data)
    Config.set_config(pipeline_config)
