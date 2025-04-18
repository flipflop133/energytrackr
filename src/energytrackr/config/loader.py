"""This module provides a function to load the pipeline configuration from a JSON file."""

import json

from energytrackr.config.config_model import PipelineConfig
from energytrackr.config.config_store import Config


def load_pipeline_config(config_path: str) -> None:
    """Load the pipeline configuration from the specified file and set it as the configuration for the pipeline.

    Args:
        config_path (str): The path to the JSON configuration file.
    """
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    pipeline_config = PipelineConfig(**data)
    Config.set_config(pipeline_config)
