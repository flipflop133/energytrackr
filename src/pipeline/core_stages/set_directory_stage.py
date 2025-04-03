import logging
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class SetDirectoryStage(PipelineStage):
    def run(self, context: dict[str, Any]) -> None:
        config = Config.get_config()
        if config.execution_plan.batch_size:
            # cd into the commit directory
            run_command(f"cd {config.repo_path}_{context['commit']}", cwd=config.repo_path)
