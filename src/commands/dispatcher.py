"""Command dispatcher for the energy measurement tool."""

import argparse

from pipeline.pipeline import measure
from plot.plot import create_energy_plots
from utils.exceptions import UnknownCommandError
from utils.sort import reorder_commits


def handle_command(args: argparse.Namespace) -> None:
    """Handle the CLI command based on the parsed arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        UnknownCommandError: If the command is not recognized.
    """
    match args.command:
        case "measure":
            # Measure energy consumption
            measure(args.config)
        case "sort":
            # Sort a result file
            reorder_commits(args.file, args.repo_path, args.output_file)
        case "plot":
            # Plot a result file
            create_energy_plots(args.file, args.repo_path)
        case _:
            raise UnknownCommandError(args.command)
