"""Energy Pipeline CLI."""

from commands.dispatcher import handle_command
from utils.args_parser import parse_args


def main() -> None:
    """Main entry point for the Energy Pipeline CLI."""
    args = parse_args()
    handle_command(args)


if __name__ == "__main__":
    main()
