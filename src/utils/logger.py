"""Logger with context support for logging messages in a structured way."""

import logging
import os
from datetime import datetime
from typing import Any, cast, override

from rich.logging import RichHandler


class ContextLogger(logging.Logger):
    """Custom logger class that supports logging with additional context.

    ContextLogger is a custom logger class that extends the standard `logging.Logger` to support
    logging with additional context. It allows for conditional logging based on the presence of
    a context dictionary, which can be used to buffer log messages or pass additional metadata.

    Methods:
        - info(msg, *args, **kwargs):
            Logs a message with level INFO. Supports an optional `context` dictionary.
        - debug(msg, *args, **kwargs):
            Logs a message with level DEBUG. Supports an optional `context` dictionary.
        - warning(msg, *args, **kwargs):
            Logs a message with level WARNING. Supports an optional `context` dictionary.
        - error(msg, *args, **kwargs):
            Logs a message with level ERROR. Supports an optional `context` dictionary.
        - critical(msg, *args, **kwargs):
            Logs a message with level CRITICAL. Supports an optional `context` dictionary.
        - exception(msg, *args, **kwargs):
            Logs a message with level ERROR and includes exception traceback information.
            Supports an optional `context` dictionary.
        - log(level, msg, *args, **kwargs):
            Logs a message with a specified logging level. Supports an optional `context` dictionary.

    Special Behavior:
        - If a `context` dictionary is provided and contains the key `"worker_process"`, log messages
          are buffered into the `"log_buffer"` key of the context instead of being immediately logged.
        - If no `context` is provided or `"worker_process"` is not set, messages are logged normally.

    Parameters:
        - level (int): The logging level (e.g., INFO, DEBUG, etc.).
        - msg (str): The message to log.
        - context (dict[str, Any] | None, optional): A dictionary containing additional context for
          the log message. Defaults to None.
        - *args (Any): Additional positional arguments for the log message.
        - **kwargs (Any): Additional keyword arguments for the log message.
    """

    def _log_with_context(
        self,
        level: int,
        msg: str,
        context: dict[str, Any] | None,
        *args: tuple[Any],
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        if context is not None and context.get("worker_process"):
            context.setdefault("log_buffer", []).append((level, msg))
        else:
            super().log(level, msg, *args, **kwargs)

    @override
    def info(self, msg: object, *args: Any, **kwargs: Any) -> None:
        context = cast(dict[str, Any] | None, kwargs.pop("context", None))
        self._log_with_context(logging.INFO, str(msg), context, *args, **kwargs)

    @override
    def debug(self, msg: object, *args: Any, **kwargs: Any) -> None:
        context = cast(dict[str, Any] | None, kwargs.pop("context", None))
        self._log_with_context(logging.DEBUG, str(msg), context, *args, **kwargs)

    @override
    def warning(self, msg: object, *args: Any, **kwargs: Any) -> None:
        context = cast(dict[str, Any] | None, kwargs.pop("context", None))
        self._log_with_context(logging.WARNING, str(msg), context, *args, **kwargs)

    @override
    def error(self, msg: object, *args: Any, **kwargs: Any) -> None:
        context = cast(dict[str, Any] | None, kwargs.pop("context", None))
        self._log_with_context(logging.ERROR, str(msg), context, *args, **kwargs)

    @override
    def critical(self, msg: object, *args: Any, **kwargs: Any) -> None:
        context = cast(dict[str, Any] | None, kwargs.pop("context", None))
        self._log_with_context(logging.CRITICAL, str(msg), context, *args, **kwargs)

    @override
    def exception(self, msg: object, *args: Any, **kwargs: Any) -> None:
        context = cast(dict[str, Any] | None, kwargs.pop("context", None))
        kwargs.setdefault("exc_info", True)  # ensures traceback is logged
        self._log_with_context(logging.ERROR, str(msg), context, *args, **kwargs)

    @override
    def log(self, level: int, msg: object, *args: Any, **kwargs: Any) -> None:
        context = cast(dict[str, Any] | None, kwargs.pop("context", None))
        self._log_with_context(level, str(msg), context, *args, **kwargs)


# Set custom logger class
logging.setLoggerClass(ContextLogger)

# Create and configure logger
logger = cast(ContextLogger, logging.getLogger("energy-pipeline"))
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Prevent messages from propagating to ancestor loggers

# Add handlers only if not already added
if not logger.handlers:
    # Rich handler for pretty stdout
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setLevel(logging.DEBUG)
    logger.addHandler(rich_handler)

    # Optional file handler
    SAVE_LOGS_TO_FILE = True
    if SAVE_LOGS_TO_FILE:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"debug_{timestamp}.log")
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.WARNING)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
