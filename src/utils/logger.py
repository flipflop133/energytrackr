import logging
from typing import Any, cast
from typing_extensions import override


class ContextLogger(logging.Logger):
    def _log_with_context(self, level: int, msg: str, context: dict[str, Any] | None, *args: Any, **kwargs: Any) -> None:
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
        self._log_with_context(logging.ERROR, str(msg), context, *args, **kwargs)


logging.setLoggerClass(ContextLogger)
logger = cast(ContextLogger, logging.getLogger("energy-pipeline"))
