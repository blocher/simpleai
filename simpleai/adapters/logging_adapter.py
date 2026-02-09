"""Central logging adapter for prompt execution telemetry."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


def _is_django_configured() -> bool:
    try:
        from django.conf import settings as django_settings  # type: ignore
    except Exception:
        return False
    return bool(getattr(django_settings, "configured", False))


class PromptLogger:
    """Structured logger for `run_prompt` lifecycle events."""

    def __init__(self, logging_settings: dict[str, Any] | None) -> None:
        self.settings = logging_settings or {}
        self.enabled = bool(self.settings.get("enabled", False))
        self.logger: logging.Logger | None = None

        if not self.enabled:
            return

        self.logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        if _is_django_configured():
            logger_name = str(self.settings.get("django_logfile") or "django")
            logger = logging.getLogger(logger_name)
            return logger

        logger = logging.getLogger("simpleai")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            logfile = Path(str(self.settings.get("logfile_location") or "./simpleai.log"))
            logfile.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(logfile)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _emit(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.logger is None:
            return

        payload.setdefault("ts", time.time())
        self.logger.info(json.dumps(payload, default=str, ensure_ascii=True, indent=2))

    def log_start(self, args: dict[str, Any], adapter_payload: dict[str, Any]) -> str:
        event_id = str(uuid4())
        self._emit(
            {
                "event": "run_prompt.start",
                "event_id": event_id,
                "args": args,
                "adapter_payload": adapter_payload,
            }
        )
        return event_id

    def log_end(
        self,
        event_id: str,
        started_at: float,
        result_preview: str,
        citations_count: int,
    ) -> None:
        ended_at = time.time()
        self._emit(
            {
                "event": "run_prompt.end",
                "event_id": event_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_seconds": ended_at - started_at,
                "result_preview": result_preview[:5000],
                "citations_count": citations_count,
            }
        )

    def log_error(
        self,
        event_id: str,
        started_at: float,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        ended_at = time.time()
        self._emit(
            {
                "event": "run_prompt.error",
                "event_id": event_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_seconds": ended_at - started_at,
                "error_type": type(error).__name__,
                "error": str(error),
                "context": context,
            }
        )
