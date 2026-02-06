"""SimpleAI exception hierarchy."""

from __future__ import annotations


class SimpleAIException(Exception):
    """Catch-all exception for errors surfaced by `run_prompt`."""

    def __init__(
        self,
        message: str,
        *,
        original_exception: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.original_exception = original_exception


class SimpleAIError(SimpleAIException):
    """Base error for SimpleAI (backward-compatible alias)."""


class SettingsError(SimpleAIError):
    """Raised for invalid or missing configuration."""


class ProviderError(SimpleAIError):
    """Raised when a provider adapter fails."""


class ModelResolutionError(SimpleAIError):
    """Raised when model/provider resolution fails."""


class FileExtractionError(SimpleAIError):
    """Raised when file extraction fails."""
