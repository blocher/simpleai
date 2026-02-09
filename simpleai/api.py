"""Public API for running prompts across providers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from .adapters import get_adapter
from .adapters.logging_adapter import PromptLogger
from .exceptions import ProviderError, SettingsError, SimpleAIException
from .files import collect_file_paths, extract_text_from_files
from .model_registry import resolve_provider_and_model
from .settings import expected_provider_env_vars, get_provider_api_key, load_settings
from .types import PromptInput
from .utils import coerce_output


def _coerce_bool(value: bool | str | None, *, name: str, allow_none: bool) -> bool | None:
    if value is None:
        if allow_none:
            return None
        raise SettingsError(f"{name} cannot be None.")

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False

    raise SettingsError(f"{name} must be a boolean value.")


def _append_extracted_files_to_prompt(prompt: PromptInput, extracted: Iterable[tuple[Path, str]]) -> PromptInput:
    blocks = []
    for path, text in extracted:
        blocks.append(f"[File: {path.name}]\n{text}")
    file_context = "\n\n".join(blocks)

    if isinstance(prompt, str):
        return f"{prompt}\n\nIncluded file text:\n{file_context}" if file_context else prompt

    prompt_list = list(prompt)
    if file_context:
        prompt_list.append(f"Included file text:\n{file_context}")
    return prompt_list



def _build_log_args(
    prompt: PromptInput,
    require_search: bool,
    return_citations: bool,
    file: str | Path | None,
    files: str | Path | Iterable[str | Path] | None,
    binary_files: bool,
    model: str | None,
    output_format: type[BaseModel] | None,
    provider_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "require_search": require_search,
        "return_citations": return_citations,
        "file": str(file) if file is not None else None,
        "files": [str(item) for item in files] if isinstance(files, (list, tuple, set)) else str(files) if files else None,
        "binary_files": binary_files,
        "model": model,
        "output_format": output_format.__name__ if output_format else None,
        "provider_kwargs": provider_kwargs,
    }



def run_prompt(
    prompt: PromptInput,
    *,
    require_search: bool = False,
    return_citations: bool | None = None,
    file: str | Path | None = None,
    files: str | Path | Iterable[str | Path] | None = None,
    binary_files: bool = True,
    model: str | None = None,
    output_format: type[BaseModel] | None = None,
    settings_file: str | Path | None = None,
    adapter_options: dict[str, Any] | None = None,
    **provider_kwargs: Any,
) -> Any:
    """Run a prompt on the resolved provider model.

    Args:
        prompt: Required prompt string or list of conversation turns.
        require_search: If True, enables provider-native search tools.
        return_citations: Defaults to True when require_search is True, else False.
        file: Optional single file path.
        files: Optional single path or list of paths.
        binary_files: If True and adapter supports it, upload files as binary attachments.
            Otherwise files are extracted to text and appended to the prompt.
        model: Provider alias (e.g. "openai") or specific model ID.
        output_format: Optional Pydantic model type for structured output validation.
        settings_file: Optional override path to ai_settings.json.
        adapter_options: Explicit provider payload overrides.
        **provider_kwargs: Additional provider payload overrides.

    Returns:
        Plain text or validated Pydantic object.
        If return_citations is True, returns (result, citations).
    """
    try:
        require_search_bool = bool(_coerce_bool(require_search, name="require_search", allow_none=False))
        return_citations_bool = _coerce_bool(return_citations, name="return_citations", allow_none=True)
        binary_files_bool = bool(_coerce_bool(binary_files, name="binary_files", allow_none=False))

        effective_return_citations = (
            require_search_bool if return_citations_bool is None else bool(return_citations_bool)
        )
        # Citations require grounded search context; citations always force search on.
        effective_require_search = require_search_bool or effective_return_citations

        settings = load_settings(settings_file)
        provider, resolved_model = resolve_provider_and_model(settings, model)

        providers = settings.get("providers", {})
        provider_settings = providers.get(provider, {}) if isinstance(providers, dict) else {}
        if not isinstance(provider_settings, dict):
            raise SettingsError(f"Invalid settings for provider '{provider}'.")

        if not provider_settings.get("api_key"):
            provider_settings = dict(provider_settings)
            provider_settings["api_key"] = get_provider_api_key(settings, provider)

        if not provider_settings.get("api_key"):
            env_vars = expected_provider_env_vars(provider)
            env_hint = ", ".join(env_vars) if env_vars else "provider-specific env var"
            raise SettingsError(
                f"Missing API key for provider '{provider}'. "
                f"Set providers.{provider}.api_key or one of: {env_hint}."
            )

        adapter = get_adapter(provider, provider_settings)

        # File handling: binary upload if supported; otherwise append extracted text.
        prompt_payload: PromptInput = prompt
        adapter_files: list[Path] | None = None
        file_paths = collect_file_paths(file=file, files=files)
        if file_paths:
            if binary_files_bool and adapter.supports_binary_files:
                adapter_files = file_paths
            else:
                extracted = extract_text_from_files(file_paths)
                prompt_payload = _append_extracted_files_to_prompt(
                    prompt_payload,
                    ((item.path, item.text) for item in extracted),
                )

        logger = PromptLogger(settings.get("logging", {}))
        started_at = time.time()

        combined_adapter_options: dict[str, Any] = {}
        if adapter_options:
            combined_adapter_options.update(adapter_options)
        combined_adapter_options.update(provider_kwargs)

        event_id = logger.log_start(
            args=_build_log_args(
                prompt=prompt,
                require_search=effective_require_search,
                return_citations=effective_return_citations,
                file=file,
                files=files,
                binary_files=binary_files_bool,
                model=model,
                output_format=output_format,
                provider_kwargs=provider_kwargs,
            ),
            adapter_payload={
                "provider": provider,
                "model": resolved_model,
                "require_search": effective_require_search,
                "return_citations": effective_return_citations,
                "binary_files": binary_files_bool,
                "adapter_supports_binary": adapter.supports_binary_files,
                "file_count": len(file_paths),
                "params": combined_adapter_options,
            },
        )

        try:
            adapter_response = adapter.run(
                prompt=prompt_payload,
                model=resolved_model,
                require_search=effective_require_search,
                return_citations=effective_return_citations,
                files=adapter_files,
                output_format=output_format,
                adapter_options=combined_adapter_options or None,
            )
        except Exception as exc:
            logger.log_error(
                event_id=event_id,
                started_at=started_at,
                error=exc,
                context={
                    "provider": provider,
                    "model": resolved_model,
                },
            )
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"Provider '{provider}' failed: {exc}") from exc

        result = coerce_output(adapter_response.text, output_format)
        citations = [item.to_dict() for item in adapter_response.citations]

        logger.log_end(
            event_id=event_id,
            started_at=started_at,
            result_preview=adapter_response.text,
            citations_count=len(citations),
        )

        if effective_return_citations:
            return result, citations
        return result
    except Exception as exc:
        if isinstance(exc, SimpleAIException):
            raise
        raise SimpleAIException(
            f"run_prompt failed: {exc.__class__.__name__}: {exc}",
            original_exception=exc,
        ) from exc
