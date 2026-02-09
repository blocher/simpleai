"""Settings loading for SimpleAI.

Priority:
1) Django settings (SIMPLEAI or SIMPLEAI_SETTINGS) when Django is configured.
2) ai_settings.json from explicit path, env var, and common app-root locations.
3) Built-in defaults.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .exceptions import SettingsError

DEFAULT_SETTINGS: dict[str, Any] = {
    "defaults": ["gemini", "openai", "claude", "grok", "perplexity"],
    "providers": {
        "gemini": {
            "api_key": None,
            "default_model": "gemini-3-pro-preview",
            "max_output_tokens": 8192,
        },
        "claude": {
            "api_key": None,
            "default_model": "claude-opus-4-6",
            "max_tokens": 4096,
            "max_retries": 3,  # retries on 429 errors (uses retry-after header)
            "skip_citation_followup": False,  # skip extra API call for citations
        },
        "openai": {
            "api_key": None,
            "default_model": "gpt-5.2",
            "max_output_tokens": 8192,
            "base_url": None,
        },
        "grok": {
            "api_key": None,
            "default_model": "grok-4-latest",
            "max_tokens": 8192,
        },
        "perplexity": {
            "api_key": None,
            "default_model": "sonar-deep-research",
            "max_output_tokens": 4096,
        },
    },
    "logging": {
        "enabled": False,
        "network_logging": False,
        "django_logfile": "django",
        "logfile_location": "./simpleai.log",
    },
}

PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "perplexity": ("PERPLEXITY_API_KEY", "PPLX_API_KEY"),
}

_PROVIDER_ALIASES = {
    "google": "gemini",
    "gemini": "gemini",
    "anthropic": "claude",
    "claude": "claude",
    "openai": "openai",
    "chatgpt": "openai",
    "grok": "grok",
    "xai": "grok",
    "perplexity": "perplexity",
    "perplexityai": "perplexity",
}

_APP_ROOT_MARKERS = ("pyproject.toml", "setup.py", "manage.py", ".git")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def canonical_provider_name(name: str) -> str | None:
    """Return canonical provider key for aliases."""

    return _PROVIDER_ALIASES.get(name.strip().lower())



def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            # Convert Mapping to dict for recursive merge
            merged[key] = _deep_merge(merged[key], dict(value))
        else:
            merged[key] = value
    return merged



def _normalize_user_settings(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(raw)

    providers_raw = normalized.get("providers") or normalized.get("provider") or {}
    providers_normalized: dict[str, Any] = {}
    for key, value in providers_raw.items():
        canonical = canonical_provider_name(str(key)) or str(key).lower()
        providers_normalized[canonical] = value
    normalized["providers"] = providers_normalized

    defaults_raw = normalized.get("defaults")
    if isinstance(defaults_raw, list):
        mapped_defaults: list[str] = []
        for item in defaults_raw:
            if not isinstance(item, str):
                continue
            canonical = canonical_provider_name(item) or item.strip().lower()
            if canonical not in mapped_defaults:
                mapped_defaults.append(canonical)
        if mapped_defaults:
            normalized["defaults"] = mapped_defaults

    return normalized



def _load_from_django() -> dict[str, Any] | None:
    try:
        from django.conf import settings as django_settings  # type: ignore
    except Exception:
        return None

    if not getattr(django_settings, "configured", False):
        return None

    for attr in ("SIMPLEAI", "SIMPLEAI_SETTINGS"):
        value = getattr(django_settings, attr, None)
        if isinstance(value, Mapping):
            return dict(value)

    return None



def _json_candidates(explicit: str | Path | None) -> list[Path]:
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(explicit))

    env_path = os.getenv("SIMPLEAI_SETTINGS_FILE")
    if env_path:
        candidates.append(Path(env_path))

    for root in _application_roots():
        candidates.append(root / "ai_settings.json")
        candidates.append(root / "config" / "ai_settings.json")
        candidates.append(root / "settings" / "ai_settings.json")

    # Allow package-local example to be copied and edited.
    candidates.append(Path(__file__).resolve().parents[1] / "ai_settings.json")

    return _dedupe_paths(candidates)


def _application_roots() -> list[Path]:
    seed_roots: list[Path] = [Path.cwd()]

    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    if main_file:
        seed_roots.append(Path(main_file).expanduser().resolve().parent)

    env_root = os.getenv("SIMPLEAI_APP_ROOT")
    if env_root:
        seed_roots.append(Path(env_root).expanduser().resolve())

    traversed: list[Path] = []
    for seed in _dedupe_paths(seed_roots):
        traversed.append(seed)
        traversed.extend(seed.parents)

    traversed = _dedupe_paths(traversed)
    marked = [root for root in traversed if any((root / marker).exists() for marker in _APP_ROOT_MARKERS)]

    return _dedupe_paths(marked + traversed)



def _load_from_json(explicit: str | Path | None) -> dict[str, Any] | None:
    for path in _json_candidates(explicit):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SettingsError(f"Invalid JSON in settings file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise SettingsError(f"Settings file {path} must contain a JSON object.")
        return data

    return None



def get_provider_api_key(settings: dict[str, Any], provider: str) -> str | None:
    """Resolve provider API key from settings first, then environment."""

    provider_config = settings.get("providers", {}).get(provider, {})
    if isinstance(provider_config, dict):
        configured = provider_config.get("api_key")
        if configured:
            return str(configured)

    for env_var in PROVIDER_ENV_VARS.get(provider, ()):
        value = os.getenv(env_var)
        if value:
            return value

    return None


def expected_provider_env_vars(provider: str) -> tuple[str, ...]:
    """Return accepted environment variable names for a provider."""

    return PROVIDER_ENV_VARS.get(provider, ())



def load_settings(settings_file: str | Path | None = None) -> dict[str, Any]:
    """Load settings from Django, JSON file, then defaults."""

    merged = deepcopy(DEFAULT_SETTINGS)

    django_data = _load_from_django()
    if django_data:
        merged = _deep_merge(merged, _normalize_user_settings(django_data))
        return merged

    json_data = _load_from_json(settings_file)
    if json_data:
        merged = _deep_merge(merged, _normalize_user_settings(json_data))

    return merged
