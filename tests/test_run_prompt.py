from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from simpleai.api import run_prompt
from simpleai.exceptions import SettingsError
from simpleai.types import AdapterResponse, Citation


class PayloadModel(BaseModel):
    value: int


class DummyAdapter:
    def __init__(self, supports_binary_files: bool = False) -> None:
        self.supports_binary_files = supports_binary_files
        self.last_kwargs: dict[str, Any] = {}

    def run(self, **kwargs: Any) -> AdapterResponse:
        self.last_kwargs = kwargs
        text = '{"value": 7}'
        citations = [Citation(provider="openai", url="https://example.com", title="Example")]
        return AdapterResponse(text=text, citations=citations, raw={"ok": True})


BASE_SETTINGS = {
    "defaults": ["openai"],
    "providers": {
        "openai": {
            "default_model": "gpt-5",
            "api_key": "sk-test",
        }
    },
    "logging": {"enabled": False},
}


def test_run_prompt_returns_model_instance(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    result = run_prompt("hello", model="openai", output_format=PayloadModel)
    assert isinstance(result, PayloadModel)
    assert result.value == 7


def test_run_prompt_returns_tuple_when_citations_enabled(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    result, citations = run_prompt("hello", model="openai", return_citations=True)
    assert isinstance(result, str)
    assert isinstance(citations, list)
    assert citations[0]["url"] == "https://example.com"
    assert adapter.last_kwargs["require_search"] is True
    assert adapter.last_kwargs["return_citations"] is True


def test_run_prompt_infers_return_citations_from_require_search(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    result, citations = run_prompt("hello", model="openai", require_search=True)
    assert isinstance(result, str)
    assert len(citations) == 1
    assert adapter.last_kwargs["require_search"] is True
    assert adapter.last_kwargs["return_citations"] is True


def test_return_citations_true_forces_require_search_even_if_false(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    run_prompt("hello", model="openai", require_search=False, return_citations=True)

    assert adapter.last_kwargs["require_search"] is True
    assert adapter.last_kwargs["return_citations"] is True


def test_run_prompt_extracts_files_when_binary_not_supported(monkeypatch, tmp_path: Path) -> None:
    adapter = DummyAdapter(supports_binary_files=False)
    note = tmp_path / "note.txt"
    note.write_text("attached content", encoding="utf-8")

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    run_prompt("base prompt", model="openai", file=note, binary_files=True)

    prompt_payload = adapter.last_kwargs["prompt"]
    assert isinstance(prompt_payload, str)
    assert "attached content" in prompt_payload
    assert adapter.last_kwargs["files"] is None


def test_run_prompt_passes_binary_files_when_supported(monkeypatch, tmp_path: Path) -> None:
    adapter = DummyAdapter(supports_binary_files=True)
    note = tmp_path / "note.txt"
    note.write_text("attached content", encoding="utf-8")

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    run_prompt("base prompt", model="openai", file=note, binary_files=True)

    files = adapter.last_kwargs["files"]
    assert files is not None
    assert files[0] == note.resolve()


def test_run_prompt_merges_provider_kwargs(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    run_prompt("hello", model="openai", temperature=0.2, adapter_options={"top_p": 0.8})

    options = adapter.last_kwargs["adapter_options"]
    assert options["top_p"] == 0.8
    assert options["temperature"] == 0.2


def test_run_prompt_missing_provider_key_raises_settings_error(monkeypatch) -> None:
    settings = {
        "defaults": ["grok"],
        "providers": {
            "grok": {
                "default_model": "grok-4-latest",
                "api_key": None,
            }
        },
        "logging": {"enabled": False},
    }

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: settings)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("grok", "grok-4-latest"))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)

    with pytest.raises(SettingsError) as exc:
        run_prompt("hello", model="grok")

    message = str(exc.value)
    assert "Missing API key for provider 'grok'" in message
    assert "XAI_API_KEY" in message
    assert "GROK_API_KEY" in message
