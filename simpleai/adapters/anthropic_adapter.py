"""Anthropic Claude adapter using Messages API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.exceptions import ProviderError
from simpleai.types import AdapterResponse, Citation, PromptInput


class AnthropicAdapter(BaseAdapter):
    provider_name = "claude"
    supports_binary_files = False

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_settings)

        try:
            from anthropic import Anthropic
        except Exception as exc:  # pragma: no cover - dependency missing path
            raise ProviderError("anthropic package is required for AnthropicAdapter.") from exc

        api_key = provider_settings.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        self.client = Anthropic(api_key=api_key)

    def _build_messages(self, prompt: PromptInput) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        messages: list[dict[str, Any]] = []
        for turn in prompt:
            messages.append({"role": "user", "content": [{"type": "text", "text": str(turn)}]})

        if not messages:
            messages.append({"role": "user", "content": [{"type": "text", "text": ""}]})

        return messages

    def _extract_citations(self, response_dict: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[Any, ...]] = set()

        def append_citation(
            *,
            url: str | None,
            title: str | None,
            source: str | None,
            snippet: str | None,
            raw: dict[str, Any],
        ) -> None:
            key = (url, title, source, snippet)
            if key in seen:
                return
            seen.add(key)
            citations.append(
                Citation(
                    provider=self.provider_name,
                    url=url,
                    title=title,
                    source=source,
                    snippet=snippet,
                    raw=raw,
                )
            )

        for block in response_dict.get("content", []):
            if block.get("type") == "text":
                for item in block.get("citations") or []:
                    source_obj = item.get("source") or {}
                    if not isinstance(source_obj, dict):
                        source_obj = {"source": source_obj}
                    url = item.get("url") or source_obj.get("url")
                    title = item.get("title") or source_obj.get("title")
                    source = url or source_obj.get("source")
                    append_citation(
                        url=url,
                        title=title,
                        source=source,
                        snippet=item.get("cited_text"),
                        raw=item,
                    )

            if block.get("type") == "web_search_tool_result":
                raw_content = block.get("content") or []
                items = [raw_content] if isinstance(raw_content, dict) else raw_content
                for result in items:
                    append_citation(
                        url=result.get("url"),
                        title=result.get("title"),
                        source=result.get("url"),
                        snippet=None,
                        raw=result,
                    )

        return citations

    def run(
        self,
        *,
        prompt: PromptInput,
        model: str,
        require_search: bool,
        return_citations: bool,
        files: Sequence[Path] | None,
        output_format: type[BaseModel] | None,
        adapter_options: dict[str, Any] | None,
    ) -> AdapterResponse:
        del files  # unsupported in this adapter; caller should pass extracted text instead

        try:
            payload: dict[str, Any] = {
                "model": model,
                "max_tokens": int(self.provider_settings.get("max_tokens", 4096)),
                "messages": self._build_messages(prompt),
            }

            if require_search:
                payload["tools"] = [
                    {
                        "name": "web_search",
                        "type": "web_search_20250305",
                    }
                ]
                payload["tool_choice"] = {"type": "any"}

            if output_format is not None:
                payload["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": output_format.model_json_schema(),
                    }
                }

            if adapter_options:
                payload.update(adapter_options)

            response = self.client.messages.create(**payload)
            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}

            texts: list[str] = []
            for block in response_dict.get("content", []):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            text = "\n".join(item for item in texts if item)

            citations = self._extract_citations(response_dict) if return_citations else []
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Anthropic adapter failed: {exc}") from exc
