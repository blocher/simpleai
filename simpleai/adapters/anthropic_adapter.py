"""Anthropic Claude adapter using Messages API."""

from __future__ import annotations

from copy import deepcopy
import json
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

    def _prompt_as_text(self, prompt: PromptInput) -> str:
        if isinstance(prompt, str):
            return prompt
        return "\n\n".join(str(item) for item in prompt)

    def _normalize_schema_for_anthropic(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Anthropic requires explicit additionalProperties for object schemas."""

        normalized = deepcopy(schema)

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                node_type = node.get("type")
                is_object = node_type == "object" or (
                    isinstance(node_type, list) and "object" in node_type
                )
                # Anthropic requires object schemas to explicitly set additionalProperties=false.
                # Enforce this even if a non-false value exists (e.g. dict-shaped schemas).
                looks_objectish = any(
                    key in node
                    for key in ("properties", "required", "patternProperties", "additionalProperties")
                )
                if is_object or looks_objectish:
                    node["additionalProperties"] = False

                for value in node.values():
                    walk(value)
                return

            if isinstance(node, list):
                for item in node:
                    walk(item)

        walk(normalized)
        return normalized

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

    def _extract_text(self, response_dict: dict[str, Any]) -> str:
        texts: list[str] = []
        for block in response_dict.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    def _has_web_search_result(self, response_dict: dict[str, Any]) -> bool:
        for block in response_dict.get("content", []):
            if block.get("type") == "web_search_tool_result":
                return True
        return False

    def _render_web_search_context(self, response_dict: dict[str, Any]) -> str:
        lines: list[str] = []
        for block in response_dict.get("content", []):
            if block.get("type") != "web_search_tool_result":
                continue
            raw_content = block.get("content") or []
            if isinstance(raw_content, dict):
                raw_content = [raw_content]
            for item in raw_content:
                title = item.get("title") or ""
                url = item.get("url") or ""
                age = item.get("page_age") or ""
                parts = [part for part in (title, url, age) if part]
                if parts:
                    lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _citation_key(self, item: Citation) -> tuple[Any, ...]:
        return (
            item.provider,
            item.url,
            item.title,
            item.source,
            item.snippet,
            item.citation_id,
            item.start_index,
            item.end_index,
        )

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
                        "schema": self._normalize_schema_for_anthropic(
                            output_format.model_json_schema()
                        ),
                    }
                }

            if adapter_options:
                payload.update(adapter_options)

            response = self.client.messages.create(**payload)
            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            text = self._extract_text(response_dict)

            citations = self._extract_citations(response_dict) if return_citations else []

            # If a forced search turn returns only tool blocks (no text), synthesize a final response.
            if not text:
                has_search_result = self._has_web_search_result(response_dict)
                if require_search and has_search_result:
                    search_context = self._render_web_search_context(response_dict)
                    prompt_text = self._prompt_as_text(prompt)
                    synthesis_text = (
                        f"{prompt_text}\n\n"
                        "Web search results already gathered:\n"
                        f"{search_context}\n\n"
                        "Return the final answer now. "
                        "If a JSON schema is required, return only valid JSON."
                    )
                    synthesis_payload: dict[str, Any] = {
                        "model": model,
                        "max_tokens": int(self.provider_settings.get("max_tokens", 4096)),
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": synthesis_text}],
                            }
                        ],
                    }
                    if output_format is not None:
                        synthesis_payload["output_config"] = {
                            "format": {
                                "type": "json_schema",
                                "schema": self._normalize_schema_for_anthropic(
                                    output_format.model_json_schema()
                                ),
                            }
                        }
                    if adapter_options:
                        passthrough = dict(adapter_options)
                        passthrough.pop("tools", None)
                        passthrough.pop("tool_choice", None)
                        synthesis_payload.update(passthrough)

                    synthesis_response = self.client.messages.create(**synthesis_payload)
                    synthesis_dict = (
                        synthesis_response.model_dump(mode="json")
                        if hasattr(synthesis_response, "model_dump")
                        else {}
                    )
                    text = self._extract_text(synthesis_dict)
                    if return_citations:
                        for extra in self._extract_citations(synthesis_dict):
                            if self._citation_key(extra) not in {self._citation_key(c) for c in citations}:
                                citations.append(extra)

                    if text:
                        response_dict = synthesis_dict

            # Last-resort fallback: some schema-compatible outputs may appear as tool-like input.
            if not text and output_format is not None:
                for block in response_dict.get("content", []):
                    if block.get("type") in {"tool_use", "server_tool_use"} and isinstance(block.get("input"), dict):
                        text = json.dumps(block["input"], ensure_ascii=True)
                        break

            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Anthropic adapter failed: {exc}") from exc
