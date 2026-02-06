"""OpenAI adapter using the Responses API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.exceptions import ProviderError
from simpleai.schema import openai_response_schema
from simpleai.types import AdapterResponse, Citation, PromptInput


class OpenAIAdapter(BaseAdapter):
    provider_name = "openai"
    supports_binary_files = True

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_settings)

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - dependency missing path
            raise ProviderError("openai package is required for OpenAIAdapter.") from exc

        api_key = provider_settings.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = provider_settings.get("base_url")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)

    def _build_input(self, prompt: PromptInput, file_ids: Sequence[str]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": str(turn)}],
                }
                for turn in prompt
            ]

        if not messages:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": ""}],
                }
            ]

        if file_ids:
            content = messages[-1]["content"]
            for file_id in file_ids:
                content.append({"type": "input_file", "file_id": file_id})

        return messages

    def _extract_citations(self, response_dict: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[Any, ...]] = set()

        def append_citation(
            *,
            url: str | None,
            title: str | None,
            source: str | None,
            start_index: int | None,
            end_index: int | None,
            raw: dict[str, Any],
        ) -> None:
            key = (url, title, source, start_index, end_index)
            if key in seen:
                return
            seen.add(key)
            citations.append(
                Citation(
                    provider=self.provider_name,
                    url=url,
                    title=title,
                    source=source,
                    start_index=start_index,
                    end_index=end_index,
                    raw=raw,
                )
            )

        # 1) Message text annotations (inline citations in generated text).
        for output in response_dict.get("output", []):
            if output.get("type") != "message":
                continue

            for part in output.get("content", []):
                for annotation in part.get("annotations") or []:
                    url_citation = annotation.get("url_citation") or {}
                    url = annotation.get("url") or url_citation.get("url")
                    title = annotation.get("title") or url_citation.get("title")
                    start_index = annotation.get("start_index")
                    if start_index is None:
                        start_index = url_citation.get("start_index")
                    end_index = annotation.get("end_index")
                    if end_index is None:
                        end_index = url_citation.get("end_index")

                    append_citation(
                        url=url,
                        title=title,
                        source=url,
                        start_index=start_index,
                        end_index=end_index,
                        raw=annotation,
                    )

        # 2) Full source list from web_search_call output (when include contains sources).
        for output in response_dict.get("output", []):
            if output.get("type") != "web_search_call":
                continue
            action = output.get("action") or {}
            for src in action.get("sources") or []:
                url = src.get("url")
                title = src.get("title")
                source_type = src.get("type") or src.get("source")
                append_citation(
                    url=url,
                    title=title,
                    source=source_type or url,
                    start_index=None,
                    end_index=None,
                    raw=src,
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
        try:
            file_ids: list[str] = []
            if files:
                for path in files:
                    with path.open("rb") as handle:
                        uploaded = self.client.files.create(file=handle, purpose="user_data")
                    file_ids.append(uploaded.id)

            payload: dict[str, Any] = {
                "model": model,
                "input": self._build_input(prompt, file_ids),
            }

            if require_search:
                payload["tools"] = [{"type": "web_search"}]
                payload["tool_choice"] = "required"
                if return_citations:
                    payload["include"] = ["web_search_call.action.sources"]

            if output_format is not None:
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "simpleai_output",
                        "schema": openai_response_schema(output_format),
                        "strict": True,
                    }
                }

            if adapter_options:
                payload.update(adapter_options)

            response = self.client.responses.create(**payload)
            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            text = getattr(response, "output_text", "")
            if not text and response_dict:
                chunks: list[str] = []
                for output in response_dict.get("output", []):
                    if output.get("type") != "message":
                        continue
                    for part in output.get("content", []):
                        if part.get("type") == "output_text":
                            chunks.append(part.get("text", ""))
                text = "".join(chunks)

            citations = self._extract_citations(response_dict) if return_citations else []
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"OpenAI adapter failed: {exc}") from exc
