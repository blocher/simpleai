"""Google Gemini adapter using google-genai SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.exceptions import ProviderError
from simpleai.types import AdapterResponse, Citation, PromptInput


class GeminiAdapter(BaseAdapter):
    provider_name = "gemini"
    supports_binary_files = True

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_settings)

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover - dependency missing path
            raise ProviderError("google-genai package is required for GeminiAdapter.") from exc

        api_key = provider_settings.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.types = types

    def _build_contents(self, prompt: PromptInput, files: Sequence[Path] | None) -> Any:
        contents: list[Any] = []

        if files:
            for path in files:
                uploaded = self.client.files.upload(file=str(path))
                contents.append(uploaded)

        if isinstance(prompt, str):
            contents.append(prompt)
        else:
            contents.extend(str(item) for item in prompt)

        if len(contents) == 1:
            return contents[0]
        return contents

    def _extract_citations(self, response_dict: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[Any, ...]] = set()

        def append_citation(
            *,
            url: str | None,
            title: str | None,
            source: str | None,
            snippet: str | None,
            start_index: int | None = None,
            end_index: int | None = None,
            raw: dict[str, Any],
        ) -> None:
            key = (url, title, source, snippet, start_index, end_index)
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
                    start_index=start_index,
                    end_index=end_index,
                    raw=raw,
                )
            )

        for candidate in response_dict.get("candidates", []):
            # Citation metadata (inline offsets + URI/title).
            citation_meta = candidate.get("citation_metadata") or candidate.get("citationMetadata") or {}
            for item in citation_meta.get("citations") or []:
                append_citation(
                    url=item.get("uri"),
                    title=item.get("title"),
                    source=item.get("uri"),
                    snippet=None,
                    start_index=item.get("start_index") or item.get("startIndex"),
                    end_index=item.get("end_index") or item.get("endIndex"),
                    raw=item,
                )

            # Grounding metadata from Google Search tool.
            grounding = candidate.get("grounding_metadata") or candidate.get("groundingMetadata") or {}
            chunks = grounding.get("grounding_chunks") or grounding.get("groundingChunks") or []
            for chunk in chunks:
                web = chunk.get("web") or {}
                if web:
                    append_citation(
                        url=web.get("uri") or web.get("url"),
                        title=web.get("title"),
                        source=web.get("domain") or web.get("uri") or web.get("url"),
                        snippet=None,
                        raw=chunk,
                    )

                retrieved = chunk.get("retrieved_context") or chunk.get("retrievedContext") or {}
                if retrieved:
                    append_citation(
                        url=retrieved.get("uri"),
                        title=retrieved.get("title") or retrieved.get("document_name") or retrieved.get("documentName"),
                        source=retrieved.get("document_name") or retrieved.get("documentName") or retrieved.get("uri"),
                        snippet=retrieved.get("text"),
                        raw=chunk,
                    )

                maps = chunk.get("maps") or {}
                if maps:
                    append_citation(
                        url=maps.get("uri"),
                        title=maps.get("title"),
                        source="google_maps",
                        snippet=maps.get("text"),
                        raw=chunk,
                    )

            # Query metadata can still be useful provenance even when chunks are absent.
            for query in grounding.get("web_search_queries") or grounding.get("webSearchQueries") or []:
                append_citation(
                    url=None,
                    title=None,
                    source="google_search_query",
                    snippet=str(query),
                    raw={"query": query},
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
            config_kwargs: dict[str, Any] = {
                "max_output_tokens": int(self.provider_settings.get("max_output_tokens", 8192)),
            }

            if require_search:
                config_kwargs["tools"] = [
                    self.types.Tool(google_search=self.types.GoogleSearch())
                ]
                config_kwargs.setdefault(
                    "system_instruction",
                    "Use Google Search to ground your answer and provide citations to sources.",
                )

            if output_format is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = output_format.model_json_schema()

            if adapter_options:
                config_kwargs.update(adapter_options)

            config = self.types.GenerateContentConfig(**config_kwargs)
            response = self.client.models.generate_content(
                model=model,
                contents=self._build_contents(prompt, files),
                config=config,
            )

            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            text = getattr(response, "text", "") or ""
            if not text and response_dict:
                chunks: list[str] = []
                for candidate in response_dict.get("candidates", []):
                    content = candidate.get("content") or {}
                    for part in content.get("parts") or []:
                        if "text" in part:
                            chunks.append(part["text"])
                text = "\n".join(chunks)

            citations = self._extract_citations(response_dict) if return_citations else []
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Gemini adapter failed: {exc}") from exc
