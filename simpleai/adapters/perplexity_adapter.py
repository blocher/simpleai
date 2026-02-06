"""Perplexity adapter using the Responses API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.exceptions import ProviderError
from simpleai.types import AdapterResponse, Citation, PromptInput


class PerplexityAdapter(BaseAdapter):
    provider_name = "perplexity"
    supports_binary_files = False

    _PRESET_ALIASES = {
        "fast-search": "fast-search",
        "pro-search": "pro-search",
        "deep-research": "deep-research",
        # Backward-compatible aliases from older Sonar naming.
        "sonar": "fast-search",
        "sonar-pro": "pro-search",
        "sonar-reasoning": "pro-search",
        "sonar-reasoning-pro": "deep-research",
        "sonar-deep-research": "deep-research",
    }

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_settings)

        try:
            from perplexity import Perplexity
        except Exception as exc:  # pragma: no cover - dependency missing path
            raise ProviderError("perplexityai package is required for PerplexityAdapter.") from exc

        api_key = (
            provider_settings.get("api_key")
            or os.getenv("PERPLEXITY_API_KEY")
            or os.getenv("PPLX_API_KEY")
        )
        self.client = Perplexity(api_key=api_key)

    def _build_input(self, prompt: PromptInput) -> str | list[dict[str, Any]]:
        if isinstance(prompt, str):
            return prompt

        messages: list[dict[str, Any]] = []
        for turn in prompt:
            messages.append({
                "type": "message",
                "role": "user",
                "content": str(turn),
            })
        if not messages:
            return ""
        return messages

    def _resolve_model_target(self, model: str) -> dict[str, str]:
        normalized = model.strip()
        lowered = normalized.lower()

        preset = self._PRESET_ALIASES.get(lowered)
        if preset:
            return {"preset": preset}

        # Responses API model names are provider/model.
        if "/" in normalized:
            return {"model": normalized}

        # Heuristic provider prefixing for common raw model names.
        if lowered.startswith(("gpt-", "o1", "o3", "o4")):
            return {"model": f"openai/{normalized}"}
        if lowered.startswith("claude"):
            return {"model": f"anthropic/{normalized}"}
        if lowered.startswith("gemini"):
            return {"model": f"google/{normalized}"}
        if lowered.startswith("grok"):
            return {"model": f"xai/{normalized}"}
        if lowered.startswith("sonar"):
            return {"model": f"perplexity/{normalized}"}

        return {"model": normalized}

    def _extract_citations(self, response_dict: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []

        for output_item in response_dict.get("output", []):
            out_type = output_item.get("type")

            if out_type == "message":
                for part in output_item.get("content", []):
                    for annotation in part.get("annotations") or []:
                        citations.append(
                            Citation(
                                provider=self.provider_name,
                                url=annotation.get("url"),
                                title=annotation.get("title"),
                                source=annotation.get("url"),
                                start_index=annotation.get("start_index"),
                                end_index=annotation.get("end_index"),
                                raw=annotation,
                            )
                        )

            if out_type == "search_results":
                for result in output_item.get("results") or []:
                    citations.append(
                        Citation(
                            provider=self.provider_name,
                            url=result.get("url"),
                            title=result.get("title"),
                            source=result.get("source"),
                            snippet=result.get("snippet"),
                            raw=result,
                        )
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
                "input": self._build_input(prompt),
            }

            payload.update(self._resolve_model_target(model))

            if require_search:
                payload["tools"] = [{"type": "web_search"}]

            if output_format is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "simpleai_output",
                        "schema": output_format.model_json_schema(),
                        "strict": True,
                    },
                }

            if adapter_options:
                payload.update(adapter_options)

            response = self.client.responses.create(**payload)
            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            text = getattr(response, "output_text", "") or ""

            citations = self._extract_citations(response_dict) if return_citations else []
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Perplexity adapter failed: {exc}") from exc
