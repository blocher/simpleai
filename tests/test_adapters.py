from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field

from simpleai.adapters.anthropic_adapter import AnthropicAdapter
from simpleai.adapters.gemini_adapter import GeminiAdapter
from simpleai.adapters.grok_adapter import GrokAdapter
from simpleai.adapters.openai_adapter import OpenAIAdapter
from simpleai.adapters.perplexity_adapter import PerplexityAdapter


class OutputModel(BaseModel):
    value: int


class OutputWithDictModel(BaseModel):
    value: int
    metadata: dict[str, str]


class OutputWithBoundedArrayModel(BaseModel):
    values: list[int] = Field(min_length=1, max_length=3)


def test_openai_adapter_payload_and_citations(tmp_path: Path) -> None:
    upload_file = tmp_path / "data.txt"
    upload_file.write_text("hello", encoding="utf-8")

    class FakeOpenAIResponse:
        output_text = "ok"

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "sources": [
                                {"url": "https://source.example", "title": "Source Title", "type": "web_page"}
                            ]
                        },
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url_citation": {
                                            "url": "https://example.com",
                                            "title": "Example",
                                            "start_index": 0,
                                            "end_index": 2,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

    class FakeFiles:
        def __init__(self) -> None:
            self.calls = []

        def create(self, file, purpose: str):
            self.calls.append((purpose, bool(file.read())))
            file.seek(0)
            return SimpleNamespace(id="file-1")

    class FakeResponses:
        def __init__(self) -> None:
            self.payload = None

        def create(self, **kwargs):
            self.payload = kwargs
            return FakeOpenAIResponse()

    fake_files = FakeFiles()
    fake_responses = FakeResponses()

    adapter = OpenAIAdapter({"api_key": "sk-test"})
    adapter.client = SimpleNamespace(files=fake_files, responses=fake_responses)

    response = adapter.run(
        prompt="hello",
        model="gpt-5",
        require_search=True,
        return_citations=True,
        files=[upload_file],
        output_format=OutputModel,
        adapter_options={"temperature": 0.2},
    )

    assert response.text == "ok"
    urls = {c.url for c in response.citations}
    assert "https://example.com" in urls
    assert "https://source.example" in urls
    assert fake_responses.payload["tools"] == [{"type": "web_search"}]
    assert fake_responses.payload["tool_choice"] == "required"
    assert fake_responses.payload["include"] == ["web_search_call.action.sources"]
    assert fake_responses.payload["text"]["format"]["type"] == "json_schema"
    assert fake_responses.payload["text"]["format"]["schema"]["additionalProperties"] is False
    assert fake_responses.payload["temperature"] == 0.2


def test_anthropic_adapter_payload_and_citations() -> None:
    class FakeAnthropicResponse:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "done",
                        "citations": [
                            {
                                "url": "https://claude.ai",
                                "title": "Claude",
                                "cited_text": "reference",
                            }
                        ],
                    }
                ]
            }

    class FakeMessages:
        def __init__(self) -> None:
            self.payload = None

        def create(self, **kwargs):
            self.payload = kwargs
            return FakeAnthropicResponse()

    fake_messages = FakeMessages()

    adapter = AnthropicAdapter({"api_key": "test", "max_tokens": 100})
    adapter.client = SimpleNamespace(messages=fake_messages)

    response = adapter.run(
        prompt=["hello", "world"],
        model="claude-opus-4-1-20250805",
        require_search=True,
        return_citations=True,
        files=None,
        output_format=OutputModel,
        adapter_options={"temperature": 0.1},
    )

    assert response.text == "done"
    assert response.citations[0].url == "https://claude.ai"
    assert fake_messages.payload["tools"][0]["type"] == "web_search_20250305"
    assert fake_messages.payload["tool_choice"] == {"type": "any"}
    assert fake_messages.payload["output_config"]["format"]["type"] == "json_schema"
    assert (
        fake_messages.payload["output_config"]["format"]["schema"]["additionalProperties"]
        is False
    )
    assert fake_messages.payload["temperature"] == 0.1


def test_anthropic_schema_normalization_forces_nested_additional_properties_false() -> None:
    adapter = AnthropicAdapter({"api_key": "test"})

    schema = adapter._normalize_schema_for_anthropic(OutputWithDictModel.model_json_schema())

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            is_object = node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            )
            looks_objectish = any(
                key in node for key in ("properties", "required", "patternProperties", "additionalProperties")
            )
            if is_object or looks_objectish:
                assert node.get("additionalProperties") is False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def test_anthropic_schema_normalization_strips_unsupported_array_keywords() -> None:
    adapter = AnthropicAdapter({"api_key": "test"})
    schema = adapter._normalize_schema_for_anthropic(OutputWithBoundedArrayModel.model_json_schema())

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            assert "minItems" not in node
            assert "maxItems" not in node
            assert "uniqueItems" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def test_anthropic_adapter_synthesizes_when_search_turn_has_no_text() -> None:
    class FakeAnthropicResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return self._payload

    class FakeMessages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return FakeAnthropicResponse(
                    {
                        "content": [
                            {
                                "type": "web_search_tool_result",
                                "content": [
                                    {"title": "Company Site", "url": "https://company.example"}
                                ],
                            }
                        ]
                    }
                )
            return FakeAnthropicResponse(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "{\"value\": 42}",
                        }
                    ]
                }
            )

    fake_messages = FakeMessages()
    adapter = AnthropicAdapter({"api_key": "test", "max_tokens": 100})
    adapter.client = SimpleNamespace(messages=fake_messages)

    response = adapter.run(
        prompt="Summarize and return JSON",
        model="claude-opus-4-6",
        require_search=True,
        return_citations=True,
        files=None,
        output_format=OutputModel,
        adapter_options=None,
    )

    assert response.text == "{\"value\": 42}"
    assert any(c.url == "https://company.example" for c in response.citations)
    assert len(fake_messages.calls) == 2


def test_anthropic_adapter_collects_citations_with_second_pass_when_schema_hides_them() -> None:
    class FakeAnthropicResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return self._payload

    class FakeMessages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return FakeAnthropicResponse(
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": "{\"value\": 9}",
                            }
                        ]
                    }
                )
            return FakeAnthropicResponse(
                {
                    "content": [
                        {
                            "type": "web_search_tool_result",
                            "content": [
                                {"title": "Ref", "url": "https://citation.example"}
                            ],
                        }
                    ]
                }
            )

    fake_messages = FakeMessages()
    adapter = AnthropicAdapter({"api_key": "test", "max_tokens": 100})
    adapter.client = SimpleNamespace(messages=fake_messages)

    response = adapter.run(
        prompt="Return JSON and cite sources",
        model="claude-opus-4-6",
        require_search=True,
        return_citations=True,
        files=None,
        output_format=OutputModel,
        adapter_options=None,
    )

    assert response.text == "{\"value\": 9}"
    assert any(c.url == "https://citation.example" for c in response.citations)
    assert len(fake_messages.calls) == 2
    assert "output_config" in fake_messages.calls[0]
    assert "output_config" not in fake_messages.calls[1]


def test_gemini_adapter_payload_and_citations(tmp_path: Path) -> None:
    upload_file = tmp_path / "data.txt"
    upload_file.write_text("hello", encoding="utf-8")

    class FakeGeminiResponse:
        text = "gemini answer"

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "candidates": [
                    {
                        "grounding_metadata": {
                            "grounding_chunks": [
                                {
                                    "web": {
                                        "uri": "https://gemini.example",
                                        "title": "Gemini Source",
                                        "domain": "gemini.example",
                                    }
                                }
                            ]
                        }
                    }
                ]
            }

    class FakeModels:
        def __init__(self) -> None:
            self.payload = None

        def generate_content(self, **kwargs):
            self.payload = kwargs
            return FakeGeminiResponse()

    class FakeUploadedFile:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeFiles:
        def upload(self, file: str):
            return FakeUploadedFile(file)

    fake_models = FakeModels()

    adapter = GeminiAdapter({"api_key": "test", "max_output_tokens": 128})
    adapter.client = SimpleNamespace(models=fake_models, files=FakeFiles())

    response = adapter.run(
        prompt="hello",
        model="gemini-2.5-pro",
        require_search=True,
        return_citations=True,
        files=[upload_file],
        output_format=OutputModel,
        adapter_options={"temperature": 0.3},
    )

    assert response.text == "gemini answer"
    assert response.citations[0].url == "https://gemini.example"
    assert fake_models.payload["model"] == "gemini-2.5-pro"
    assert fake_models.payload["config"].system_instruction == (
        "Use Google Search to ground your answer and provide citations to sources."
    )


def test_grok_adapter_payload_and_citations(tmp_path: Path) -> None:
    upload_file = tmp_path / "data.txt"
    upload_file.write_text("hello", encoding="utf-8")

    class FakeInlineCitation:
        def __init__(self) -> None:
            self.id = "1"
            self.start_index = 0
            self.end_index = 5
            self.title = "Grok Article"
            self.web_citation = SimpleNamespace(url="https://grok.example")

        def HasField(self, field: str) -> bool:
            return field == "web_citation"

    class FakeGrokResponse:
        def __init__(self) -> None:
            self.content = "grok answer"
            self.citations = ["https://grok.example"]
            self.inline_citations = [FakeInlineCitation()]

    class FakeChatSession:
        def sample(self):
            return FakeGrokResponse()

    class FakeChatClient:
        def __init__(self) -> None:
            self.payload = None

        def create(self, **kwargs):
            self.payload = kwargs
            return FakeChatSession()

    class FakeFilesClient:
        def upload(self, file_path: str):
            return SimpleNamespace(id=f"file::{file_path}")

    class FakeChatHelpers:
        @staticmethod
        def file(file_id: str):
            return f"file:{file_id}"

        @staticmethod
        def system(text: str):
            return {"role": "system", "parts": [text]}

        @staticmethod
        def user(*parts):
            return {"role": "user", "parts": list(parts)}

    class FakeTools:
        @staticmethod
        def web_search():
            return "web_search_tool"

    fake_chat = FakeChatClient()

    adapter = GrokAdapter({"api_key": "test", "max_tokens": 256})
    adapter.client = SimpleNamespace(chat=fake_chat, files=FakeFilesClient())
    adapter.chat_helpers = FakeChatHelpers()
    adapter.xai_tools = FakeTools()

    response = adapter.run(
        prompt="hello",
        model="grok-4-latest",
        require_search=True,
        return_citations=True,
        files=[upload_file],
        output_format=OutputModel,
        adapter_options={"temperature": 0.4},
    )

    assert response.text == "grok answer"
    assert response.citations[0].source == "https://grok.example"
    assert response.citations[1].title == "Grok Article"
    assert response.citations[1].raw["title"] == "Grok Article"
    assert response.citations[1].url == "https://grok.example"
    assert fake_chat.payload["model"] == "grok-4-latest"
    assert fake_chat.payload["tools"] == ["web_search_tool"]
    assert fake_chat.payload["tool_choice"] == "required"
    assert fake_chat.payload["max_turns"] == 12
    assert fake_chat.payload["include"] == ["inline_citations", "web_search_call_output"]
    assert fake_chat.payload["temperature"] == 0.4


def test_perplexity_adapter_payload_and_citations() -> None:
    class FakePerplexityResponse:
        output_text = "perplexity answer"

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "perplexity answer",
                                "annotations": [
                                    {
                                        "url": "https://pplx.example",
                                        "title": "PPLX",
                                        "start_index": 0,
                                        "end_index": 4,
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "type": "search_results",
                        "results": [
                            {
                                "url": "https://pplx-search.example",
                                "title": "Search",
                                "source": "web",
                                "snippet": "snippet",
                            }
                        ],
                    },
                ]
            }

    class FakeResponses:
        def __init__(self) -> None:
            self.payload = None

        def create(self, **kwargs):
            self.payload = kwargs
            return FakePerplexityResponse()

    fake_responses = FakeResponses()

    adapter = PerplexityAdapter({"api_key": "test"})
    adapter.client = SimpleNamespace(responses=fake_responses)

    response = adapter.run(
        prompt="hello",
        model="sonar-pro",
        require_search=True,
        return_citations=True,
        files=None,
        output_format=OutputModel,
        adapter_options={"reasoning": {"effort": "low"}},
    )

    assert response.text == "perplexity answer"
    assert len(response.citations) == 2
    assert fake_responses.payload["preset"] == "pro-search"
    assert "tools" not in fake_responses.payload
    assert fake_responses.payload["response_format"]["json_schema"]["schema"]["additionalProperties"] is False


def test_perplexity_adapter_prefixes_provider_for_raw_model() -> None:
    class FakePerplexityResponse:
        output_text = "ok"

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"output": []}

    class FakeResponses:
        def __init__(self) -> None:
            self.payload = None

        def create(self, **kwargs):
            self.payload = kwargs
            return FakePerplexityResponse()

    fake_responses = FakeResponses()

    adapter = PerplexityAdapter({"api_key": "test"})
    adapter.client = SimpleNamespace(responses=fake_responses)

    adapter.run(
        prompt="hello",
        model="gpt-5.2",
        require_search=False,
        return_citations=False,
        files=None,
        output_format=None,
        adapter_options=None,
    )

    assert fake_responses.payload["model"] == "openai/gpt-5.2"


def test_perplexity_adapter_retries_without_response_format_on_invalid_request() -> None:
    class FakePerplexityResponse:
        output_text = '{"value": 11}'

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"output": []}

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("Error code: 400 - {'error': {'message': 'invalid request'}}")
            return FakePerplexityResponse()

    fake_responses = FakeResponses()
    adapter = PerplexityAdapter({"api_key": "test"})
    adapter.client = SimpleNamespace(responses=fake_responses)

    response = adapter.run(
        prompt="hello",
        model="perplexity/deep-research",
        require_search=True,
        return_citations=False,
        files=None,
        output_format=OutputModel,
        adapter_options=None,
    )

    assert response.text == '{"value": 11}'
    assert len(fake_responses.calls) == 2
    assert "response_format" in fake_responses.calls[0]
    assert "response_format" not in fake_responses.calls[1]
