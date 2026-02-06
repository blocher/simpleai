"""Provider-safe JSON schema helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from pydantic import BaseModel

# Anthropic output schemas currently reject these JSON Schema keywords.
# Source: docs.anthropic.com (Structured outputs limitations), accessed 2026-02-06.
ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def output_model_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """Return the Pydantic-generated JSON schema for an output model."""

    return output_format.model_json_schema()


def enforce_closed_objects(schema: dict[str, Any]) -> dict[str, Any]:
    """Set additionalProperties=false on all object-like schema nodes."""

    normalized = deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            is_object = node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            )
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


def strip_schema_keywords(schema: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Remove unsupported JSON Schema keywords recursively."""

    keys_set = set(keys)
    normalized = deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in list(node.keys()):
                if key in keys_set:
                    node.pop(key, None)
                    continue
                walk(node[key])
            return

        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(normalized)
    return normalized


def openai_response_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """Build strict-schema payload for OpenAI Responses API."""

    return enforce_closed_objects(output_model_schema(output_format))


def anthropic_response_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """Build output schema compatible with Anthropic output_config constraints."""

    schema = enforce_closed_objects(output_model_schema(output_format))
    return strip_schema_keywords(schema, ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS)


def perplexity_response_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """Build JSON schema payload for Perplexity responses."""

    return enforce_closed_objects(output_model_schema(output_format))
