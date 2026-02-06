# simpleai

`simpleai` is a Python library with one public function, `run_prompt`, that abstracts prompt execution across major public GenAI providers.

Supported providers:
- Anthropic Claude
- Google Gemini
- OpenAI ChatGPT
- xAI Grok
- Perplexity

The package works as a standalone Python library and as an installable Django app.

## Install

From GitHub (recommended until PyPI release):

```bash
pip install "simpleai @ git+https://github.com/blocher/simpleai.git"
```

With `uv`:

```bash
uv pip install "simpleai @ git+https://github.com/blocher/simpleai.git"
```

From local source:

```bash
pip install -e .
```

From local source with `uv`:

```bash
uv pip install -e .
```

Add to your own project's `requirements.txt` (while this repo is still the source):

```txt
simpleai @ git+https://github.com/blocher/simpleai.git@main
```

Optional: pin to an exact commit for reproducible installs:

```txt
simpleai @ git+https://github.com/blocher/simpleai.git@<commit-sha>
```

## Public API

```python
from simpleai import run_prompt
```

### Signature

```python
def run_prompt(
    prompt,
    *,
    require_search=False,
    return_citations=None,
    file=None,
    files=None,
    binary_files=True,
    model=None,
    output_format=None,
    settings_file=None,
    adapter_options=None,
    **provider_kwargs,
)
```

### Arguments

- `prompt` (required): `str` or `list[str]` conversation history.
- `require_search` (default `False`): force provider-native search tool usage.
- `return_citations` (default `True` when `require_search=True`, else `False`): include normalized citations.
  - If `return_citations=True`, `require_search` is always forced to `True` (even if passed as `False`).
  - String booleans like `"True"` / `"False"` are accepted.
- `file` / `files`: one file path or multiple file paths.
- `binary_files` (default `True`): upload binary attachments when adapter supports them; otherwise text extraction fallback.
- `model`: provider alias or model name.
  - Provider aliases: `google`, `gemini`, `anthropic`, `claude`, `openai`, `chatgpt`, `grok`, `xai`, `perplexityai`, `perplexity`.
  - If provider alias is passed, provider default model from settings is used.
  - If omitted, first provider in `defaults` with credentials is selected.
- `output_format`: a Pydantic model class for structured output validation.
- `settings_file`: explicit path to `ai_settings.json`.
- `adapter_options` / `**provider_kwargs`: provider-specific passthrough params.

### Return value

- Default: `str` or validated Pydantic model instance (when `output_format` is provided).
- If citations requested: `(result, citations)` tuple.

`citations` is a normalized list of dicts with common keys like:
- `provider`
- `url`
- `title`
- `source`
- `snippet`
- `citation_id`
- `start_index`
- `end_index`
- `raw`

## Configuration

Settings are loaded in this order:
1. Django settings (`SIMPLEAI` or `SIMPLEAI_SETTINGS`) when Django is configured.
2. `ai_settings.json` (explicit path, then `$SIMPLEAI_SETTINGS_FILE`, then common app-root locations like current working directory, script directory, and parent project roots).
3. Built-in defaults.

### Non-Django (`ai_settings.json`)

Use `simpleai/settings_examples/ai_settings.example.json` as a template.

A root copy is also provided in this repo: `ai_settings.example.json`.

### Django

1. Add app:

```python
INSTALLED_APPS = [
    # ...
    "simpleai",
]
```

2. Add config in `settings.py`:

```python
SIMPLEAI = {
    "defaults": ["gemini", "openai", "claude", "grok", "perplexity"],
    "providers": {
        "gemini": {"api_key": "...", "default_model": "gemini-3-pro-preview"},
        "openai": {"api_key": "...", "default_model": "gpt-5.2"},
        "claude": {"api_key": "...", "default_model": "claude-opus-4-6"},
        "grok": {"api_key": "...", "default_model": "grok-4-latest"},
        "perplexity": {"api_key": "...", "default_model": "deep-research"},
    },
    "logging": {
        "enabled": True,
        "django_logfile": "django",
        "logfile_location": "./simpleai.log",
    },
}
```

Use `simpleai/settings_examples/django_settings_example.py` for the full template.

## Credentials

Provider auth is API-key based for normal usage.
- OpenAI: API key
- Anthropic: API key
- Gemini (Google AI Studio API): API key
- xAI: API key
- Perplexity: API key

No separate app ID is required for basic API usage. Some optional enterprise/alternate paths (for example Google Vertex AI) may require project configuration in addition to credentials.

API key setup instructions are in [README_API_KEYS.md](README_API_KEYS.md).

## File handling

When files are supplied and text extraction is needed (either `binary_files=False` or adapter binary unsupported), SimpleAI extracts text from:
- `pdf`
- `doc`
- `docx`
- `md`
- `txt`
- `json`
- `rtf`

Extracted text is appended to the prompt with file labels.

## Logging

Logging is handled in `simpleai/adapters/logging_adapter.py`.

When enabled, each `run_prompt` call logs:
- call start/end timestamps
- elapsed time
- all arguments
- provider/model resolution
- adapter payload params
- result preview
- detailed errors

## Provider adapters

Adapters are in `simpleai/adapters/`:
- `openai_adapter.py` (OpenAI Responses API + `web_search` tool)
- `anthropic_adapter.py` (Anthropic Messages API + `web_search_20250305` tool)
- `gemini_adapter.py` (`google-genai` + `GoogleSearch` tool)
- `grok_adapter.py` (`xai-sdk` Agent Tools API + `web_search` tool)
- `perplexity_adapter.py` (Perplexity Responses API)

## Example usage

```python
from pydantic import BaseModel
from simpleai import run_prompt

class Summary(BaseModel):
    topic: str
    key_points: list[str]

result, citations = run_prompt(
    "Summarize the latest GPU architecture announcements.",
    model="openai",
    require_search=True,
    return_citations=True,
    output_format=Summary,
)

print(result)
print(citations)
```

## Tests

Run with:

```bash
pytest
```

The test suite is built to run offline by mocking provider SDK clients.

## Packaging / PyPI

This repo includes required packaging files:
- `pyproject.toml`
- `MANIFEST.in`
- `requirements.txt`
- `LICENSE`

PyPI publishing steps are documented in [README_PYPI.md](README_PYPI.md).

## Defaults (as of 2026-02-06)

Current default models in bundled settings:
- Gemini: `gemini-3-pro-preview`
- OpenAI: `gpt-5.2`
- Claude: `claude-opus-4-6`
- Grok: `grok-4-latest`
- Perplexity: `deep-research`

These can be changed in your settings file at any time.
