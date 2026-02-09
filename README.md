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
pip install "simpleai @ git+https://github.com/HireAnEsquire/simpleai.git"
```

With `uv`:

```bash
uv pip install "simpleai @ git+https://github.com/HireAnEsquire/simpleai.git"
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
simpleai @ git+https://github.com/HireAnEsquire/simpleai.git@main
```

Optional: pin to an exact commit for reproducible installs:

```txt
simpleai @ git+https://github.com/HireAnEsquire/simpleai.git@<commit-sha>
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

### Error handling

`run_prompt` always raises `SimpleAIException` (or one of its subclasses) for failures, so you can catch all runtime/config/provider errors with one exception type.

```python
from simpleai import run_prompt, SimpleAIException

try:
    result = run_prompt("Hello", model="openai")
except SimpleAIException as exc:
    # Full original exception is preserved for debugging.
    print(exc)
    print(exc.original_exception)
    raise
```

Specific exception subclasses (for targeted handling) are still available from `simpleai.exceptions`, for example `SettingsError` and `ProviderError`.

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
        "perplexity": {"api_key": "...", "default_model": "sonar-deep-research"},
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

Provider auth is API-key-based for normal usage.
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

### Anthropic Rate Limiting (Tier 1 Accounts)

Anthropic Tier 1 accounts have strict rate limits (e.g., 30,000 input tokens per minute). When using `return_citations=True` with structured output, the adapter may make multiple API calls, which can exceed these limits.

The adapter automatically handles rate limiting by reading the `retry-after` header from 429 responses and waiting the exact time specified by the API before retrying.

Configuration options (set in `ai_settings.json` under `providers.claude`):

```json
{
  "providers": {
    "claude": {
      "api_key": "...",
      "default_model": "claude-opus-4-6",
      "max_tokens": 4096,
      "max_retries": 3,
      "skip_citation_followup": false
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `max_retries` | `3` | Number of retries on 429 rate limit errors |
| `skip_citation_followup` | `false` | Skip the secondary citation-gathering API call (reduces API calls but may return fewer citations) |

**Tips for Tier 1 accounts:**
- The adapter automatically respects the `retry-after` header from Anthropic's API
- Set `skip_citation_followup: true` to reduce API calls when using structured output with citations
- Consider upgrading to a higher tier for production workloads

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

## Manual Provider Smoke Runner

This repo includes a manual smoke runner that executes the same resume+search+citations prompt across all providers and prints:
- per-provider output
- per-provider citations
- per-provider file handling mode (`binary upload` vs `parsed text`)
- final summary with:
  - green `SUCCESS`
  - red `FAILED`
  - yellow `API KEY NOT SET`

Definition of success in this runner:
- no exception
- structured result validates to `JobHistory`
- citations list is non-empty

### Standalone Python script

```bash
python ./simpleai/scripts/run_provider_smoke.py
```

If `--file` is omitted, the runner looks in this order:
- bundled package sample: `simpleai/samples/functionalsample.pdf` (works from installed package, including Django projects)
- `../hae/api/functionalsample.pdf`
- current working directory: `./functionalsample.pdf`
- repo root: `./simpleai/functionalsample.pdf`

Optional args:

```bash
python ./simpleai/scripts/run_provider_smoke.py \
  --file ../hae/api/functionalsample.pdf \
  --providers openai anthropic gemini grok perplexity \
  --settings-file /path/to/ai_settings.json
```

### Django management command

If `simpleai` is in `INSTALLED_APPS`, run:

```bash
python manage.py run_provider_smoke
```

Optional args:

```bash
python manage.py run_provider_smoke \
  --file ../hae/api/functionalsample.pdf \
  --providers openai anthropic gemini grok perplexity \
  --settings-file /path/to/ai_settings.json
```

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
- Perplexity: `sonar-deep-research`

These can be changed in your settings file at any time.
