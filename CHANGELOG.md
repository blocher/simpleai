# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Updated bundled default models to latest documented options as of 2026-02-06:
  - OpenAI: `gpt-5.2`
  - Gemini: `gemini-3-pro-preview`
  - Claude: `claude-opus-4-6`
  - Perplexity: `deep-research` (Responses API preset)
- Expanded model registry with newer/preview model IDs (OpenAI, Gemini, Claude).
- Extended non-Django settings discovery for `ai_settings.json` across common app-root locations.
- Updated install docs for GitHub-based installation before PyPI release.
- Fixed Perplexity adapter model routing by mapping legacy Sonar model names to valid Responses API presets and adding provider-prefix heuristics for raw model names.
- Added API-key preflight validation in `run_prompt` with clearer `SettingsError` messages, and added env-var aliases (`GROK_API_KEY`, `PPLX_API_KEY`, etc.).
- Added `xai` as a provider alias equivalent to `grok` (including settings normalization and model alias resolution).
- Updated `run_prompt` so `return_citations=True` always forces `require_search=True`, even when `require_search=False` is passed.
- Updated citation/search implementations across adapters:
  - OpenAI: switched to `web_search` tool with required tool usage and source extraction.
  - Anthropic: forced web search tool usage for grounded responses and improved citation parsing.
  - Gemini: improved grounding/citation extraction from grounding metadata and citation metadata.
  - Grok: migrated from deprecated live search parameters to xAI Agent Tools `web_search`.
- Fixed Anthropic structured output schema handling by explicitly normalizing JSON Schema object nodes with `additionalProperties: false`.
- Added boolean-string coercion for `require_search`, `return_citations`, and `binary_files`.
- Added Anthropic fallback synthesis pass when a forced web-search turn returns no final text block, preventing downstream JSON parsing failures for structured output.

### Added
- `README_API_KEYS.md` with provider key acquisition/setup instructions.
- `README_PYPI.md` with build/publish steps for PyPI.

## [0.1.0] - 2026-02-06

### Added
- Initial `simpleai` library release with one public API: `run_prompt`.
- Provider adapters for OpenAI, Anthropic, Gemini, Grok, and Perplexity.
- Model/provider resolution with provider aliases, known model map, and heuristic fallback.
- Django-first settings loading with JSON fallback (`ai_settings.json`).
- Bundled settings examples for Django and non-Django usage.
- Binary attachment support with text extraction fallback.
- Text extraction support for `pdf`, `doc`, `docx`, `md`, `txt`, `json`, and `rtf`.
- Structured output validation via Pydantic models.
- Normalized cross-provider citation format.
- Centralized structured logging adapter for call lifecycle and errors.
- Django app integration (`SimpleAIConfig`).
- Packaging setup for pip/PyPI (`pyproject.toml`, `MANIFEST.in`, `requirements.txt`).
- Comprehensive test suite compatible with pytest.
- MIT license and project `.gitignore`.
