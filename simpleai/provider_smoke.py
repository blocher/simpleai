"""Manual provider smoke-runner for resume/search/citation validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, Field

from .adapters import ADAPTER_CLASSES
from .api import run_prompt
from .settings import canonical_provider_name, expected_provider_env_vars, get_provider_api_key, load_settings

PROMPT = (
    "Parse the attached resume to return latest 3 job experiences, "
    "and use a search to find the company urls"
)


class JobExperience(BaseModel):
    """Single job history item expected from model output."""

    company_name: str = Field(description="Company name from resume.")
    role_title: str = Field(description="Job title/role.")
    start_date: str = Field(description="Start date as shown in resume.")
    end_date: str | None = Field(default=None, description="End date or null if current role.")
    company_url: str | None = Field(default=None, description="Company homepage URL discovered via search.")


class JobHistory(BaseModel):
    """Structured output target for smoke validation."""

    latest_job_experiences: list[JobExperience] = Field(
        description="Most recent three jobs in reverse-chronological order.",
        min_length=1,
        max_length=3,
    )


@dataclass(slots=True)
class ProviderTarget:
    display_name: str
    model_arg: str
    settings_provider: str


@dataclass(slots=True)
class ProviderRunResult:
    display_name: str
    model_arg: str
    status: Literal["success", "failed", "missing_key"]
    message: str
    file_handling: str
    job_history: JobHistory | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)


PROVIDER_TARGETS: tuple[ProviderTarget, ...] = (
    ProviderTarget(display_name="OpenAI", model_arg="openai", settings_provider="openai"),
    ProviderTarget(display_name="Anthropic", model_arg="anthropic", settings_provider="claude"),
    ProviderTarget(display_name="Gemini", model_arg="gemini", settings_provider="gemini"),
    ProviderTarget(display_name="Grok", model_arg="grok", settings_provider="grok"),
    ProviderTarget(display_name="Perplexity", model_arg="perplexity", settings_provider="perplexity"),
)

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}


def colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI[color]}{text}{ANSI['reset']}"


def resolve_sample_file_path(file_path: str | Path | None = None) -> Path:
    """Resolve the sample resume path from explicit input or known defaults."""

    candidates: list[Path] = []

    if file_path is not None:
        candidates.append(Path(file_path).expanduser())

    if env_val := os.getenv("SAMPLE_PDF_PATH"):
        env_path = Path(env_val).expanduser()
        if env_path.exists():
            candidates.append(env_path)

    candidates.extend(
        [
            Path(__file__).resolve().parent / "samples" / "functionalsample.pdf",
            Path.cwd() / "functionalsample.pdf",
            Path(__file__).resolve().parents[1] / "functionalsample.pdf",
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    searched = "\n".join(f"- {str(path)}" for path in candidates)
    raise FileNotFoundError(f"Could not find functionalsample.pdf. Checked:\n{searched}")


def _short_error(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    if not text:
        text = exc.__class__.__name__
    return text[:220]


def _provider_filter(requested: Sequence[str] | None) -> set[str] | None:
    if not requested:
        return None

    selected: set[str] = set()
    for raw in requested:
        item = raw.strip().lower()
        canonical = canonical_provider_name(item) or item
        selected.add(canonical)
        selected.add(item)
    return selected


def _emit_provider_header(
    emit: Callable[[str], None],
    use_color: bool,
    target: ProviderTarget,
    file_path: Path,
) -> None:
    emit("\n" + "=" * 88)
    emit(
        f"{colorize(target.display_name, 'cyan', use_color)} "
        f"(model=\"{target.model_arg}\", file=\"{file_path}\")"
    )
    emit("-" * 88)


def _file_handling_mode(provider: str) -> str:
    adapter_cls = ADAPTER_CLASSES.get(provider)
    supports_binary = bool(getattr(adapter_cls, "supports_binary_files", False))
    return "binary upload" if supports_binary else "parsed text"


def run_provider_matrix(
    *,
    file_path: Path,
    settings_file: str | Path | None = None,
    providers: Sequence[str] | None = None,
    emit: Callable[[str], None] = print,
    use_color: bool = True,
) -> list[ProviderRunResult]:
    """Run the same structured+search prompt against each configured provider."""

    settings = load_settings(settings_file)
    requested = _provider_filter(providers)
    results: list[ProviderRunResult] = []

    for target in PROVIDER_TARGETS:
        if requested is not None and target.settings_provider not in requested and target.model_arg not in requested:
            continue

        _emit_provider_header(emit, use_color, target, file_path)
        file_handling = _file_handling_mode(target.settings_provider)
        emit(f"File handling: {file_handling}")

        api_key = get_provider_api_key(settings, target.settings_provider)
        if not api_key:
            envs = expected_provider_env_vars(target.settings_provider)
            msg = (
                f"API key not set. Configure providers.{target.settings_provider}.api_key "
                f"or env vars: {', '.join(envs)}"
            )
            emit(colorize(msg, "yellow", use_color))
            results.append(
                ProviderRunResult(
                    display_name=target.display_name,
                    model_arg=target.model_arg,
                    status="missing_key",
                    message=msg,
                    file_handling=file_handling,
                )
            )
            continue

        try:
            response = run_prompt(
                PROMPT,
                output_format=JobHistory,
                return_citations=True,
                binary_files=True,
                model=target.model_arg,
                file=str(file_path),
                settings_file=settings_file,
            )

            if not isinstance(response, tuple) or len(response) != 2:
                raise ValueError("run_prompt did not return (result, citations) tuple")

            result_obj, citations_obj = response

            if not isinstance(result_obj, JobHistory):
                raise TypeError(f"Expected JobHistory, got {type(result_obj).__name__}")
            if not isinstance(citations_obj, list):
                raise TypeError(f"Expected citations list, got {type(citations_obj).__name__}")
            if not citations_obj:
                raise ValueError("No citations returned")

            citations: list[dict[str, Any]] = [
                item if isinstance(item, dict) else {"raw": item}
                for item in citations_obj
            ]

            emit(colorize("SUCCESS", "green", use_color))
            emit("JobHistory:")
            emit(result_obj.model_dump_json(indent=2))
            emit(f"Citations returned: {len(citations)}")
            for index, citation in enumerate(citations[:5], start=1):
                label = citation.get("title") or citation.get("url") or citation.get("source") or "(no source label)"
                emit(f"  {index}. {label}")
            if len(citations) > 5:
                emit(f"  ... and {len(citations) - 5} more")

            results.append(
                ProviderRunResult(
                    display_name=target.display_name,
                    model_arg=target.model_arg,
                    status="success",
                    message=(
                        f"Structured output validated; "
                        f"{len(result_obj.latest_job_experiences)} experiences; {len(citations)} citations"
                    ),
                    file_handling=file_handling,
                    job_history=result_obj,
                    citations=citations,
                )
            )
        except Exception as exc:  # pragma: no cover - intentionally runtime-facing
            msg = _short_error(exc)
            emit(colorize(f"FAILED: {msg}", "red", use_color))
            results.append(
                ProviderRunResult(
                    display_name=target.display_name,
                    model_arg=target.model_arg,
                    status="failed",
                    message=msg,
                    file_handling=file_handling,
                )
            )

    emit("\n" + "=" * 88)
    emit(colorize("Provider Summary", "bold", use_color))
    emit("=" * 88)

    for item in results:
        if item.status == "success":
            status = colorize("SUCCESS", "green", use_color)
        elif item.status == "missing_key":
            status = colorize("API KEY NOT SET", "yellow", use_color)
        else:
            status = colorize("FAILED", "red", use_color)
        emit(f"{item.display_name:<12} {status:<20} [{item.file_handling}] {item.message}")

    return results
