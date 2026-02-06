from __future__ import annotations

from pathlib import Path

from simpleai.provider_smoke import JobExperience, JobHistory, resolve_sample_file_path, run_provider_matrix



def test_resolve_sample_file_path_explicit(tmp_path: Path) -> None:
    path = tmp_path / "resume.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    resolved = resolve_sample_file_path(path)
    assert resolved == path.resolve()



def test_run_provider_matrix_classifies_outcomes(monkeypatch, tmp_path: Path) -> None:
    sample_file = tmp_path / "resume.pdf"
    sample_file.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("simpleai.provider_smoke.load_settings", lambda settings_file=None: {})

    def fake_key(settings: dict, provider: str):
        if provider in {"openai", "claude"}:
            return "key"
        return None

    monkeypatch.setattr("simpleai.provider_smoke.get_provider_api_key", fake_key)

    def fake_run_prompt(prompt: str, **kwargs):
        model = kwargs["model"]
        if model == "openai":
            return (
                JobHistory(
                    latest_job_experiences=[
                        JobExperience(
                            company_name="Example",
                            role_title="Engineer",
                            start_date="2020",
                            end_date="2022",
                            company_url="https://example.com",
                        )
                    ]
                ),
                [{"url": "https://example.com"}],
            )
        if model == "anthropic":
            raise RuntimeError("provider exploded")
        raise RuntimeError(f"unexpected model: {model}")

    monkeypatch.setattr("simpleai.provider_smoke.run_prompt", fake_run_prompt)

    output_lines: list[str] = []
    results = run_provider_matrix(
        file_path=sample_file,
        providers=["openai", "anthropic", "grok"],
        emit=output_lines.append,
        use_color=False,
    )

    statuses = {item.display_name: item.status for item in results}
    assert statuses["OpenAI"] == "success"
    assert statuses["Anthropic"] == "failed"
    assert statuses["Grok"] == "missing_key"



def test_run_provider_matrix_requires_citations(monkeypatch, tmp_path: Path) -> None:
    sample_file = tmp_path / "resume.pdf"
    sample_file.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("simpleai.provider_smoke.load_settings", lambda settings_file=None: {})
    monkeypatch.setattr("simpleai.provider_smoke.get_provider_api_key", lambda settings, provider: "key")
    monkeypatch.setattr(
        "simpleai.provider_smoke.run_prompt",
        lambda prompt, **kwargs: (
            JobHistory(
                latest_job_experiences=[
                    JobExperience(
                        company_name="Example",
                        role_title="Engineer",
                        start_date="2020",
                        end_date="2022",
                    )
                ]
            ),
            [],
        ),
    )

    results = run_provider_matrix(
        file_path=sample_file,
        providers=["openai"],
        emit=lambda _: None,
        use_color=False,
    )

    assert results[0].status == "failed"
    assert "No citations returned" in results[0].message



def test_resolve_sample_file_path_missing_falls_back_to_bundled_sample(tmp_path: Path) -> None:
    resolved = resolve_sample_file_path(tmp_path / "missing.pdf")
    assert resolved.exists()
    assert resolved.name == "functionalsample.pdf"
    assert "simpleai/samples" in str(resolved)
