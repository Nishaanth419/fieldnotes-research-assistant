"""Test validation of user-edited research findings before graph resume."""

import pytest

from main import _validated_findings, main


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"task": "results must be a list"},
        {"task": ["each result must be an object"]},
        {1: [{"url": "https://example.org"}]},
    ],
)
def test_rejects_invalid_edited_findings(value: object) -> None:
    """Reject malformed findings with actionable validation errors."""
    with pytest.raises(ValueError):
        _validated_findings(value)


def test_accepts_valid_edited_findings() -> None:
    """Preserve a well-shaped user-edited evidence object."""
    findings = {"heat risk": [{"title": "Source", "url": "https://example.org"}]}

    assert _validated_findings(findings) == findings


def test_cli_reports_missing_openai_key_before_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require OpenAI credentials in the terminal interface as well as the web app."""
    monkeypatch.setattr("main.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        main()
