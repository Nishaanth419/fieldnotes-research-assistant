"""Test domain restrictions and result normalization at the Tavily boundary."""

from typing import Any

import pytest

from tools import search


def test_search_restricts_results_to_default_trusted_domains(
    monkeypatch: Any,
) -> None:
    """Pass the built-in climate and public-health allowlist in restrict mode."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("TAVILY_INCLUDE_DOMAINS", raising=False)
    captured: dict[str, Any] = {}

    class FakeTavilyClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        def search(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "results": [
                    {
                        "title": "Heat guidance",
                        "url": "https://climate.gov/heat",
                        "content": "Protective measures.",
                        "score": 0.95,
                    }
                ]
            }

    monkeypatch.setattr(search, "TavilyClient", FakeTavilyClient)

    results = search.search_web("urban heat risk", max_results=2)

    assert captured["include_domains"] == list(search.DEFAULT_CLIMATE_DOMAINS)
    assert captured["include_domains_mode"] == "restrict"
    assert captured["max_results"] == 2
    assert results == [
        {
            "query": "urban heat risk",
            "title": "Heat guidance",
            "url": "https://climate.gov/heat",
            "content": "Protective measures.",
            "score": 0.95,
        }
    ]


def test_search_uses_custom_domain_allowlist(monkeypatch: Any) -> None:
    """Support explicitly configured research domains."""
    monkeypatch.setenv("TAVILY_INCLUDE_DOMAINS", "example.org, climate.example ")

    assert search._research_domains() == ["example.org", "climate.example"]


def test_search_requires_tavily_key(monkeypatch: Any) -> None:
    """Fail explicitly instead of attempting an unauthenticated search."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        search.search_web("urban heat risk")
