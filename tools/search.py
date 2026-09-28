"""Provide the shared Tavily web-search boundary used by research agents."""

import os
from typing import Any

from tavily import TavilyClient

DEFAULT_CLIMATE_DOMAINS = (
    "climate.gov",
    "noaa.gov",
    "epa.gov",
    "cdc.gov",
    "ipcc.ch",
    "who.int",
    "unhabitat.org",
    "wri.org",
    "worldbank.org",
    "nature.com",
    "sciencedirect.com",
)


def _research_domains() -> list[str]:
    """Return a configurable domain allowlist for urban climate research."""
    configured_domains = os.getenv("TAVILY_INCLUDE_DOMAINS", "")
    if configured_domains.strip():
        domains = [domain.strip() for domain in configured_domains.split(",") if domain.strip()]
        if not domains:
            raise ValueError("TAVILY_INCLUDE_DOMAINS must contain at least one domain.")
        return domains
    return list(DEFAULT_CLIMATE_DOMAINS)


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search trusted climate and public-health domains for citation-ready evidence."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add it to the project's .env file."
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_domains=_research_domains(),
        include_domains_mode="restrict",
    )
    results = response.get("results", [])
    return [
        {
            "query": query,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "score": result.get("score"),
        }
        for result in results
    ]
