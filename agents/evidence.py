"""Bound evidence context before it is passed to language models."""

from typing import Any


MAX_RESULTS_PER_SUBTASK = 3
MAX_EVIDENCE_FIELD_CHARS = 1_000


def compact_findings(
    findings: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep a small, citation-ready evidence sample within the model context."""
    compacted: dict[str, list[dict[str, Any]]] = {}
    for task, results in findings.items():
        compacted[task[:MAX_EVIDENCE_FIELD_CHARS]] = [
            {
                key: (
                    value[:MAX_EVIDENCE_FIELD_CHARS]
                    if isinstance(value, str)
                    else value
                )
                for key, value in result.items()
                if key in {"query", "title", "url", "content", "score"}
            }
            for result in results[-MAX_RESULTS_PER_SUBTASK:]
        ]
    return compacted
