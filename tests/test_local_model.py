"""Test local Ollama model configuration and structured-output integration."""

import os

import pytest

from agents.model import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    get_chat_model,
)
from agents.planner import ResearchPlan


def test_local_model_can_bind_the_planner_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the Pydantic structured-output runnable without network access."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    model = get_chat_model()
    structured_model = model.with_structured_output(ResearchPlan)

    assert model.model == DEFAULT_OLLAMA_MODEL
    assert model.base_url.rstrip("/") == DEFAULT_OLLAMA_BASE_URL
    assert structured_model is not None


@pytest.mark.ollama
@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_MODEL_TESTS") != "1",
    reason="Set RUN_LOCAL_MODEL_TESTS=1 to run the local Ollama inference test.",
)
def test_local_model_returns_a_structured_research_plan() -> None:
    """Verify a running local Ollama model returns a valid typed research plan."""
    plan = get_chat_model().with_structured_output(ResearchPlan).invoke(
        [
            (
                "system",
                "Return exactly three short, distinct research sub-tasks as a plan.",
            ),
            ("human", "How can cities reduce extreme heat risk for vulnerable groups?"),
        ]
    )

    assert 3 <= len(plan.subtasks) <= 5
    assert all(task.strip() for task in plan.subtasks)
