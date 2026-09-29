"""Test OpenAI model configuration and structured-output integration."""

import os

import pytest

from agents.model import (
    DEFAULT_OPENAI_MODEL,
    get_chat_model,
)
from agents.planner import ResearchPlan


def test_openai_model_can_bind_the_planner_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the Pydantic structured-output runnable without network access."""
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    model = get_chat_model()
    structured_model = model.with_structured_output(ResearchPlan)

    assert model.model_name == DEFAULT_OPENAI_MODEL
    assert structured_model is not None


@pytest.mark.openai
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_MODEL_TESTS") != "1",
    reason="Set RUN_OPENAI_MODEL_TESTS=1 to run the OpenAI inference test.",
)
def test_openai_model_returns_a_structured_research_plan() -> None:
    """Verify OpenAI returns a valid typed research plan."""
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
