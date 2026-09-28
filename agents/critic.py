"""Assess research evidence and decide whether another web-search round is needed."""

import json

from pydantic import BaseModel, Field

from agents.evidence import compact_findings
from agents.model import get_chat_model
from state import AgentState


class TaskReview(BaseModel):
    """Score whether the evidence for one sub-task is useful and sufficient."""

    subtask: str
    relevance_score: int = Field(ge=1, le=5)
    completeness_score: int = Field(ge=1, le=5)
    gap: str


class CriticAssessment(BaseModel):
    """Capture the critic's overall quality decision and task-level reasoning."""

    quality_score: int = Field(ge=1, le=5)
    complete: bool
    task_reviews: list[TaskReview]
    feedback: str


def critic_agent(state: AgentState) -> dict[str, object]:
    """Review the relevance and coverage of findings and request retries if needed."""
    model = get_chat_model()
    critic = model.with_structured_output(CriticAssessment)
    evidence = json.dumps(
        compact_findings(state["findings"]),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assessment = critic.invoke(
        [
            (
                "system",
                "You are a rigorous research critic. Review each planned sub-task's "
                "search results for relevance and completeness. Score each from 1–5, "
                "identify meaningful gaps, and provide an overall quality score from "
                "1–5. Mark complete only if the evidence is sufficient to answer the "
                "user's question. Prefer authoritative climate science, public health, "
                "and government evidence for urban heat and climate adaptation. "
                "Request another search round when evidence is weak. For every "
                "task_reviews item, copy its subtask value exactly from the supplied "
                "planned sub-task list; do not paraphrase the sub-task.",
            ),
            (
                "human",
                f"Question: {state['user_question']}\n"
                f"Planned sub-tasks: {json.dumps(state['subtasks'], ensure_ascii=False)}\n"
                f"Current research round: {state['research_round']}\n"
                f"Evidence: {evidence}",
            ),
        ]
    )
    reviewed_subtasks = {review.subtask for review in assessment.task_reviews}
    if not set(state["subtasks"]).issubset(reviewed_subtasks):
        raise ValueError("Critic did not provide scores for every planned sub-task.")

    needs_research = assessment.quality_score < 4 or not assessment.complete
    return {
        "quality_score": assessment.quality_score,
        "task_reviews": [review.model_dump() for review in assessment.task_reviews],
        "critic_feedback": assessment.feedback,
        "needs_research": needs_research,
    }
