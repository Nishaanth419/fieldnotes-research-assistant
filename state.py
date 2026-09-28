"""Define the shared workflow state that lets research agents coordinate findings."""

from typing import Any, TypedDict


class AgentState(TypedDict):
    """Carry the question, evidence, review decisions, and report through the graph."""

    user_question: str
    subtasks: list[str]
    findings: dict[str, list[dict[str, Any]]]
    research_round: int
    max_research_rounds: int
    quality_score: int
    task_reviews: list[dict[str, Any]]
    critic_feedback: str
    needs_research: bool
    final_report: dict[str, Any]
