"""Turn broad research questions into a small, actionable investigation plan."""

from pydantic import BaseModel, Field

from agents.model import get_chat_model
from state import AgentState


class ResearchPlan(BaseModel):
    """Represent the focused questions that will guide web research."""

    subtasks: list[str] = Field(
        description="Three to five distinct, answerable web research sub-tasks."
    )


def planner_agent(state: AgentState) -> dict[str, list[str]]:
    """Ask the OpenAI model to create focused urban climate research sub-tasks."""
    planner = get_chat_model().with_structured_output(ResearchPlan)
    system_prompt = (
        "Plan urban heat and climate adaptation research. Create 3 to 5 short, "
        "distinct, independently searchable sub-tasks that answer the user's "
        "question. Return non-empty subtask strings."
    )
    messages = [
        ("system", system_prompt),
        ("human", state["user_question"]),
    ]

    for attempt in range(2):
        plan = planner.invoke(messages)
        if 3 <= len(plan.subtasks) <= 5 and all(
            task.strip() for task in plan.subtasks
        ):
            return {"subtasks": plan.subtasks}
        if attempt == 0:
            messages = [
                ("system", system_prompt),
                (
                    "human",
                    f"Question: {state['user_question']}\n"
                    f"Your previous plan had {len(plan.subtasks)} tasks. "
                    "Return a corrected plan with exactly 3 distinct, non-empty tasks.",
                ),
            ]

    raise ValueError("Planner failed to return 3 to 5 non-empty research sub-tasks.")
