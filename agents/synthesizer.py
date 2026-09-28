"""Synthesize reviewed web findings into a structured, evidence-grounded report."""

import json
from typing import Any

from pydantic import BaseModel, Field

from agents.evidence import compact_findings
from agents.model import get_chat_model
from state import AgentState


class ReportSection(BaseModel):
    """Represent one coherent report section with supporting source URLs."""

    heading: str
    content: str
    source_urls: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Define the final report shape presented to the research requester."""

    title: str
    executive_summary: str
    sections: list[ReportSection]
    conclusion: str
    limitations: list[str]


def synthesizer_agent(state: AgentState) -> dict[str, dict[str, Any]]:
    """Combine reviewed evidence into a balanced report with traceable citations."""
    model = get_chat_model()
    synthesizer = model.with_structured_output(ResearchReport)
    report = synthesizer.invoke(
        [
            (
                "system",
                "You are a research synthesizer. Write a clear, balanced report that "
                "answers the user's question using only the provided findings. "
                "Distinguish established facts from uncertainty, preserve important "
                "disagreements, and cite sources by including their URLs in the "
                "relevant section's source_urls. For urban heat and climate "
                "adaptation, explain practical implications and equity considerations "
                "when supported by evidence. Never invent sources or details.",
            ),
            (
                "human",
                f"Question: {state['user_question']}\n"
                f"Critic quality score: {state['quality_score']}/5\n"
                f"Per-task review scores: "
                f"{json.dumps(state['task_reviews'], ensure_ascii=False)}\n"
                f"Critic feedback: {state['critic_feedback']}\n"
                f"Findings: "
                f"{json.dumps(compact_findings(state['findings']), ensure_ascii=False, separators=(',', ':'))}",
            ),
        ]
    )
    return {"final_report": report.model_dump()}
