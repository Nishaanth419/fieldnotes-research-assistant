"""Gather source-backed web evidence for each planned research sub-task."""

from typing import Any

from state import AgentState
from tools.search import search_web


def researcher_agent(state: AgentState) -> dict[str, Any]:
    """Search every planned task and retain evidence from any prior review round."""
    findings = {
        task: list(results) for task, results in state["findings"].items()
    }
    for task in state["subtasks"]:
        query = task
        if state["research_round"] and state["critic_feedback"]:
            query = f"{task}. Additional focus: {state['critic_feedback']}"
        results = search_web(query, max_results=3)
        combined_results = findings.get(task, []) + results
        unique_results: dict[str, dict[str, Any]] = {}
        for result in combined_results:
            result_url = result.get("url")
            if result_url:
                unique_results[result_url] = result
        findings[task] = list(unique_results.values())[-3:]

    return {
        "findings": findings,
        "research_round": state["research_round"] + 1,
    }
