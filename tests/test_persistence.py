"""Test restoring paused and completed research runs after graph recreation."""

from typing import Any

from graph import workflow
from state import AgentState
from webapp import _jobs, _restore_job
import webapp


def test_webapp_restores_paused_review_and_completed_report(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Recreate app graph instances and recover saved review and final report data."""
    subtasks = ["heat exposure", "health impacts", "adaptation options"]

    def planner(state: AgentState) -> dict[str, list[str]]:
        return {"subtasks": subtasks}

    def researcher(state: AgentState) -> dict[str, Any]:
        return {
            "findings": {
                task: [{"title": f"Evidence for {task}", "url": "https://source.test"}]
                for task in subtasks
            },
            "research_round": state["research_round"] + 1,
        }

    def critic(state: AgentState) -> dict[str, Any]:
        return {
            "quality_score": 4,
            "task_reviews": [],
            "critic_feedback": "Evidence is sufficient",
            "needs_research": False,
        }

    def synthesizer(state: AgentState) -> dict[str, Any]:
        return {"final_report": {"findings": state["findings"]}}

    monkeypatch.setattr(workflow, "planner_agent", planner)
    monkeypatch.setattr(workflow, "researcher_agent", researcher)
    monkeypatch.setattr(workflow, "critic_agent", critic)
    monkeypatch.setattr(workflow, "synthesizer_agent", synthesizer)

    database_path = tmp_path / "webapp-persistence.sqlite"
    config = {"configurable": {"thread_id": "webapp-restart-test"}}
    state: AgentState = {
        "user_question": "How can cities reduce heat risk?",
        "subtasks": [],
        "findings": {},
        "research_round": 0,
        "max_research_rounds": 3,
        "quality_score": 0,
        "task_reviews": [],
        "critic_feedback": "",
        "needs_research": False,
        "final_report": {},
    }
    first_graph = workflow.build_research_graph(checkpoint_db_path=database_path)
    list(first_graph.stream(state, config, stream_mode="updates"))
    expected_findings = first_graph.get_state(config).values["findings"]
    first_graph.checkpointer.conn.close()

    _jobs.clear()
    paused_graph = workflow.build_research_graph(checkpoint_db_path=database_path)
    monkeypatch.setattr(webapp, "_graph", paused_graph)
    restored_review = _restore_job("webapp-restart-test")
    assert restored_review is not None
    assert restored_review["status"] == "waiting_review"
    assert restored_review["events"][0]["findings"] == expected_findings

    edited_findings = {
        "health impacts": [{"title": "Reviewed source", "url": "https://reviewed.test"}]
    }
    paused_graph.update_state(config, {"findings": edited_findings})
    list(paused_graph.stream(None, config, stream_mode="updates"))
    paused_graph.checkpointer.conn.close()

    _jobs.clear()
    completed_graph = workflow.build_research_graph(checkpoint_db_path=database_path)
    monkeypatch.setattr(webapp, "_graph", completed_graph)
    restored_report = _restore_job("webapp-restart-test")
    assert restored_report is not None
    assert restored_report["status"] == "complete"
    assert restored_report["events"][0]["report"]["findings"] == edited_findings
    completed_graph.checkpointer.conn.close()
