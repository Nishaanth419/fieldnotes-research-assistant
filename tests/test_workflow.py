"""Test retry routing and the checkpointed human-review workflow."""

from typing import Any

from graph import workflow
from state import AgentState


def _initial_state() -> AgentState:
    """Create a fully populated graph state for deterministic node stubs."""
    return {
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


def test_critic_retries_then_pauses_for_editable_human_review(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Retry low-quality research, pause before synthesis, then resume edited evidence."""
    calls = {"researcher": 0, "critic": 0, "synthesizer": 0}
    subtasks = ["heat exposure", "public health", "city interventions"]

    def planner(state: AgentState) -> dict[str, list[str]]:
        return {"subtasks": subtasks}

    def researcher(state: AgentState) -> dict[str, Any]:
        calls["researcher"] += 1
        return {
            "findings": {
                **state["findings"],
                "heat exposure": [
                    {"round": calls["researcher"], "url": "https://example.org"}
                ],
            },
            "research_round": state["research_round"] + 1,
        }

    def critic(state: AgentState) -> dict[str, Any]:
        calls["critic"] += 1
        needs_research = calls["critic"] == 1
        return {
            "quality_score": 2 if needs_research else 4,
            "task_reviews": [],
            "critic_feedback": "More evidence needed" if needs_research else "Sufficient",
            "needs_research": needs_research,
        }

    def synthesizer(state: AgentState) -> dict[str, Any]:
        calls["synthesizer"] += 1
        return {"final_report": {"findings": state["findings"]}}

    monkeypatch.setattr(workflow, "planner_agent", planner)
    monkeypatch.setattr(workflow, "researcher_agent", researcher)
    monkeypatch.setattr(workflow, "critic_agent", critic)
    monkeypatch.setattr(workflow, "synthesizer_agent", synthesizer)

    graph = workflow.build_research_graph(
        checkpoint_db_path=tmp_path / "retry-review.sqlite"
    )
    config = {"configurable": {"thread_id": "retry-review-test"}}
    list(graph.stream(_initial_state(), config, stream_mode="updates"))

    paused_state = graph.get_state(config)
    assert paused_state.next == ("human_in_the_loop",)
    assert calls == {"researcher": 2, "critic": 2, "synthesizer": 0}
    assert paused_state.values["research_round"] == 2

    edited_findings = {
        "heat exposure": [{"title": "Reviewed source", "url": "https://reviewed.org"}]
    }
    graph.update_state(config, {"findings": edited_findings})
    list(graph.stream(None, config, stream_mode="updates"))

    completed_state = graph.get_state(config)
    assert completed_state.next == ()
    assert completed_state.values["final_report"]["findings"] == edited_findings
    assert calls["synthesizer"] == 1


def test_sqlite_checkpoint_survives_graph_recreation(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Restore a paused run in a new graph instance and complete with edited evidence."""
    calls = {"synthesizer": 0}
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
        calls["synthesizer"] += 1
        return {"final_report": {"findings": state["findings"]}}

    monkeypatch.setattr(workflow, "planner_agent", planner)
    monkeypatch.setattr(workflow, "researcher_agent", researcher)
    monkeypatch.setattr(workflow, "critic_agent", critic)
    monkeypatch.setattr(workflow, "synthesizer_agent", synthesizer)

    database_path = tmp_path / "persistent-research.sqlite"
    config = {"configurable": {"thread_id": "sqlite-restart-test"}}
    first_graph = workflow.build_research_graph(checkpoint_db_path=database_path)
    list(first_graph.stream(_initial_state(), config, stream_mode="updates"))

    first_snapshot = first_graph.get_state(config)
    assert first_snapshot.next == ("human_in_the_loop",)
    saved_findings = first_snapshot.values["findings"]
    assert saved_findings["health impacts"][0]["title"] == "Evidence for health impacts"
    first_graph.checkpointer.conn.close()

    restarted_graph = workflow.build_research_graph(checkpoint_db_path=database_path)
    restored_snapshot = restarted_graph.get_state(config)
    assert restored_snapshot.next == ("human_in_the_loop",)
    assert restored_snapshot.values["findings"] == saved_findings
    assert restored_snapshot.values["user_question"] == _initial_state()["user_question"]

    edited_findings = {
        "health impacts": [{"title": "Reviewed source", "url": "https://reviewed.test"}]
    }
    restarted_graph.update_state(config, {"findings": edited_findings})
    list(restarted_graph.stream(None, config, stream_mode="updates"))

    completed_snapshot = restarted_graph.get_state(config)
    assert completed_snapshot.next == ()
    assert completed_snapshot.values["final_report"]["findings"] == edited_findings
    assert calls["synthesizer"] == 1
    restarted_graph.checkpointer.conn.close()


def test_retry_stops_when_maximum_research_round_is_reached() -> None:
    """Do not route back to research after the configured retry limit."""
    state = _initial_state()
    state.update(
        {
            "needs_research": True,
            "research_round": 3,
            "max_research_rounds": 3,
        }
    )

    assert workflow._route_after_critic(state) == "human_in_the_loop"
