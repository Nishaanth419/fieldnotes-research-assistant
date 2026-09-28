"""Test provider health checks, actionable errors, and interrupted-run recovery."""

import json
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.request import Request, urlopen

import pytest
from graph import workflow
from state import AgentState
from webapp import (
    _check_ollama_health,
    _check_tavily_health,
    _jobs,
    _provider_error,
    _restore_job,
    _run_graph,
    _resume_checkpointed_run,
)
import webapp


@pytest.fixture
def local_app_server() -> Any:
    """Serve the app handler on an isolated ephemeral port for API tests."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), webapp.ResearchRequestHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_health_endpoint_reports_both_provider_states(
    monkeypatch: Any,
    local_app_server: str,
) -> None:
    """Expose Ollama and Tavily readiness together through the health API."""
    monkeypatch.setattr(webapp, "_check_ollama_health", lambda: {"status": "ok"})
    monkeypatch.setattr(
        webapp,
        "_check_tavily_health",
        lambda: {
            "status": "error",
            "error": {"code": "tavily_limit_reached", "action": "Wait for quota reset."},
        },
    )

    with urlopen(f"{local_app_server}/api/health") as response:
        health = json.load(response)

    assert health["status"] == "degraded"
    assert health["providers"]["ollama"]["status"] == "ok"
    assert health["providers"]["tavily"]["error"]["code"] == "tavily_limit_reached"


def test_resume_endpoint_routes_to_saved_checkpoint(
    monkeypatch: Any,
    local_app_server: str,
) -> None:
    """Expose the persisted-run resume operation through the HTTP API."""
    called_with: list[str] = []

    def resume(thread_id: str) -> tuple[HTTPStatus, dict[str, str]]:
        called_with.append(thread_id)
        return HTTPStatus.ACCEPTED, {"status": "resuming"}

    monkeypatch.setattr(webapp, "_resume_checkpointed_run", resume)
    request = Request(
        f"{local_app_server}/api/research/saved-thread/resume",
        data=b"",
        method="POST",
    )

    with urlopen(request) as response:
        body = json.load(response)

    assert called_with == ["saved-thread"]
    assert body == {"status": "resuming"}


def test_ollama_health_reports_ready_model(monkeypatch: Any) -> None:
    """Report the configured model only when it is present in the local service."""
    class FakeModel:
        model = "llama3.1:8b"

    class FakeResponse:
        models = [FakeModel()]

    class FakeOllamaClient:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["host"] == "http://ollama.test"

        def list(self) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    monkeypatch.setattr(webapp, "OllamaClient", FakeOllamaClient)

    assert _check_ollama_health() == {"status": "ok", "model": "llama3.1:8b"}


def test_ollama_health_gives_pull_command_for_missing_model(monkeypatch: Any) -> None:
    """Tell users how to install the configured model when Ollama is reachable."""
    class FakeResponse:
        models: list[object] = []

    class FakeOllamaClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def list(self) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setenv("OLLAMA_MODEL", "missing-model")
    monkeypatch.setattr(webapp, "OllamaClient", FakeOllamaClient)

    result = _check_ollama_health()

    assert result["status"] == "error"
    assert result["error"]["code"] == "ollama_model_missing"
    assert "ollama pull missing-model" in result["error"]["action"]


def test_tavily_health_requires_key_without_calling_provider(monkeypatch: Any) -> None:
    """Report missing Tavily configuration directly."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = _check_tavily_health()

    assert result["status"] == "error"
    assert result["error"]["code"] == "tavily_key_missing"


def test_tavily_health_checks_search_access_and_sanitizes_auth_error(
    monkeypatch: Any,
) -> None:
    """Check the provider using a minimal search without returning credential text."""
    secret = "never-return-this-secret"
    monkeypatch.setenv("TAVILY_API_KEY", secret)

    class UnauthorizedError(Exception):
        status_code = 401

    class FakeTavilyClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == secret

        def search(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["max_results"] == 1
            raise UnauthorizedError(secret)

    monkeypatch.setattr(webapp, "TavilyClient", FakeTavilyClient)

    result = _check_tavily_health()

    assert result["status"] == "error"
    assert result["error"]["code"] == "tavily_auth_failed"
    assert secret not in str(result)
    assert "TAVILY_API_KEY" in result["error"]["action"]


def test_provider_failures_include_actionable_recovery_steps() -> None:
    """Classify common Ollama and Tavily failures into safe next steps."""
    ollama_error = _provider_error("ollama", ConnectionError("connection refused"))
    tavily_error = _provider_error(
        "tavily",
        type("RateLimitError", (Exception,), {"status_code": 429})("rate limited"),
    )

    assert ollama_error["code"] == "ollama_unavailable"
    assert "ollama serve" in ollama_error["action"]
    assert tavily_error["code"] == "tavily_limit_reached"
    assert "quota" in tavily_error["action"].lower()


def test_checkpoint_store_failure_is_reported_as_a_worker_error(
    monkeypatch: Any,
) -> None:
    """Convert graph/checkpoint startup failures into a visible recoverable event."""
    run_id = "checkpoint-store-failure"

    def fail_graph_creation() -> Any:
        raise OSError("database file is not writable")

    monkeypatch.setattr(webapp, "_get_graph", fail_graph_creation)
    _jobs[run_id] = {
        "status": "running",
        "events": [],
        "condition": threading.Condition(webapp._jobs_lock),
    }

    _run_graph(run_id, None)

    assert _jobs[run_id]["status"] == "error"
    assert _jobs[run_id]["events"][-1]["code"] == "checkpoint_store_unavailable"
    assert _jobs[run_id]["events"][-1]["resumable"] is False


def test_interrupted_graph_run_can_be_restored_and_resumed(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Preserve failed task state in SQLite and offer a worker retry after restart."""
    run_id = "recoverable-planner-run"
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

    def failing_planner(_: AgentState) -> dict[str, list[str]]:
        raise ConnectionError("Ollama is unavailable")

    monkeypatch.setattr(workflow, "planner_agent", failing_planner)
    graph = workflow.build_research_graph(
        checkpoint_db_path=tmp_path / "recoverable-run.sqlite"
    )
    monkeypatch.setattr(webapp, "_get_graph", lambda: graph)
    _jobs[run_id] = {
        "status": "running",
        "events": [],
        "condition": threading.Condition(webapp._jobs_lock),
    }

    _run_graph(run_id, state)

    failed_job = _jobs[run_id]
    assert failed_job["status"] == "error"
    assert failed_job["events"][-1]["code"] == "ollama_unavailable"
    assert failed_job["events"][-1]["resumable"] is True
    graph.checkpointer.conn.close()

    _jobs.clear()
    restarted_graph = workflow.build_research_graph(
        checkpoint_db_path=tmp_path / "recoverable-run.sqlite"
    )
    monkeypatch.setattr(webapp, "_get_graph", lambda: restarted_graph)
    restored_job = _restore_job(run_id)
    assert restored_job is not None
    assert restored_job["status"] == "interrupted"
    assert restored_job["events"][0]["resumable"] is True

    resumed: list[str] = []
    monkeypatch.setattr(webapp, "_launch_resume_worker", resumed.append)
    status, body = _resume_checkpointed_run(run_id)

    assert status == HTTPStatus.ACCEPTED
    assert body == {"status": "resuming"}
    assert resumed == [run_id]
    assert restored_job["status"] == "running"
    restarted_graph.checkpointer.conn.close()
