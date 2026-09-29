"""Serve the browser interface and expose a local API for resumable research runs."""

from __future__ import annotations

import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from tavily import TavilyClient
from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

from main import _validated_findings
from state import AgentState
from tools.search import DEFAULT_CLIMATE_DOMAINS

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
HOST = "127.0.0.1"
PORT = int(os.getenv("RESEARCH_PORT", "8765"))

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_graph: Any = None
_graph_lock = threading.Lock()


def _openai_model_name() -> str:
    """Read the model setting after dotenv has been loaded."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _provider_error(provider: str, exc: Exception) -> dict[str, Any]:
    """Map provider exceptions to safe, actionable messages without exposing secrets."""
    status_code = getattr(exc, "status_code", None)
    error_text = str(exc).lower()

    if provider == "openai":
        if status_code == 404 or "not found" in error_text:
            return {
                "provider": provider,
                "code": "openai_model_unavailable",
                "message": f"The OpenAI model '{_openai_model_name()}' is unavailable.",
                "action": "Check OPENAI_MODEL and confirm the model is available to your OpenAI project.",
                "retryable": True,
            }
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)) or status_code in (
            401,
            403,
        ):
            return {
                "provider": provider,
                "code": "openai_auth_failed",
                "message": "OpenAI rejected the configured API key or project access.",
                "action": "Check OPENAI_API_KEY and your OpenAI project permissions, then restart the app.",
                "retryable": True,
            }
        if isinstance(exc, RateLimitError) or status_code == 429:
            return {
                "provider": provider,
                "code": "openai_rate_limited",
                "message": "The OpenAI request limit or quota has been reached.",
                "action": "Check your OpenAI project limits or retry after the limit resets.",
                "retryable": True,
            }
        if (
            isinstance(exc, (APIConnectionError, ConnectionError, TimeoutError))
            or type(exc).__name__ in {"ConnectError", "ConnectTimeout", "TimeoutException"}
            or "connect" in error_text
        ):
            return {
                "provider": provider,
                "code": "openai_unavailable",
                "message": "The OpenAI API could not be reached.",
                "action": "Check your network connection and OpenAI service status, then resume the run.",
                "retryable": True,
            }
        return {
            "provider": provider,
            "code": "openai_request_failed",
            "message": "OpenAI could not complete this step.",
            "action": "Check your OpenAI model and project configuration, then retry the run.",
            "retryable": True,
        }

    if isinstance(exc, InvalidAPIKeyError) or status_code in (401, 403):
        return {
            "provider": "tavily",
            "code": "tavily_auth_failed",
            "message": "Tavily rejected the configured API key.",
            "action": "Update TAVILY_API_KEY in .env, restart the app, and resume.",
            "retryable": True,
        }
    if isinstance(exc, UsageLimitExceededError) or status_code == 429:
        return {
            "provider": "tavily",
            "code": "tavily_limit_reached",
            "message": "The Tavily request limit has been reached.",
            "action": "Check your Tavily plan or retry after the quota resets.",
            "retryable": True,
        }
    if (
        isinstance(exc, (ConnectionError, TimeoutError))
        or type(exc).__name__ in {"ConnectError", "ConnectTimeout", "TimeoutException"}
        or "connect" in error_text
    ):
        return {
            "provider": "tavily",
            "code": "tavily_unavailable",
            "message": "Tavily could not be reached to run the web search.",
            "action": "Check your network connection, then resume the run.",
            "retryable": True,
        }
    return {
        "provider": "tavily",
        "code": "tavily_search_failed",
        "message": "Tavily could not complete the web search.",
        "action": "Check Tavily service status and your search configuration, then retry.",
        "retryable": True,
    }


def _check_openai_health() -> dict[str, Any]:
    """Verify the OpenAI key and configured model are available."""
    model_name = _openai_model_name()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "model": model_name,
            "error": {
                "provider": "openai",
                "code": "openai_key_missing",
                "message": "OPENAI_API_KEY is not configured.",
                "action": "Add a valid OPENAI_API_KEY to .env, then restart the app.",
            },
        }
    try:
        OpenAI(api_key=api_key, timeout=3).models.retrieve(model_name)
        return {"status": "ok", "model": model_name}
    except Exception as exc:
        return {"status": "error", "model": model_name, "error": _provider_error("openai", exc)}


def _check_tavily_health() -> dict[str, Any]:
    """Validate Tavily credentials and search availability with a minimal request."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error": {
                "provider": "tavily",
                "code": "tavily_key_missing",
                "message": "TAVILY_API_KEY is not configured.",
                "action": "Add a valid Tavily key to .env, then restart the app.",
            },
        }
    try:
        TavilyClient(api_key=api_key).search(
            query="urban heat health",
            search_depth="basic",
            max_results=1,
            include_domains=list(DEFAULT_CLIMATE_DOMAINS),
            include_domains_mode="restrict",
            include_answer=False,
            timeout=5,
        )
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "error": _provider_error("tavily", exc)}


def _resume_checkpointed_run(thread_id: str) -> tuple[HTTPStatus, dict[str, Any]]:
    """Validate a persisted interrupted run and relaunch its unfinished graph step."""
    try:
        job = _restore_job(thread_id)
    except Exception as exc:
        print(f"Could not restore research run ({type(exc).__name__}).", file=sys.stderr)
        return (
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {
                "error": {
                    "code": "checkpoint_read_failed",
                    "message": "The saved research checkpoint could not be opened.",
                    "action": "Check the SQLite database path and file permissions.",
                }
            },
        )
    if job is None:
        return HTTPStatus.NOT_FOUND, {"error": "Research run not found."}

    with _jobs_lock:
        if job["status"] not in {"interrupted", "error"}:
            return (
                HTTPStatus.CONFLICT,
                {"error": "This research run is not interrupted or failed."},
            )
        previous_status = job["status"]
        job["status"] = "resuming"

    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _get_graph().get_state(config)
    except Exception as exc:
        with _jobs_lock:
            job["status"] = previous_status
        print(f"Could not load research checkpoint ({type(exc).__name__}).", file=sys.stderr)
        return (
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {
                "error": {
                    "code": "checkpoint_read_failed",
                    "message": "The saved research checkpoint could not be opened.",
                    "action": "Check the SQLite database path and file permissions.",
                }
            },
        )
    if not snapshot.values or not snapshot.next:
        with _jobs_lock:
            job["status"] = previous_status
        return (
            HTTPStatus.CONFLICT,
            {
                "error": {
                    "code": "checkpoint_not_resumable",
                    "message": "No unfinished graph step is available to resume.",
                    "action": "Start a new research run.",
                }
            },
        )

    with _jobs_lock:
        job["status"] = "running"
        job["events"].clear()
    _launch_resume_worker(thread_id)
    return HTTPStatus.ACCEPTED, {"status": "resuming"}


def _get_graph() -> Any:
    """Create one checkpointed graph instance shared by all local requests."""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                from graph.workflow import build_research_graph

                _graph = build_research_graph()
    return _graph


def _restore_job(thread_id: str) -> dict[str, Any] | None:
    """Rebuild an in-memory event view from a persisted graph checkpoint."""
    with _jobs_lock:
        existing_job = _jobs.get(thread_id)
        if existing_job is not None:
            return existing_job

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = _get_graph().get_state(config)
    if not snapshot.values:
        return None

    if "human_in_the_loop" in snapshot.next:
        status = "waiting_review"
        event = {
            "type": "review",
            "findings": snapshot.values["findings"],
            "subtasks": snapshot.values["subtasks"],
            "quality_score": snapshot.values["quality_score"],
            "task_reviews": snapshot.values["task_reviews"],
            "critic_feedback": snapshot.values["critic_feedback"],
        }
    elif not snapshot.next and snapshot.values.get("final_report"):
        status = "complete"
        event = {"type": "complete", "report": snapshot.values["final_report"]}
    else:
        status = "interrupted"
        event = {
            "type": "error",
            "message": (
                "This run was interrupted while an agent was working. "
                "Its last checkpoint is preserved and can be resumed."
            ),
            "action": "Resume the run to retry the unfinished agent step.",
            "resumable": bool(snapshot.next),
        }

    restored_job = {
        "status": status,
        "events": [event],
        "condition": threading.Condition(_jobs_lock),
    }
    with _jobs_lock:
        return _jobs.setdefault(thread_id, restored_job)


def _append_event(thread_id: str, event_type: str, **data: Any) -> None:
    """Publish an ordered event for the browser to read through Server-Sent Events."""
    with _jobs_lock:
        job = _jobs[thread_id]
        job["events"].append({"type": event_type, **data})
        job["condition"].notify_all()


def _launch_resume_worker(thread_id: str) -> None:
    """Start the worker that continues a graph from the current checkpoint."""
    threading.Thread(
        target=_run_graph,
        args=(thread_id, None),
        daemon=True,
    ).start()


def _run_graph(thread_id: str, input_state: AgentState | None) -> None:
    """Run or resume a graph in a worker thread and publish each node's update."""
    config = {"configurable": {"thread_id": thread_id}}
    graph: Any = None
    current_node = "workflow"
    try:
        graph = _get_graph()
        current_node = "planner" if input_state is not None else "research"
        for update in graph.stream(input_state, config, stream_mode="updates"):
            for node, node_update in update.items():
                if node != "__interrupt__":
                    current_node = node
                _append_event(
                    thread_id,
                    "node",
                    node=node,
                    update=node_update,
                )

        snapshot = graph.get_state(config)
        if "human_in_the_loop" in snapshot.next:
            with _jobs_lock:
                _jobs[thread_id]["status"] = "waiting_review"
            _append_event(
                thread_id,
                "review",
                findings=snapshot.values["findings"],
                subtasks=snapshot.values["subtasks"],
                quality_score=snapshot.values["quality_score"],
                task_reviews=snapshot.values["task_reviews"],
                critic_feedback=snapshot.values["critic_feedback"],
            )
        else:
            with _jobs_lock:
                _jobs[thread_id]["status"] = "complete"
            _append_event(
                thread_id,
                "complete",
                report=snapshot.values["final_report"],
            )
    except Exception as exc:
        if graph is None:
            resumable = False
            provider_error = {
                "provider": "workflow",
                "code": "checkpoint_store_unavailable",
                "message": "The research workflow could not open its checkpoint store.",
                "action": "Check CHECKPOINT_DB_PATH and database file permissions, then restart the app.",
                "retryable": False,
            }
        else:
            try:
                snapshot = graph.get_state(config)
                resumable = bool(snapshot.values and snapshot.next)
                failed_node = snapshot.next[0] if snapshot.next else current_node
            except Exception as checkpoint_error:
                print(
                    "Could not inspect failed run checkpoint "
                    f"({type(checkpoint_error).__name__}).",
                    file=sys.stderr,
                )
                resumable = False
                failed_node = current_node
            if failed_node == "researcher":
                provider_error = _provider_error("tavily", exc)
            elif failed_node in {"planner", "critic", "synthesizer"}:
                provider_error = _provider_error("openai", exc)
            else:
                provider_error = {
                    "provider": "workflow",
                    "code": "workflow_step_failed",
                    "message": "The research workflow could not complete this step.",
                    "action": "Inspect application logs; resume from the saved checkpoint if possible.",
                    "retryable": True,
                }
        with _jobs_lock:
            _jobs[thread_id]["status"] = "error"
        _append_event(
            thread_id,
            "error",
            **provider_error,
            resumable=resumable,
        )


class ResearchRequestHandler(BaseHTTPRequestHandler):
    """Handle the static UI and JSON/SSE endpoints for local research runs."""

    server_version = "ResearchAssistant/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        """Keep request logs concise without suppressing application errors."""
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        """Write one JSON response with explicit content and cache headers."""
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> dict[str, Any]:
        """Parse a bounded JSON request body and reject malformed inputs."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if content_length <= 0 or content_length > 1_000_000:
            raise ValueError("Request body must be between 1 byte and 1 MB.")
        try:
            payload = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        """Serve the single-page research UI or stream run progress."""
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
        elif parsed.path.startswith("/api/events/"):
            self._stream_events(parsed.path.removeprefix("/api/events/"), parsed.query)
        elif parsed.path == "/api/health":
            providers = {
                "openai": _check_openai_health(),
                "tavily": _check_tavily_health(),
            }
            status = (
                "ok"
                if all(provider["status"] == "ok" for provider in providers.values())
                else "degraded"
            )
            self._send_json(HTTPStatus.OK, {"status": status, "providers": providers})
        elif parsed.path.startswith("/api/research/"):
            thread_id = parsed.path.removeprefix("/api/research/").strip("/")
            self._get_research_status(thread_id)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        """Start a research run or submit edited evidence for synthesis."""
        parsed = urlparse(self.path)
        if parsed.path == "/api/research":
            self._start_research()
            return
        if parsed.path.startswith("/api/research/") and parsed.path.endswith("/review"):
            thread_id = parsed.path.removeprefix("/api/research/").removesuffix("/review").strip("/")
            self._submit_review(thread_id)
            return
        if parsed.path.startswith("/api/research/") and parsed.path.endswith("/resume"):
            thread_id = parsed.path.removeprefix("/api/research/").removesuffix("/resume").strip("/")
            self._resume_research(thread_id)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def _serve_index(self) -> None:
        """Return the browser application without exposing other project files."""
        content = (STATIC_DIR / "index.html").read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _start_research(self) -> None:
        """Validate the question, create a checkpointed run, and start its worker."""
        try:
            request = self._read_json()
            question = request.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Enter a research question to continue.")
            if not os.getenv("TAVILY_API_KEY"):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": {
                            "code": "tavily_key_missing",
                            "message": "TAVILY_API_KEY is not configured.",
                            "action": "Add a valid Tavily key to .env, then restart the app.",
                        }
                    },
                )
                return

            thread_id = str(uuid4())
            condition = threading.Condition(_jobs_lock)
            with _jobs_lock:
                _jobs[thread_id] = {
                    "status": "running",
                    "events": [],
                    "condition": condition,
                }

            initial_state: AgentState = {
                "user_question": question.strip(),
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
            threading.Thread(
                target=_run_graph,
                args=(thread_id, initial_state),
                daemon=True,
            ).start()
            self._send_json(HTTPStatus.ACCEPTED, {"thread_id": thread_id})
        except ValueError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "code": "invalid_research_request",
                        "message": str(exc),
                    }
                },
            )

    def _submit_review(self, thread_id: str) -> None:
        """Validate reviewed findings, update the paused checkpoint, and resume."""
        try:
            request = self._read_json()
            findings = _validated_findings(request.get("findings"))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        job = _restore_job(thread_id)
        if job is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Research run not found."})
            return
        with _jobs_lock:
            if job["status"] != "waiting_review":
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": "This research run is not waiting for review."},
                )
                return
            job["status"] = "running"

        config = {"configurable": {"thread_id": thread_id}}
        try:
            _get_graph().update_state(config, {"findings": findings})
        except Exception as exc:
            with _jobs_lock:
                _jobs[thread_id]["status"] = "waiting_review"
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Could not save the review: {exc}"},
            )
            return
        threading.Thread(
            target=_run_graph,
            args=(thread_id, None),
            daemon=True,
        ).start()
        self._send_json(HTTPStatus.ACCEPTED, {"status": "resuming"})

    def _get_research_status(self, thread_id: str) -> None:
        """Return persisted review or completion data so clients can recover after restart."""
        try:
            job = _restore_job(thread_id)
        except Exception as exc:
            print(f"Could not restore research status ({type(exc).__name__}).", file=sys.stderr)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": {
                        "code": "checkpoint_read_failed",
                        "message": "The saved research run could not be read.",
                        "action": "Check the SQLite database path and file permissions.",
                    }
                },
            )
            return
        if job is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Research run not found."})
            return
        latest_event = job["events"][-1] if job["events"] else {}
        self._send_json(
            HTTPStatus.OK,
            {
                "status": job["status"],
                "cursor": len(job["events"]),
                **{
                    key: value
                    for key, value in latest_event.items()
                    if key != "type"
                },
            },
        )

    def _resume_research(self, thread_id: str) -> None:
        """Retry the unfinished graph step from its most recent SQLite checkpoint."""
        status, response = _resume_checkpointed_run(thread_id)
        self._send_json(status, response)

    def _stream_events(self, thread_id: str, query: str) -> None:
        """Stream ordered workflow events until review, completion, or failure."""
        job = _restore_job(thread_id)
        if job is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Research run not found."})
            return
        with _jobs_lock:
            try:
                after = int(parse_qs(query).get("after", ["0"])[0])
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid event cursor."})
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        self.close_connection = True
        cursor = max(after, 0)
        try:
            while True:
                with _jobs_lock:
                    job = _jobs.get(thread_id)
                    if job is None:
                        return
                    if cursor < len(job["events"]):
                        event = job["events"][cursor]
                        cursor += 1
                        should_close = event["type"] in {"review", "complete", "error"}
                    else:
                        job["condition"].wait(timeout=15)
                        event = None
                        should_close = False

                if event is None:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue

                encoded = json.dumps(event, ensure_ascii=False, default=str)
                frame = f"id: {cursor}\ndata: {encoded}\n\n".encode("utf-8")
                self.wfile.write(frame)
                self.wfile.flush()
                if should_close:
                    return
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    """Load local credentials and start the loopback-only browser server."""
    load_dotenv(ROOT / ".env")
    server = ThreadingHTTPServer((HOST, PORT), ResearchRequestHandler)
    print(f"Research Assistant is running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Research Assistant.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
