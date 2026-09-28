"""Serve the browser interface and expose a local API for resumable research runs."""

from __future__ import annotations

import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from dotenv import load_dotenv

from main import _validated_findings
from state import AgentState

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
HOST = "127.0.0.1"
PORT = int(os.getenv("RESEARCH_PORT", "8765"))

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_graph: Any = None
_graph_lock = threading.Lock()


def _get_graph() -> Any:
    """Create one checkpointed graph instance shared by all local requests."""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                from graph.workflow import build_research_graph

                _graph = build_research_graph()
    return _graph


def _append_event(thread_id: str, event_type: str, **data: Any) -> None:
    """Publish an ordered event for the browser to read through Server-Sent Events."""
    with _jobs_lock:
        job = _jobs[thread_id]
        job["events"].append({"type": event_type, **data})
        job["condition"].notify_all()


def _run_graph(thread_id: str, input_state: AgentState | None) -> None:
    """Run or resume a graph in a worker thread and publish each node's update."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        for update in graph.stream(input_state, config, stream_mode="updates"):
            for node, node_update in update.items():
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
        with _jobs_lock:
            _jobs[thread_id]["status"] = "error"
        _append_event(thread_id, "error", message=str(exc))


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
            self._send_json(HTTPStatus.OK, {"status": "ok"})
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
            missing_keys = [
                key
                for key in ("TAVILY_API_KEY",)
                if not os.getenv(key)
            ]
            if missing_keys:
                raise ValueError(
                    "Add the required key(s) to .env before researching: "
                    + ", ".join(missing_keys)
                )

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
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _submit_review(self, thread_id: str) -> None:
        """Validate reviewed findings, update the paused checkpoint, and resume."""
        try:
            request = self._read_json()
            findings = _validated_findings(request.get("findings"))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        with _jobs_lock:
            job = _jobs.get(thread_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Research run not found."})
                return
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

    def _stream_events(self, thread_id: str, query: str) -> None:
        """Stream ordered workflow events until review, completion, or failure."""
        with _jobs_lock:
            job = _jobs.get(thread_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Research run not found."})
                return
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
