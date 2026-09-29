"""Run the research graph interactively and let users edit evidence before synthesis."""

import json
import os
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from state import AgentState


def _print_updates(events: Any) -> None:
    """Display each graph node's state update as it streams."""
    for event in events:
        for node_name, update in event.items():
            print(f"\n=== {node_name} ===")
            print(json.dumps(update, ensure_ascii=False, indent=2, default=str))


def _validated_findings(raw_findings: object) -> dict[str, list[dict[str, Any]]]:
    """Check user-edited evidence has the same shape expected by downstream nodes."""
    if not isinstance(raw_findings, dict):
        raise ValueError("Edited findings must be a JSON object keyed by sub-task.")

    findings: dict[str, list[dict[str, Any]]] = {}
    for task, results in raw_findings.items():
        if not isinstance(task, str) or not isinstance(results, list):
            raise ValueError(
                "Each findings entry must map a sub-task string to a list of results."
            )
        if any(not isinstance(result, dict) for result in results):
            raise ValueError("Each search result must be a JSON object.")
        findings[task] = results
    return findings


def _review_findings(graph: Any, config: dict[str, Any]) -> None:
    """Pause at the graph checkpoint and optionally replace findings with user edits."""
    snapshot = graph.get_state(config)
    if "human_in_the_loop" not in snapshot.next:
        return

    print("\n=== Human review ===")
    print("Review the gathered findings below:")
    print(json.dumps(snapshot.values["findings"], ensure_ascii=False, indent=2))
    edited_json = input(
        "\nPress Enter to continue unchanged, or paste a replacement findings JSON object: "
    ).strip()
    if edited_json:
        edited_findings = _validated_findings(json.loads(edited_json))
        graph.update_state(config, {"findings": edited_findings})


def main() -> None:
    """Load credentials, stream graph progress, pause for review, and print the report."""
    load_dotenv()
    missing_keys = [
        key for key in ("OPENAI_API_KEY", "TAVILY_API_KEY") if not os.getenv(key)
    ]
    if missing_keys:
        raise SystemExit(
            "Missing required API key(s): "
            + ", ".join(missing_keys)
            + ". Set them in research-agent/.env."
        )

    from graph.workflow import build_research_graph

    question = input("What would you like to research? ").strip()
    if not question:
        raise SystemExit("A non-empty research question is required.")

    graph = build_research_graph()
    config = {"configurable": {"thread_id": str(uuid4())}}
    initial_state: AgentState = {
        "user_question": question,
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

    _print_updates(graph.stream(initial_state, config, stream_mode="updates"))
    _review_findings(graph, config)
    _print_updates(graph.stream(None, config, stream_mode="updates"))

    final_state = graph.get_state(config).values
    print("\n=== Final research report ===")
    print(json.dumps(final_state["final_report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
