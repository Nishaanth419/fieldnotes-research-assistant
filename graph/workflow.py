"""Connect research agents into a reviewable graph with bounded quality retries."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.critic import critic_agent
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.synthesizer import synthesizer_agent
from state import AgentState


def human_in_the_loop(state: AgentState) -> dict[str, object]:
    """Mark the explicit review point; the graph pauses before this node runs."""
    return {}


def _route_after_critic(state: AgentState) -> str:
    """Retry research only when the critic requests it and rounds remain."""
    if state["needs_research"] and state["research_round"] < state["max_research_rounds"]:
        return "researcher"
    return "human_in_the_loop"


def build_research_graph():
    """Compile the planner→research→review→human approval→report workflow."""
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_agent)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("critic", critic_agent)
    builder.add_node("human_in_the_loop", human_in_the_loop)
    builder.add_node("synthesizer", synthesizer_agent)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "critic")
    builder.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "researcher": "researcher",
            "human_in_the_loop": "human_in_the_loop",
        },
    )
    builder.add_edge("human_in_the_loop", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_in_the_loop"],
    )
