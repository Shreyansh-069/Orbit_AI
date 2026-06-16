from typing import Literal
from langgraph.graph import StateGraph

from config import AgentState
from agents import (
    doc_analysis_node,
    vision_agent_node,
    web_research_node,
)


def planner_node(state: AgentState):
    """
    Supervisor / Routing Node.
    Inspects uploaded inputs and enqueues the appropriate agents in pending_tasks.
    """
    pending_tasks = []

    if state.get("pdf_path") and state["pdf_path"]:
        pending_tasks.append("doc_agent")

    if state.get("image_path") and state["image_path"]:
        pending_tasks.append("vision_agent")

    # Web research always runs for grounding
    pending_tasks.append("web_agent")

    first_task = pending_tasks[0] if pending_tasks else "response_generator"

    return {
        "pending_tasks": pending_tasks,
        "current_task": f"Routing to {first_task}..."
    }


def routing_conditional_edge(
    state: AgentState,
) -> Literal["doc_agent", "vision_agent", "web_agent", "response_generator"]:
    """Picks the next node from the pending_tasks queue."""
    pending = state.get("pending_tasks", [])
    if not pending:
        return "response_generator"
    return pending[0]


# ── Graph Construction ─────────────────────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("doc_agent", doc_analysis_node)
builder.add_node("vision_agent", vision_agent_node)
builder.add_node("web_agent", web_research_node)

builder.set_entry_point("planner")

_edges_config = {
    "doc_agent": "doc_agent",
    "vision_agent": "vision_agent",
    "web_agent": "web_agent",
    "response_generator": "response_generator",
}

# Each agent node re-evaluates the queue after execution
for node in ["planner", "doc_agent", "vision_agent", "web_agent"]:
    builder.add_conditional_edges(node, routing_conditional_edge, _edges_config)