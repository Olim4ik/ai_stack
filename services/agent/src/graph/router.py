"""Conditional routing logic for the LangGraph agent."""

from ..state import AgentState


def route_by_query_type(state: AgentState) -> str:
    """Route to the appropriate node based on query classification."""
    match state.get("query_type", "simple"):
        case "simple":
            return "retrieve"
        case "multi_step":
            return "plan"
        case "action":
            return "execute"
        case _:
            return "retrieve"


def route_after_execute(state: AgentState) -> str:
    """After execute, check if confirmation is needed."""
    if state.get("pending_confirmation"):
        return "confirm"
    return "synthesize"
