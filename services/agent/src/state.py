"""Agent state definition for the LangGraph graph."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for the LangGraph agent."""

    # Conversation history — automatically accumulated via add_messages
    messages: Annotated[list, add_messages]

    # Current query info
    query: str
    query_type: str  # "simple" | "multi_step" | "action"

    # Planning
    plan: list[str]

    # Retrieval results
    retrieved_chunks: list[dict]

    # MCP tool results
    tool_results: list[dict]

    # Human-in-the-loop
    pending_confirmation: dict | None

    # Output
    final_answer: str

    # Observability
    reasoning_trace: list[dict]

    # Context
    team: str
