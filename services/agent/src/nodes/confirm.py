"""Confirm node — human-in-the-loop checkpoint."""

import structlog

from ..state import AgentState

logger = structlog.get_logger()


async def confirm_node(state: AgentState) -> dict:
    """Human-in-the-loop node.

    This node is used with LangGraph's interrupt_before mechanism.
    When reached, the graph pauses and waits for external confirmation.
    The gateway sends the pending_confirmation to the frontend,
    and resumes the graph with the user's decision.
    """
    pending = state.get("pending_confirmation")

    if pending is None:
        # No confirmation needed, pass through
        return {}

    logger.info("Confirmation checkpoint", action=pending)

    # The graph will be interrupted before this node.
    # When resumed, the pending_confirmation will have been
    # updated by the gateway with the user's decision.
    return {
        "reasoning_trace": state.get("reasoning_trace", []) + [
            {"node": "confirm", "action": f"Awaiting confirmation for: {pending.get('description', '')}"}
        ],
    }
