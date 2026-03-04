"""Chat endpoints — SSE streaming via LangGraph Agent Service."""

import json

import structlog
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from ..models.requests import ChatRequest, ChatConfirmRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = structlog.get_logger()


@router.post("")
async def chat(request: Request, body: ChatRequest):
    """Send a message and stream the agent's response via SSE."""
    agent_client = request.app.state.agent_client

    # Fallback: if agent not connected, use direct retrieval
    if not agent_client:
        return await _fallback_retrieval(request, body)

    async def event_generator():
        try:
            async for event_type, data in agent_client.chat(
                session_id=body.session_id,
                message=body.message,
                team=body.team,
            ):
                yield {"event": event_type, "data": data}
        except Exception as e:
            logger.error("Agent stream error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}),
            }

    return EventSourceResponse(event_generator())


@router.post("/{session_id}/confirm")
async def confirm_action(
    request: Request,
    session_id: str,
    body: ChatConfirmRequest,
):
    """Submit human-in-the-loop confirmation and resume the agent stream."""
    agent_client = request.app.state.agent_client
    if not agent_client:
        return {"error": "Agent service not available"}

    async def event_generator():
        try:
            async for event_type, data in agent_client.resume_chat(
                session_id=session_id,
                action_id=body.action_id,
                approved=body.approved,
            ):
                yield {"event": event_type, "data": data}
        except Exception as e:
            logger.error("Agent resume error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}),
            }

    return EventSourceResponse(event_generator())


@router.get("/history")
async def chat_history(session_id: str | None = None):
    """Retrieve conversation history for a session.

    In-memory sessions are managed by the Agent service.
    This endpoint returns a stub until persistent storage is added.
    """
    return {"sessions": [], "session_id": session_id}


async def _fallback_retrieval(request: Request, body: ChatRequest):
    """Fallback when Agent Service is unavailable — direct retrieval."""
    retrieval_client = request.app.state.retrieval_client
    if not retrieval_client:
        return {"response": "Services not available", "sources": []}

    collection = f"team_{body.team}"
    result = await retrieval_client.search(
        query=body.message,
        collection=collection,
        top_k=5,
    )

    chunks = result.get("results", [])
    if chunks:
        context = "\n\n".join(c["text"] for c in chunks[:3])
        response_text = f"Based on the knowledge base:\n\n{context}"
    else:
        response_text = "No relevant documents found in the knowledge base."

    return {
        "session_id": body.session_id,
        "response": response_text,
        "sources": [
            {
                "chunk_id": c["chunk_id"],
                "score": c["score"],
                "metadata": c["metadata"],
            }
            for c in chunks
        ],
    }
