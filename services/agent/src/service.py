"""AgentService gRPC implementation with streaming."""

import json
import time
import uuid

import structlog

from .state import AgentState

logger = structlog.get_logger()

# In-memory session store (upgrade to Redis for production)
_sessions: dict[str, dict] = {}


class AgentServicer:
    """gRPC servicer for the Agent Service."""

    def __init__(self, graph) -> None:
        self._graph = graph

    async def Chat(self, request, context):
        """Handle a chat request — run the graph and stream events."""
        from .generated import agent_pb2

        session_id = request.session_id or str(uuid.uuid4())
        message = request.message
        team = request.team or "platform"
        start_time = time.monotonic()

        logger.info(
            "Chat request started",
            session_id=session_id,
            team=team,
            query_length=len(message),
            query_preview=message[:80],
        )

        # Build initial state
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": message}],
            "query": message,
            "query_type": "",
            "plan": [],
            "retrieved_chunks": [],
            "tool_results": [],
            "pending_confirmation": None,
            "final_answer": "",
            "reasoning_trace": [],
            "team": team,
        }

        # Store session for potential resume
        _sessions[session_id] = {"state": initial_state}

        # Stream graph execution
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in self._graph.astream(initial_state, config=config):
                for node_name, node_output in event.items():
                    # Stream reasoning trace
                    trace = node_output.get("reasoning_trace", [])
                    if trace:
                        latest = trace[-1]
                        yield agent_pb2.ChatResponse(
                            event_type="reasoning",
                            data=json.dumps(latest),
                        )

                    # Stream sources when retrieved
                    chunks = node_output.get("retrieved_chunks", [])
                    for chunk in chunks:
                        yield agent_pb2.ChatResponse(
                            event_type="source",
                            data=json.dumps({
                                "chunk_id": chunk.get("chunk_id", ""),
                                "score": chunk.get("score", 0),
                                "metadata": chunk.get("metadata", {}),
                            }),
                        )

                    # Check for pending confirmation (graph will interrupt)
                    pending = node_output.get("pending_confirmation")
                    if pending:
                        action_id = str(uuid.uuid4())
                        _sessions[session_id]["pending_action_id"] = action_id
                        _sessions[session_id]["config"] = config
                        yield agent_pb2.ChatResponse(
                            event_type="confirm_required",
                            data=json.dumps({
                                "action_id": action_id,
                                "tool": pending.get("tool", ""),
                                "description": pending.get("description", ""),
                            }),
                        )

                    # Stream final answer
                    answer = node_output.get("final_answer", "")
                    if answer:
                        yield agent_pb2.ChatResponse(
                            event_type="token",
                            data=json.dumps({"content": answer}),
                        )

        except Exception as e:
            logger.error("Graph execution failed", session_id=session_id, error=str(e))
            yield agent_pb2.ChatResponse(
                event_type="error",
                data=json.dumps({"message": str(e)}),
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Chat request completed",
            session_id=session_id,
            total_ms=round(elapsed_ms, 1),
        )

        yield agent_pb2.ChatResponse(
            event_type="done",
            data=json.dumps({"session_id": session_id}),
        )

    async def ResumeChat(self, request, context):
        """Resume a paused graph after human-in-the-loop confirmation."""
        from .generated import agent_pb2

        session_id = request.session_id
        action_id = request.action_id
        approved = request.approved

        session = _sessions.get(session_id)
        if not session:
            yield agent_pb2.ChatResponse(
                event_type="error",
                data=json.dumps({"message": "Session not found"}),
            )
            return

        logger.info("Resuming chat", session_id=session_id, approved=approved)

        config = session.get("config", {"configurable": {"thread_id": session_id}})

        if not approved:
            yield agent_pb2.ChatResponse(
                event_type="token",
                data=json.dumps({"content": "Action was rejected by user."}),
            )
            yield agent_pb2.ChatResponse(
                event_type="done",
                data=json.dumps({"session_id": session_id}),
            )
            return

        # Resume graph execution — clear pending confirmation
        try:
            resume_state = {"pending_confirmation": None}
            async for event in self._graph.astream(resume_state, config=config):
                for node_name, node_output in event.items():
                    answer = node_output.get("final_answer", "")
                    if answer:
                        yield agent_pb2.ChatResponse(
                            event_type="token",
                            data=json.dumps({"content": answer}),
                        )
        except Exception as e:
            logger.error("Resume failed", error=str(e))
            yield agent_pb2.ChatResponse(
                event_type="error",
                data=json.dumps({"message": str(e)}),
            )

        yield agent_pb2.ChatResponse(
            event_type="done",
            data=json.dumps({"session_id": session_id}),
        )
