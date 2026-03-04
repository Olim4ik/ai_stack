import grpc
import structlog

from ..generated import agent_pb2, agent_pb2_grpc

logger = structlog.get_logger()


class AgentClient:
    """gRPC client for the LangGraph Agent Service."""

    def __init__(self, host: str) -> None:
        self._host = host
        self._channel: grpc.aio.Channel | None = None
        self._stub: agent_pb2_grpc.AgentServiceStub | None = None

    async def connect(self) -> None:
        self._channel = grpc.aio.insecure_channel(self._host)
        self._stub = agent_pb2_grpc.AgentServiceStub(self._channel)
        logger.info("Connected to Agent Service", host=self._host)

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()

    async def chat(self, session_id: str, message: str, team: str):
        """Stream chat responses from the Agent.

        Yields (event_type, data) tuples from the gRPC stream.
        """
        request = agent_pb2.ChatRequest(
            session_id=session_id,
            message=message,
            team=team,
        )
        async for response in self._stub.Chat(request):
            yield response.event_type, response.data

    async def resume_chat(self, session_id: str, action_id: str, approved: bool):
        """Resume a paused chat after human-in-the-loop confirmation.

        Yields (event_type, data) tuples from the gRPC stream.
        """
        request = agent_pb2.ResumeRequest(
            session_id=session_id,
            action_id=action_id,
            approved=approved,
        )
        async for response in self._stub.ResumeChat(request):
            yield response.event_type, response.data
