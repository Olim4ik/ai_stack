import asyncio
import signal

import grpc
import structlog
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from langchain_openai import ChatOpenAI

from .config import settings
from .generated import agent_pb2_grpc
from .graph.builder import build_graph
from .grpc_clients.retrieval import RetrievalClient
from .mcp_client.client import MCPClientWrapper
from .service import AgentServicer

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


async def serve() -> None:
    # Initialize LLM via LangChain (swap provider by changing langchain-openai → langchain-anthropic etc.)
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
    )

    retrieval_client = RetrievalClient(host=settings.retrieval_service_host)
    await retrieval_client.connect()

    mcp_client = MCPClientWrapper(
        server_command=settings.mcp_server_command,
        server_cwd=settings.mcp_server_cwd,
    )
    await mcp_client.connect()

    # Build the LangGraph graph
    graph = build_graph(
        llm=llm,
        retrieval_client=retrieval_client,
        mcp_client=mcp_client,
    )

    # Create gRPC server
    server = grpc.aio.server()

    servicer = AgentServicer(graph=graph)
    agent_pb2_grpc.add_AgentServiceServicer_to_server(servicer, server)

    # Health service
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set(
        "agent.AgentService",
        health_pb2.HealthCheckResponse.SERVING,
    )

    listen_addr = f"0.0.0.0:{settings.agent_port}"
    server.add_insecure_port(listen_addr)

    logger.info(
        "Starting Agent Service",
        port=settings.agent_port,
        model=settings.llm_model,
        retrieval=settings.retrieval_service_host,
    )

    await server.start()

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()
    logger.info("Shutting down gracefully")
    await retrieval_client.close()
    await server.stop(grace=5)


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
