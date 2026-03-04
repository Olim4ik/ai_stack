from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from .config import settings
from .grpc_clients.agent import AgentClient
from .grpc_clients.embedding import EmbeddingClient
from .grpc_clients.retrieval import RetrievalClient
from .ingestion.pipeline import IngestionPipeline
from .middleware.errors import global_error_handler
from .middleware.logging import LoggingMiddleware
from .routers import chat, documents, health

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to services. Shutdown: close connections."""
    logger.info("Starting FastAPI Gateway")

    # Qdrant direct client (for ingestion & document management)
    qdrant = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    app.state.qdrant = qdrant

    # gRPC clients
    embedding_client = EmbeddingClient(host=settings.embedding_service_host)
    await embedding_client.connect()
    app.state.embedding_client = embedding_client

    retrieval_client = RetrievalClient(host=settings.retrieval_service_host)
    await retrieval_client.connect()
    app.state.retrieval_client = retrieval_client

    # Agent client
    agent_client = AgentClient(host=settings.agent_service_host)
    await agent_client.connect()
    app.state.agent_client = agent_client

    # Ingestion pipeline
    app.state.ingestion_pipeline = IngestionPipeline(
        qdrant=qdrant,
        embedding_client=embedding_client,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    logger.info("Gateway ready", port=settings.gateway_port)
    yield

    # Shutdown
    logger.info("Shutting down Gateway")
    await agent_client.close()
    await embedding_client.close()
    await retrieval_client.close()
    await qdrant.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Base Assistant — API Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(LoggingMiddleware)
    app.add_exception_handler(Exception, global_error_handler)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    return app


app = create_app()
