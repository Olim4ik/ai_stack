import asyncio
import signal

import grpc
import structlog
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from .config import settings
from .generated import retrieval_pb2_grpc
from .grpc_clients.embedding import EmbeddingClient
from .qdrant_client import QdrantManager
from .service import RetrievalServicer

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


async def serve() -> None:
    # Initialize dependencies
    qdrant = QdrantManager(host=settings.qdrant_host, port=settings.qdrant_port)
    await qdrant.connect()

    embedding_client = EmbeddingClient(host=settings.embedding_service_host)
    await embedding_client.connect()

    # Create gRPC server
    server = grpc.aio.server()

    servicer = RetrievalServicer(
        qdrant=qdrant,
        embedding_client=embedding_client,
        default_top_k=settings.default_top_k,
        hybrid_dense_weight=settings.hybrid_dense_weight,
    )
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(servicer, server)

    # Health service
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set(
        "retrieval.RetrievalService",
        health_pb2.HealthCheckResponse.SERVING,
    )

    listen_addr = f"0.0.0.0:{settings.retrieval_port}"
    server.add_insecure_port(listen_addr)

    logger.info(
        "Starting Retrieval Service",
        port=settings.retrieval_port,
        qdrant=f"{settings.qdrant_host}:{settings.qdrant_port}",
        embedding=settings.embedding_service_host,
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
    await embedding_client.close()
    await qdrant.close()
    await server.stop(grace=5)


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
