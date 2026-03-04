import asyncio
import signal

import grpc
import structlog
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from .config import settings
from .providers import create_provider
from .service import EmbeddingServicer

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


async def serve() -> None:
    provider = create_provider(
        provider=settings.embedding_model_provider,
        model=settings.embedding_model_name,
        api_key=settings.openai_api_key,
    )

    server = grpc.aio.server()

    # Register embedding service
    from .generated import embedding_pb2_grpc

    servicer = EmbeddingServicer(provider=provider, max_batch_size=settings.embedding_max_batch_size)
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(servicer, server)

    # Register health service
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set(
        "embedding.EmbeddingService",
        health_pb2.HealthCheckResponse.SERVING,
    )

    listen_addr = f"0.0.0.0:{settings.embedding_port}"
    server.add_insecure_port(listen_addr)

    logger.info(
        "Starting Embedding Service",
        port=settings.embedding_port,
        provider=settings.embedding_model_provider,
        model=settings.embedding_model_name,
    )

    await server.start()

    # Graceful shutdown on SIGTERM/SIGINT
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()
    logger.info("Shutting down gracefully")
    await server.stop(grace=5)


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
