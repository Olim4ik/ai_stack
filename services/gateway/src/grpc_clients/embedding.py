import grpc
import structlog

from ..generated import embedding_pb2, embedding_pb2_grpc

logger = structlog.get_logger()


class EmbeddingClient:
    """gRPC client for the Embedding Service."""

    def __init__(self, host: str) -> None:
        self._host = host
        self._channel: grpc.aio.Channel | None = None
        self._stub: embedding_pb2_grpc.EmbeddingServiceStub | None = None

    async def connect(self) -> None:
        self._channel = grpc.aio.insecure_channel(self._host)
        self._stub = embedding_pb2_grpc.EmbeddingServiceStub(self._channel)
        logger.info("Connected to Embedding Service", host=self._host)

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()

    async def embed(self, text: str) -> list[float]:
        response = await self._stub.Embed(embedding_pb2.EmbedRequest(text=text))
        return list(response.vector)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._stub.EmbedBatch(embedding_pb2.EmbedBatchRequest(texts=texts))
        return [list(e.vector) for e in response.embeddings]
