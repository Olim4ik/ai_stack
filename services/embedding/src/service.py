import structlog
from grpc import StatusCode

from .providers.base import EmbeddingProvider

logger = structlog.get_logger()


class EmbeddingServicer:
    """gRPC servicer for the Embedding Service."""

    def __init__(self, provider: EmbeddingProvider, max_batch_size: int = 64) -> None:
        self._provider = provider
        self._max_batch_size = max_batch_size

    async def Embed(self, request, context):
        from .generated import embedding_pb2

        text = request.text
        if not text.strip():
            await context.abort(StatusCode.INVALID_ARGUMENT, "Text must not be empty")

        logger.info("Embedding single text", length=len(text))
        try:
            vector = await self._provider.embed(text)
            return embedding_pb2.EmbedResponse(
                vector=vector,
                dimensions=self._provider.dimensions,
            )
        except Exception as e:
            logger.error("Embedding failed", error=str(e))
            await context.abort(StatusCode.INTERNAL, f"Embedding failed: {e}")

    async def EmbedBatch(self, request, context):
        from .generated import embedding_pb2

        texts = list(request.texts)
        if not texts:
            await context.abort(StatusCode.INVALID_ARGUMENT, "Texts list must not be empty")
        if len(texts) > self._max_batch_size:
            await context.abort(
                StatusCode.INVALID_ARGUMENT,
                f"Batch size {len(texts)} exceeds max {self._max_batch_size}",
            )

        logger.info("Embedding batch", count=len(texts))
        try:
            vectors = await self._provider.embed_batch(texts)
            embeddings = [
                embedding_pb2.EmbedResponse(vector=v, dimensions=self._provider.dimensions)
                for v in vectors
            ]
            return embedding_pb2.EmbedBatchResponse(embeddings=embeddings)
        except Exception as e:
            logger.error("Batch embedding failed", error=str(e))
            await context.abort(StatusCode.INTERNAL, f"Batch embedding failed: {e}")

    async def GetModelInfo(self, request, context):
        from .generated import embedding_pb2

        return embedding_pb2.ModelInfo(
            model_name=self._provider.model_name,
            provider=self._provider.provider_name,
            dimensions=self._provider.dimensions,
        )
