"""gRPC client for the Retrieval Service."""

import grpc
import structlog

from ..generated import retrieval_pb2, retrieval_pb2_grpc

logger = structlog.get_logger()


class RetrievalClient:
    def __init__(self, host: str) -> None:
        self._host = host
        self._channel: grpc.aio.Channel | None = None
        self._stub: retrieval_pb2_grpc.RetrievalServiceStub | None = None

    async def connect(self) -> None:
        self._channel = grpc.aio.insecure_channel(self._host)
        self._stub = retrieval_pb2_grpc.RetrievalServiceStub(self._channel)
        logger.info("Connected to Retrieval Service", host=self._host)

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()

    async def search(self, query: str, collection: str, top_k: int = 5) -> list[dict]:
        response = await self._stub.Search(
            retrieval_pb2.SearchRequest(
                query=query,
                collection=collection,
                top_k=top_k,
                mode=retrieval_pb2.DENSE,
            )
        )
        return [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "metadata": dict(r.metadata),
            }
            for r in response.results
        ]
