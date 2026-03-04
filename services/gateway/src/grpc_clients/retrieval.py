import json

import grpc
import structlog

from ..generated import retrieval_pb2, retrieval_pb2_grpc

logger = structlog.get_logger()


class RetrievalClient:
    """gRPC client for the Retrieval Service."""

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

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        mode: str = "dense",
        filters: list[dict] | None = None,
    ) -> dict:
        search_mode = (
            retrieval_pb2.HYBRID if mode == "hybrid" else retrieval_pb2.DENSE
        )

        grpc_filters = []
        for f in (filters or []):
            grpc_filters.append(
                retrieval_pb2.Filter(
                    field=f["field"],
                    operator=f["operator"],
                    value=json.dumps(f["value"]),
                )
            )

        response = await self._stub.Search(
            retrieval_pb2.SearchRequest(
                query=query,
                collection=collection,
                top_k=top_k,
                mode=search_mode,
                filters=grpc_filters,
            )
        )

        return {
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "score": r.score,
                    "metadata": dict(r.metadata),
                }
                for r in response.results
            ],
            "search_time_ms": response.search_time_ms,
        }

    async def health_check(self) -> dict:
        response = await self._stub.HealthCheck(retrieval_pb2.Empty())
        return {"healthy": response.healthy, "qdrant_status": response.qdrant_status}
