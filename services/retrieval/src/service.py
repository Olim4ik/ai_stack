"""Retrieval gRPC service implementation."""

import time

import structlog
from grpc import StatusCode

from .grpc_clients.embedding import EmbeddingClient
from .qdrant_client import QdrantManager
from .search.dense import dense_search
from .search.filters import build_qdrant_filter
from .search.hybrid import hybrid_search

logger = structlog.get_logger()


class RetrievalServicer:
    """gRPC servicer for the Retrieval Service."""

    def __init__(
        self,
        qdrant: QdrantManager,
        embedding_client: EmbeddingClient,
        default_top_k: int = 5,
        hybrid_dense_weight: float = 0.7,
    ) -> None:
        self._qdrant = qdrant
        self._embedding = embedding_client
        self._default_top_k = default_top_k
        self._hybrid_dense_weight = hybrid_dense_weight

    async def Search(self, request, context):
        from .generated import retrieval_pb2

        collection = request.collection
        if not collection:
            await context.abort(StatusCode.INVALID_ARGUMENT, "Collection name is required")

        query = request.query
        if not query.strip():
            await context.abort(StatusCode.INVALID_ARGUMENT, "Query must not be empty")

        top_k = request.top_k if request.top_k > 0 else self._default_top_k

        # Extract trace ID from gRPC metadata for correlation
        metadata = dict(context.invocation_metadata()) if context.invocation_metadata() else {}
        trace_id = metadata.get("x-trace-id", "")

        start = time.monotonic()

        # Embed the query
        query_vector = await self._embedding.embed(query)
        embed_ms = (time.monotonic() - start) * 1000

        # Build Qdrant filters from gRPC filters
        query_filter = build_qdrant_filter(list(request.filters))

        search_start = time.monotonic()

        # Execute search based on mode
        mode_str = "hybrid" if request.mode == retrieval_pb2.HYBRID else "dense"
        if request.mode == retrieval_pb2.HYBRID:
            results = await hybrid_search(
                client=self._qdrant.client,
                collection=collection,
                query_vector=query_vector,
                sparse_query_text=query,
                top_k=top_k,
                dense_weight=self._hybrid_dense_weight,
                query_filter=query_filter,
            )
        else:
            results = await dense_search(
                client=self._qdrant.client,
                collection=collection,
                query_vector=query_vector,
                top_k=top_k,
                query_filter=query_filter,
            )

        search_ms = (time.monotonic() - search_start) * 1000
        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Search completed",
            trace_id=trace_id,
            collection=collection,
            mode=mode_str,
            top_k=top_k,
            result_count=len(results),
            embed_ms=round(embed_ms, 1),
            search_ms=round(search_ms, 1),
            total_ms=round(elapsed_ms, 1),
        )

        search_results = [
            retrieval_pb2.SearchResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in results
        ]

        return retrieval_pb2.SearchResponse(
            results=search_results,
            search_time_ms=elapsed_ms,
        )

    async def CreateCollection(self, request, context):
        from .generated import retrieval_pb2

        name = request.name
        vector_size = request.vector_size
        if not name:
            await context.abort(StatusCode.INVALID_ARGUMENT, "Collection name is required")
        if vector_size <= 0:
            await context.abort(StatusCode.INVALID_ARGUMENT, "Vector size must be positive")

        success = await self._qdrant.create_collection(name, vector_size)
        return retrieval_pb2.CreateCollectionResponse(success=success)

    async def DeleteCollection(self, request, context):
        from .generated import retrieval_pb2

        name = request.name
        if not name:
            await context.abort(StatusCode.INVALID_ARGUMENT, "Collection name is required")

        success = await self._qdrant.delete_collection(name)
        return retrieval_pb2.DeleteCollectionResponse(success=success)

    async def HealthCheck(self, request, context):
        from .generated import retrieval_pb2

        healthy, status = await self._qdrant.is_healthy()
        return retrieval_pb2.HealthResponse(healthy=healthy, qdrant_status=status)
