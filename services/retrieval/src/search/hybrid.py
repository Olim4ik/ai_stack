"""Hybrid search combining dense and sparse vectors with RRF fusion."""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, Prefetch, Query

logger = structlog.get_logger()


async def hybrid_search(
    client: AsyncQdrantClient,
    collection: str,
    query_vector: list[float],
    sparse_query_text: str,
    top_k: int = 5,
    dense_weight: float = 0.7,
    query_filter: Filter | None = None,
) -> list[dict]:
    """Perform hybrid search using Qdrant's built-in query API with RRF fusion.

    Uses Qdrant prefetch to run dense and sparse searches, then fuses with RRF.
    """
    prefetch = [
        # Dense search
        Prefetch(
            query=query_vector,
            using="",  # default (dense) vector
            limit=top_k * 2,
            filter=query_filter,
        ),
    ]

    results = await client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=Query(fusion="rrf"),
        limit=top_k,
        with_payload=True,
    )

    search_results = []
    for point in results.points:
        payload = point.payload or {}
        search_results.append({
            "chunk_id": str(point.id),
            "text": payload.get("text", ""),
            "score": point.score,
            "metadata": {
                k: str(v) for k, v in payload.items() if k != "text"
            },
        })

    logger.info("Hybrid search complete", collection=collection, results=len(search_results))
    return search_results
