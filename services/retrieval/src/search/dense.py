"""Dense vector search against Qdrant."""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter

logger = structlog.get_logger()


async def dense_search(
    client: AsyncQdrantClient,
    collection: str,
    query_vector: list[float],
    top_k: int = 5,
    query_filter: Filter | None = None,
) -> list[dict]:
    """Perform dense vector similarity search in Qdrant.

    Returns a list of results with chunk_id, text, score, and metadata.
    """
    results = await client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=query_filter,
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

    logger.info("Dense search complete", collection=collection, results=len(search_results))
    return search_results
