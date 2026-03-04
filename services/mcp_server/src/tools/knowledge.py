"""Knowledge base search tool — wraps the Retrieval Service via gRPC."""

import json

import grpc
import structlog

logger = structlog.get_logger()

# Lazy-loaded gRPC stubs
_channel = None
_stub = None


async def _get_stub(host: str):
    global _channel, _stub
    if _stub is None:
        # Import generated stubs
        import sys
        import os
        # Add parent to path for generated imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from generated import retrieval_pb2_grpc
        _channel = grpc.aio.insecure_channel(host)
        _stub = retrieval_pb2_grpc.RetrievalServiceStub(_channel)
    return _stub


async def knowledge_search(
    query: str,
    team: str = "platform",
    top_k: int = 5,
    *,
    retrieval_host: str,
) -> str:
    """Search the internal knowledge base for relevant documents."""
    from generated import retrieval_pb2

    try:
        stub = await _get_stub(retrieval_host)
        response = await stub.Search(
            retrieval_pb2.SearchRequest(
                query=query,
                collection=f"team_{team}",
                top_k=top_k,
                mode=retrieval_pb2.DENSE,
            )
        )

        results = []
        for r in response.results:
            results.append({
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "metadata": dict(r.metadata),
            })

        return json.dumps({
            "results": results,
            "total": len(results),
            "search_time_ms": response.search_time_ms,
        })
    except Exception as e:
        logger.error("Knowledge search failed", error=str(e))
        return json.dumps({"error": True, "code": "KB_SEARCH_FAILED", "message": str(e)})
