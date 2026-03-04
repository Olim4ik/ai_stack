"""Retrieve node — search the knowledge base via the Retrieval Service."""

import structlog

from ..grpc_clients.retrieval import RetrievalClient
from ..state import AgentState

logger = structlog.get_logger()


async def retrieve_node(state: AgentState, *, retrieval_client: RetrievalClient) -> dict:
    """Retrieve relevant document chunks from the knowledge base."""
    query = state["query"]
    team = state.get("team", "platform")
    collection = f"team_{team}"

    chunks = await retrieval_client.search(query=query, collection=collection, top_k=5)

    logger.info("Retrieved chunks", query=query[:50], count=len(chunks))

    return {
        "retrieved_chunks": chunks,
        "reasoning_trace": state.get("reasoning_trace", []) + [
            {"node": "retrieve", "action": f"Retrieved {len(chunks)} chunks from {collection}"}
        ],
    }
