"""Ingestion pipeline: parse → chunk → embed → store in Qdrant."""

import hashlib
import uuid
from datetime import datetime, timezone

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from ..grpc_clients.embedding import EmbeddingClient
from .chunker import chunk_text
from .parser import parse_file

logger = structlog.get_logger()


class IngestionPipeline:
    """Orchestrates document ingestion into the vector store."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedding_client: EmbeddingClient,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_batch_size: int = 32,
    ) -> None:
        self._qdrant = qdrant
        self._embedding = embedding_client
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_batch_size = max_batch_size

    async def ingest(
        self,
        filename: str,
        content: bytes,
        team: str,
        tags: list[str] | None = None,
    ) -> dict:
        """Ingest a document: parse, chunk, embed, store.

        Returns metadata about the ingested document.
        """
        tags = tags or []
        collection = f"team_{team}"
        doc_id = _make_doc_id(team, filename)

        # 1. Parse
        title, text = parse_file(filename, content)
        logger.info("Parsed document", filename=filename, title=title, length=len(text))

        # 2. Chunk
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise ValueError(f"No content to ingest from {filename}")

        # 3. Delete existing chunks for this doc (re-ingestion)
        await self._delete_existing(collection, doc_id)

        # 4. Embed in batches
        now = datetime.now(timezone.utc).isoformat()
        points = []

        for batch_start in range(0, len(chunks), self._max_batch_size):
            batch = chunks[batch_start : batch_start + self._max_batch_size]
            texts = [c["text"] for c in batch]
            vectors = await self._embedding.embed_batch(texts)

            for chunk, vector in zip(batch, vectors):
                point_id = str(uuid.uuid4())
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": chunk["text"],
                            "doc_id": doc_id,
                            "title": title,
                            "section": chunk["section"],
                            "chunk_index": chunk["chunk_index"],
                            "source_file": filename,
                            "tags": tags,
                            "ingested_at": now,
                        },
                    )
                )

        # 5. Upsert into Qdrant
        await self._qdrant.upsert(collection_name=collection, points=points)

        logger.info(
            "Document ingested",
            doc_id=doc_id,
            title=title,
            chunks=len(points),
            collection=collection,
        )

        return {
            "doc_id": doc_id,
            "title": title,
            "chunks_created": len(points),
            "source_file": filename,
            "team": team,
            "tags": tags,
            "ingested_at": now,
        }

    async def _delete_existing(self, collection: str, doc_id: str) -> None:
        """Delete all chunks for a given doc_id (for re-ingestion)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        await self._qdrant.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )


def _make_doc_id(team: str, filename: str) -> str:
    """Create a deterministic doc_id from team + filename."""
    return hashlib.sha256(f"{team}:{filename}".encode()).hexdigest()[:16]
