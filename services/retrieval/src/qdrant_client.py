"""Qdrant connection and collection management."""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

logger = structlog.get_logger()


class QdrantManager:
    """Manages Qdrant connection and collection lifecycle."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(host=self._host, port=self._port)
        logger.info("Connected to Qdrant", host=self._host, port=self._port)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    @property
    def client(self) -> AsyncQdrantClient:
        if not self._client:
            raise RuntimeError("QdrantManager not connected. Call connect() first.")
        return self._client

    async def create_collection(self, name: str, vector_size: int) -> bool:
        if await self._client.collection_exists(name):
            logger.info("Collection already exists", collection=name)
            return False

        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

        # Create payload indexes
        for field, schema in [
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("tags", PayloadSchemaType.KEYWORD),
            ("source_file", PayloadSchemaType.KEYWORD),
            ("ingested_at", PayloadSchemaType.DATETIME),
        ]:
            await self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
            )

        logger.info("Collection created", collection=name, vector_size=vector_size)
        return True

    async def delete_collection(self, name: str) -> bool:
        if not await self._client.collection_exists(name):
            logger.info("Collection does not exist", collection=name)
            return False

        await self._client.delete_collection(name)
        logger.info("Collection deleted", collection=name)
        return True

    async def is_healthy(self) -> tuple[bool, str]:
        try:
            collections = await self._client.get_collections()
            return True, f"{len(collections.collections)} collections"
        except Exception as e:
            return False, str(e)
