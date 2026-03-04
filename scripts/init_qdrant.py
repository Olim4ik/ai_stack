"""Initialize Qdrant collections and payload indexes.

Run after Qdrant is healthy:
    python scripts/init_qdrant.py
"""

import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

DEFAULT_TEAMS = ["platform", "backend", "frontend", "devops"]


def init_collection(client: QdrantClient, name: str) -> None:
    collection_name = f"team_{name}"

    if client.collection_exists(collection_name):
        print(f"  Collection '{collection_name}' already exists, skipping.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )

    # Create payload indexes for filtered search
    for field, schema in [
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("tags", PayloadSchemaType.KEYWORD),
        ("source_file", PayloadSchemaType.KEYWORD),
        ("ingested_at", PayloadSchemaType.DATETIME),
    ]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema,
        )

    print(f"  Created collection '{collection_name}' with payload indexes.")


def main() -> None:
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Verify connectivity
    try:
        collections = client.get_collections()
        print(f"Connected. Existing collections: {len(collections.collections)}")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}", file=sys.stderr)
        sys.exit(1)

    print("Initializing default team collections...")
    for team in DEFAULT_TEAMS:
        init_collection(client, team)

    print("Done.")


if __name__ == "__main__":
    main()
