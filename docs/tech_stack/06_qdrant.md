# 06. Qdrant: Vector Database

## Table of Contents

1. [Qdrant Overview](#1-qdrant-overview)
2. [Collections](#2-collections)
3. [Points (Data Management)](#3-points-data-management)
4. [Payload & Filtering](#4-payload--filtering)
5. [HNSW Indexing](#5-hnsw-indexing)
6. [Quantization](#6-quantization)
7. [Search Operations](#7-search-operations)
8. [Sparse Vectors](#8-sparse-vectors)
9. [Snapshots & Backup](#9-snapshots--backup)
10. [Production Setup](#10-production-setup)
11. [Python Client (qdrant-client)](#11-python-client-qdrant-client)
12. [Q&A Section](#12-qa-section)

---

## 1. Qdrant Overview

### What is Qdrant?

Qdrant (pronounced "quadrant") is an open-source, high-performance vector similarity search engine and
vector database written in **Rust**. It provides a production-ready service with a convenient API for
storing, searching, and managing vectors with additional payloads (metadata).

```
+------------------------------------------------------------------+
|                        Qdrant Architecture                       |
+------------------------------------------------------------------+
|                                                                  |
|  Client (Python/REST/gRPC)                                       |
|       |                                                          |
|       v                                                          |
|  +------------------+                                            |
|  |   API Gateway    |  <-- REST (port 6333) / gRPC (port 6334)  |
|  +------------------+                                            |
|       |                                                          |
|       v                                                          |
|  +------------------+     +------------------+                   |
|  | Collection Mgr   |---->|   Shard Manager  |                   |
|  +------------------+     +------------------+                   |
|       |                        |          |                      |
|       v                        v          v                      |
|  +-----------+          +---------+  +---------+                 |
|  | HNSW Index|          | Shard 0 |  | Shard 1 |                |
|  +-----------+          +---------+  +---------+                 |
|       |                      |            |                      |
|       v                      v            v                      |
|  +------------------+  +----------+  +----------+                |
|  |  Payload Index   |  | Segments |  | Segments |               |
|  +------------------+  +----------+  +----------+                |
|       |                      |            |                      |
|       v                      v            v                      |
|  +------------------+  +----------+  +----------+                |
|  | Write-Ahead Log  |  | Storage  |  | Storage  |               |
|  |     (WAL)        |  | (mmap)   |  | (mmap)   |               |
|  +------------------+  +----------+  +----------+                |
|                                                                  |
+------------------------------------------------------------------+
```

### Core Features

| Feature              | Description                                          |
|----------------------|------------------------------------------------------|
| Written in Rust      | Memory-safe, high performance, no GC pauses          |
| HNSW Index           | State-of-the-art approximate nearest neighbor search  |
| Payload Filtering    | Rich filtering on metadata during vector search       |
| Quantization         | Scalar, product, and binary quantization support      |
| Distributed Mode     | Horizontal scaling with sharding and replication      |
| Multiple Vectors     | Store multiple named vectors per point                |
| Sparse Vectors       | Native sparse vector support for hybrid search        |
| Snapshots            | Point-in-time backup and restore                      |
| WAL                  | Write-Ahead Log for crash recovery                    |
| REST + gRPC          | Dual API interface                                    |

### Deployment Options

```
+---------------------+---------------------+---------------------+
|      Local/Dev      |       Docker        |    Qdrant Cloud     |
+---------------------+---------------------+---------------------+
|                     |                     |                     |
|  pip install        |  docker run -p      |  Managed service    |
|    qdrant-client    |    6333:6333        |  on cloud infra     |
|                     |    qdrant/qdrant    |                     |
|  In-memory mode     |  Persistent volume  |  Auto-scaling       |
|  (for prototyping)  |  support            |  High availability  |
|                     |                     |  Built-in backups   |
|  No server needed   |  Single node or     |  Monitoring/metrics |
|  (":memory:" mode)  |  docker-compose     |  No ops overhead    |
|                     |  for clusters       |                     |
+---------------------+---------------------+---------------------+
```

```python
from qdrant_client import QdrantClient

# Option 1: In-memory (prototyping, tests)
client = QdrantClient(":memory:")

# Option 2: Local server
client = QdrantClient(host="localhost", port=6333)

# Option 3: With URL
client = QdrantClient(url="http://localhost:6333")

# Option 4: Qdrant Cloud
client = QdrantClient(
    url="https://your-cluster-id.aws.cloud.qdrant.io",
    api_key="your-api-key"
)

# Option 5: Local persistent storage (no server, embedded mode)
client = QdrantClient(path="/path/to/local/storage")
```

### Comparison with Other Vector Databases

```
+---------------+--------+---------+----------+----------+----------+
| Feature       | Qdrant | Pinecone| Milvus   | Weaviate | ChromaDB |
+---------------+--------+---------+----------+----------+----------+
| Language      | Rust   | N/A*    | Go/C++   | Go       | Python   |
| Open Source   | Yes    | No      | Yes      | Yes      | Yes      |
| Self-hosted   | Yes    | No      | Yes      | Yes      | Yes      |
| Managed Cloud | Yes    | Yes     | Yes      | Yes      | Yes      |
| Filtering     | Rich   | Basic   | Rich     | GraphQL  | Basic    |
| Sparse Vecs   | Yes    | Yes     | Yes      | No       | No       |
| Multi-vector  | Yes    | No      | Yes      | No       | No       |
| Quantization  | S/P/B  | N/A     | S/P      | P/B      | No       |
| gRPC          | Yes    | Yes     | Yes      | Yes      | No       |
| Distributed   | Yes    | Yes**   | Yes      | Yes      | No       |
| Performance   | High   | High    | High     | Medium   | Low      |
| Best For      | Prod   | NoOps   | Large    | Graph    | Proto-   |
|               | AI     | cloud   | scale    | + vector | typing   |
+---------------+--------+---------+----------+----------+----------+
* Pinecone is proprietary        S=Scalar, P=Product, B=Binary
** Pinecone handles distribution automatically (serverless)
```

**When to Choose Qdrant:**
- You need a self-hosted, high-performance vector DB
- Rich payload filtering is a core requirement
- You need multiple vectors per point (e.g., title + content embeddings)
- Hybrid search (dense + sparse) is required
- You want Rust-level performance with crash safety
- Memory efficiency matters (quantization support)

---

## 2. Collections

A **collection** is the primary organizational unit in Qdrant. It holds points (vectors + payloads)
and defines how vectors are stored, indexed, and compared.

### Creating Collections

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    WalConfigDiff,
)

client = QdrantClient(host="localhost", port=6333)

# --- Basic collection ---
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(
        size=384,               # Dimension of vectors (must match your embedding model)
        distance=Distance.COSINE
    ),
)

# --- Collection with full configuration ---
client.create_collection(
    collection_name="production_collection",
    vectors_config=VectorParams(
        size=1536,                          # OpenAI text-embedding-3-small dimension
        distance=Distance.COSINE,
        on_disk=True,                       # Store vectors on disk (saves RAM)
    ),
    hnsw_config=HnswConfigDiff(
        m=16,                               # Number of edges per node
        ef_construct=100,                   # Index build quality
        full_scan_threshold=10000,          # Below this count, skip index
        on_disk=True,                       # Store HNSW graph on disk
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000,           # Min points before indexing
        memmap_threshold=50000,             # Switch to mmap after this many points
    ),
    wal_config=WalConfigDiff(
        wal_capacity_mb=64,                 # WAL size in MB
        wal_segments_ahead=0,
    ),
    on_disk_payload=True,                   # Store payloads on disk
    shard_number=2,                         # Number of shards
    replication_factor=1,                   # Replication factor
)
```

### Distance Metrics

```
+------------------------------------------------------------------+
|                      Distance Metrics                            |
+------------------------------------------------------------------+
|                                                                  |
|  COSINE SIMILARITY                                               |
|  ==================                                              |
|  Measures angle between vectors (direction, not magnitude)       |
|                                                                  |
|        B                                                         |
|       /                                                          |
|      / ) theta                                                   |
|     /  )                                                         |
|    +---------> A        cos(theta) = (A . B) / (|A| * |B|)      |
|                                                                  |
|  Range: [-1, 1]  (Qdrant stores as 1 - cosine, so [0, 2])      |
|  Best for: NLP embeddings, semantic similarity                   |
|  Use when: Vector magnitude is irrelevant                        |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  DOT PRODUCT                                                     |
|  ============                                                    |
|  Raw dot product of two vectors                                  |
|                                                                  |
|  dot(A, B) = sum(a_i * b_i)                                     |
|                                                                  |
|  Range: (-inf, +inf)                                             |
|  Best for: Recommendation systems, when magnitude matters        |
|  Use when: Vectors are NOT normalized, and magnitude encodes     |
|            importance or relevance                                |
|  Note: Qdrant returns negative dot product as distance           |
|        (smaller = more similar, to keep consistent with others)  |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  EUCLIDEAN DISTANCE                                              |
|  ==================                                              |
|  Straight-line distance in vector space                          |
|                                                                  |
|    B *                                                           |
|      |  \                                                        |
|      |    \  d = sqrt(sum((a_i - b_i)^2))                        |
|      |      \                                                    |
|      +--------* A                                                |
|                                                                  |
|  Range: [0, +inf)                                                |
|  Best for: Image embeddings, spatial data                        |
|  Use when: Absolute positions matter, not just direction         |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  MANHATTAN DISTANCE                                              |
|  ==================                                              |
|  Sum of absolute differences (L1 norm)                           |
|                                                                  |
|    B *--+                                                        |
|         |                                                        |
|         +--* A        d = sum(|a_i - b_i|)                       |
|                                                                  |
|  Range: [0, +inf)                                                |
|  Best for: High-dimensional sparse data, certain ML features     |
|                                                                  |
+------------------------------------------------------------------+
```

**Decision Guide:**

| Scenario                                    | Recommended Metric |
|---------------------------------------------|--------------------|
| OpenAI / Sentence-Transformers embeddings   | Cosine             |
| Pre-normalized embeddings                   | Dot Product        |
| Image feature vectors                       | Euclidean          |
| Recommendation with relevance scores        | Dot Product        |
| General-purpose text search                 | Cosine             |
| Geospatial or coordinate-like data          | Euclidean          |

### Multiple Named Vectors per Point

Qdrant supports storing multiple vectors per point, each with its own configuration. This is
useful when you have different embedding models for different aspects of the same data (e.g.,
title vs. body, image vs. text).

```python
from qdrant_client.models import VectorParams, Distance

# Create collection with multiple named vectors
client.create_collection(
    collection_name="multi_vector_collection",
    vectors_config={
        "title": VectorParams(size=384, distance=Distance.COSINE),
        "content": VectorParams(size=1536, distance=Distance.COSINE),
        "image": VectorParams(size=512, distance=Distance.EUCLIDEAN),
    },
)
```

### Collection Management

```python
# List all collections
collections = client.get_collections()
print(collections)

# Get collection info
info = client.get_collection("my_collection")
print(f"Points count: {info.points_count}")
print(f"Vectors count: {info.vectors_count}")
print(f"Status: {info.status}")
print(f"Segments: {info.segments_count}")

# Check if collection exists
exists = client.collection_exists("my_collection")

# Update collection parameters (partial update)
client.update_collection(
    collection_name="my_collection",
    hnsw_config=HnswConfigDiff(ef_construct=200),
    optimizers_config=OptimizersConfigDiff(indexing_threshold=50000),
)

# Delete collection
client.delete_collection("my_collection")

# Rename collection (via alias)
client.update_collection_aliases(
    change_aliases_operations=[
        {"create_alias": {"alias_name": "prod_embeddings", "collection_name": "my_collection"}}
    ]
)
```

---

## 3. Points (Data Management)

A **point** is the fundamental data unit in Qdrant. Each point consists of:
- **ID**: Unique identifier (integer or UUID string)
- **Vector(s)**: One or more embedding vectors
- **Payload**: Optional JSON metadata

```
+------------------------------------------------------------------+
|                          Point Structure                          |
+------------------------------------------------------------------+
|                                                                  |
|  Point {                                                         |
|    id: 42,                         // Integer or UUID            |
|    vector: [0.1, 0.23, -0.5, ...], // Dense vector (float32)   |
|    payload: {                      // Arbitrary JSON metadata    |
|      "title": "Intro to Qdrant",                                |
|      "category": "database",                                    |
|      "year": 2024,                                               |
|      "tags": ["vector", "search"],                               |
|      "metadata": {                                               |
|        "author": "Jane Doe",                                     |
|        "rating": 4.8                                             |
|      }                                                           |
|    }                                                             |
|  }                                                               |
|                                                                  |
+------------------------------------------------------------------+
```

### Upserting Points

```python
from qdrant_client.models import PointStruct
import uuid

# --- Upsert single point with integer ID ---
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, ...],  # Must match collection vector size
            payload={"title": "First document", "category": "tech"},
        )
    ],
)

# --- Upsert with UUID ---
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=str(uuid.uuid4()),  # e.g., "550e8400-e29b-41d4-a716-446655440000"
            vector=[0.4, 0.5, 0.6, ...],
            payload={"title": "Second document"},
        )
    ],
)

# --- Batch upsert ---
points = [
    PointStruct(id=i, vector=embedding, payload={"text": text, "index": i})
    for i, (embedding, text) in enumerate(zip(embeddings_list, texts_list))
]

# Upsert in batches (recommended for large datasets)
BATCH_SIZE = 100
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i : i + BATCH_SIZE]
    client.upsert(collection_name="my_collection", points=batch)
    print(f"Upserted batch {i // BATCH_SIZE + 1}")

# --- Upsert with multiple named vectors ---
client.upsert(
    collection_name="multi_vector_collection",
    points=[
        PointStruct(
            id=1,
            vector={
                "title": [0.1, 0.2, ...],     # 384-dim
                "content": [0.3, 0.4, ...],   # 1536-dim
                "image": [0.5, 0.6, ...],     # 512-dim
            },
            payload={"title": "AI Paper", "source": "arxiv"},
        )
    ],
)
```

### Point IDs: Integer vs UUID

```
+---------------------------+---------------------------+
|       Integer IDs         |        UUID IDs           |
+---------------------------+---------------------------+
| id: 1, 2, 3, ...         | id: "550e8400-e29b..."   |
|                           |                           |
| Pros:                     | Pros:                     |
| - Compact storage         | - Globally unique         |
| - Sequential, easy to     | - No collision across     |
|   manage                  |   distributed systems     |
| - Faster comparison       | - Can be generated        |
|                           |   independently           |
| Cons:                     |                           |
| - Needs coordination to   | Cons:                     |
|   avoid duplicates        | - More storage space      |
| - Not safe for            | - Slightly slower         |
|   distributed generation  |   comparison              |
+---------------------------+---------------------------+

Recommendation:
- Use integers for simple sequential data
- Use UUIDs for distributed systems or when IDs come from external sources
```

### Updating Payloads

```python
from qdrant_client.models import PointIdsList

# Set payload (merges with existing payload)
client.set_payload(
    collection_name="my_collection",
    payload={"category": "updated_category", "reviewed": True},
    points=[1, 2, 3],  # Point IDs to update
)

# Overwrite payload (replaces entire payload)
client.overwrite_payload(
    collection_name="my_collection",
    payload={"new_field": "only this remains"},
    points=[1],
)

# Delete specific payload keys
client.delete_payload(
    collection_name="my_collection",
    keys=["category", "reviewed"],
    points=[1, 2, 3],
)
```

### Deleting Points

```python
# Delete by IDs
client.delete(
    collection_name="my_collection",
    points_selector=PointIdsList(points=[1, 2, 3]),
)

# Delete by filter (delete all points matching a condition)
from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue

client.delete(
    collection_name="my_collection",
    points_selector=FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value="deprecated"))
            ]
        )
    ),
)
```

### Scrolling Through Points

```python
# Scroll through all points (paginated retrieval)
offset = None
all_points = []

while True:
    results, next_offset = client.scroll(
        collection_name="my_collection",
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,  # Set True if you need vectors
    )
    all_points.extend(results)

    if next_offset is None:
        break
    offset = next_offset

print(f"Total points retrieved: {len(all_points)}")

# Scroll with filter
results, _ = client.scroll(
    collection_name="my_collection",
    scroll_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
    ),
    limit=50,
)
```

### Retrieve Points by ID

```python
# Get specific points
points = client.retrieve(
    collection_name="my_collection",
    ids=[1, 2, 3],
    with_payload=True,
    with_vectors=True,
)

for point in points:
    print(f"ID: {point.id}, Payload: {point.payload}")
```

---

## 4. Payload & Filtering

### Payload Types

Qdrant supports rich payload types. Payloads are stored as JSON and can be indexed for
high-performance filtering during search.

```
+------------------------------------------------------------------+
|                       Payload Types                              |
+------------------------------------------------------------------+
|                                                                  |
|  Type       | Example                  | Index Type              |
|  -----------+--------------------------+------------------------ |
|  keyword    | "science"                | Exact match index       |
|  integer    | 2024                     | Range index             |
|  float      | 4.5                      | Range index             |
|  bool       | true                     | Exact match index       |
|  text       | "long text content..."   | Full-text index         |
|  geo        | {"lat": 40.7, "lon":-74} | Geo index               |
|  datetime   | "2024-01-15T10:30:00Z"   | Range index             |
|  uuid       | "550e8400-e29b-..."      | UUID index              |
|  keyword[]  | ["ai", "ml", "nlp"]      | Exact match (any elem)  |
|  nested     | {"meta": {"k": "v"}}     | Nested key access       |
|                                                                  |
+------------------------------------------------------------------+
```

### Payload Indexes

Creating payload indexes drastically improves filter performance. Without an index, Qdrant
performs a full scan on payloads during filtered search.

```python
from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType

# Keyword index (for exact match)
client.create_payload_index(
    collection_name="my_collection",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD,
)

# Integer index (for range queries)
client.create_payload_index(
    collection_name="my_collection",
    field_name="year",
    field_schema=PayloadSchemaType.INTEGER,
)

# Float index
client.create_payload_index(
    collection_name="my_collection",
    field_name="rating",
    field_schema=PayloadSchemaType.FLOAT,
)

# Geo index
client.create_payload_index(
    collection_name="my_collection",
    field_name="location",
    field_schema=PayloadSchemaType.GEO,
)

# Full-text index
client.create_payload_index(
    collection_name="my_collection",
    field_name="description",
    field_schema=TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WORD,
        min_token_len=2,
        max_token_len=20,
        lowercase=True,
    ),
)

# Datetime index
client.create_payload_index(
    collection_name="my_collection",
    field_name="created_at",
    field_schema=PayloadSchemaType.DATETIME,
)
```

### Filter Conditions

```
+------------------------------------------------------------------+
|                    Filter Logic Overview                          |
+------------------------------------------------------------------+
|                                                                  |
|  Filter {                                                        |
|    must: [        ]  <-- AND logic: ALL conditions must match     |
|    should: [      ]  <-- OR logic: AT LEAST ONE must match       |
|    must_not: [    ]  <-- NOT logic: NONE of these can match      |
|  }                                                               |
|                                                                  |
|  Example:                                                        |
|  Find points where:                                              |
|    category IS "science"                                         |
|    AND year >= 2020                                              |
|    AND tags contain "ai" OR "ml"                                 |
|    AND author is NOT "unknown"                                   |
|                                                                  |
|  Filter {                                                        |
|    must: [                                                       |
|      category == "science",                                      |
|      year >= 2020                                                |
|    ],                                                            |
|    should: [                                                     |
|      tags contains "ai",                                         |
|      tags contains "ml"                                          |
|    ],                                                            |
|    must_not: [                                                   |
|      author == "unknown"                                         |
|    ]                                                             |
|  }                                                               |
|                                                                  |
+------------------------------------------------------------------+
```

### Filter Code Examples

```python
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    MatchExcept,
    MatchText,
    Range,
    GeoBoundingBox,
    GeoPoint,
    GeoRadius,
    ValuesCount,
    IsEmptyCondition,
    IsNullCondition,
    HasIdCondition,
    NestedCondition,
    DatetimeRange,
)

# --- Match (exact value) ---
filter_exact = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="science"))
    ]
)

# --- Match Any (value in list) ---
filter_any = Filter(
    must=[
        FieldCondition(key="category", match=MatchAny(any=["science", "tech", "math"]))
    ]
)

# --- Match Except (value NOT in list) ---
filter_except = Filter(
    must=[
        FieldCondition(key="status", match=MatchExcept(except_=["deleted", "archived"]))
    ]
)

# --- Full-text Match (requires text index) ---
filter_text = Filter(
    must=[
        FieldCondition(key="description", match=MatchText(text="neural network"))
    ]
)

# --- Range (numeric) ---
filter_range = Filter(
    must=[
        FieldCondition(
            key="year",
            range=Range(gte=2020, lte=2025)  # gte, gt, lte, lt
        )
    ]
)

# --- Datetime Range ---
filter_datetime = Filter(
    must=[
        FieldCondition(
            key="created_at",
            range=DatetimeRange(
                gte="2024-01-01T00:00:00Z",
                lt="2025-01-01T00:00:00Z",
            ),
        )
    ]
)

# --- Geo: Bounding Box ---
filter_geo_bbox = Filter(
    must=[
        FieldCondition(
            key="location",
            geo_bounding_box=GeoBoundingBox(
                top_left=GeoPoint(lat=41.0, lon=-74.5),
                bottom_right=GeoPoint(lat=40.5, lon=-73.5),
            ),
        )
    ]
)

# --- Geo: Radius ---
filter_geo_radius = Filter(
    must=[
        FieldCondition(
            key="location",
            geo_radius=GeoRadius(
                center=GeoPoint(lat=40.7128, lon=-74.0060),
                radius=5000.0,  # meters
            ),
        )
    ]
)

# --- Values Count (for array fields) ---
filter_values_count = Filter(
    must=[
        FieldCondition(
            key="tags",
            values_count=ValuesCount(gte=2, lte=10)  # tags array has 2-10 elements
        )
    ]
)

# --- Is Empty (field has no value or does not exist) ---
filter_empty = Filter(
    must_not=[
        IsEmptyCondition(is_empty={"key": "description"})  # exclude empty descriptions
    ]
)

# --- Is Null ---
filter_null = Filter(
    must_not=[
        IsNullCondition(is_null={"key": "rating"})  # exclude null ratings
    ]
)

# --- Has ID (filter by point IDs) ---
filter_ids = Filter(
    must=[
        HasIdCondition(has_id=[1, 5, 10, 42])
    ]
)

# --- Nested Filter (for nested objects) ---
filter_nested = Filter(
    must=[
        NestedCondition(
            nested={
                "key": "metadata",
                "filter": Filter(
                    must=[
                        FieldCondition(
                            key="metadata.author",
                            match=MatchValue(value="John"),
                        )
                    ]
                ),
            }
        )
    ]
)

# --- Complex Combined Filter ---
complex_filter = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="science")),
        FieldCondition(key="year", range=Range(gte=2020)),
    ],
    should=[
        FieldCondition(key="tags", match=MatchValue(value="ai")),
        FieldCondition(key="tags", match=MatchValue(value="ml")),
    ],
    must_not=[
        FieldCondition(key="status", match=MatchValue(value="draft")),
    ],
)

# Use filter in search
results = client.query_points(
    collection_name="my_collection",
    query=[0.1, 0.2, 0.3, ...],
    query_filter=complex_filter,
    limit=10,
)
```

### Visual Example: How Filtering Works

```
+------------------------------------------------------------------+
|              Filtered Vector Search Pipeline                     |
+------------------------------------------------------------------+
|                                                                  |
|  Query Vector: [0.1, 0.2, 0.3, ...]                             |
|  Filter: must=[category=="science"], must_not=[year < 2020]     |
|                                                                  |
|  Step 1: Apply Payload Filter (pre-filter)                       |
|  +----+----+----+----+----+----+----+----+----+----+             |
|  | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |P10|           |
|  +----+----+----+----+----+----+----+----+----+----+             |
|    sci  tech sci  sci  art  sci  sci  tech sci  sci              |
|   2024  2023 2019 2021 2024 2018 2024 2024 2022 2020            |
|    v     x    x    v    x    x    v    x    v    v              |
|                                                                  |
|  Candidates after filter: P1, P4, P7, P9, P10                   |
|                                                                  |
|  Step 2: Vector Similarity Search (HNSW on filtered subset)      |
|  P7  ---> distance: 0.12  (closest)                              |
|  P1  ---> distance: 0.25                                         |
|  P9  ---> distance: 0.31                                         |
|  P10 ---> distance: 0.45                                         |
|  P4  ---> distance: 0.58                                         |
|                                                                  |
|  Result: [P7, P1, P9, P10, P4] (ordered by similarity)          |
|                                                                  |
+------------------------------------------------------------------+
```

> **Note:** Qdrant uses a smart strategy for filtering. When the filter matches a large
> portion of points, it applies the filter during HNSW traversal. When the filter is very
> selective (few matches), it first identifies matching points and then compares vectors.
> This hybrid approach ensures optimal performance in both cases.

---

## 5. HNSW Indexing

### How HNSW Works

HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest
neighbor (ANN) search. It builds a multi-layer graph where:

- **Layer 0** contains ALL points
- Each higher layer contains a random subset of points from the layer below
- Higher layers have fewer points, enabling long-distance jumps
- Lower layers have more points, enabling fine-grained search

```
+------------------------------------------------------------------+
|                    HNSW Graph Structure                           |
+------------------------------------------------------------------+
|                                                                  |
|  Layer 3:  A ──────────────────────────── H                      |
|            (very sparse: ~2 nodes, fast long-range jumps)        |
|                                                                  |
|  Layer 2:  A ──────────── D ──────────── H                      |
|            (sparse: ~4 nodes)                                    |
|                                                                  |
|  Layer 1:  A ──── C ──── D ──── F ──── H ──── J                |
|            (moderate: ~8 nodes)                                  |
|                                                                  |
|  Layer 0:  A ─ B ─ C ─ D ─ E ─ F ─ G ─ H ─ I ─ J ─ K ─ L     |
|            (dense: ALL nodes, fine-grained connections)           |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Search Process (finding nearest neighbor to query Q):           |
|                                                                  |
|  1. Start at entry point on top layer (A on Layer 3)             |
|  2. Greedily move to closest neighbor on same layer              |
|     Layer 3: A -> H  (H is closer to Q)                         |
|  3. Drop down to Layer 2 at node H                               |
|     Layer 2: H -> D  (D is closer to Q)                         |
|  4. Drop down to Layer 1 at node D                               |
|     Layer 1: D -> F  (F is closer to Q)                         |
|  5. Drop down to Layer 0 at node F                               |
|     Layer 0: F -> E -> E is closest!                             |
|                                                                  |
|  Result: E (with ef=1). With higher ef, explore more candidates. |
|                                                                  |
+------------------------------------------------------------------+
```

### HNSW Parameters

```
+------------------------------------------------------------------+
|                    HNSW Parameters                                |
+------------------------------------------------------------------+
|                                                                  |
|  Parameter        | Default | Description                       |
|  -----------------+---------+----------------------------------- |
|  m                | 16      | Max connections per node per layer |
|  ef_construct     | 100     | Beam width during index building   |
|  ef               | (auto)  | Beam width during search           |
|  full_scan_thresh | 10000   | Points below this -> brute force   |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  m (Connections per Node)                                        |
|  ========================                                        |
|                                                                  |
|  Low m (4-8):            High m (32-64):                         |
|   A --- B                 A --- B                                |
|   |                       |\ /| \                                |
|   C                       C--D--E--F                             |
|                            \ |X| /                               |
|  - Less memory              G--H                                 |
|  - Faster build           - More memory                          |
|  - Lower recall           - Slower build                         |
|  - Good for low dims      - Higher recall                        |
|                           - Good for high dims                   |
|                                                                  |
|  Recommended: m=16 for most use cases                            |
|               m=32-64 for high-dimensional (>1000) vectors       |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  ef_construct (Build Quality)                                    |
|  ============================                                    |
|                                                                  |
|  Low (50):    Faster build, lower quality graph                  |
|  Medium (100): Good balance (DEFAULT)                            |
|  High (200+): Slower build, higher quality graph                 |
|                                                                  |
|  Rule of thumb: ef_construct >= 2 * m                            |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  ef (Search Quality / Speed)                                     |
|  ============================                                    |
|                                                                  |
|  Controls how many candidates HNSW explores during search.       |
|  Must be >= limit (number of results requested).                 |
|                                                                  |
|  ef=50:  Fast search, possibly lower recall                      |
|  ef=128: Balanced (good default for most queries)                |
|  ef=256: Higher recall, slower search                            |
|  ef=512: Near-perfect recall, slowest                            |
|                                                                  |
|  Set per-query via search params:                                |
|    search_params={"hnsw_ef": 128}                                |
|                                                                  |
+------------------------------------------------------------------+
```

### Trade-offs

```
+------------------------------------------------------------------+
|             HNSW Parameter Trade-offs                             |
+------------------------------------------------------------------+
|                                                                  |
|            Recall (Accuracy)                                     |
|              ^                                                   |
|         100% |          * (m=64, ef=512)                         |
|              |      *                                            |
|          95% |    *     <-- Sweet spot (m=16, ef=128)            |
|              |  *                                                 |
|          90% | *                                                  |
|              |*                                                   |
|          80% *                                                   |
|              +----+----+----+----+-----> Search Latency          |
|              1ms  2ms  5ms  10ms 20ms                            |
|                                                                  |
|                                                                  |
|            Memory Usage                                          |
|              ^                                                   |
|         High |              * m=64                               |
|              |          *                                        |
|              |      * m=16                                       |
|              |  *                                                 |
|         Low  * m=4                                               |
|              +----+----+----+----+-----> Recall                  |
|              80%  85%  90%  95%  100%                            |
|                                                                  |
+------------------------------------------------------------------+
```

### Configuration Recommendations

| Use Case                     | m    | ef_construct | ef (search) | Notes                    |
|------------------------------|------|-------------|-------------|--------------------------|
| Prototyping / small data     | 16   | 100         | 50          | Default, fast build      |
| Production / balanced        | 16   | 128         | 128         | Good balance             |
| High recall critical         | 32   | 200         | 256         | More memory, slower      |
| Very high-dimensional (>1K)  | 48   | 200         | 256         | Higher m helps           |
| Memory constrained           | 8    | 64          | 64          | Reduce m, accept recall  |
| Maximum throughput           | 16   | 100         | 50          | Lower ef = faster search |

```python
from qdrant_client.models import HnswConfigDiff, SearchParams

# Set HNSW config at collection level
client.update_collection(
    collection_name="my_collection",
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=128,
        on_disk=False,        # Keep HNSW graph in RAM for speed
    ),
)

# Set ef per query (overrides collection default)
results = client.query_points(
    collection_name="my_collection",
    query=[0.1, 0.2, ...],
    search_params=SearchParams(hnsw_ef=256),  # Higher ef for this query
    limit=10,
)
```

### On-Disk HNSW Index

For large collections that exceed RAM, the HNSW graph can be stored on disk:

```python
client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
        on_disk=True,          # Vectors on disk
    ),
    hnsw_config=HnswConfigDiff(
        on_disk=True,          # HNSW graph on disk
    ),
    on_disk_payload=True,      # Payloads on disk
)
```

```
+------------------------------------------------------------------+
|           Memory Layout: In-RAM vs On-Disk                       |
+------------------------------------------------------------------+
|                                                                  |
|  Full In-RAM:                 On-Disk (mmap):                    |
|  +------------------+        +------------------+                |
|  |     RAM          |        |     RAM          |                |
|  | +----------+     |        | +----------+     |                |
|  | |HNSW Graph|     |        | | Hot pages |     |               |
|  | +----------+     |        | | (cached)  |     |               |
|  | +----------+     |        | +----------+     |                |
|  | | Vectors  |     |        +------------------+                |
|  | +----------+     |                |                           |
|  | +----------+     |                v                           |
|  | | Payloads |     |        +------------------+                |
|  | +----------+     |        |     Disk         |                |
|  +------------------+        | +----------+     |                |
|                              | |HNSW Graph|     |                |
|  Fast but needs lots         | +----------+     |                |
|  of RAM                      | | Vectors  |     |                |
|                              | +----------+     |                |
|                              | | Payloads |     |                |
|                              | +----------+     |                |
|                              +------------------+                |
|                                                                  |
|                              Slower but handles                  |
|                              much larger datasets                |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 6. Quantization

Quantization reduces the memory footprint of vectors by representing them with fewer bits.
This trades a small amount of accuracy for significant memory savings and potentially faster
search.

### Types of Quantization

```
+------------------------------------------------------------------+
|                    Quantization Methods                           |
+------------------------------------------------------------------+
|                                                                  |
|  ORIGINAL (float32):                                             |
|  [0.1234, 0.5678, 0.9012, 0.3456]                               |
|  4 bytes per dimension = 128 bytes for 32 dims                   |
|                                                                  |
|  ================================================================|
|                                                                  |
|  SCALAR QUANTIZATION (float32 -> uint8):                         |
|  [31, 145, 230, 88]                                              |
|  1 byte per dimension = 32 bytes for 32 dims                     |
|  4x compression, ~1-2% recall loss                               |
|                                                                  |
|  How it works:                                                   |
|  min_val = min(all values across all vectors)                    |
|  max_val = max(all values across all vectors)                    |
|  quantized = round((original - min_val) / (max_val - min_val)   |
|              * 255)                                              |
|                                                                  |
|  ================================================================|
|                                                                  |
|  PRODUCT QUANTIZATION (PQ):                                      |
|  Vector split into sub-vectors, each mapped to codebook entry    |
|  [0.12, 0.56] [0.90, 0.34] -> [codebook_id_1, codebook_id_2]   |
|  Much higher compression (8-64x), more recall loss               |
|                                                                  |
|  Original:    [0.12, 0.56, 0.90, 0.34, 0.78, 0.23, 0.45, 0.67] |
|  Sub-vectors: [0.12, 0.56] [0.90, 0.34] [0.78, 0.23] [0.45,.67]|
|  PQ codes:    [   42     ] [   17     ] [   238    ] [   91    ] |
|  4 bytes total for 8 dimensions (vs 32 bytes original)           |
|                                                                  |
|  ================================================================|
|                                                                  |
|  BINARY QUANTIZATION:                                            |
|  [0.1234, -0.5678, 0.9012, -0.3456] -> [1, 0, 1, 0]            |
|  1 bit per dimension = 4 bytes for 32 dims                       |
|  32x compression, uses Hamming distance                          |
|  Best for high-dimensional, normalized vectors (e.g., OpenAI)    |
|                                                                  |
|  How it works:                                                   |
|  value > 0 -> 1                                                  |
|  value <= 0 -> 0                                                 |
|                                                                  |
+------------------------------------------------------------------+
```

### Compression Comparison

```
+------------------------------------------------------------------+
|     Memory Usage Comparison (1M vectors, 1536 dims)              |
+------------------------------------------------------------------+
|                                                                  |
|  Method          | Per Vector  | 1M Vectors | Compression | Recall|
|  ----------------+-------------+------------+-------------+------|
|  No quantization | 6,144 bytes | ~5.7 GB    | 1x          | 100% |
|  Scalar (uint8)  | 1,536 bytes | ~1.4 GB    | 4x          | ~99% |
|  Product (PQ)    | ~192 bytes  | ~183 MB    | ~32x        | ~95% |
|  Binary          | 192 bytes   | ~183 MB    | 32x         | ~92%*|
|                                                                  |
|  * Binary quantization recall depends heavily on the embedding   |
|    model. Works best with OpenAI, Cohere embeddings.             |
|                                                                  |
+------------------------------------------------------------------+
```

### Configuration Examples

```python
from qdrant_client.models import (
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    ProductQuantization,
    ProductQuantizationConfig,
    CompressionRatio,
    BinaryQuantization,
    BinaryQuantizationConfig,
    QuantizationSearchParams,
    SearchParams,
)

# --- Scalar Quantization ---
client.create_collection(
    collection_name="scalar_quantized",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,       # Clip outliers (use 99th percentile for range)
            always_ram=True,     # Keep quantized vectors in RAM even if on_disk=True
        ),
    ),
)

# --- Product Quantization ---
client.create_collection(
    collection_name="product_quantized",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ProductQuantization(
        product=ProductQuantizationConfig(
            compression=CompressionRatio.X16,  # X4, X8, X16, X32, X64
            always_ram=True,
        ),
    ),
)

# --- Binary Quantization ---
client.create_collection(
    collection_name="binary_quantized",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=BinaryQuantization(
        binary=BinaryQuantizationConfig(
            always_ram=True,
        ),
    ),
)

# --- Search with quantization oversampling (re-scoring) ---
# Qdrant first searches quantized vectors (fast), then re-scores top
# candidates using original vectors (accurate).
results = client.query_points(
    collection_name="scalar_quantized",
    query=[0.1, 0.2, ...],
    search_params=SearchParams(
        quantization=QuantizationSearchParams(
            rescore=True,       # Re-score using original vectors
            oversampling=2.0,   # Fetch 2x candidates before re-scoring
            ignore=False,       # Set True to bypass quantization for this query
        )
    ),
    limit=10,
)
```

### When to Use Quantization

```
+------------------------------------------------------------------+
|              Quantization Decision Guide                         |
+------------------------------------------------------------------+
|                                                                  |
|  Do you have memory constraints?                                 |
|    |                                                             |
|    No --> Don't quantize (full float32 is best accuracy)         |
|    |                                                             |
|    Yes --> How much compression do you need?                     |
|      |                                                           |
|      Moderate (4x) --> Scalar Quantization                       |
|      |   Best general-purpose option                             |
|      |   Minimal recall loss                                     |
|      |   Works with any embedding model                          |
|      |                                                           |
|      High (16-64x) --> Product Quantization                      |
|      |   Good for very large datasets                            |
|      |   Accepts more recall trade-off                           |
|      |   Needs enough data to train codebook                     |
|      |                                                           |
|      Maximum (32x) + Speed --> Binary Quantization               |
|          Works best with high-dimensional embeddings             |
|          (OpenAI text-embedding-3-*, Cohere embed-v3)            |
|          Hamming distance is extremely fast (CPU bitwise ops)    |
|          Significant recall loss for some models                 |
|                                                                  |
|  ALWAYS enable rescore=True for production to recover accuracy!  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 7. Search Operations

### Basic Nearest Neighbor Search

```python
from qdrant_client.models import SearchParams

# Simple search
results = client.query_points(
    collection_name="my_collection",
    query=[0.1, 0.2, 0.3, ...],  # Query vector
    limit=10,                     # Number of results
)

for point in results.points:
    print(f"ID: {point.id}, Score: {point.score}, Payload: {point.payload}")

# Search with named vector
results = client.query_points(
    collection_name="multi_vector_collection",
    query=[0.1, 0.2, ...],
    using="title",  # Search against the "title" vector
    limit=5,
)

# Search with HNSW parameters
results = client.query_points(
    collection_name="my_collection",
    query=[0.1, 0.2, ...],
    search_params=SearchParams(
        hnsw_ef=256,     # Higher ef = better recall
        exact=False,      # Set True for brute-force (exact) search
    ),
    limit=10,
)
```

### Search with Filters

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Search with payload filter
results = client.query_points(
    collection_name="my_collection",
    query=[0.1, 0.2, 0.3, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="science")),
            FieldCondition(key="year", range=Range(gte=2022)),
        ],
        must_not=[
            FieldCondition(key="status", match=MatchValue(value="archived")),
        ],
    ),
    limit=10,
)
```

### Search with Score Threshold

```python
# Only return results above a similarity threshold
results = client.query_points(
    collection_name="my_collection",
    query=[0.1, 0.2, ...],
    score_threshold=0.8,  # Only results with score >= 0.8
    limit=10,
)
```

### Batch Search

```python
from qdrant_client.models import QueryRequest

# Search multiple queries in a single request (more efficient)
results = client.query_batch_points(
    collection_name="my_collection",
    requests=[
        QueryRequest(
            query=[0.1, 0.2, ...],
            limit=5,
        ),
        QueryRequest(
            query=[0.4, 0.5, ...],
            filter=Filter(
                must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
            ),
            limit=10,
        ),
        QueryRequest(
            query=[0.7, 0.8, ...],
            limit=3,
            score_threshold=0.9,
        ),
    ],
)

for i, batch_result in enumerate(results):
    print(f"\nQuery {i} results:")
    for point in batch_result.points:
        print(f"  ID: {point.id}, Score: {point.score}")
```

### Recommendation API

The recommendation API finds points similar to positive examples and dissimilar to negative
examples, without needing to provide an explicit query vector.

```
+------------------------------------------------------------------+
|                  Recommendation API                              |
+------------------------------------------------------------------+
|                                                                  |
|  Positive examples: Points you want MORE results like            |
|  Negative examples: Points you want FEWER results like           |
|                                                                  |
|  Conceptual:                                                     |
|                                                                  |
|       (+) Point A          (-) Point C                           |
|          *                    x                                  |
|         / \                                                      |
|        /   \    Target                                           |
|       /     \   Region                                           |
|      (+) Point B  ****                                           |
|          *       * ?? *     Results found here:                  |
|                  * ?? *     similar to A, B                      |
|                   ****      dissimilar to C                      |
|                                                                  |
+------------------------------------------------------------------+
```

```python
# Recommend by point IDs (uses existing vectors)
results = client.query_points(
    collection_name="my_collection",
    query=RecommendInput(
        positive=[1, 42],     # Point IDs user liked
        negative=[7],         # Point IDs user disliked
    ),
    limit=10,
)

# Recommend with strategy
from qdrant_client.models import RecommendStrategy, RecommendInput

results = client.query_points(
    collection_name="my_collection",
    query=RecommendInput(
        positive=[1, 42],
        negative=[7],
        strategy=RecommendStrategy.AVERAGE_VECTOR,  # or BEST_SCORE
    ),
    limit=10,
)
```

**Strategies:**
- `AVERAGE_VECTOR` (default): Averages positive vectors, subtracts negative. Single search pass. Fast.
- `BEST_SCORE`: Performs a search per example, combines scores. More accurate for diverse positives.

### Discovery Search

Discovery search explores the vector space with context. It uses positive/negative pairs to
define a "direction" and a target to define the "region."

```python
from qdrant_client.models import DiscoverInput, ContextPair

results = client.query_points(
    collection_name="my_collection",
    query=DiscoverInput(
        target=[0.1, 0.2, ...],  # The anchor point (what to search near)
        context=[
            ContextPair(
                positive=1,   # Point ID: prefer this direction
                negative=2,   # Point ID: avoid this direction
            ),
        ],
    ),
    limit=10,
)
```

### Grouping Results

Group search results by a payload field. Useful for deduplication, e.g., when multiple chunks
from the same document appear in results.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Group results by "document_id" field
results = client.query_points_groups(
    collection_name="my_collection",
    query=[0.1, 0.2, ...],
    group_by="document_id",   # Payload field to group by
    group_size=2,             # Max points per group
    limit=5,                  # Number of groups
)

for group in results.groups:
    print(f"\nGroup: {group.id}")
    for point in group.hits:
        print(f"  ID: {point.id}, Score: {point.score}")
```

```
+------------------------------------------------------------------+
|              Grouped Search Results                              |
+------------------------------------------------------------------+
|                                                                  |
|  Without Grouping:              With group_by="doc_id":          |
|  1. chunk_3 (doc_A) 0.95       Group doc_A:                     |
|  2. chunk_7 (doc_A) 0.93         chunk_3 (0.95)                 |
|  3. chunk_1 (doc_B) 0.91         chunk_7 (0.93)                 |
|  4. chunk_5 (doc_A) 0.89       Group doc_B:                     |
|  5. chunk_2 (doc_B) 0.87         chunk_1 (0.91)                 |
|                                   chunk_2 (0.87)                 |
|  Problem: doc_A dominates!      Group doc_C:                     |
|                                   chunk_4 (0.85)                 |
|                                                                  |
|                                 Better diversity!                 |
+------------------------------------------------------------------+
```

---

## 8. Sparse Vectors

### What Are Sparse Vectors?

Sparse vectors have mostly zero values. They are the natural representation for traditional
information retrieval methods like **BM25** and **TF-IDF**, where each dimension corresponds
to a specific term in the vocabulary. Only non-zero dimensions are stored.

```
+------------------------------------------------------------------+
|            Dense vs Sparse Vectors                               |
+------------------------------------------------------------------+
|                                                                  |
|  Dense Vector (from embedding model):                            |
|  [0.12, -0.45, 0.78, 0.23, -0.56, 0.34, 0.91, -0.12, ...]    |
|  All dimensions have values. Typically 384-3072 dims.            |
|  Captures semantic meaning.                                      |
|                                                                  |
|  Sparse Vector (from BM25/TF-IDF/SPLADE):                       |
|  {                                                               |
|    "indices": [42, 1337, 5891, 12045],                           |
|    "values":  [0.8, 1.2,  0.3,  2.1]                            |
|  }                                                               |
|  Most dimensions are zero. Vocabulary-sized (30K-100K dims).     |
|  Captures exact keyword matching / lexical similarity.           |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Why Both?                                                       |
|                                                                  |
|  Query: "Python asyncio tutorial"                                |
|                                                                  |
|  Dense: Finds semantically similar results                       |
|         -> "concurrent programming in Python guide"              |
|         -> "async/await patterns for beginners"                  |
|         (good at understanding MEANING)                          |
|                                                                  |
|  Sparse: Finds exact keyword matches                             |
|          -> "Python asyncio tutorial for beginners"              |
|          -> "asyncio documentation"                              |
|          (good at finding EXACT TERMS)                           |
|                                                                  |
|  Hybrid: Combines both for best results!                         |
|                                                                  |
+------------------------------------------------------------------+
```

### Creating a Collection with Sparse Vectors

```python
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    SparseIndexParams,
)

# Collection with both dense and sparse vectors
client.create_collection(
    collection_name="hybrid_collection",
    vectors_config={
        "dense": VectorParams(size=1536, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(
                on_disk=False,
            )
        ),
    },
)
```

### Upserting Points with Sparse Vectors

```python
from qdrant_client.models import PointStruct, SparseVector

client.upsert(
    collection_name="hybrid_collection",
    points=[
        PointStruct(
            id=1,
            vector={
                "dense": [0.1, 0.2, 0.3, ...],  # Dense embedding
                "sparse": SparseVector(
                    indices=[42, 1337, 5891],    # Non-zero dimension indices
                    values=[0.8, 1.2, 0.3],      # Corresponding values
                ),
            },
            payload={"text": "Python asyncio tutorial", "category": "programming"},
        ),
        PointStruct(
            id=2,
            vector={
                "dense": [0.4, 0.5, 0.6, ...],
                "sparse": SparseVector(
                    indices=[42, 2048, 9999],
                    values=[0.5, 0.9, 1.5],
                ),
            },
            payload={"text": "Async programming patterns", "category": "programming"},
        ),
    ],
)
```

### Hybrid Search (Dense + Sparse)

```python
from qdrant_client.models import (
    Prefetch,
    Query,
    FusionQuery,
    Fusion,
    SparseVector,
)

# Hybrid search using Reciprocal Rank Fusion (RRF)
results = client.query_points(
    collection_name="hybrid_collection",
    prefetch=[
        # Dense search
        Prefetch(
            query=[0.1, 0.2, 0.3, ...],     # Dense query vector
            using="dense",
            limit=20,
        ),
        # Sparse search
        Prefetch(
            query=SparseVector(
                indices=[42, 1337],
                values=[0.8, 1.2],
            ),
            using="sparse",
            limit=20,
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
    limit=10,
)

for point in results.points:
    print(f"ID: {point.id}, Score: {point.score}, Payload: {point.payload}")
```

```
+------------------------------------------------------------------+
|           Hybrid Search with Reciprocal Rank Fusion              |
+------------------------------------------------------------------+
|                                                                  |
|  Dense Search Results:     Sparse Search Results:                |
|  Rank 1: Doc_A (0.95)     Rank 1: Doc_C (2.1)                   |
|  Rank 2: Doc_B (0.91)     Rank 2: Doc_A (1.8)                   |
|  Rank 3: Doc_C (0.87)     Rank 3: Doc_E (1.5)                   |
|  Rank 4: Doc_D (0.82)     Rank 4: Doc_B (1.2)                   |
|                                                                  |
|  RRF Score = sum(1 / (k + rank_i))  where k=60 (constant)       |
|                                                                  |
|  Doc_A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325         |
|  Doc_B: 1/(60+2) + 1/(60+4) = 0.0161 + 0.0156 = 0.0317         |
|  Doc_C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323         |
|  Doc_D: 1/(60+4) + 0        = 0.0156             = 0.0156       |
|  Doc_E: 0        + 1/(60+3) = 0.0159             = 0.0159       |
|                                                                  |
|  Final Ranking:                                                  |
|  1. Doc_A (0.0325) -- appeared in both, boosted!                 |
|  2. Doc_C (0.0323) -- appeared in both                           |
|  3. Doc_B (0.0317) -- appeared in both                           |
|  4. Doc_E (0.0159) -- only in sparse                             |
|  5. Doc_D (0.0156) -- only in dense                              |
|                                                                  |
+------------------------------------------------------------------+
```

### Generating Sparse Vectors

In practice, you need a model to generate sparse vectors. Common approaches:

```python
# Option 1: Using SPLADE (via transformers)
# SPLADE generates learned sparse representations
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")

def get_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    logits = output.logits
    # Apply log(1 + ReLU(x)) activation (SPLADE transformation)
    sparse = torch.max(
        torch.log(1 + torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1),
        dim=1,
    )[0].squeeze()
    # Get non-zero indices and values
    nonzero = sparse.nonzero().squeeze()
    indices = nonzero.tolist()
    values = sparse[nonzero].tolist()
    return indices, values

# Option 2: Using Qdrant's built-in FastEmbed (simpler)
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")

# FastEmbed supports sparse models natively
# The client can generate embeddings directly
```

---

## 9. Snapshots & Backup

### Creating Snapshots

Snapshots are point-in-time backups of a collection or the entire Qdrant storage.

```python
# Create a snapshot of a single collection
snapshot_info = client.create_snapshot(collection_name="my_collection")
print(f"Snapshot created: {snapshot_info.name}")
# Output: Snapshot created: my_collection-2024-01-15-10-30-00.snapshot

# List snapshots for a collection
snapshots = client.list_snapshots(collection_name="my_collection")
for snap in snapshots:
    print(f"Name: {snap.name}, Size: {snap.size}, Created: {snap.creation_time}")

# Delete a snapshot
client.delete_snapshot(
    collection_name="my_collection",
    snapshot_name="my_collection-2024-01-15-10-30-00.snapshot",
)
```

### Full Storage Snapshot

```python
# Create a full storage snapshot (all collections)
snapshot_info = client.create_full_snapshot()
print(f"Full snapshot: {snapshot_info.name}")

# List full snapshots
full_snapshots = client.list_full_snapshots()
```

### Downloading and Restoring Snapshots

```bash
# Download snapshot via REST API
curl -o backup.snapshot \
  "http://localhost:6333/collections/my_collection/snapshots/my_collection-2024-01-15.snapshot"

# Restore collection from snapshot (via REST API)
curl -X POST "http://localhost:6333/collections/my_collection/snapshots/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@backup.snapshot"
```

```python
# Restore from a snapshot URL (Python client)
client.recover_snapshot(
    collection_name="my_collection",
    location="http://backup-server/my_collection-2024-01-15.snapshot",
)

# Restore from local file path (if Qdrant can access it)
client.recover_snapshot(
    collection_name="my_collection",
    location="file:///path/to/backup.snapshot",
)
```

### Backup Strategies

```
+------------------------------------------------------------------+
|                    Backup Strategies                              |
+------------------------------------------------------------------+
|                                                                  |
|  Strategy 1: Periodic Snapshots                                  |
|  ============================                                    |
|  - Schedule cron job to create snapshots every N hours            |
|  - Download and store in S3/GCS/Azure Blob                       |
|  - Retain last N snapshots, delete older ones                     |
|                                                                  |
|  # Example cron (every 6 hours):                                 |
|  0 */6 * * * curl -X POST \                                     |
|    "http://localhost:6333/collections/my_collection/snapshots"   |
|                                                                  |
|  Strategy 2: Replication                                         |
|  ========================                                        |
|  - Run Qdrant in cluster mode with replication_factor > 1        |
|  - Automatic failover if a node goes down                        |
|  - Combine with periodic snapshots for disaster recovery         |
|                                                                  |
|  Strategy 3: Volume Snapshots (Docker/K8s)                       |
|  ==========================================                      |
|  - Use persistent volume snapshots                               |
|  - Works at the storage level (filesystem)                       |
|  - Faster than Qdrant-level snapshots for very large datasets    |
|                                                                  |
|  Strategy 4: WAL-based Recovery                                  |
|  =============================                                   |
|  - Qdrant uses Write-Ahead Log for crash recovery                |
|  - On restart, uncommitted operations are replayed from WAL      |
|  - Not a backup strategy per se, but provides crash safety       |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 10. Production Setup

### Docker Deployment

```yaml
# docker-compose.yml (Single Node)
version: "3.8"

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC API
    volumes:
      - qdrant_data:/qdrant/storage
      - ./config/config.yaml:/qdrant/config/production.yaml
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G

volumes:
  qdrant_data:
    driver: local
```

```yaml
# config/config.yaml (Qdrant configuration)
storage:
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/snapshots
  on_disk_payload: true
  performance:
    max_search_threads: 0  # Auto (use all available cores)
    max_optimization_threads: 2

service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  enable_tls: false
  api_key: "your-secret-api-key"  # Optional: enable API key auth

log_level: INFO
```

### Cluster Mode (Distributed Qdrant)

```yaml
# docker-compose-cluster.yml
version: "3.8"

services:
  qdrant-node-0:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
      - "6335:6335"   # Internal cluster port
    volumes:
      - qdrant_node0:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334

  qdrant-node-1:
    image: qdrant/qdrant:latest
    ports:
      - "6343:6333"
      - "6344:6334"
      - "6345:6335"
    volumes:
      - qdrant_node1:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    command: ./qdrant --bootstrap "http://qdrant-node-0:6335"

  qdrant-node-2:
    image: qdrant/qdrant:latest
    ports:
      - "6353:6333"
      - "6354:6334"
      - "6355:6335"
    volumes:
      - qdrant_node2:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    command: ./qdrant --bootstrap "http://qdrant-node-0:6335"

volumes:
  qdrant_node0:
  qdrant_node1:
  qdrant_node2:
```

### Sharding and Replication

```
+------------------------------------------------------------------+
|             Sharding & Replication in Qdrant                     |
+------------------------------------------------------------------+
|                                                                  |
|  SHARDING: Splits data across nodes for horizontal scaling       |
|                                                                  |
|  Collection "products" (shard_number=3, replication_factor=2)    |
|                                                                  |
|  +----------+    +----------+    +----------+                    |
|  |  Node 0  |    |  Node 1  |    |  Node 2  |                   |
|  +----------+    +----------+    +----------+                    |
|  | Shard 0  |    | Shard 1  |    | Shard 2  |  <-- Primary      |
|  | (primary)|    | (primary)|    | (primary)|      shards       |
|  +----------+    +----------+    +----------+                    |
|  | Shard 1  |    | Shard 2  |    | Shard 0  |  <-- Replica      |
|  | (replica)|    | (replica)|    | (replica)|      shards       |
|  +----------+    +----------+    +----------+                    |
|                                                                  |
|  - Each shard holds ~1/3 of the data                             |
|  - Replicas provide redundancy and read scaling                  |
|  - If Node 1 goes down: Shard 1 replica on Node 0 takes over    |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Shard Types:                                                    |
|  - Automatic sharding: Qdrant distributes points by hash         |
|  - Custom sharding: You choose which shard each point goes to    |
|                                                                  |
|  Write Consistency:                                              |
|  - Default: Write to majority of replicas before acknowledging   |
|  - Can configure: All, Majority, Quorum                          |
|                                                                  |
|  Read Consistency:                                               |
|  - Default: Read from any available replica                      |
|  - Can configure: All, Majority, Quorum                          |
|                                                                  |
+------------------------------------------------------------------+
```

```python
# Create a distributed collection
client.create_collection(
    collection_name="distributed_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    shard_number=6,            # Number of shards
    replication_factor=2,      # Each shard has 2 copies
    write_consistency_factor=1, # Writes acknowledged after 1 replica confirms
)
```

### Performance Tuning

```
+------------------------------------------------------------------+
|              Performance Tuning Checklist                         |
+------------------------------------------------------------------+
|                                                                  |
|  1. VECTOR STORAGE                                               |
|     - Keep vectors in RAM if possible (fastest)                  |
|     - Use on_disk=True for vectors if RAM is limited             |
|     - Use quantization to reduce memory footprint                |
|                                                                  |
|  2. HNSW INDEX                                                   |
|     - m=16 for most cases, increase for high recall needs        |
|     - ef_construct=100-200 (build once, search many)             |
|     - Set hnsw_ef per query based on accuracy needs              |
|     - Use on_disk=True for HNSW if RAM is limited                |
|                                                                  |
|  3. PAYLOAD INDEXES                                              |
|     - Create indexes on frequently filtered fields               |
|     - Use appropriate field types (keyword, integer, etc.)       |
|     - on_disk_payload=True if payloads are large                 |
|                                                                  |
|  4. BATCH OPERATIONS                                             |
|     - Batch upserts (100-500 points per batch)                   |
|     - Use batch search for multiple queries                      |
|     - Avoid single-point operations in loops                     |
|                                                                  |
|  5. OPTIMIZER SETTINGS                                           |
|     - indexing_threshold: Min points before building HNSW        |
|       (lower = faster index, higher = fewer rebuilds)            |
|     - memmap_threshold: When to switch from RAM to mmap          |
|     - max_optimization_threads: CPU cores for background tasks   |
|                                                                  |
|  6. gRPC vs REST                                                 |
|     - Use gRPC for high-throughput applications                  |
|     - REST is easier to debug and prototype with                 |
|     - gRPC has lower latency and binary serialization            |
|                                                                  |
|  7. CONNECTION POOLING                                           |
|     - Reuse client instances (don't create per request)          |
|     - Use async client for high-concurrency applications         |
|                                                                  |
+------------------------------------------------------------------+
```

### Monitoring

```python
# Health check
health = client.get_collections()

# Collection metrics
info = client.get_collection("my_collection")
print(f"Points: {info.points_count}")
print(f"Indexed vectors: {info.indexed_vectors_count}")
print(f"Segments: {info.segments_count}")
print(f"Status: {info.status}")  # "green" = fully indexed
print(f"Optimizer status: {info.optimizer_status}")
```

```bash
# Qdrant exposes Prometheus metrics at /metrics
curl http://localhost:6333/metrics

# Key metrics to monitor:
# - qdrant_collection_points_total
# - qdrant_search_latency_seconds
# - qdrant_grpc_responses_total
# - qdrant_rest_responses_total
# - process_resident_memory_bytes
```

### Memory Management

```
+------------------------------------------------------------------+
|           Memory Management Guidelines                           |
+------------------------------------------------------------------+
|                                                                  |
|  Memory consumption formula (approximate):                       |
|                                                                  |
|  Total RAM = HNSW graph + Vectors + Payloads + Payload Indexes   |
|                                                                  |
|  HNSW graph per point: ~(4 * m * 2) bytes = ~128 bytes (m=16)   |
|  Vector per point: dim * 4 bytes (float32)                       |
|                                                                  |
|  Example: 10M points, 1536 dims, m=16                            |
|  HNSW:    10M * 128 bytes    = ~1.2 GB                           |
|  Vectors: 10M * 1536 * 4     = ~57.3 GB                          |
|  Total: ~58.5 GB (without payloads)                              |
|                                                                  |
|  With scalar quantization:                                       |
|  Vectors: 10M * 1536 * 1     = ~14.3 GB                          |
|  Total: ~15.5 GB   (3.8x savings!)                               |
|                                                                  |
|  Strategies to reduce memory:                                    |
|  1. Quantization (scalar = 4x, binary = 32x)                    |
|  2. on_disk vectors (mmap, OS manages caching)                   |
|  3. on_disk HNSW index                                           |
|  4. on_disk_payload                                              |
|  5. Increase shard count across more nodes                       |
|                                                                  |
+------------------------------------------------------------------+
```

### WAL Configuration

```
+------------------------------------------------------------------+
|              Write-Ahead Log (WAL)                               |
+------------------------------------------------------------------+
|                                                                  |
|  WAL ensures crash safety:                                       |
|                                                                  |
|  1. Client sends write request                                   |
|  2. Qdrant writes to WAL (sequential, fast)                      |
|  3. Qdrant acknowledges the write to client                      |
|  4. Background: WAL entries applied to segments                  |
|  5. Applied WAL entries are garbage collected                     |
|                                                                  |
|  If crash occurs between steps 3-4:                              |
|  -> On restart, unapplied WAL entries are replayed               |
|  -> No data loss                                                 |
|                                                                  |
|  Configuration:                                                  |
|  wal_capacity_mb: Size of WAL segment (default: 32MB)            |
|  wal_segments_ahead: Pre-allocated WAL segments (default: 0)     |
|                                                                  |
+------------------------------------------------------------------+
```

```python
from qdrant_client.models import WalConfigDiff

client.update_collection(
    collection_name="my_collection",
    wal_config=WalConfigDiff(
        wal_capacity_mb=64,      # Larger WAL for high write throughput
        wal_segments_ahead=2,    # Pre-allocate 2 segments
    ),
)
```

---

## 11. Python Client (qdrant-client)

### Installation

```bash
pip install qdrant-client

# With optional dependencies
pip install "qdrant-client[fastembed]"  # Includes FastEmbed for local embeddings
```

### Sync vs Async Client

```python
# === Synchronous Client ===
from qdrant_client import QdrantClient

sync_client = QdrantClient(host="localhost", port=6333)
results = sync_client.query_points(
    collection_name="my_collection",
    query=[0.1, 0.2, ...],
    limit=10,
)

# === Asynchronous Client ===
from qdrant_client import AsyncQdrantClient
import asyncio

async_client = AsyncQdrantClient(host="localhost", port=6333)

async def search():
    results = await async_client.query_points(
        collection_name="my_collection",
        query=[0.1, 0.2, ...],
        limit=10,
    )
    return results

results = asyncio.run(search())
```

### Connection Management

```python
# With timeout and connection settings
client = QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,      # Use gRPC for better performance
    timeout=30,            # Connection timeout in seconds
    api_key="your-key",    # Optional API key
)

# Using the client as a context manager (auto-closes connection)
# Not natively supported, but you can manage lifecycle:
client = QdrantClient(host="localhost", port=6333)
try:
    # ... operations ...
    pass
finally:
    client.close()
```

### Error Handling

```python
from qdrant_client.http.exceptions import (
    UnexpectedResponse,
    ResponseHandlingException,
)
from grpc import RpcError
import httpx

def safe_search(client, collection_name, query_vector, limit=10):
    """Search with comprehensive error handling."""
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )
        return results.points

    except UnexpectedResponse as e:
        # HTTP error from Qdrant (4xx, 5xx)
        print(f"Qdrant error: {e.status_code} - {e.reason_phrase}")
        if e.status_code == 404:
            print(f"Collection '{collection_name}' not found")
        elif e.status_code == 400:
            print(f"Bad request: check vector dimensions and parameters")
        raise

    except RpcError as e:
        # gRPC error
        print(f"gRPC error: {e.code()} - {e.details()}")
        raise

    except httpx.ConnectError:
        print("Cannot connect to Qdrant. Is the server running?")
        raise

    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        raise


def safe_upsert(client, collection_name, points, max_retries=3):
    """Upsert with retry logic."""
    for attempt in range(max_retries):
        try:
            client.upsert(
                collection_name=collection_name,
                points=points,
            )
            return True
        except (UnexpectedResponse, RpcError, httpx.ConnectError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                import time
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                raise
    return False
```

### FastAPI Integration Example

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid

# --- Application Setup ---

qdrant_client: AsyncQdrantClient | None = None
COLLECTION_NAME = "documents"
VECTOR_SIZE = 1536


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage Qdrant client lifecycle."""
    global qdrant_client
    qdrant_client = AsyncQdrantClient(host="localhost", port=6333)

    # Ensure collection exists
    collections = await qdrant_client.get_collections()
    collection_names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in collection_names:
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    yield  # Application runs here

    # Cleanup
    if qdrant_client:
        await qdrant_client.close()


app = FastAPI(title="Vector Search API", lifespan=lifespan)


# --- Schemas ---

class DocumentCreate(BaseModel):
    text: str
    category: str
    metadata: dict = {}


class SearchRequest(BaseModel):
    query_vector: list[float]
    category: str | None = None
    limit: int = 10


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    category: str


# --- Helper ---

async def get_embedding(text: str) -> list[float]:
    """Get embedding from your model. Placeholder implementation."""
    # Replace with actual embedding generation:
    # e.g., OpenAI, Sentence-Transformers, FastEmbed
    import random
    return [random.uniform(-1, 1) for _ in range(VECTOR_SIZE)]


# --- Endpoints ---

@app.post("/documents", status_code=201)
async def create_document(doc: DocumentCreate):
    """Store a document with its vector embedding."""
    doc_id = str(uuid.uuid4())
    embedding = await get_embedding(doc.text)

    await qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "text": doc.text,
                    "category": doc.category,
                    **doc.metadata,
                },
            )
        ],
    )
    return {"id": doc_id, "status": "created"}


@app.post("/search", response_model=list[SearchResult])
async def search_documents(req: SearchRequest):
    """Search for similar documents."""
    query_filter = None
    if req.category:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="category", match=MatchValue(value=req.category)
                )
            ]
        )

    results = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=req.query_vector,
        query_filter=query_filter,
        limit=req.limit,
    )

    return [
        SearchResult(
            id=str(point.id),
            score=point.score,
            text=point.payload.get("text", ""),
            category=point.payload.get("category", ""),
        )
        for point in results.points
    ]


@app.post("/search/semantic")
async def semantic_search(query: str, category: str | None = None, limit: int = 10):
    """Search by text (generates embedding automatically)."""
    query_vector = await get_embedding(query)

    query_filter = None
    if category:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="category", match=MatchValue(value=category)
                )
            ]
        )

    results = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
    )

    return {
        "query": query,
        "results": [
            {
                "id": str(point.id),
                "score": round(point.score, 4),
                "payload": point.payload,
            }
            for point in results.points
        ],
    }


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document by ID."""
    from qdrant_client.models import PointIdsList

    await qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[doc_id]),
    )
    return {"id": doc_id, "status": "deleted"}


@app.get("/health")
async def health_check():
    """Check Qdrant connection health."""
    try:
        info = await qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "status": "healthy",
            "qdrant_status": info.status.value,
            "points_count": info.points_count,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unhealthy: {e}")
```

### Best Practices

```
+------------------------------------------------------------------+
|              Python Client Best Practices                        |
+------------------------------------------------------------------+
|                                                                  |
|  1. REUSE CLIENT INSTANCES                                       |
|     - Create one client at app startup, share across requests    |
|     - Never create a client per request                          |
|                                                                  |
|  2. USE ASYNC CLIENT FOR WEB APPS                                |
|     - AsyncQdrantClient for FastAPI / async frameworks           |
|     - Sync client is fine for scripts and batch jobs             |
|                                                                  |
|  3. PREFER gRPC FOR PERFORMANCE                                  |
|     - prefer_grpc=True reduces serialization overhead            |
|     - Especially beneficial for large batch operations           |
|                                                                  |
|  4. BATCH YOUR OPERATIONS                                        |
|     - Upsert in batches of 100-500 points                        |
|     - Use query_batch_points for multiple searches               |
|                                                                  |
|  5. SET APPROPRIATE TIMEOUTS                                     |
|     - Increase timeout for large upserts/snapshots               |
|     - Default timeout may be too short for big operations        |
|                                                                  |
|  6. HANDLE ERRORS GRACEFULLY                                     |
|     - Implement retry with exponential backoff                   |
|     - Check collection existence before operations               |
|     - Validate vector dimensions match collection config         |
|                                                                  |
|  7. USE PAYLOAD INDEXES                                          |
|     - Always create indexes on filtered fields                   |
|     - Check collection info to verify indexes exist              |
|                                                                  |
|  8. MONITOR COLLECTION STATUS                                    |
|     - "green" = fully indexed and optimized                      |
|     - "yellow" = optimization in progress                        |
|     - Wait for green status before benchmarking                  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 12. Q&A Section

### Q1: What is Qdrant and why would you choose it over alternatives?

**A:** Qdrant is an open-source vector similarity search engine written in Rust. It stores,
indexes, and searches high-dimensional vectors with optional metadata (payloads).

**Reasons to choose Qdrant:**
- **Performance:** Written in Rust, no garbage collector pauses, memory-safe.
- **Rich filtering:** Advanced payload filtering (must/should/must_not) during vector search,
  with multiple index types.
- **Multi-vector support:** Store multiple named vectors per point (e.g., title + body
  embeddings), which other DBs like Pinecone and ChromaDB do not support.
- **Hybrid search:** Native sparse vector support for combining semantic and keyword search.
- **Quantization:** Scalar, product, and binary quantization for memory optimization.
- **Self-hosted + cloud:** You can run it locally, in Docker, or use Qdrant Cloud.
- **Flexible deployment:** Single node for dev, distributed cluster for production.

Compared to Pinecone (managed-only, no self-hosting), Milvus (heavier operational complexity),
Weaviate (less flexible filtering), and ChromaDB (not production-ready at scale).

---

### Q2: Explain the HNSW algorithm and how Qdrant uses it.

**A:** HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor search
algorithm that builds a multi-layer graph:

1. **Layer 0** contains all points connected to their nearest neighbors.
2. Each higher layer contains a random subset of points from the layer below, with longer-range
   connections.
3. Search starts at the topmost layer's entry point and greedily navigates to the closest node.
4. At each layer, it moves to the nearest neighbor until no closer node exists, then drops to
   the next lower layer.
5. At Layer 0, it performs a more thorough local search (controlled by `ef` parameter).

**Key parameters in Qdrant:**
- `m`: Maximum edges per node per layer (default 16). Higher m = more memory but better recall.
- `ef_construct`: Search width during index build (default 100). Higher = better graph quality
  but slower build.
- `ef`: Search width during query (set per-query via `search_params`). Higher = better recall
  but slower search.

HNSW provides sub-linear search time: O(log N) compared to O(N) for brute force, making it
practical for millions of vectors.

---

### Q3: What are the different distance metrics and when should you use each?

**A:**

| Metric    | Formula                         | Range      | Use Case                      |
|-----------|---------------------------------|------------|-------------------------------|
| Cosine    | 1 - (A.B)/(|A|*|B|)            | [0, 2]     | NLP embeddings (default)      |
| Dot       | -(A.B)                          | (-inf,inf) | Recommendations, magnitude matters |
| Euclidean | sqrt(sum((a_i-b_i)^2))          | [0, inf)   | Image features, spatial data  |
| Manhattan | sum(|a_i-b_i|)                  | [0, inf)   | Sparse features, L1-optimized |

**Guidelines:**
- Use **Cosine** for text embeddings from OpenAI, Sentence-Transformers, Cohere (these models
  typically output normalized or near-normalized vectors).
- Use **Dot Product** when vector magnitude encodes information (e.g., popularity score baked
  into the vector), common in recommendation systems.
- Use **Euclidean** for image feature vectors or when absolute position in vector space matters.
- If your vectors are already L2-normalized, Cosine and Dot Product produce equivalent rankings.

---

### Q4: How does payload filtering work in Qdrant? How does it stay efficient?

**A:** Qdrant supports filtering on metadata (payloads) during vector search. Filters use
a `must` / `should` / `must_not` structure:
- **must:** All conditions must match (AND logic).
- **should:** At least one condition must match (OR logic).
- **must_not:** None of the conditions can match (NOT logic).

**Efficiency strategies:**
1. **Payload indexes:** Create indexes on frequently filtered fields (`keyword`, `integer`,
   `float`, `geo`, `text`, `datetime`). Without indexes, filtering requires scanning all
   payloads.
2. **Adaptive filtering:** Qdrant automatically chooses the best strategy:
   - If the filter is **broad** (matches many points): filter is applied during HNSW
     traversal (check filter as each node is visited).
   - If the filter is **selective** (matches few points): first identify matching point IDs
     from the payload index, then compare only those vectors.
3. This adaptive approach prevents the common problem of other systems where highly selective
   filters force scanning many irrelevant HNSW nodes.

---

### Q5: What is quantization and when should you use it?

**A:** Quantization reduces vector precision to save memory:

- **Scalar quantization (INT8):** Compresses float32 to uint8. 4x memory reduction. Minimal
  recall loss (~1%). Best general-purpose option.
- **Product quantization (PQ):** Splits vectors into sub-vectors and maps each to a codebook
  entry. 8-64x compression. Higher recall loss. Best for very large datasets.
- **Binary quantization:** Converts each dimension to a single bit (positive = 1, negative = 0).
  32x compression. Uses fast Hamming distance. Works best with high-dimensional, normalized
  vectors (OpenAI, Cohere models).

**When to use:**
- You have millions of vectors and RAM is the bottleneck.
- You can tolerate small accuracy trade-offs (mitigated by `rescore=True`).
- Binary quantization is particularly effective with OpenAI embeddings (1536 dims).

**Always use `rescore=True` in production:** Qdrant first searches quantized vectors (fast
approximate), then re-ranks the top candidates using original full-precision vectors.

---

### Q6: How do you optimize search performance in Qdrant?

**A:** Key optimization strategies:

1. **Payload indexes:** Create indexes on all fields used in filters. This is the single
   biggest performance improvement for filtered search.
2. **HNSW tuning:** Set `ef` per query based on accuracy needs. Lower ef = faster, higher
   ef = more accurate. Use `ef=128` as a starting point.
3. **Quantization with rescoring:** Scalar quantization (4x RAM savings) with
   `rescore=True` and `oversampling=1.5-2.0` gives near-original accuracy at much
   lower memory.
4. **gRPC over REST:** Use `prefer_grpc=True` for lower serialization overhead.
5. **Batch operations:** Use `query_batch_points` for multiple queries instead of
   individual calls.
6. **On-disk storage:** For datasets larger than RAM, use `on_disk=True` for vectors and
   HNSW. The OS page cache will keep frequently accessed data in memory.
7. **Shard count:** More shards enable parallel search within a single query. Set
   shard_number equal to the number of CPU cores for single-node deployments.
8. **Separate hot/cold data:** Use multiple collections or payload filters to avoid
   searching cold data.

---

### Q7: Explain Qdrant's architecture and how data is organized internally.

**A:** Qdrant's internal architecture:

1. **Collections:** Top-level containers. Each collection has its own vector configuration,
   HNSW index, and payload schema.
2. **Shards:** Each collection is divided into shards. Shards can be distributed across
   cluster nodes. Each shard is independent and searchable.
3. **Segments:** Each shard contains multiple segments. Segments are the actual storage units.
   There are two types:
   - **Mutable segments:** Accept new writes (in-memory, small).
   - **Immutable segments:** Optimized for reads (indexed, potentially mmap-backed).
4. **Optimizer:** Background process that merges small segments into larger, indexed segments.
   Also builds HNSW indexes when segment size exceeds the threshold.
5. **WAL (Write-Ahead Log):** All writes go to WAL first, then are applied to segments. This
   ensures crash safety.
6. **HNSW index:** Built per segment for vector similarity search.
7. **Payload index:** Per-field indexes for fast filtering.

Data flow: Client -> API Gateway -> Shard Router -> Target Shard -> WAL -> Segment -> Response.

---

### Q8: How does sharding work in Qdrant?

**A:** Sharding distributes data within a collection across multiple partitions:

- **Automatic sharding:** Points are distributed by hashing their IDs. Each shard holds
  approximately 1/N of the data (N = shard_number).
- **Custom sharding:** You can specify a `shard_key` to control which shard a point goes to.
  Useful when you want to co-locate related data (e.g., all points from one tenant).

**In a cluster:**
- Shards are distributed across nodes. The Raft consensus protocol manages shard placement.
- `replication_factor` determines how many copies of each shard exist.
- When a search query arrives, it is fan-out to all shards. Results from each shard are
  merged and the top-K are returned.
- Write operations go to the primary shard, then replicate to replicas.

**Configuration:**
```python
# Custom sharding
client.create_collection(
    collection_name="multi_tenant",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    shard_number=1,  # Let Qdrant handle number per shard key
    sharding_method="custom",
)

# Upsert with shard key
client.upsert(
    collection_name="multi_tenant",
    points=[PointStruct(id=1, vector=[...], payload={"tenant": "acme"})],
    shard_key_selector="acme",
)
```

---

### Q9: What are sparse vectors and how do they enable hybrid search?

**A:** Sparse vectors have mostly zero values, with only a few non-zero dimensions. They
represent traditional lexical/keyword matching (like BM25, TF-IDF, or learned sparse models
like SPLADE).

**Key differences from dense vectors:**
- Dense: All dimensions have values. Capture semantic meaning. Generated by embedding models.
- Sparse: Most dimensions are zero. Capture exact keyword overlap. Generated by BM25/SPLADE.

**Hybrid search** combines both:
1. Dense search finds semantically similar results (understands meaning).
2. Sparse search finds lexically similar results (exact keyword matches).
3. Results are fused using Reciprocal Rank Fusion (RRF) or other fusion methods.

This is powerful because dense search might miss exact keyword matches (e.g., product codes,
proper nouns), while sparse search misses semantic similarity (e.g., synonyms, paraphrases).

In Qdrant, you use `prefetch` to run both searches, then `FusionQuery(fusion=Fusion.RRF)` to
merge results.

---

### Q10: How do you handle collection updates in production without downtime?

**A:** Several strategies:

1. **Collection aliases:** Create a new collection, populate it, then atomically switch the
   alias to point to the new collection. Zero-downtime.
   ```python
   client.update_collection_aliases(
       change_aliases_operations=[
           {"delete_alias": {"alias_name": "production"}},
           {"create_alias": {"alias_name": "production", "collection_name": "v2"}},
       ]
   )
   ```

2. **Upserts are atomic:** Upserting to an existing collection updates vectors and payloads
   in place. Points remain searchable during updates.

3. **Payload index creation:** Creating a new payload index does not block searches. It is
   built in the background.

4. **HNSW re-indexing:** Updating HNSW parameters triggers a re-build in the background.
   Searches continue on the old index until the new one is ready.

5. **Replication:** With replication_factor > 1, you can take down one replica at a time
   for maintenance while others serve traffic.

---

### Q11: What is the recommendation API and how does it differ from regular search?

**A:** The recommendation API finds points similar to "positive" examples and dissimilar to
"negative" examples. Unlike regular search where you provide a query vector, you provide
point IDs.

**Two strategies:**
- **AVERAGE_VECTOR:** Averages all positive vectors and subtracts the average of negative
  vectors. Creates a single combined query vector. Fast (one search pass) but may not
  capture diversity well.
- **BEST_SCORE:** Performs separate searches from each positive/negative example and combines
  scores. More accurate for diverse positive examples but slower.

**Use cases:**
- "More like this" features (positive = items user clicked).
- Content moderation (negative = flagged content).
- Exploration with feedback (positive = liked, negative = disliked).

---

### Q12: How do you backup and restore Qdrant data?

**A:** Qdrant provides snapshot-based backups:

1. **Collection snapshots:** Point-in-time backup of a single collection including vectors,
   payloads, and indexes.
2. **Full storage snapshots:** Backup of all collections at once.
3. **Download snapshots** via REST API and store in S3/GCS for disaster recovery.
4. **Restore** by uploading a snapshot file or pointing to a URL.

**Best practices:**
- Schedule periodic snapshots via cron or job scheduler.
- Store snapshots in a different storage system (S3, GCS) from where Qdrant runs.
- Combine with replication for high availability (replication handles node failures;
  snapshots handle data corruption or accidental deletion).
- Test restoration periodically to ensure backups are valid.

---

### Q13: How do you tune HNSW parameters for your specific use case?

**A:** Tuning HNSW requires balancing recall, speed, and memory:

1. **Start with defaults** (m=16, ef_construct=100, ef=auto).
2. **Measure recall:** Run test queries with `exact=True` (brute force) and compare to
   HNSW results. Calculate recall@K.
3. **If recall is too low:**
   - Increase `ef` (search-time parameter, no rebuild needed). This is the cheapest knob.
   - Increase `ef_construct` and rebuild the index (one-time cost at build).
   - Increase `m` (requires more memory, also requires rebuild).
4. **If search is too slow:**
   - Decrease `ef`.
   - Enable quantization with rescoring to reduce memory and speed up distance calculations.
5. **General guidelines:**
   - `ef_construct` should be at least `2 * m`.
   - `ef` at search time should be at least equal to `limit` (number of requested results).
   - For 1M vectors, m=16, ef=128 typically gives >98% recall@10.

---

### Q14: What is the difference between on_disk storage and mmap in Qdrant?

**A:** Both involve storing data on disk, but they differ in how data is accessed:

- **on_disk=True (for vectors/HNSW):** Vectors or the HNSW graph are stored on disk and
  accessed via memory-mapped files (mmap). The OS manages which pages are cached in RAM.
  Hot (frequently accessed) data stays in RAM; cold data is read from disk on demand.

- **memmap_threshold:** When the number of points in a segment exceeds this threshold,
  the segment switches from fully in-memory to mmap-backed storage.

**Practical implications:**
- If your dataset fits in RAM, keep everything in-memory (fastest).
- If your dataset is 2-5x RAM, use mmap (the OS page cache will keep hot data in RAM).
- If your dataset is 10x+ RAM, combine mmap with quantization to reduce working set size.
- SSD storage is essential for on-disk mode; HDD will be too slow for random reads.

---

### Q15: How do you implement multi-tenancy in Qdrant?

**A:** Three approaches:

1. **Separate collections per tenant:**
   - Strongest isolation.
   - Overhead: each collection has its own HNSW index and optimizer.
   - Best for: few tenants with large data per tenant.

2. **Payload-based filtering:**
   - Add a `tenant_id` field to every point's payload.
   - Create a payload index on `tenant_id`.
   - Filter by `tenant_id` on every search query.
   - Best for: many tenants, simple implementation.
   ```python
   results = client.query_points(
       collection_name="shared",
       query=vector,
       query_filter=Filter(must=[
           FieldCondition(key="tenant_id", match=MatchValue(value="tenant_123"))
       ]),
       limit=10,
   )
   ```

3. **Custom sharding (shard-per-tenant):**
   - Each tenant gets its own shard within one collection.
   - Good isolation without collection overhead.
   - Best for: many tenants, need some data isolation.
   ```python
   client.upsert(
       collection_name="multi_tenant",
       points=points,
       shard_key_selector="tenant_123",
   )
   ```

---

### Q16: What happens when you upsert a point with an existing ID?

**A:** Upserting a point with an existing ID performs a full replacement:
- The old vector is replaced with the new vector.
- The old payload is replaced with the new payload.
- The operation is atomic per point.
- The HNSW index is updated (the old node is marked as deleted and a new one is inserted).

If you only want to update the payload without changing the vector, use `set_payload` or
`overwrite_payload` instead, which is more efficient because it does not touch the vector
index.

---

### Q17: How do you handle large-scale data ingestion into Qdrant?

**A:** Best practices for bulk ingestion:

1. **Disable indexing during bulk load:**
   ```python
   client.update_collection(
       collection_name="my_collection",
       optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
   )
   ```
   This prevents HNSW from being rebuilt after every batch.

2. **Upsert in batches of 100-500 points** to balance network overhead and memory usage.

3. **Use gRPC** (`prefer_grpc=True`) for lower serialization overhead.

4. **Use parallel uploads** from multiple threads/processes.

5. **Re-enable indexing after load:**
   ```python
   client.update_collection(
       collection_name="my_collection",
       optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
   )
   ```

6. **Wait for indexing to complete:**
   ```python
   import time
   while True:
       info = client.get_collection("my_collection")
       if info.status == "green":
           break
       time.sleep(5)
   ```

---

### Q18: What is the Prefetch mechanism and how does it work for multi-stage search?

**A:** Prefetch enables multi-stage search pipelines in a single query. It works by:

1. **Stage 1 (Prefetch):** Runs one or more searches (dense, sparse, or filtered) to collect
   a large set of candidates.
2. **Stage 2 (Main query):** Re-ranks, fuses, or refines the prefetched candidates.

**Use cases:**
- **Hybrid search:** Prefetch from dense and sparse, then fuse with RRF.
- **Re-ranking:** Prefetch with a fast model, then re-rank with a more accurate (but slower)
  named vector.
- **Multi-vector search:** Prefetch from "title" vectors, then refine with "content" vectors.

```python
# Multi-stage: fast prefetch with quantized vectors, then re-rank with full vectors
results = client.query_points(
    collection_name="my_collection",
    prefetch=[
        Prefetch(
            query=[0.1, 0.2, ...],
            limit=100,  # Get 100 candidates
            params=SearchParams(quantization=QuantizationSearchParams(rescore=False)),
        )
    ],
    query=[0.1, 0.2, ...],  # Re-score top candidates with original vectors
    limit=10,
)
```

---

### Q19: How does Qdrant ensure data durability and crash recovery?

**A:** Qdrant uses multiple mechanisms:

1. **Write-Ahead Log (WAL):** Every write operation is first recorded in the WAL before being
   applied to segments. On crash recovery, unapplied WAL entries are replayed.

2. **Segment immutability:** Once a mutable segment is optimized into an immutable segment,
   it is never modified in place. Updates create new segments.

3. **Atomic segment switching:** When the optimizer creates a new optimized segment, the
   switch from old to new segment is atomic. If a crash occurs during optimization, the old
   segment is still valid.

4. **Replication (cluster mode):** With replication_factor > 1, data exists on multiple nodes.
   Even if one node's disk fails, data is recoverable from replicas.

5. **Snapshots:** Point-in-time backups for disaster recovery beyond what WAL and replication
   handle (e.g., accidental collection deletion).

---

### Q20: Compare Qdrant's query_points vs the older search method. What changed?

**A:** Qdrant recently unified its search APIs under `query_points` (and `query_batch_points`):

- **Old API:** Separate methods for `search`, `recommend`, `discover`, `search_groups`.
- **New API:** A single `query_points` method that accepts different query types.

```python
# Old way
results = client.search(collection_name="col", query_vector=[...], limit=10)

# New way (query_points)
results = client.query_points(collection_name="col", query=[...], limit=10)
```

The `query` parameter accepts:
- A plain vector (nearest neighbor search).
- `RecommendInput(positive=[...], negative=[...])` for recommendations.
- `DiscoverInput(target=[...], context=[...])` for discovery search.
- `FusionQuery(fusion=Fusion.RRF)` for result fusion (with prefetch).
- A point ID to find similar points.

The `query_points` API is more flexible and composable, especially with `prefetch` for
multi-stage pipelines.

---

### Q21: How do you monitor Qdrant in production?

**A:** Key monitoring approaches:

1. **Prometheus metrics** (exposed at `/metrics`):
   - `qdrant_collection_points_total`: Track ingestion rate.
   - `qdrant_search_latency_seconds`: Monitor search performance.
   - `qdrant_grpc_responses_total` / `qdrant_rest_responses_total`: Request rates and errors.
   - `process_resident_memory_bytes`: Memory usage.
   - Set up Grafana dashboards for visualization.

2. **Collection status checks:**
   - `green`: All segments indexed, ready for optimal search.
   - `yellow`: Optimizer running (rebuilding indexes).
   - Monitor transition from yellow to green after large upserts.

3. **Health endpoints:** `GET /healthz` for liveness, `GET /readyz` for readiness.

4. **Alerts to set:**
   - Memory usage > 80% of available RAM.
   - Search latency p99 exceeds acceptable threshold.
   - Collection status stuck on "yellow" for too long.
   - Disk usage approaching capacity.
   - Error rate spike in API responses.

---

### Q22: When should you use exact (brute-force) search instead of HNSW?

**A:** Use exact search (`exact=True`) when:

1. **Small collections:** Under 10,000 points (controlled by `full_scan_threshold`). Brute
   force on small data can be faster than HNSW graph traversal overhead.

2. **Recall must be 100%:** Critical applications where approximate results are not acceptable
   (e.g., deduplication, compliance checks).

3. **Highly selective filters:** When your filter reduces candidates to a few hundred points,
   brute force on the filtered subset is faster than HNSW traversal.

4. **Benchmarking:** To measure the recall of your HNSW configuration, compare its results
   against exact search.

```python
# Exact search
results = client.query_points(
    collection_name="my_collection",
    query=[0.1, 0.2, ...],
    search_params=SearchParams(exact=True),
    limit=10,
)
```

Note: Qdrant automatically uses brute force when the number of points in a collection is
below `full_scan_threshold` (default 10,000), regardless of the `exact` parameter.

---

### Q23: How do you choose the right vector dimension for your application?

**A:** Vector dimension is determined by your embedding model, not by Qdrant. However,
choosing the right model (and thus dimension) matters:

| Model                          | Dimensions | Notes                        |
|--------------------------------|-----------|------------------------------|
| OpenAI text-embedding-3-small  | 1536      | Good quality, moderate size  |
| OpenAI text-embedding-3-large  | 3072      | Best quality, large          |
| Sentence-Transformers (mini)   | 384       | Fast, lightweight            |
| Sentence-Transformers (base)   | 768       | Good balance                 |
| Cohere embed-v3                | 1024      | Multilingual, versatile      |
| BGE-large                      | 1024      | Open-source, high quality    |

**Trade-offs:**
- Higher dimensions: Better representation, more memory, slower search.
- Lower dimensions: Less memory, faster search, possibly lower quality.
- Some models (OpenAI text-embedding-3-*) support Matryoshka embeddings, allowing you to
  truncate vectors to lower dimensions with graceful quality degradation.

**Qdrant-specific consideration:** Higher dimensions benefit more from quantization. For
1536-dim with binary quantization, you go from ~6KB to ~192 bytes per vector.

---

### Q24: Explain the difference between segments and how the optimizer works.

**A:** Segments are the internal storage units within a shard:

- **Mutable (writable) segments:** Small, in-memory segments that accept new writes. No HNSW
  index (uses brute-force search within the segment).
- **Immutable (read-optimized) segments:** Larger, indexed segments. Have HNSW index built.
  May be mmap-backed for disk storage.

**The optimizer** is a background process that:
1. Monitors mutable segments.
2. When a mutable segment exceeds the `indexing_threshold`, the optimizer creates a new
   immutable segment.
3. Builds the HNSW index for the new immutable segment.
4. Atomically swaps the old mutable segment for the new immutable one.
5. Merges small immutable segments into larger ones (reducing total segment count).

This architecture allows Qdrant to accept writes continuously while maintaining search
performance. New data is immediately searchable in mutable segments (via brute force), and
eventually indexed in immutable segments for optimal search.

---

### Q25: What are collection aliases and how are they useful?

**A:** Aliases are alternative names that point to a collection. They enable zero-downtime
collection updates:

```
+------------------------------------------------------------------+
|                    Alias-Based Deployment                        |
+------------------------------------------------------------------+
|                                                                  |
|  Before migration:                                               |
|  "production" alias ---> collection_v1                           |
|                                                                  |
|  During migration:                                               |
|  "production" alias ---> collection_v1  (still serving)          |
|  collection_v2 being populated in background                     |
|                                                                  |
|  After migration (atomic switch):                                |
|  "production" alias ---> collection_v2  (now serving)            |
|  collection_v1 can be deleted                                    |
|                                                                  |
+------------------------------------------------------------------+
```

**Use cases:**
- **Blue-green deployment:** Build a new collection with updated embeddings, then switch.
- **A/B testing:** Point different aliases to different collections.
- **Rollback:** If the new collection has issues, switch the alias back.

```python
# Your application always queries the alias "production"
results = client.query_points(
    collection_name="production",  # This is the alias
    query=vector,
    limit=10,
)
```

The application code never changes; only the alias mapping is updated.
