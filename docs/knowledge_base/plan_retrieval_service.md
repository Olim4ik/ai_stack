# Service Plan — Retrieval Service

**Service**: Retrieval Service
**Type**: gRPC
**Port**: 50051
**Build Phase**: Phase 2 (Week 2)
**Parent Plan**: [implementation_plan.md](implementation_plan.md)

---

## 1. Purpose

The Retrieval Service handles all vector search operations against Qdrant. It accepts a query, embeds it via the Embedding Service, performs semantic (and optionally hybrid) search in Qdrant, and returns ranked document chunks with metadata and relevance scores.

---

## 2. Responsibilities

- Accept search queries via gRPC
- Embed queries by calling the Embedding Service
- Perform dense vector search in Qdrant
- Support hybrid search (dense + sparse vectors)
- Apply metadata filters (team, document type, tags, date range)
- Return ranked results with scores and source attribution
- Manage Qdrant collection lifecycle (create, configure, delete)

---

## 3. Search Modes

### 3.1 Dense Search (Default)

Standard semantic similarity search using cosine distance on dense vectors.

```
Query → Embed (gRPC) → Qdrant dense search → Ranked results
```

### 3.2 Hybrid Search

Combines dense semantic search with sparse keyword matching for better precision.

```
Query → Embed (gRPC) + Sparse encode → Qdrant hybrid search → RRF fusion → Ranked results
```

- Sparse encoding: BM25 via Qdrant's built-in sparse vector support
- Fusion: Reciprocal Rank Fusion (RRF) to merge dense and sparse rankings
- Weight: Configurable dense/sparse weight ratio (default 0.7/0.3)

### 3.3 Filtered Search

All search modes support metadata filtering via Qdrant payload filters:

```python
# Example filter: platform team, runbook docs, last 30 days
filter = Filter(
    must=[
        FieldCondition(key="team", match=MatchValue(value="platform")),
        FieldCondition(key="tags", match=MatchAny(any=["runbook"])),
        FieldCondition(key="ingested_at", range=Range(gte="2026-02-01")),
    ]
)
```

---

## 4. gRPC Interface

### 4.1 Proto Definition

```protobuf
syntax = "proto3";
package retrieval;

service RetrievalService {
  rpc Search(SearchRequest) returns (SearchResponse);
  rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);
  rpc DeleteCollection(DeleteCollectionRequest) returns (DeleteCollectionResponse);
  rpc HealthCheck(Empty) returns (HealthResponse);
}

message Empty {}

message SearchRequest {
  string query = 1;
  string collection = 2;          // Team/tenant collection name
  int32 top_k = 3;                // Number of results (default 5)
  SearchMode mode = 4;            // DENSE or HYBRID
  repeated Filter filters = 5;    // Metadata filters
}

enum SearchMode {
  DENSE = 0;
  HYBRID = 1;
}

message Filter {
  string field = 1;
  string operator = 2;            // "eq", "in", "gte", "lte", "range"
  string value = 3;               // JSON-encoded value
}

message SearchResponse {
  repeated SearchResult results = 1;
  float search_time_ms = 2;
}

message SearchResult {
  string chunk_id = 1;
  string text = 2;
  float score = 3;
  map<string, string> metadata = 4;   // doc_id, title, section, source_file, tags
}

message CreateCollectionRequest {
  string name = 1;
  int32 vector_size = 2;
}

message CreateCollectionResponse {
  bool success = 1;
}

message DeleteCollectionRequest {
  string name = 1;
}

message DeleteCollectionResponse {
  bool success = 1;
}

message HealthResponse {
  bool healthy = 1;
  string qdrant_status = 2;
}
```

---

## 5. Qdrant Collection Schema

Each team gets a dedicated collection:

```python
client.create_collection(
    collection_name="team_platform",
    vectors_config=VectorParams(
        size=1536,                    # Match embedding model dimensions
        distance=Distance.COSINE,
    ),
    sparse_vectors_config={           # For hybrid search
        "sparse": SparseVectorParams()
    },
)
```

### Payload Schema

| Field          | Type       | Description                     |
|----------------|------------|---------------------------------|
| `doc_id`       | string     | Hash of team + source_file      |
| `title`        | string     | Document title                  |
| `section`      | string     | Section heading (if any)        |
| `chunk_index`  | integer    | Position within the document    |
| `source_file`  | string     | Original file name              |
| `tags`         | string[]   | User-provided tags              |
| `ingested_at`  | string     | ISO timestamp of ingestion      |

---

## 6. Directory Structure

```
services/retrieval/
├── Dockerfile
├── requirements.txt
└── src/
    ├── main.py                   # gRPC server entry point
    ├── config.py                 # Configuration
    ├── service.py                # RetrievalService gRPC implementation
    ├── qdrant_client.py          # Qdrant connection and operations
    ├── search/
    │   ├── dense.py              # Dense search implementation
    │   ├── hybrid.py             # Hybrid search implementation
    │   └── filters.py            # Metadata filter builder
    └── grpc_clients/
        └── embedding.py          # Embedding service gRPC client
```

---

## 7. Dependencies

```
grpcio>=1.60.0
grpcio-tools>=1.60.0
grpcio-health-checking>=1.60.0
qdrant-client>=1.12.0
pydantic>=2.0
pydantic-settings>=2.0
structlog>=24.0
```

---

## 8. Configuration

| Variable                  | Default          | Description                              |
|---------------------------|------------------|------------------------------------------|
| `RETRIEVAL_PORT`          | `50051`          | gRPC server port                         |
| `QDRANT_HOST`             | `localhost`      | Qdrant server hostname                   |
| `QDRANT_PORT`             | `6333`           | Qdrant REST port                         |
| `QDRANT_GRPC_PORT`        | `6334`           | Qdrant gRPC port                         |
| `EMBEDDING_SERVICE_HOST`  | `localhost:50052` | Embedding service address               |
| `DEFAULT_TOP_K`           | `5`              | Default number of search results         |
| `HYBRID_DENSE_WEIGHT`     | `0.7`            | Dense vector weight in hybrid search     |

---

## 9. Implementation Steps

1. [ ] Define `retrieval.proto` with `Search`, `CreateCollection`, `DeleteCollection` RPCs
2. [ ] Generate Python gRPC stubs
3. [ ] Implement Qdrant client wrapper (connection, collection management)
4. [ ] Implement dense search — embed query, search Qdrant, return ranked results
5. [ ] Implement metadata filter builder (translate gRPC filters to Qdrant filters)
6. [ ] Implement hybrid search with RRF fusion
7. [ ] Implement `RetrievalService` gRPC servicer
8. [ ] Create gRPC client for Embedding Service
9. [ ] Add gRPC health checking with Qdrant connectivity check
10. [ ] Add structured logging
11. [ ] Write unit tests (mock Qdrant client)
12. [ ] Write integration tests against real Qdrant instance
13. [ ] Create Dockerfile

---

## 10. Key Design Decisions

| Decision                        | Choice                      | Rationale                                                    |
|---------------------------------|-----------------------------|--------------------------------------------------------------|
| Collection per team             | Yes                         | Clean isolation, independent scaling, easy deletion          |
| Default search mode             | Dense only                  | Simpler, hybrid is opt-in when precision matters             |
| Qdrant client connection        | gRPC (port 6334)            | Better performance than REST for service-to-service calls    |
| Embedding via gRPC vs. local    | Via Embedding Service       | Consistent model, single source of truth for embeddings      |
