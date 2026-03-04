# Service Plan — Embedding Service

**Service**: Embedding Service
**Type**: gRPC
**Port**: 50052
**Build Phase**: Phase 1 (Week 1)
**Parent Plan**: [implementation_plan.md](implementation_plan.md)

---

## 1. Purpose

The Embedding Service is a lightweight gRPC microservice that accepts text input and returns vector embeddings. It is used by both the ingestion pipeline (to embed document chunks) and the Retrieval Service (to embed user queries). By centralizing embedding logic in a single service, we ensure consistent vector dimensions and model versions across the system.

---

## 2. Responsibilities

- Accept single or batched text inputs via gRPC
- Generate vector embeddings using a configured model
- Return embeddings as float arrays with metadata (model name, dimensions)
- Support model hot-swapping via configuration (OpenAI, Sentence Transformers)
- Expose health check endpoint

---

## 3. Embedding Models

| Option                    | Model                          | Dimensions | Pros                         | Cons                      |
|---------------------------|--------------------------------|------------|------------------------------|---------------------------|
| OpenAI (default)          | `text-embedding-3-small`       | 1536       | High quality, easy setup     | API cost, network latency |
| Sentence Transformers     | `all-MiniLM-L6-v2`            | 384        | Free, local, fast            | Lower quality             |
| Sentence Transformers     | `all-mpnet-base-v2`           | 768        | Good quality, local          | Higher memory usage       |

**Default**: OpenAI `text-embedding-3-small` for quality. Configurable via `EMBEDDING_MODEL_PROVIDER` env var.

---

## 4. gRPC Interface

### 4.1 Proto Definition

```protobuf
syntax = "proto3";
package embedding;

service EmbeddingService {
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);
  rpc GetModelInfo(Empty) returns (ModelInfo);
}

message Empty {}

message EmbedRequest {
  string text = 1;
}

message EmbedResponse {
  repeated float vector = 1;
  int32 dimensions = 2;
}

message EmbedBatchRequest {
  repeated string texts = 1;
}

message EmbedBatchResponse {
  repeated EmbedResponse embeddings = 1;
}

message ModelInfo {
  string model_name = 1;
  string provider = 2;
  int32 dimensions = 3;
}
```

### 4.2 Batch Processing

- `EmbedBatch` accepts up to 64 texts per request
- For OpenAI: sends a single batched API call (their API supports batch)
- For Sentence Transformers: batches through the model in one forward pass
- Returns embeddings in the same order as input texts

---

## 5. Directory Structure

```
services/embedding/
├── Dockerfile
├── requirements.txt
└── src/
    ├── main.py                 # gRPC server entry point
    ├── config.py               # Configuration (model, provider, port)
    ├── service.py              # EmbeddingService gRPC implementation
    └── providers/
        ├── base.py             # Abstract embedding provider interface
        ├── openai_provider.py  # OpenAI embeddings implementation
        └── st_provider.py     # Sentence Transformers implementation
```

---

## 6. Provider Interface

```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
```

Factory pattern selects provider based on `EMBEDDING_MODEL_PROVIDER` env var.

---

## 7. Dependencies

```
grpcio>=1.60.0
grpcio-tools>=1.60.0
grpcio-health-checking>=1.60.0
openai>=1.50.0
sentence-transformers>=3.0.0    # Optional, only if using local models
pydantic>=2.0
pydantic-settings>=2.0
structlog>=24.0
```

---

## 8. Configuration

| Variable                   | Default                     | Description                        |
|----------------------------|-----------------------------|------------------------------------|
| `EMBEDDING_PORT`           | `50052`                     | gRPC server port                   |
| `EMBEDDING_MODEL_PROVIDER` | `openai`                    | `openai` or `sentence_transformers`|
| `EMBEDDING_MODEL_NAME`     | `text-embedding-3-small`    | Model identifier                   |
| `OPENAI_API_KEY`           | —                           | Required if provider is `openai`   |
| `EMBEDDING_MAX_BATCH_SIZE` | `64`                        | Max texts per batch request        |

---

## 9. Implementation Steps

1. [ ] Define `embedding.proto` with `Embed`, `EmbedBatch`, `GetModelInfo` RPCs
2. [ ] Generate Python gRPC stubs from proto
3. [ ] Implement `EmbeddingProvider` abstract base class
4. [ ] Implement `OpenAIProvider` — calls OpenAI embeddings API
5. [ ] Implement `SentenceTransformersProvider` — local model inference
6. [ ] Implement provider factory based on config
7. [ ] Implement `EmbeddingService` gRPC servicer
8. [ ] Add gRPC health checking
9. [ ] Add structured logging
10. [ ] Write unit tests (mock providers)
11. [ ] Write integration test with real embeddings
12. [ ] Create Dockerfile

---

## 10. Key Design Decisions

| Decision                          | Choice                    | Rationale                                              |
|-----------------------------------|---------------------------|--------------------------------------------------------|
| Separate service vs. library      | Separate gRPC service     | Single source of truth for model + dimensions          |
| Async vs. sync gRPC               | Async (`grpc.aio`)        | Non-blocking, better throughput for batch requests     |
| Provider pattern                  | Abstract base + factory   | Easy to swap models without changing service interface |
| Default model                     | OpenAI `text-embedding-3-small` | Best quality/cost ratio for the use case         |
