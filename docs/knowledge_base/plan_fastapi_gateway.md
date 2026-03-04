# Service Plan — FastAPI Gateway

**Service**: FastAPI Gateway
**Type**: REST / SSE
**Port**: 8000
**Build Phase**: Phase 2 (Week 2)
**Parent Plan**: [implementation_plan.md](implementation_plan.md)

---

## 1. Purpose

The FastAPI Gateway is the single entry point for all client requests. It exposes REST endpoints for the Vue.js frontend, proxies chat requests to the LangGraph Agent via gRPC, handles document ingestion, and streams responses via Server-Sent Events (SSE).

---

## 2. Responsibilities

- Expose REST API for chat, document management, and session history
- Stream agent responses to the frontend via SSE
- Orchestrate document ingestion (chunking → embedding → Qdrant storage)
- Proxy requests to internal gRPC services (Agent, Embedding, Retrieval)
- Handle authentication, rate limiting, and input validation
- Manage human-in-the-loop confirmation state

---

## 3. API Endpoints

### 3.1 Chat

| Method | Path                      | Description                                      |
|--------|---------------------------|--------------------------------------------------|
| POST   | `/api/chat`               | Send a message, receive streamed SSE response    |
| GET    | `/api/chat/history`       | Retrieve conversation history for a session      |
| POST   | `/api/chat/{id}/confirm`  | Submit human-in-the-loop confirmation             |

**`POST /api/chat` Request Body:**
```json
{
  "session_id": "uuid",
  "message": "How do I restart the auth service?",
  "team": "platform"
}
```

**SSE Response Stream:**
```
event: token
data: {"content": "To restart the auth service..."}

event: source
data: {"title": "runbook-auth.md", "chunk_id": "abc123", "score": 0.92}

event: reasoning
data: {"node": "retrieve", "action": "Searching knowledge base..."}

event: confirm_required
data: {"action_id": "xyz", "tool": "jira", "description": "Create ticket PLAT-456"}

event: done
data: {"session_id": "uuid", "message_id": "uuid"}
```

### 3.2 Documents

| Method | Path                      | Description                                      |
|--------|---------------------------|--------------------------------------------------|
| POST   | `/api/documents/ingest`   | Upload and ingest a document                     |
| GET    | `/api/documents`          | List documents with pagination and filtering     |
| DELETE | `/api/documents/{id}`     | Delete document and its chunks from Qdrant       |

**`POST /api/documents/ingest` — Multipart Form:**
```
file: <binary>
team: "platform"
tags: ["runbook", "auth"]
```

### 3.3 Health

| Method | Path          | Description            |
|--------|---------------|------------------------|
| GET    | `/health`     | Gateway health check   |
| GET    | `/health/all` | Health of all services |

---

## 4. Document Ingestion Pipeline

The ingestion pipeline runs within the gateway process:

```
Upload → Parse → Chunk → Embed (gRPC) → Store in Qdrant
```

### 4.1 Steps

1. **Parse**: Extract text from uploaded file (Markdown, PDF, HTML, plain text)
   - Libraries: `markdown`, `pypdf2`, `beautifulsoup4`
2. **Chunk**: Split text into overlapping chunks
   - Default: 512 tokens, 50-token overlap
   - Preserve section headings as metadata
3. **Embed**: Call Embedding Service via gRPC to generate vectors
   - Batch chunks for efficiency (max 32 per request)
4. **Store**: Upsert vectors + metadata into Qdrant
   - Collection per team (multi-tenancy)
   - Payload: `doc_id`, `title`, `section`, `chunk_index`, `source_file`, `tags`, `ingested_at`
5. **Track**: Store document metadata in a lightweight SQLite/JSON registry for listing and deletion

### 4.2 Re-ingestion

- On re-upload of same `source_file` for a team, delete existing chunks first
- Match by `doc_id` (hash of team + source_file)

---

## 5. gRPC Client Connections

The gateway maintains gRPC client stubs to internal services:

| Target Service    | Proto                   | Usage                                        |
|-------------------|-------------------------|----------------------------------------------|
| Embedding Service | `embedding.proto`       | Generate embeddings during ingestion         |
| Retrieval Service | `retrieval.proto`       | Direct search queries (bypassing agent)      |
| LangGraph Agent   | `agent.proto`           | Forward chat messages, stream responses back |

### Connection Management

- Use `grpc.aio` for async gRPC clients
- Connection pooling with configurable max channels
- Retry policy: 3 retries with exponential backoff
- Health check before first request at startup

---

## 6. Directory Structure

```
services/gateway/
├── Dockerfile
├── requirements.txt
└── src/
    ├── main.py                 # FastAPI app, lifespan, middleware
    ├── config.py               # pydantic-settings configuration
    ├── routers/
    │   ├── chat.py             # /api/chat endpoints
    │   ├── documents.py        # /api/documents endpoints
    │   └── health.py           # /health endpoints
    ├── ingestion/
    │   ├── parser.py           # File format parsers
    │   ├── chunker.py          # Text chunking strategies
    │   └── pipeline.py         # Orchestrates parse → chunk → embed → store
    ├── grpc_clients/
    │   ├── embedding.py        # Embedding service client
    │   ├── retrieval.py        # Retrieval service client
    │   └── agent.py            # Agent service client
    ├── models/
    │   ├── requests.py         # Pydantic request models
    │   └── responses.py        # Pydantic response models
    └── middleware/
        ├── logging.py          # structlog request logging
        └── errors.py           # Global error handler
```

---

## 7. Dependencies

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
grpcio>=1.60.0
grpcio-tools>=1.60.0
qdrant-client>=1.12.0
pydantic>=2.0
pydantic-settings>=2.0
structlog>=24.0
python-multipart>=0.0.9
pypdf2>=3.0
beautifulsoup4>=4.12
sse-starlette>=2.0
```

---

## 8. Implementation Steps

1. [ ] Scaffold FastAPI project with `main.py`, config, and health endpoint
2. [ ] Define Pydantic request/response models
3. [ ] Implement file parsers (Markdown, PDF, HTML, plain text)
4. [ ] Implement text chunker with configurable strategy
5. [ ] Create gRPC client for Embedding Service
6. [ ] Build ingestion pipeline (parse → chunk → embed → store)
7. [ ] Implement `POST /api/documents/ingest` endpoint
8. [ ] Implement `GET /api/documents` and `DELETE /api/documents/{id}`
9. [ ] Create gRPC client for Agent Service
10. [ ] Implement `POST /api/chat` with SSE streaming
11. [ ] Implement `GET /api/chat/history`
12. [ ] Implement `POST /api/chat/{id}/confirm` for human-in-the-loop
13. [ ] Add structlog middleware and error handlers
14. [ ] Write unit tests for parsers, chunker, and ingestion pipeline
15. [ ] Write integration tests against running gRPC services
16. [ ] Create Dockerfile

---

## 9. Key Design Decisions

| Decision                        | Choice                 | Rationale                                              |
|---------------------------------|------------------------|--------------------------------------------------------|
| Ingestion in gateway vs. separate service | In gateway   | Simplicity — ingestion is infrequent, no need for a separate service |
| Streaming protocol              | SSE                    | Simpler than WebSockets, sufficient for server→client streaming |
| Document registry               | SQLite                 | Lightweight, no extra infra, only used for listing/deletion |
| Multi-tenancy                   | Collection per team    | Qdrant natively supports collection isolation          |
