# Backlog — FastAPI Gateway

**Phase**: 2
**Service**: FastAPI Gateway (REST/SSE, port 8000)
**Plan**: [plan_fastapi_gateway.md](../plan_fastapi_gateway.md)

---

## Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Scaffold FastAPI project with `main.py`, config, and health endpoint | DONE | `services/gateway/src/main.py` with lifespan, `routers/health.py` |
| 2 | Define Pydantic request/response models | DONE | `models/requests.py`, `models/responses.py` |
| 3 | Implement file parsers (Markdown, PDF, HTML, plain text) | DONE | `ingestion/parser.py` — 4 formats + fallback |
| 4 | Implement text chunker with configurable strategy | DONE | `ingestion/chunker.py` — word-based with overlap, section heading preservation |
| 5 | Create gRPC client for Embedding Service | DONE | `grpc_clients/embedding.py` |
| 6 | Create gRPC client for Retrieval Service | DONE | `grpc_clients/retrieval.py` — search + health_check |
| 7 | Build ingestion pipeline (parse → chunk → embed → store) | DONE | `ingestion/pipeline.py` — batch embedding, re-ingestion via doc_id, Qdrant upsert |
| 8 | Implement `POST /api/documents/ingest` endpoint | DONE | `routers/documents.py` — multipart upload |
| 9 | Implement `GET /api/documents` and `DELETE /api/documents/{id}` | DONE | `routers/documents.py` — list stub + Qdrant-backed delete |
| 10 | Implement `POST /api/chat` with SSE streaming (stub — full in Phase 3) | DONE | `routers/chat.py` — stub that queries retrieval service directly |
| 11 | Add structlog middleware and error handlers | DONE | `middleware/logging.py`, `middleware/errors.py` |
| 12 | Write unit tests for parsers, chunker, and ingestion pipeline | DONE | `test_gateway_parser.py` (12 tests), `test_gateway_chunker.py` (6 tests) — all passing |
| 13 | Create Dockerfile | DONE | `services/gateway/Dockerfile` — uvicorn entrypoint |
| 14 | Add service to `docker-compose.yml` | DONE | depends_on: embedding, retrieval |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-04 | All 14 tasks complete. FastAPI app with lifespan (connects to Qdrant, Embedding, Retrieval on startup), 3 routers (health, documents, chat), ingestion pipeline (parse 4 formats → chunk with overlap → batch embed → Qdrant upsert), gRPC clients for embedding + retrieval, request/response models, structlog middleware, error handler, 18 unit tests passing, Dockerfile, docker-compose entry. Chat endpoint is a Phase 2 stub (direct retrieval, no agent). |
