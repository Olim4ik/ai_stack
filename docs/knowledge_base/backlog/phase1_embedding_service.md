# Backlog — Embedding Service

**Phase**: 1
**Service**: Embedding Service (gRPC, port 50052)
**Plan**: [plan_embedding_service.md](../plan_embedding_service.md)

---

## Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Define `embedding.proto` with `Embed`, `EmbedBatch`, `GetModelInfo` RPCs | DONE | `proto/embedding.proto` |
| 2 | Generate Python gRPC stubs from proto | DONE | `services/embedding/src/generated/` via `scripts/generate_protos.sh` |
| 3 | Implement `EmbeddingProvider` abstract base class | DONE | `services/embedding/src/providers/base.py` |
| 4 | Implement `OpenAIProvider` — calls OpenAI embeddings API | DONE | `services/embedding/src/providers/openai_provider.py` |
| 5 | Implement `SentenceTransformersProvider` — local model inference | DONE | `services/embedding/src/providers/st_provider.py` — lazy import |
| 6 | Implement provider factory based on config | DONE | `services/embedding/src/providers/factory.py` |
| 7 | Implement `EmbeddingService` gRPC servicer | DONE | `services/embedding/src/service.py` |
| 8 | Implement pydantic-settings configuration | DONE | `services/embedding/src/config.py` |
| 9 | Add gRPC health checking | DONE | In `services/embedding/src/main.py` via `grpc_health` |
| 10 | Add structured logging with structlog | DONE | JSON output configured in `main.py` |
| 11 | Write unit tests (mock providers) | DONE | `tests/unit/test_embedding_providers.py` — 8/8 passing |
| 12 | Create Dockerfile | DONE | `services/embedding/Dockerfile` |
| 13 | Add service to `docker-compose.yml` | DONE | `docker-compose.yml` |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-04 | Phase 1 complete. All 13 tasks done. Proto defined, stubs generated, providers implemented (OpenAI + SentenceTransformers with lazy import), gRPC servicer with health checking, structured logging, config, Dockerfile, docker-compose entry, and 8 unit tests all passing. |
