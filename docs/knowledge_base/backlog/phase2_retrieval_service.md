# Backlog — Retrieval Service

**Phase**: 2
**Service**: Retrieval Service (gRPC, port 50051)
**Plan**: [plan_retrieval_service.md](../plan_retrieval_service.md)

---

## Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Define `retrieval.proto` with `Search`, `CreateCollection`, `DeleteCollection` RPCs | DONE | `proto/retrieval.proto` |
| 2 | Generate Python gRPC stubs | DONE | `services/retrieval/src/generated/` (retrieval + embedding stubs) |
| 3 | Implement Qdrant client wrapper (connection, collection management) | DONE | `services/retrieval/src/qdrant_client.py` — QdrantManager with async client |
| 4 | Implement dense search — embed query, search Qdrant, return ranked results | DONE | `services/retrieval/src/search/dense.py` |
| 5 | Implement metadata filter builder (translate gRPC filters to Qdrant filters) | DONE | `services/retrieval/src/search/filters.py` — supports eq, in, gte, lte, range with DatetimeRange |
| 6 | Implement hybrid search with RRF fusion | DONE | `services/retrieval/src/search/hybrid.py` — Qdrant prefetch + RRF |
| 7 | Implement `RetrievalService` gRPC servicer | DONE | `services/retrieval/src/service.py` — Search, CreateCollection, DeleteCollection, HealthCheck |
| 8 | Create gRPC client for Embedding Service | DONE | `services/retrieval/src/grpc_clients/embedding.py` |
| 9 | Implement pydantic-settings configuration | DONE | `services/retrieval/src/config.py` |
| 10 | Add gRPC health checking with Qdrant connectivity check | DONE | In `main.py` + QdrantManager.is_healthy() |
| 11 | Add structured logging | DONE | structlog JSON across all modules |
| 12 | Write unit tests (mock Qdrant client) | DONE | `tests/unit/test_retrieval_filters.py` — 9 tests passing |
| 13 | Create Dockerfile | DONE | `services/retrieval/Dockerfile` |
| 14 | Add service to `docker-compose.yml` | DONE | depends_on: qdrant, embedding |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-04 | All 14 tasks complete. Proto defined, gRPC stubs generated, QdrantManager with collection CRUD, dense + hybrid search, filter builder with DatetimeRange for date fields, gRPC servicer, embedding client, config, health checks, logging, 9 unit tests passing, Dockerfile, docker-compose entry. |
