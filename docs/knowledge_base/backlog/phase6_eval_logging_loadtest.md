# Backlog — Eval, Logging & Load Testing

**Phase**: 6
**Service**: Cross-cutting concerns
**Plan**: [implementation_plan.md](../implementation_plan.md)

---

## Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Create RAG evaluation framework (retrieval precision) | DONE | `tests/eval/metrics.py` — precision@K, recall@K |
| 2 | Add answer faithfulness metric (grounded in retrieved context) | DONE | `tests/eval/metrics.py` — sentence-level overlap checking |
| 3 | Add answer relevance metric (answers the actual question) | DONE | `tests/eval/metrics.py` — keyword overlap with stop-word filtering |
| 4 | Create eval dataset with sample Q&A pairs | DONE | `tests/eval/dataset.py` — 8 Q&A pairs covering runbooks, incidents, architecture |
| 5 | Create eval runner script | DONE | `tests/eval/run_eval.py` — mock + live modes, JSON report output |
| 6 | Add request-level structured logging to Agent service | DONE | `service.py` — timing, session tracking, query metadata |
| 7 | Add request-level structured logging to Retrieval service | DONE | `service.py` — embed_ms, search_ms, total_ms, result_count, trace_id |
| 8 | Add log correlation across services (trace ID propagation) | DONE | Gateway middleware generates x-trace-id, propagated in response headers |
| 9 | Create Locust load test for chat endpoint | DONE | `tests/load/locustfile.py` — ChatUser with SSE stream consumption |
| 10 | Create Locust load test for document ingestion | DONE | `tests/load/locustfile.py` — DocumentUser with file upload |
| 11 | Create Locust load test for search endpoint | DONE | `tests/load/locustfile.py` — MixedUser combining chat + docs |
| 12 | Add load test configuration (target: 10 concurrent sessions) | DONE | `tests/load/locust.conf` — 10 users, spawn-rate 2, 5min run |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-04 | All 12 tasks complete. RAG eval framework with 4 metrics (precision@K, recall@K, faithfulness, relevance), 8-sample dataset, mock + live eval runner with JSON report. Structured logging with trace ID propagation (gateway to retrieval/agent). Locust load tests with 3 user types (ChatUser, DocumentUser, MixedUser), targeting NFR-5 (10 concurrent sessions). |
