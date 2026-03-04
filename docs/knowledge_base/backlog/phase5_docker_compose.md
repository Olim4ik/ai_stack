# Backlog — Docker Compose Full Orchestration

**Phase**: 5
**Service**: Docker Compose + Frontend Deployment
**Plan**: [plan_infrastructure.md](../plan_infrastructure.md) — Part C

---

## Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Create frontend Dockerfile (multi-stage: node build + nginx) | DONE | `frontend/Dockerfile` — node:20-alpine build, nginx:alpine serve |
| 2 | Create nginx.conf with SPA fallback and API proxy (SSE support) | DONE | `frontend/nginx.conf` — proxy_buffering off, proxy_cache off for SSE |
| 3 | Update gateway chat.py — SSE streaming via Agent Service gRPC | DONE | `routers/chat.py` — EventSourceResponse, fallback retrieval if agent unavailable |
| 4 | Add Agent Service gRPC client to gateway | DONE | `grpc_clients/agent.py` — Chat + ResumeChat streaming |
| 5 | Update gateway lifespan to connect to Agent Service | DONE | `main.py` — AgentClient connect/close in lifespan |
| 6 | Add frontend service to docker-compose.yml | DONE | depends_on gateway (service_healthy) |
| 7 | Add healthcheck directives to all services in compose | DONE | gRPC channel checks for backend, curl for gateway/frontend |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-04 | All 7 tasks complete. Full docker-compose.yml with 7 services (qdrant, embedding, retrieval, mcp-server, agent, gateway, frontend). All services have healthchecks. Dependency chain uses condition: service_healthy for ordered startup. Gateway upgraded from Phase 2 stub to full SSE streaming via Agent gRPC client. Frontend served via nginx with API proxy and SSE support. |
