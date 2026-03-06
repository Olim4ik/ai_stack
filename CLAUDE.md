# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered internal knowledge base assistant using RAG with agentic workflows. Python 3.12 microservices communicate via gRPC, with a FastAPI REST/SSE gateway and Vue.js frontend. LangGraph orchestrates query classification, multi-step reasoning, and tool execution via MCP.

## Commands

### Full stack
```bash
docker compose up --build          # Start everything
docker compose up qdrant embedding retrieval  # Backend infra only
```

### Proto generation
```bash
bash scripts/generate_protos.sh    # Generate gRPC stubs into services/{name}/src/generated/
```

### Tests
```bash
pytest tests/unit/                          # All unit tests
pytest tests/unit/test_embedding_providers.py  # Single test file
pytest tests/integration/                   # Integration tests (requires running services)
python tests/eval/run_eval.py               # RAG evaluation
locust -f tests/load/locustfile.py          # Load tests
```

### Frontend
```bash
cd frontend
npm install
npm run dev       # Dev server with API proxy to localhost:8000
npm run build     # Production build
```

## Architecture

**Request flow:** Vue.js (:8080/nginx) → FastAPI Gateway (:8000) → LangGraph Agent (:50053) → Retrieval Service (:50051) / MCP Server (stdio)

**Internal communication:** All backend services use gRPC with shared proto definitions in `proto/`. Each service generates its own stubs into `src/generated/`. The gateway is the only service exposing REST/SSE.

**Key architectural patterns:**
- **Query routing:** Agent classifies queries as simple/multi-step/action, then routes through different LangGraph graph paths
- **Human-in-the-loop:** MCP tools annotated with `requires_confirmation: True` trigger graph interruption; gateway sends SSE `confirm_required` event; user confirms via `POST /api/chat/{id}/confirm`
- **Multi-tenancy:** One Qdrant collection per team (`team_{name}`), filtered at query time
- **Document ingestion** runs inside the gateway process: parse → chunk (512 tokens, 50 overlap) → embed via gRPC → store in Qdrant
- **Embedding provider pattern:** Abstract base + factory, swappable via `EMBEDDING_MODEL_PROVIDER` env var (openai or sentence_transformers)

## Service Layout

Each service follows the same structure: `services/{name}/Dockerfile`, `requirements.txt`, `src/main.py`, `src/config.py` (pydantic-settings), `src/service.py`.

**Startup dependency order:** qdrant → embedding → retrieval → mcp-server → agent → gateway → frontend

## Conventions

- **Config:** pydantic-settings loading from env vars (see `.env.example`)
- **Logging:** structlog with JSON output
- **gRPC:** async via `grpc.aio`, health checks on all services
- **Linting:** ruff
- **Testing:** pytest
- **Error handling:** gRPC status codes internally, HTTP status codes at the gateway; MCP tools return structured error objects, never raise exceptions

## Agent & Workflow Guidance

### When to use parallel agents

This is a microservices repo — many tasks touch multiple services independently. Use parallel agents when:

- **Proto changes:** Updating a `.proto` file affects multiple services. After editing the proto, launch parallel agents to update each consuming service's generated stubs, service implementation, and tests simultaneously.
- **Cross-service feature work:** e.g., adding a new search filter requires changes in retrieval (filter logic), gateway (API parameter), and frontend (UI control) — these can be developed in parallel agents since each service is self-contained.
- **Bulk test runs:** Run unit tests for different services in parallel background agents: `pytest tests/unit/test_embedding_providers.py`, `pytest tests/unit/test_retrieval_filters.py`, etc.

### When to use background agents

- **Run tests in background** while continuing to code. After making changes to a service, launch a background agent to run its tests and keep working on the next service.
- **Explore agent in background** to research how a feature works across services while you edit a specific file in the foreground.

### Common multi-step workflows

**Adding a new MCP tool** (touches 3 services):

1. Add tool function in `services/mcp_server/src/tools/` with schema and annotations
2. Update agent's tool handling if the tool needs human-in-the-loop (`services/agent/src/nodes/execute.py`)
3. Add SSE event handling in gateway if new event types are needed
4. Add unit test in `tests/unit/test_mcp_tools.py`

Steps 1-3 can run as parallel agents since they edit different services.

**Changing a proto definition** (cascading change):

1. Edit the `.proto` file in `proto/`
2. Run `bash scripts/generate_protos.sh` to regenerate stubs
3. Update the gRPC service implementation in the owning service
4. Update all gRPC clients in consuming services
5. Update affected tests

Step 1-2 are sequential. Steps 3-5 can run as parallel agents per service.

**Adding a new REST endpoint** (gateway-focused):

1. Add Pydantic models in `services/gateway/src/models/`
2. Add route handler in `services/gateway/src/routers/`
3. If it calls a gRPC service, update or create client in `services/gateway/src/grpc_clients/`
4. Add frontend API call in `frontend/src/api/client.ts` and wire up the view

Steps 1-3 are sequential (same service). Step 4 can be a parallel agent.

### Explore agent usage

Use the Explore agent for cross-service investigations:
- "Trace how a chat message flows from the frontend through all services to Qdrant and back"
- "Find all places that reference a specific proto message type across services"
- "How does human-in-the-loop confirmation propagate from MCP tool annotation to the frontend dialog"

### What NOT to parallelize

- Changes within a single service — these often have interdependencies (config → service → route)
- Docker Compose changes — single file, do it directly
- Frontend component work — Vue components often import each other, handle sequentially
