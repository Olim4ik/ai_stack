# Implementation Plan — Internal Knowledge Base / DevOps Assistant

## Overview

This document is the master implementation plan for the Internal Knowledge Base / DevOps Assistant. It describes the overall architecture, service boundaries, build order, and links to the detailed plan for each service.

See [Functional Requirements](functional_requirements.md) for the full list of requirements this plan implements.

---

## Service Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        Docker Compose                          │
 │                                                                │
 │  ┌──────────┐    REST/SSE    ┌───────────────┐                 │
 │  │  Vue.js  │ ─────────────► │   FastAPI     │                 │
 │  │ (nginx)  │                │   Gateway     │                 │
 │  │ :8080    │                │   :8000       │                 │
 │  └──────────┘                └───────┬───────┘                 │
 │                                      │                         │
 │                         ┌────────────┼────────────┐            │
 │                         │            │            │            │
 │                         ▼            │            ▼            │
 │               ┌──────────────┐       │   ┌──────────────┐     │
 │               │  LangGraph   │       │   │   Document   │     │
 │               │    Agent     │       │   │   Ingestion  │     │
 │               │   :50053     │       │   │  (in gateway)│     │
 │               └──────┬───┬──┘       │   └──────┬───────┘     │
 │                      │   │          │          │              │
 │              ┌───────┘   └──────┐   │          │              │
 │              ▼                  ▼   │          │              │
 │     ┌──────────────┐  ┌──────────┐ │          │              │
 │     │  MCP Server  │  │ Retrieval│ │          │              │
 │     │   (stdio)    │  │ Service  │◄┘          │              │
 │     └──────┬───────┘  │  :50051  │            │              │
 │            │          └────┬─────┘            │              │
 │        External            │                  │              │
 │         APIs               ▼                  ▼              │
 │     (Jira, GitHub,   ┌──────────┐      ┌──────────┐         │
 │      PagerDuty)      │  Qdrant  │      │ Embedding│         │
 │                       │  :6333   │      │ Service  │         │
 │                       └──────────┘      │  :50052  │         │
 │                                         └──────────┘         │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Services

| # | Service            | Type           | Port  | Detailed Plan                                          |
|---|--------------------|----------------|-------|--------------------------------------------------------|
| 1 | FastAPI Gateway    | REST / SSE     | 8000  | [plan_fastapi_gateway.md](plan_fastapi_gateway.md)     |
| 2 | LangGraph Agent    | gRPC           | 50053 | [plan_langgraph_agent.md](plan_langgraph_agent.md)     |
| 3 | Embedding Service  | gRPC           | 50052 | [plan_embedding_service.md](plan_embedding_service.md) |
| 4 | Retrieval Service  | gRPC           | 50051 | [plan_retrieval_service.md](plan_retrieval_service.md) |
| 5 | MCP Server         | stdio / SSE    | —     | [plan_mcp_server.md](plan_mcp_server.md)               |
| 6 | Qdrant + Vue.js    | Infrastructure | 6333 / 8080 | [plan_infrastructure.md](plan_infrastructure.md) |

---

## Build Order & Dependencies

The services must be built in dependency order — lower layers first, consumers last.

```
Phase 1 (Week 1)     Phase 2 (Week 2)      Phase 3 (Week 3-4)     Phase 4 (Week 5)     Phase 6 (Week 6)
─────────────────    ──────────────────    ────────────────────    ─────────────────    ─────────────────
 Qdrant (infra)       Retrieval Service     LangGraph Agent         Vue.js Frontend      Eval & Logging
 Embedding Service    FastAPI Gateway        MCP Server              Docker Compose       Load Testing
                      Ingestion Pipeline     Human-in-the-loop       full orchestration
```

### Dependency Graph

```
Vue.js Frontend
    └── FastAPI Gateway
            ├── LangGraph Agent
            │       ├── Retrieval Service
            │       │       ├── Qdrant
            │       │       └── Embedding Service
            │       └── MCP Server
            │               └── External APIs
            └── Document Ingestion
                    ├── Qdrant
                    └── Embedding Service
```

---

## Project Structure

```
knowledge-base-assistant/
├── docker-compose.yml
├── .env.example
├── proto/                          # Shared protobuf definitions
│   ├── embedding.proto
│   ├── retrieval.proto
│   └── agent.proto
├── services/
│   ├── gateway/                    # FastAPI Gateway
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── main.py
│   │       ├── routers/
│   │       ├── ingestion/
│   │       └── grpc_clients/
│   ├── agent/                      # LangGraph Agent
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── main.py
│   │       ├── graph/
│   │       ├── nodes/
│   │       └── state.py
│   ├── embedding/                  # Embedding gRPC Service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── main.py
│   │       └── service.py
│   ├── retrieval/                  # Retrieval gRPC Service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── main.py
│   │       └── service.py
│   └── mcp_server/                 # MCP Tool Server
│       ├── Dockerfile
│       ├── requirements.txt
│       └── src/
│           ├── main.py
│           └── tools/
│               ├── jira.py
│               ├── github.py
│               └── pagerduty.py
├── frontend/                       # Vue.js Client
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── src/
│       ├── App.vue
│       ├── views/
│       └── components/
├── tests/
│   ├── integration/
│   └── load/
└── docs/
    └── knowledge_base/
        ├── functional_requirements.md
        └── implementation_plan.md
```

---

## Shared Conventions

- **Language**: Python 3.12 for all backend services
- **Package Manager**: pip with `requirements.txt` (or Poetry if preferred)
- **Protobuf**: Shared `.proto` files in `proto/`, each service generates its own stubs
- **Config**: Environment variables via `.env`, loaded with `pydantic-settings`
- **Logging**: `structlog` with JSON output across all services
- **Error Handling**: gRPC status codes internally, HTTP status codes at the gateway
- **Testing**: `pytest` for unit/integration tests, `locust` for load tests
- **Linting**: `ruff` for formatting and linting

---

## Environment Variables

| Variable                  | Used By               | Description                          |
|---------------------------|-----------------------|--------------------------------------|
| `QDRANT_HOST`             | Retrieval, Gateway    | Qdrant server hostname               |
| `QDRANT_PORT`             | Retrieval, Gateway    | Qdrant gRPC/HTTP port                |
| `EMBEDDING_SERVICE_HOST`  | Retrieval, Gateway    | Embedding service gRPC address       |
| `RETRIEVAL_SERVICE_HOST`  | Agent, Gateway        | Retrieval service gRPC address       |
| `AGENT_SERVICE_HOST`      | Gateway               | Agent service gRPC address           |
| `ANTHROPIC_API_KEY`       | Agent                 | Claude API key                       |
| `OPENAI_API_KEY`          | Embedding             | OpenAI embeddings API key (optional) |
| `JIRA_API_TOKEN`          | MCP Server            | Jira API credentials                 |
| `GITHUB_TOKEN`            | MCP Server            | GitHub personal access token         |
| `PAGERDUTY_API_KEY`       | MCP Server            | PagerDuty API key                    |
